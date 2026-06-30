# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import os

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)

SPARSE_DECODE_RETRIEVE_TOKENS = int(
    os.environ.get("LMCACHE_SPARSE_DECODE_RETRIEVE_TOKENS", "2048")
)


def _dsa_debug_enabled() -> bool:
    return os.environ.get("VLLM_ASCEND_DSA_SHRINK_DEBUG", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _dsa_debug_summary_enabled() -> bool:
    return os.environ.get(
        "VLLM_ASCEND_DSA_SHRINK_DEBUG_MODE", "fail_only"
    ).lower() in ("summary", "trace", "verbose", "all")


def _dsa_debug_limit() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_ASCEND_DSA_SHRINK_DEBUG_LIMIT", "8")))
    except ValueError:
        return 8


def _dsa_debug_should_log(owner: Any, site: str) -> bool:
    if not _dsa_debug_enabled():
        return False
    if not _dsa_debug_summary_enabled():
        return False
    counts = getattr(owner, "_dsa_shrink_debug_counts", None)
    if counts is None:
        counts = {}
        setattr(owner, "_dsa_shrink_debug_counts", counts)
    count = counts.get(site, 0)
    if count >= _dsa_debug_limit():
        return False
    counts[site] = count + 1
    return True


def _dsa_debug_shape(value: Any) -> Any:
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(shape)
    try:
        return len(value)
    except TypeError:
        return type(value).__name__


def _dsa_debug_sample(value: Any, limit: Optional[int] = None) -> Any:
    if value is None:
        return None
    limit = _dsa_debug_limit() if limit is None else limit
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return []
            return value.detach().reshape(-1)[:limit].to(device="cpu").tolist()
        return list(value[:limit])
    except Exception as exc:
        return f"{type(value).__name__}:sample_failed:{exc}"


def _dsa_debug_minmax_count(value: Any) -> Any:
    if value is None:
        return None
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            flat = value.detach().reshape(-1)
            return (
                flat.min().to(device="cpu").item(),
                flat.max().to(device="cpu").item(),
                int(flat.numel()),
            )
        seq = list(value)
        if not seq:
            return None
        return (min(seq), max(seq), len(seq))
    except Exception as exc:
        return f"{type(value).__name__}:minmax_failed:{exc}"


def _sparse_slot_mapping_len(prompt_tokens: int) -> int:
    return min(SPARSE_DECODE_RETRIEVE_TOKENS, prompt_tokens)


def _build_slot_mapping(
    block_ids: list[int], block_size: int, num_tokens: int
) -> torch.Tensor:
    if num_tokens <= 0:
        return torch.empty(0, dtype=torch.long)
    num_blocks = utils.cdiv(num_tokens, block_size)
    block_ids_t = torch.tensor(block_ids[:num_blocks], dtype=torch.long)
    block_offsets = torch.arange(0, block_size, dtype=torch.long)
    slots = (
        block_offsets.reshape((1, block_size))
        + block_ids_t.reshape((num_blocks, 1)) * block_size
    ).flatten()
    return slots[:num_tokens]


def _dsa_decode_window_tokens() -> int:
    try:
        return max(
            0,
            int(os.environ.get("VLLM_ASCEND_DSA_DECODE_WINDOW_TOKENS", "1024")),
        )
    except ValueError:
        return 1024


def _dsa_decode_lmcache_len(prompt_len: int, input_token_len: int) -> int:
    window_tokens = _dsa_decode_window_tokens()
    decoded_tokens = max(0, input_token_len - prompt_len)
    if window_tokens <= 0 or decoded_tokens <= 0:
        return prompt_len
    return prompt_len + ((decoded_tokens - 1) // window_tokens) * window_tokens


def _dsa_decode_window_should_save(prompt_len: int, input_token_len: int) -> bool:
    window_tokens = _dsa_decode_window_tokens()
    decoded_tokens = input_token_len - prompt_len
    return window_tokens > 0 and decoded_tokens > 0 and decoded_tokens % window_tokens == 0


def _build_dsa_decode_window_slot_mapping(
    block_ids: list[int],
    block_size: int,
    num_tokens: int,
    prompt_len: int,
) -> torch.Tensor:
    if num_tokens <= 0:
        return torch.empty(0, dtype=torch.long)
    window_tokens = _dsa_decode_window_tokens()
    if window_tokens <= 0 or num_tokens <= prompt_len:
        return _build_slot_mapping(block_ids, block_size, num_tokens)

    positions = torch.arange(num_tokens, dtype=torch.long)
    logical_positions = positions.clone()
    decode_mask = positions >= int(prompt_len)
    logical_positions[decode_mask] = int(prompt_len) + torch.remainder(
        positions[decode_mask] - int(prompt_len),
        int(window_tokens),
    )
    logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
    offsets = torch.remainder(logical_positions, block_size)
    if block_ids:
        max_block_index = len(block_ids) - 1
        logical_blocks = torch.clamp(logical_blocks, min=0, max=max_block_index)
        block_ids_t = torch.tensor(block_ids, dtype=torch.long)
        physical_blocks = block_ids_t.index_select(0, logical_blocks)
    else:
        physical_blocks = torch.zeros_like(logical_blocks)
    return physical_blocks * block_size + offsets


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    # key of cached object
    cached_keys: list[list] = field(default_factory=list)
    cached_starts: list[int] = field(default_factory=list)
    cached_ends: list[int] = field(default_factory=list)
    cached_memory_objs: list[list] = field(default_factory=list)
    cached_tensors: list[list] = field(default_factory=list)
    # Sparse decode only: NPU device ptr per cached chunk, parallel to cached_tensors.
    cached_chunk_dev_ptrs: list[list[int]] = field(default_factory=list)
    # Sparse decode only: prebuilt NPU tensor of chunk device ptrs, one entry per layer.
    cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = field(default_factory=list)
    # Sparse decode only: prompt token ids for retrieve keys, built once.
    sparse_token_ids: list[int] = field(default_factory=list, repr=False)
    # Sparse decode only: single-element list holding CPU then NPU slot_mapping.
    sparse_slot_mapping: list[torch.Tensor] = field(default_factory=list, repr=False)
    # Sparse decode only: reused across decode steps to avoid per-step allocation.
    sparse_decode_token_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    sparse_decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            # If input is a list, flatten it to handle potential nesting.
            # This also correctly processes already-flat lists.
            new_block_ids = [
                i
                for elem in new_block_ids
                for i in (elem if isinstance(elem, list) else [elem])
            ]
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            self.sparse_token_ids.clear()
            self.sparse_slot_mapping.clear()
            self.sparse_decode_token_mask = None
            self.sparse_decode_ret_mask = None
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class WorkerRetrieveState:
    """Worker-local retrieve cache; survives scheduler/worker IPC each decode step."""

    cached_keys: list[list] = field(default_factory=list)
    cached_starts: list[int] = field(default_factory=list)
    cached_ends: list[int] = field(default_factory=list)
    cached_memory_objs: list[list] = field(default_factory=list)
    cached_tensors: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = field(default_factory=list)
    location: Optional[str] = None
    metadata_warm: bool = False
    token_count: int = 0


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Single-element list; sparse decode reuses tracker.sparse_slot_mapping by reference.
    slot_mapping: list[torch.Tensor] = field(default_factory=list)

    # key of cached object
    cached_keys: list[list] = field(default_factory=list)
    cached_starts: list[int] = field(default_factory=list)
    cached_ends: list[int] = field(default_factory=list)
    cached_memory_objs: list[list] = field(default_factory=list)
    cached_tensors: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = field(default_factory=list)
    # Sparse decode only: shared with RequestTracker, reused across decode steps.
    decode_token_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)

    # Set by scheduler when a cached request resumes after preemption.
    resumed_from_preemption: bool = False

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Whether is sparse attention and decode or not
    is_sparse_decode: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        is_sparse_decode: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)
        force_decode_window_save = is_sparse_decode and _dsa_decode_window_should_save(
            tracker.prompt_len,
            input_token_len,
        )

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (
                not force_decode_window_save
                and tracker.num_saved_tokens > 0
                and input_token_len < chunk_boundary
            )
            or (
                tracker.is_decode_phase
                and not save_decode_cache
                and not force_decode_window_save
            )
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if force_decode_window_save:
            num_tokens_to_save = input_token_len
        elif not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        if is_sparse_decode and load_spec is not None and skip_save:
            if (
                not tracker.sparse_token_ids
                or len(tracker.sparse_token_ids) != load_spec.lmcache_cached_tokens
            ):
                tracker.sparse_token_ids = input_token_ids[
                    : load_spec.lmcache_cached_tokens
                ]
                if tracker.mm_hashes:
                    token_ids_tensor = torch.tensor(tracker.sparse_token_ids)
                    assert tracker.mm_positions is not None, (
                        "tracker got mm_hashes but no mm_positions"
                    )
                    apply_mm_hashes_to_token_ids(
                        token_ids_tensor, tracker.mm_hashes, tracker.mm_positions
                    )
                    tracker.sparse_token_ids = token_ids_tensor.tolist()
            token_ids = tracker.sparse_token_ids
        else:
            token_ids = input_token_ids[:num_tokens_to_save]

            # If the request has multimodal hashes, apply them to the token ids
            if tracker.mm_hashes:
                # TODO: Optimize this
                token_ids = torch.tensor(token_ids)
                assert tracker.mm_positions is not None, (
                    "tracker got mm_hashes but no mm_positions"
                )
                apply_mm_hashes_to_token_ids(
                    token_ids, tracker.mm_hashes, tracker.mm_positions
                )
                token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size and not is_sparse_decode:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        if is_sparse_decode and load_spec is not None and skip_save:
            num_slots = _sparse_slot_mapping_len(load_spec.lmcache_cached_tokens)
            if (
                not tracker.sparse_slot_mapping
                or int(tracker.sparse_slot_mapping[0].numel()) != num_slots
            ):
                tracker.sparse_slot_mapping.clear()
                tracker.sparse_slot_mapping.append(
                    _build_slot_mapping(
                        tracker.allocated_block_ids, block_size, num_slots
                    )
                )
            slot_mapping = tracker.sparse_slot_mapping
        elif is_sparse_decode and not skip_save:
            slot_mapping = [
                _build_dsa_decode_window_slot_mapping(
                    tracker.allocated_block_ids,
                    block_size,
                    len(token_ids),
                    tracker.prompt_len,
                )
            ]
        else:
            slot_mapping = [
                _build_slot_mapping(
                    tracker.allocated_block_ids, block_size, len(token_ids)
                )
            ]

        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        decode_token_mask: Optional[torch.Tensor] = None
        decode_ret_mask: Optional[torch.Tensor] = None
        if is_sparse_decode and load_spec is not None:
            num_retrieve_tokens = len(token_ids)
            if (
                tracker.sparse_decode_token_mask is None
                or tracker.sparse_decode_token_mask.numel() != num_retrieve_tokens
            ):
                tracker.sparse_decode_token_mask = torch.ones(
                    num_retrieve_tokens, dtype=torch.bool
                )
            if (
                tracker.sparse_decode_ret_mask is None
                or tracker.sparse_decode_ret_mask.numel() != num_retrieve_tokens
            ):
                tracker.sparse_decode_ret_mask = torch.zeros(
                    num_retrieve_tokens, dtype=torch.bool, device="cpu"
                )
            decode_token_mask = tracker.sparse_decode_token_mask
            decode_ret_mask = tracker.sparse_decode_ret_mask

        if is_sparse_decode and load_spec is not None and _dsa_debug_should_log(
            tracker, "reqmeta_sparse"
        ):
            logger.warning(
                "[DSA_SHRINK_CHECK] lmcache_reqmeta_sparse req=%s "
                "input_token_len=%s token_ids_len=%s skip_save=%s "
                "num_blocks=%s block_size=%s vllm_cached=%s "
                "lmcache_cached=%s can_load=%s slot_mapping_shape=%s "
                "slot_mapping_sample=%s slot_mapping_minmax_count=%s "
                "decode_token_mask_shape=%s decode_ret_mask_shape=%s",
                tracker.req_id,
                input_token_len,
                len(token_ids),
                skip_save,
                num_blocks,
                block_size,
                load_spec.vllm_cached_tokens,
                load_spec.lmcache_cached_tokens,
                load_spec.can_load,
                _dsa_debug_shape(slot_mapping[0] if slot_mapping else None),
                _dsa_debug_sample(slot_mapping[0] if slot_mapping else None),
                _dsa_debug_minmax_count(slot_mapping[0] if slot_mapping else None),
                _dsa_debug_shape(decode_token_mask),
                _dsa_debug_shape(decode_ret_mask),
            )

        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            is_sparse_decode=is_sparse_decode,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            cached_keys=tracker.cached_keys,
            cached_starts=tracker.cached_starts,
            cached_ends=tracker.cached_ends,
            cached_memory_objs=tracker.cached_memory_objs,
            cached_tensors=tracker.cached_tensors,
            cached_chunk_dev_ptrs=tracker.cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu=tracker.cached_chunk_ptrs_npu,
            decode_token_mask=decode_token_mask,
            decode_ret_mask=decode_ret_mask,
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.enable_sparse_attention = config.enable_sparse_attention

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._kvcaches_list: list[torch.Tensor] = []
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
            and not self.enable_sparse_attention
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        if role != KVConnectorRole.SCHEDULER:
            self._worker_retrieve_state: dict[str, WorkerRetrieveState] = {}

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self):
        """Setup metrics for monitoring data structures in the connector."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "connector metrics will not be collected"
            )
            return

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    def _build_kv_layer_groups(self):
        # Build KV layer groups structure if not already built
        if self.lmcache_engine is not None:
            assert len(self.kv_caches) > 0
            kv_layer_groups_manager = (
                self.lmcache_engine.metadata.kv_layer_groups_manager
            )
            kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches)

    def _refresh_kvcaches_list(self) -> None:
        self._kvcaches_list = list(self.kv_caches.values())

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

        self._build_kv_layer_groups()
        self._refresh_kvcaches_list()

    ####################
    # Worker side APIs
    ####################
    @staticmethod
    def _load_tokens_for_retrieve(
        tokens: list[int], lmcache_cached_tokens: int, *, is_sparse_decode: bool
    ) -> list[int]:
        """Return token ids that can be retrieved from LMCache."""
        if is_sparse_decode:
            return tokens[:lmcache_cached_tokens]
        if lmcache_cached_tokens >= len(tokens):
            return tokens
        return tokens[:lmcache_cached_tokens]

    @staticmethod
    def _load_token_mask_for_retrieve(
        request: "ReqMeta",
        token_count: int,
        lmcache_chunk_size: int,
    ) -> torch.Tensor:
        """Build or reuse the token mask for a retrieve call."""
        if request.is_sparse_decode and request.decode_token_mask is not None:
            mask = request.decode_token_mask
            if mask.numel() == token_count:
                return mask

        token_mask = torch.ones(token_count, dtype=torch.bool)
        if request.load_spec is not None:
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // lmcache_chunk_size
                * lmcache_chunk_size
            )
            if masked_token_count:
                token_mask[:masked_token_count] = False

        if request.is_sparse_decode:
            request.decode_token_mask = token_mask
        return token_mask

    def _drain_layerwise_retrievers(self) -> None:
        """Finish suspended layerwise generators to avoid GC cost on reset."""
        if _dsa_debug_enabled():
            logger.warning(
                "[DSA_LOAD_DBG] lmcache_drain_retrievers enter step=%s "
                "current_layer=%s num_layers=%s retrievers=%s",
                getattr(self, "_dsa_forward_step", None),
                getattr(self, "current_layer", None),
                getattr(self, "num_layers", None),
                len(self.layerwise_retrievers),
            )
        for idx, retriever in enumerate(self.layerwise_retrievers):
            try:
                if _dsa_debug_enabled():
                    logger.warning(
                        "[DSA_LOAD_DBG] lmcache_drain_retrievers drain step=%s "
                        "idx=%s current_layer=%s num_layers=%s retrievers=%s",
                        getattr(self, "_dsa_forward_step", None),
                        idx,
                        getattr(self, "current_layer", None),
                        getattr(self, "num_layers", None),
                        len(self.layerwise_retrievers),
                    )
                if self.current_layer < self.num_layers:
                    retriever.close()
                else:
                    next(retriever)
            except StopIteration:
                if _dsa_debug_enabled():
                    logger.warning(
                        "[DSA_LOAD_DBG] lmcache_drain_retrievers stopped step=%s "
                        "idx=%s current_layer=%s num_layers=%s",
                        getattr(self, "_dsa_forward_step", None),
                        idx,
                        getattr(self, "current_layer", None),
                        getattr(self, "num_layers", None),
                    )
                pass
            except Exception:
                if _dsa_debug_enabled():
                    logger.exception(
                        "[DSA_LOAD_DBG] lmcache_drain_retrievers exception step=%s "
                        "idx=%s current_layer=%s num_layers=%s retrievers=%s",
                        getattr(self, "_dsa_forward_step", None),
                        idx,
                        getattr(self, "current_layer", None),
                        getattr(self, "num_layers", None),
                        len(self.layerwise_retrievers),
                    )
                raise
        self.layerwise_retrievers.clear()
        if _dsa_debug_enabled():
            logger.warning(
                "[DSA_LOAD_DBG] lmcache_drain_retrievers cleared step=%s "
                "current_layer=%s num_layers=%s retrievers=%s",
                getattr(self, "_dsa_forward_step", None),
                getattr(self, "current_layer", None),
                getattr(self, "num_layers", None),
                len(self.layerwise_retrievers),
            )

    def _should_defer_lookup_unpin_for_sparse_decode(self, request: ReqMeta) -> bool:
        """Keep lookup pins across decode steps while sparse retrieve is active."""
        return (
            getattr(request, "is_sparse_decode", False)
            and request.load_spec is not None
            and request.load_spec.can_load
        )

    def _release_request_lookup_pins(self, req_id: str) -> None:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return
        engine = manager.lmcache_engine
        if engine is not None:
            engine.lookup_unpin(req_id)

    def _maybe_lookup_unpin_for_request(self, request: ReqMeta) -> None:
        if self._should_defer_lookup_unpin_for_sparse_decode(request):
            return
        self._release_request_lookup_pins(request.req_id)

    def _prune_worker_retrieve_state(self, active_req_ids: set[str]) -> None:
        if not hasattr(self, "_worker_retrieve_state"):
            return
        dropped_req_ids = set(self._worker_retrieve_state) - active_req_ids
        for req_id in dropped_req_ids:
            self._release_request_lookup_pins(req_id)
        self._worker_retrieve_state = {
            req_id: state
            for req_id, state in self._worker_retrieve_state.items()
            if req_id in active_req_ids
        }

    def _drop_worker_retrieve_state(self, req_id: str) -> None:
        if hasattr(self, "_worker_retrieve_state"):
            self._worker_retrieve_state.pop(req_id, None)
        self._release_request_lookup_pins(req_id)

    def _should_invalidate_worker_retrieve_state(
        self, request: ReqMeta, token_count: int
    ) -> bool:
        if request.resumed_from_preemption:
            return True
        state = self._worker_retrieve_state.get(request.req_id)
        if state is None:
            return False
        if state.cached_ends and token_count < state.cached_ends[-1]:
            return True
        return False

    def _bind_worker_retrieve_state_to_request(
        self, request: ReqMeta
    ) -> Optional[WorkerRetrieveState]:
        state = self._worker_retrieve_state.get(request.req_id)
        if state is None or not (state.metadata_warm or state.cached_keys):
            return None
        request.cached_keys = state.cached_keys
        request.cached_starts = state.cached_starts
        request.cached_ends = state.cached_ends
        request.cached_memory_objs = state.cached_memory_objs
        request.cached_tensors = state.cached_tensors
        request.cached_chunk_dev_ptrs = state.cached_chunk_dev_ptrs
        request.cached_chunk_ptrs_npu = state.cached_chunk_ptrs_npu
        return state

    def _save_worker_retrieve_state_from_request(
        self,
        request: ReqMeta,
        *,
        location: Optional[str],
        metadata_warm: bool,
        token_count: int,
    ) -> None:
        if not hasattr(self, "_worker_retrieve_state"):
            return
        if not metadata_warm and not request.cached_keys:
            return
        self._worker_retrieve_state[request.req_id] = WorkerRetrieveState(
            cached_keys=request.cached_keys,
            cached_starts=request.cached_starts,
            cached_ends=request.cached_ends,
            cached_memory_objs=request.cached_memory_objs,
            cached_tensors=request.cached_tensors,
            cached_chunk_dev_ptrs=request.cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu=request.cached_chunk_ptrs_npu,
            location=location,
            metadata_warm=metadata_warm,
            token_count=token_count,
        )

    def _finalize_worker_retrieve_state_from_metadata(
        self, metadata: LMCacheConnectorMetadata
    ) -> None:
        if not hasattr(self, "_worker_retrieve_state"):
            return
        for request in metadata.requests:
            if not request.is_sparse_decode:
                continue
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            if not request.cached_keys:
                continue
            existing = self._worker_retrieve_state.get(request.req_id)
            location = existing.location if existing is not None else None
            metadata_warm = (
                existing.metadata_warm if existing is not None else True
            )
            self._save_worker_retrieve_state_from_request(
                request,
                location=location,
                metadata_warm=metadata_warm or bool(request.cached_keys),
                token_count=len(request.token_ids),
            )

    def _sparse_decode_retrieve_warm_kwargs(
        self,
        request: ReqMeta,
        token_count: int,
        bound_state: Optional[WorkerRetrieveState],
    ) -> dict[str, Any]:
        warm_kwargs: dict[str, Any] = {}
        if bound_state is None:
            return warm_kwargs
        if bound_state.location is not None:
            warm_kwargs["cached_retrieve_location"] = bound_state.location
        if (
            bound_state.metadata_warm
            and bound_state.cached_keys
            and bound_state.cached_ends
            and token_count <= bound_state.cached_ends[-1]
        ):
            warm_kwargs["_retrieve_metadata_warm"] = True
        return warm_kwargs

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._build_kv_layer_groups()
        self._refresh_kvcaches_list()
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self.current_layer = 0
        self._dsa_forward_step = getattr(self, "_dsa_forward_step", 0) + 1

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        active_req_ids = {req.req_id for req in metadata.requests}
        self._prune_worker_retrieve_state(active_req_ids)

        assert len(self.kv_caches) > 0
        if not self._kvcaches_list:
            self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_list

        attn_metadata = forward_context.attn_metadata
        if _dsa_debug_should_log(self, "start_load_kv"):
            request_summaries = []
            for request in metadata.requests[: _dsa_debug_limit()]:
                load_spec = request.load_spec
                save_spec = request.save_spec
                request_summaries.append(
                    {
                        "req_id": request.req_id,
                        "sparse": request.is_sparse_decode,
                        "token_ids": len(request.token_ids),
                        "load": None
                        if load_spec is None
                        else {
                            "can_load": load_spec.can_load,
                            "vllm_cached": load_spec.vllm_cached_tokens,
                            "lmcache_cached": load_spec.lmcache_cached_tokens,
                        },
                        "save": None
                        if save_spec is None
                        else {
                            "can_save": save_spec.can_save,
                            "skip_leading": save_spec.skip_leading_tokens,
                        },
                    }
                )
            logger.warning(
                "[DSA_LOAD_DBG] lmcache_start_load metadata_requests=%s "
                "step=%s impl_id=%s parent_id=%s metadata_id=%s "
                "parent_metadata_id=%s active_req_ids=%s attn_metadata=%s "
                "attn_metadata_id=%s use_layerwise=%s kv_caches=%s "
                "current_layer=%s requests=%s",
                len(metadata.requests),
                self._dsa_forward_step,
                id(self),
                id(self._parent),
                id(metadata),
                id(self._parent._connector_metadata)
                if self._parent._connector_metadata is not None else None,
                sorted(active_req_ids),
                attn_metadata.__class__.__name__ if attn_metadata is not None else None,
                id(attn_metadata) if attn_metadata is not None else None,
                self.use_layerwise,
                len(self.kv_caches),
                self.current_layer,
                request_summaries,
            )
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self._drain_layerwise_retrievers()

        last_idx = -1
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is not None and request.load_spec.can_load:
                last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                continue

            tokens = request.token_ids
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            assert request.slot_mapping

            if request.is_sparse_decode:
                if request.slot_mapping[0].device.type != torch.device(
                    self.device
                ).type:
                    request.slot_mapping[0] = request.slot_mapping[0].to(
                        device=self.device, dtype=torch.long
                    )
                slot_mapping = request.slot_mapping[0]
            else:
                slot_mapping = request.slot_mapping[0].to(
                    device=self.device, dtype=torch.long
                )

            if not request.is_sparse_decode:
                assert len(tokens) == len(slot_mapping)

            retrieve_tokens = self._load_tokens_for_retrieve(
                tokens,
                lmcache_cached_tokens,
                is_sparse_decode=request.is_sparse_decode,
            )
            token_count = len(retrieve_tokens)
            token_mask = self._load_token_mask_for_retrieve(
                request, token_count, self._lmcache_chunk_size
            )

            if self.use_layerwise or request.is_sparse_decode:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        retrieve_tokens,
                        token_mask,
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping,
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                elif request.is_sparse_decode:
                    if hasattr(self, "_worker_retrieve_state"):
                        if self._should_invalidate_worker_retrieve_state(
                            request, token_count
                        ):
                            self._drop_worker_retrieve_state(request.req_id)
                        bound_state = self._bind_worker_retrieve_state_to_request(
                            request
                        )
                    else:
                        bound_state = None

                    retrieve_kwargs: dict[str, Any] = {
                        "kvcaches": kvcaches,
                        "slot_mapping": slot_mapping,
                        "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                        "lmcache_cached_tokens": request.load_spec.lmcache_cached_tokens,
                        "sync": sync,
                        "cached_keys": request.cached_keys,
                        "cached_starts": request.cached_starts,
                        "cached_ends": request.cached_ends,
                        "cached_memory_objs": request.cached_memory_objs,
                        "cached_tensors": request.cached_tensors,
                        "cached_chunk_dev_ptrs": request.cached_chunk_dev_ptrs,
                        "cached_chunk_ptrs_npu": request.cached_chunk_ptrs_npu,
                        "req_id": request.req_id,
                    }
                    retrieve_kwargs.update(
                        self._sparse_decode_retrieve_warm_kwargs(
                            request, token_count, bound_state
                        )
                    )
                    if request.decode_ret_mask is not None:
                        retrieve_kwargs["ret_mask"] = request.decode_ret_mask

                    if _dsa_debug_should_log(self, "start_sparse_decode"):
                        logger.warning(
                            "[DSA_SHRINK_CHECK] lmcache_start_sparse_decode "
                            "req=%s idx=%s last_idx=%s sync=%s "
                            "tokens_len=%s retrieve_token_count=%s "
                            "token_mask_shape=%s token_mask_minmax_count=%s "
                            "slot_mapping_shape=%s slot_mapping_sample=%s "
                            "slot_mapping_minmax_count=%s vllm_cached=%s "
                            "lmcache_cached=%s cached_keys=%s cached_starts=%s "
                            "cached_ends=%s cached_tensors_layers=%s "
                            "cached_chunk_ptr_layers=%s metadata_warm=%s "
                            "bound_state=%s ret_mask_shape=%s",
                            request.req_id,
                            idx,
                            last_idx,
                            sync,
                            len(tokens),
                            token_count,
                            _dsa_debug_shape(token_mask),
                            _dsa_debug_minmax_count(token_mask),
                            _dsa_debug_shape(slot_mapping),
                            _dsa_debug_sample(slot_mapping),
                            _dsa_debug_minmax_count(slot_mapping),
                            request.load_spec.vllm_cached_tokens,
                            request.load_spec.lmcache_cached_tokens,
                            len(request.cached_keys)
                            if request.cached_keys is not None else None,
                            len(request.cached_starts)
                            if request.cached_starts is not None else None,
                            len(request.cached_ends)
                            if request.cached_ends is not None else None,
                            len(request.cached_tensors)
                            if request.cached_tensors is not None else None,
                            len(request.cached_chunk_ptrs_npu)
                            if request.cached_chunk_ptrs_npu is not None else None,
                            bool(retrieve_kwargs.get("_retrieve_metadata_warm")),
                            bound_state is not None,
                            _dsa_debug_shape(request.decode_ret_mask),
                        )

                    layerwise_retriever = (
                        self.lmcache_engine.retrieve_layer_head_token_wise(
                            retrieve_tokens,
                            token_mask,
                            **retrieve_kwargs,
                        )
                    )
                    # NOTE: retrieve layers one by one with cpu prefetch
                    next(layerwise_retriever)
                    location = retrieve_kwargs.get("cached_retrieve_location")
                    metadata_warm = bool(
                        retrieve_kwargs.get("_retrieve_metadata_warm")
                        or request.cached_keys
                    )
                    self._save_worker_retrieve_state_from_request(
                        request,
                        location=location,
                        metadata_warm=metadata_warm,
                        token_count=token_count,
                    )
                    self.layerwise_retrievers.append(layerwise_retriever)
                else:
                    retrieve_slot_mapping = (
                        slot_mapping
                        if lmcache_cached_tokens >= len(slot_mapping)
                        else slot_mapping[:lmcache_cached_tokens]
                    )
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        retrieve_tokens,
                        token_mask,
                        kvcaches=kvcaches,
                        slot_mapping=retrieve_slot_mapping,
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                retrieve_slot_mapping = (
                    slot_mapping
                    if lmcache_cached_tokens >= len(slot_mapping)
                    else slot_mapping[:lmcache_cached_tokens]
                )
                ret_token_mask = self.lmcache_engine.retrieve(
                    retrieve_tokens,
                    token_mask,
                    kvcaches=kvcaches,
                    slot_mapping=retrieve_slot_mapping,
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask,
                        ret_token_mask,
                        retrieve_slot_mapping,
                    )
                    self._invalid_block_ids.update(missing_blocks)

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(
        self,
        layer_name: str,
        selected_tokens: list = None,
        token_start_index: list = None,
        request_ids: list = None,
    ) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
            selected_tokens: batched sparse token indices per decode request.
            token_start_index: per-request start offset into slot_mapping.
            request_ids: req_id for each selected_tokens row (input_batch order).
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)
        if not self.layerwise_retrievers:
            return

        if _dsa_debug_enabled():
            logger.warning(
                "[DSA_SHRINK_CHECK] lmcache_wait_layer_enter step=%s "
                "layer=%s current_layer=%s num_layers=%s retrievers=%s "
                "metadata_id=%s selected_shape=%s token_start_index=%s "
                "request_ids=%s",
                getattr(self, "_dsa_forward_step", None),
                layer_name,
                self.current_layer,
                self.num_layers,
                len(self.layerwise_retrievers),
                id(metadata),
                _dsa_debug_shape(selected_tokens),
                _dsa_debug_sample(token_start_index),
                _dsa_debug_sample(request_ids),
            )

        row_of_req = (
            {rid: row for row, rid in enumerate(request_ids)}
            if request_ids is not None
            else None
        )
        trace_wait = _dsa_debug_should_log(self, "wait_sparse_decode")
        if trace_wait:
            sparse_reqs = [
                request.req_id for request in metadata.requests
                if request.is_sparse_decode
            ]
            logger.warning(
                "[DSA_SHRINK_CHECK] lmcache_wait_selected layer=%s "
                "current_layer=%s retrievers=%s metadata_reqs=%s "
                "sparse_reqs=%s selected_shape=%s selected_sample=%s "
                "selected_minmax_count=%s token_start_index=%s "
                "request_ids=%s row_of_req=%s",
                layer_name,
                self.current_layer,
                len(self.layerwise_retrievers),
                len(metadata.requests),
                sparse_reqs[:_dsa_debug_limit()],
                _dsa_debug_shape(selected_tokens),
                _dsa_debug_sample(selected_tokens),
                _dsa_debug_minmax_count(selected_tokens),
                _dsa_debug_sample(token_start_index),
                _dsa_debug_sample(request_ids),
                dict(list(row_of_req.items())[:_dsa_debug_limit()])
                if row_of_req is not None else None,
            )

        idx = 0
        decode_row = 0
        for request in metadata.requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
            if idx >= len(self.layerwise_retrievers):
                logger.warning(
                    "wait_for_layer_load: missing retriever for request %s "
                    "(idx=%d, retrievers=%d)",
                    request.req_id,
                    idx,
                    len(self.layerwise_retrievers),
                )
                break
            layerwise_retriever = self.layerwise_retrievers[idx]
            if request.is_sparse_decode:
                if selected_tokens is None:
                    selected_tokens_per_req = None
                    token_start_index_per_req = 0
                else:
                    if row_of_req is not None and request.req_id not in row_of_req:
                        raise RuntimeError(
                            "[DSA_SHRINK_CHECK] lmcache_wait_row_missing "
                            f"layer={layer_name} req={request.req_id} "
                            f"request_ids={_dsa_debug_sample(request_ids)} "
                            f"sparse_decode_row={decode_row}"
                        )
                    row = (
                        row_of_req[request.req_id]
                        if row_of_req is not None
                        else decode_row
                    )
                    selected_rows = (
                        int(selected_tokens.shape[0])
                        if hasattr(selected_tokens, "shape")
                        and len(selected_tokens.shape) > 0
                        else len(selected_tokens)
                    )
                    if row >= selected_rows:
                        raise RuntimeError(
                            "[DSA_SHRINK_CHECK] lmcache_wait_row_oob "
                            f"layer={layer_name} req={request.req_id} "
                            f"row={row} selected_rows={selected_rows} "
                            f"request_ids={_dsa_debug_sample(request_ids)}"
                        )
                    selected_tokens_per_req = selected_tokens[row]
                    token_start_index_per_req = (
                        0 if token_start_index is None else token_start_index[row]
                    )
                if trace_wait:
                    logger.warning(
                        "[DSA_SHRINK_CHECK] lmcache_wait_send_sparse "
                        "layer=%s req=%s idx=%s decode_row=%s row=%s "
                        "selected_shape=%s selected_sample=%s "
                        "selected_minmax_count=%s token_start_index=%s "
                        "slot_mapping_shape=%s slot_mapping_sample=%s "
                        "slot_mapping_minmax_count=%s vllm_cached=%s "
                        "lmcache_cached=%s",
                        layer_name,
                        request.req_id,
                        idx,
                        decode_row,
                        None if selected_tokens is None else row,
                        _dsa_debug_shape(selected_tokens_per_req),
                        _dsa_debug_sample(selected_tokens_per_req),
                        _dsa_debug_minmax_count(selected_tokens_per_req),
                        token_start_index_per_req,
                        _dsa_debug_shape(
                            request.slot_mapping[0] if request.slot_mapping else None
                        ),
                        _dsa_debug_sample(
                            request.slot_mapping[0] if request.slot_mapping else None
                        ),
                        _dsa_debug_minmax_count(
                            request.slot_mapping[0] if request.slot_mapping else None
                        ),
                        request.load_spec.vllm_cached_tokens,
                        request.load_spec.lmcache_cached_tokens,
                    )
                ret_token_mask = layerwise_retriever.send(
                    (selected_tokens_per_req, token_start_index_per_req)
                )
                if _dsa_debug_enabled():
                    logger.warning(
                        "[DSA_SHRINK_CHECK] lmcache_wait_send_sparse_done step=%s "
                        "layer=%s req=%s idx=%s current_layer=%s num_layers=%s "
                        "retrievers=%s ret_mask_shape=%s",
                        getattr(self, "_dsa_forward_step", None),
                        layer_name,
                        request.req_id,
                        idx,
                        self.current_layer,
                        self.num_layers,
                        len(self.layerwise_retrievers),
                        _dsa_debug_shape(ret_token_mask),
                    )
                decode_row += 1
            else:
                ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1 and not request.is_sparse_decode:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")
            idx += 1

        if self.layerwise_retrievers:
            if _dsa_debug_enabled():
                logger.warning(
                    "[DSA_SHRINK_CHECK] lmcache_wait_layer_advance step=%s "
                    "layer=%s current_layer_before=%s num_layers=%s retrievers=%s",
                    getattr(self, "_dsa_forward_step", None),
                    layer_name,
                    self.current_layer,
                    self.num_layers,
                    len(self.layerwise_retrievers),
                )
            self.current_layer += 1
            if _dsa_debug_enabled():
                logger.warning(
                    "[DSA_SHRINK_CHECK] lmcache_wait_layer_advanced step=%s "
                    "layer=%s current_layer_after=%s num_layers=%s retrievers=%s",
                    getattr(self, "_dsa_forward_step", None),
                    layer_name,
                    self.current_layer,
                    self.num_layers,
                    len(self.layerwise_retrievers),
                )
            if self.current_layer >= self.num_layers:
                if _dsa_debug_enabled():
                    logger.warning(
                        "[DSA_SHRINK_CHECK] lmcache_wait_layer_finalize step=%s "
                        "layer=%s current_layer=%s num_layers=%s retrievers=%s",
                        getattr(self, "_dsa_forward_step", None),
                        layer_name,
                        self.current_layer,
                        self.num_layers,
                        len(self.layerwise_retrievers),
                    )
                self._finalize_worker_retrieve_state_from_metadata(metadata)
                self._drain_layerwise_retrievers()

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if _dsa_debug_should_log(self, "save_kv_layer"):
            request_summaries = []
            for request in connector_metadata.requests[: _dsa_debug_limit()]:
                save_spec = request.save_spec
                request_summaries.append(
                    {
                        "req_id": request.req_id,
                        "sparse": request.is_sparse_decode,
                        "token_ids": len(request.token_ids),
                        "slot_mapping": _dsa_debug_shape(request.slot_mapping[0])
                        if request.slot_mapping else None,
                        "can_save": save_spec.can_save
                        if save_spec is not None else None,
                        "skip_leading": save_spec.skip_leading_tokens
                        if save_spec is not None else None,
                    }
                )
            logger.warning(
                "[DSA_STORE_DBG] lmcache_save_kv_layer enter layer=%s "
                "step=%s impl_id=%s parent_id=%s metadata_id=%s "
                "parent_metadata_id=%s attn_metadata_id=%s kv_role=%s "
                "use_layerwise=%s requests=%s",
                layer_name,
                getattr(self, "_dsa_forward_step", None),
                id(self),
                id(self._parent),
                id(connector_metadata),
                id(self._parent._connector_metadata)
                if self._parent._connector_metadata is not None else None,
                id(attn_metadata) if attn_metadata is not None else None,
                self.kv_role,
                self.use_layerwise,
                request_summaries,
            )

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        is_first = True

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                token_ids = request.token_ids
                assert isinstance(token_ids, list)
                assert request.slot_mapping is not None and len(request.slot_mapping) > 0
                if request.is_sparse_decode:
                    if request.slot_mapping[0].device.type != torch.device(
                        self.device
                    ).type:
                        request.slot_mapping[0] = request.slot_mapping[0].to(
                            device=self.device, dtype=torch.long
                        )
                    slot_mapping = request.slot_mapping[0]
                else:
                    slot_mapping = request.slot_mapping[0].to(
                        device=self.device, dtype=torch.long
                    )

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    if not request.is_sparse_decode:
                        skip_leading_tokens = (
                            skip_leading_tokens
                            // self._lmcache_chunk_size
                            * self._lmcache_chunk_size
                        )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                if request.is_sparse_decode and _dsa_debug_should_log(
                    self, "save_kv_layer_sparse_store"
                ):
                    logger.warning(
                        "[DSA_STORE_DBG] lmcache_save_kv_layer sparse_store "
                        "layer=%s req=%s token_ids=%s skip_leading=%s "
                        "store_tokens=%s slot_mapping=%s",
                        layer_name,
                        request.req_id,
                        len(token_ids),
                        skip_leading_tokens,
                        len(token_ids) - skip_leading_tokens,
                        _dsa_debug_shape(slot_mapping),
                    )

                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                    cached_keys = request.cached_keys,
                    cached_starts=request.cached_starts,
                    cached_ends=request.cached_ends,
                    cached_memory_objs=request.cached_memory_objs,
                    cached_tensors=request.cached_tensors,
                    is_sparse_decode=request.is_sparse_decode,
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            next(layerwise_storer)

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if _dsa_debug_should_log(self, "wait_for_save"):
            request_summaries = []
            for request in connector_metadata.requests[: _dsa_debug_limit()]:
                save_spec = request.save_spec
                request_summaries.append(
                    {
                        "req_id": request.req_id,
                        "sparse": request.is_sparse_decode,
                        "token_ids": len(request.token_ids),
                        "slot_mapping": _dsa_debug_shape(request.slot_mapping[0])
                        if request.slot_mapping else None,
                        "can_save": save_spec.can_save
                        if save_spec is not None else None,
                        "skip_leading": save_spec.skip_leading_tokens
                        if save_spec is not None else None,
                    }
                )
            logger.warning(
                "[DSA_STORE_DBG] lmcache_wait_for_save enter step=%s "
                "impl_id=%s parent_id=%s metadata_id=%s parent_metadata_id=%s "
                "kv_role=%s use_layerwise=%s requests=%s",
                getattr(self, "_dsa_forward_step", None),
                id(self),
                id(self._parent),
                id(connector_metadata),
                id(self._parent._connector_metadata)
                if self._parent._connector_metadata is not None else None,
                self.kv_role,
                self.use_layerwise,
                request_summaries,
            )

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self._maybe_lookup_unpin_for_request(request)

            return

        if self.use_layerwise:
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    next(layerwise_storer)
                self._maybe_lookup_unpin_for_request(request)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        for request in connector_metadata.requests:
            self._maybe_lookup_unpin_for_request(request)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            assert request.slot_mapping
            if request.is_sparse_decode:
                if request.slot_mapping[0].device.type != torch.device(
                    self.device
                ).type:
                    request.slot_mapping[0] = request.slot_mapping[0].to(
                        device=self.device, dtype=torch.long
                    )
                slot_mapping = request.slot_mapping[0]
            else:
                slot_mapping = request.slot_mapping[0].to(
                    device=self.device, dtype=torch.long
                )

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            if not request.is_sparse_decode:
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        # lookup_client is always initialized for scheduler role
        assert self.lookup_client is not None

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        assert self.lookup_client is not None
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
            )
            if req_meta is not None:
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )

                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                )
                if req_meta is not None:
                    req_meta.resumed_from_preemption = req.resumed_from_preemption
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                assert request.num_computed_tokens == expected, (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    f"but expected {expected} "
                    f"(full_hit_adj={full_hit_adj})"
                )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )

            is_sparse_decode = self.enable_sparse_attention and (request.num_computed_tokens > request.num_prompt_tokens)
            if is_sparse_decode:
                lmcache_cached_tokens = _dsa_decode_lmcache_len(
                    request.num_prompt_tokens,
                    len(request_tracker.token_ids),
                )
                load_spec = LoadSpec(
                    vllm_cached_tokens=0,
                    lmcache_cached_tokens=min(
                        lmcache_cached_tokens,
                        len(request_tracker.token_ids),
                    ),
                    can_load=True,
                )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
                is_sparse_decode=is_sparse_decode
            )
            if req_meta is not None:
                req_meta.resumed_from_preemption = preempted
                if is_sparse_decode and _dsa_debug_should_log(
                    self, "build_connector_meta_sparse"
                ):
                    save_spec = req_meta.save_spec
                    load_spec_dbg = req_meta.load_spec
                    logger.warning(
                        "[DSA_META_DBG] build_connector_meta_sparse req=%s "
                        "in_unfinished=%s num_new_tokens=%s new_block_ids=%s "
                        "num_current_tokens=%s tracker_len=%s new_token_ids_len=%s "
                        "token_ids_len=%s load_can_load=%s vllm_cached=%s "
                        "lmcache_cached=%s save_can_save=%s "
                        "save_skip_leading=%s resumed=%s",
                        req_id,
                        req_id in self._unfinished_requests,
                        num_new_tokens,
                        new_block_ids,
                        num_current_tokens,
                        tracker_len,
                        len(new_token_ids),
                        len(req_meta.token_ids),
                        load_spec_dbg.can_load if load_spec_dbg is not None else None,
                        load_spec_dbg.vllm_cached_tokens
                        if load_spec_dbg is not None else None,
                        load_spec_dbg.lmcache_cached_tokens
                        if load_spec_dbg is not None else None,
                        save_spec.can_save if save_spec is not None else None,
                        save_spec.skip_leading_tokens
                        if save_spec is not None else None,
                        preempted,
                    )
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._layerwise_save_storers.pop(request.request_id, None)

        self._drop_worker_retrieve_state(request.request_id)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED and self.async_loading:
            # Cancel any ongoing async lookup and prefetch tasks on workers
            lookup_id = request.request_id
            assert self.lookup_client is not None
            self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
