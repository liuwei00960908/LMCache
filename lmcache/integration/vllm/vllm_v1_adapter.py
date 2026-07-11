# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
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
    calculate_draft_layers,
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

_DSA_PROF = os.getenv("VLLM_ASCEND_DSA_PROF", "0") == "1"

SPARSE_DECODE_RETRIEVE_TOKENS = int(
    os.environ.get("LMCACHE_SPARSE_DECODE_RETRIEVE_TOKENS", "2048")
)
SPARSE_DECODE_SHARED_CPU_PHASE = "sparse_decode_bootstrap"


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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def _sparse_slot_mapping_len(prompt_tokens: int) -> int:
    return min(SPARSE_DECODE_RETRIEVE_TOKENS, prompt_tokens)


def _am_get(attn_metadata, key, default=None):
    """Read a field from attn_metadata that may be a dict or an object."""
    if isinstance(attn_metadata, dict):
        return attn_metadata.get(key, default)
    return getattr(attn_metadata, key, default)


def _ensure_list_attr(obj: Any, name: str) -> list:
    value = getattr(obj, name, None)
    if value is None:
        value = []
        try:
            setattr(obj, name, value)
        except Exception:
            pass
    return value


def _retrieve_cache_kwargs(
    obj: Any,
    *,
    kv_group: int,
    dsa_two_groups: bool,
) -> dict[str, Any]:
    """Return per-group cached retrieve/store kwargs for two-group DSA."""
    if dsa_two_groups and kv_group == 1:
        return {
            "cached_keys": _ensure_list_attr(obj, "cached_keys_indexer"),
            "cached_starts": _ensure_list_attr(obj, "cached_starts_indexer"),
            "cached_ends": _ensure_list_attr(obj, "cached_ends_indexer"),
            "cached_memory_objs": _ensure_list_attr(
                obj, "cached_memory_objs_indexer"
            ),
            "cached_tensors": _ensure_list_attr(obj, "cached_tensors_indexer"),
            "cached_chunk_dev_ptrs": _ensure_list_attr(
                obj, "cached_chunk_dev_ptrs_indexer"
            ),
            "cached_chunk_ptrs_npu": _ensure_list_attr(
                obj, "cached_chunk_ptrs_npu_indexer"
            ),
            "cached_shared_handles": _ensure_list_attr(
                obj, "cached_shared_handles_indexer"
            ),
        }
    return {
        "cached_keys": _ensure_list_attr(obj, "cached_keys"),
        "cached_starts": _ensure_list_attr(obj, "cached_starts"),
        "cached_ends": _ensure_list_attr(obj, "cached_ends"),
        "cached_memory_objs": _ensure_list_attr(obj, "cached_memory_objs"),
        "cached_tensors": _ensure_list_attr(obj, "cached_tensors"),
        "cached_chunk_dev_ptrs": _ensure_list_attr(
            obj, "cached_chunk_dev_ptrs"
        ),
        "cached_chunk_ptrs_npu": _ensure_list_attr(
            obj, "cached_chunk_ptrs_npu"
        ),
        "cached_shared_handles": _ensure_list_attr(obj, "cached_shared_handles"),
    }


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


def _dsa_payload_event_list(payload_event: Any) -> list[Any]:
    if payload_event is None:
        return []
    if isinstance(payload_event, (list, tuple)):
        return [event for event in payload_event if event is not None]
    return [payload_event]


def _dsa_wait_payload_event(payload_event: Any) -> None:
    payload_events = _dsa_payload_event_list(payload_event)
    if not payload_events:
        return
    if not (hasattr(torch, "npu") and hasattr(torch.npu, "current_stream")):
        raise RuntimeError(
            "DSA selected-token payload event was provided, but torch.npu "
            "stream support is unavailable."
        )
    try:
        current_stream = torch.npu.current_stream()
        for event in payload_events:
            current_stream.wait_event(event)
    except Exception as exc:
        raise RuntimeError(
            "Failed to wait on DSA selected-token producer event before "
            "LMCache row selection."
        ) from exc


def _dsa_device_tensor_types(value: Any) -> set[str]:
    if isinstance(value, torch.Tensor):
        return set() if value.device.type == "cpu" else {value.device.type}
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_dsa_device_tensor_types(item))
        return out
    return set()


def _dsa_record_payload_event_if_needed(*values: Any) -> Optional[Any]:
    device_types: set[str] = set()
    for value in values:
        device_types.update(_dsa_device_tensor_types(value))
    if not device_types:
        return None
    needs_npu_event = bool(device_types & {"npu", "privateuseone"})
    needs_cuda_event = "cuda" in device_types
    if needs_npu_event and needs_cuda_event:
        raise RuntimeError(
            "DSA payload contains both NPU and CUDA tensors; refusing to "
            "record a single ordering event for mixed device backends."
        )
    if needs_npu_event:
        if not (hasattr(torch, "npu") and hasattr(torch.npu, "Event")):
            raise RuntimeError(
                "DSA reordered payload contains NPU tensors but torch.npu.Event "
                "is unavailable."
            )
        event = torch.npu.Event()
        event.record(torch.npu.current_stream())
        return event
    if needs_cuda_event:
        if not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            raise RuntimeError(
                "DSA reordered payload contains CUDA tensors but CUDA stream "
                "support is unavailable."
            )
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream())
        return event
    raise RuntimeError(
        "DSA reordered payload contains device tensors with unsupported "
        f"device types {sorted(device_types)}."
    )


def _row_select(value: Any, rows: list[int]):
    if hasattr(value, "__getitem__"):
        if isinstance(value, torch.Tensor):
            if len(rows) == 1:
                row = rows[0]
                return value[row]
            return value[rows]
        return [value[row] for row in rows]
    raise TypeError(f"Unsupported row-indexed value type: {type(value)!r}")


def _single_row_select(value: Any, row: int):
    if hasattr(value, "__getitem__"):
        if isinstance(value, torch.Tensor):
            return value[row]
        return value[row]
    raise TypeError(f"Unsupported row-indexed value type: {type(value)!r}")


def _sparse_payload_value(value: Any):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, torch.Tensor):
                out.extend(item.reshape(-1).tolist())
            elif isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out
    return value


def _flatten_block_ids(block_ids) -> list[int]:
    if block_ids is None:
        return []
    flattened: list[int] = []
    for elem in block_ids:
        if isinstance(elem, (list, tuple)):
            flattened.extend(elem)
        else:
            flattened.append(elem)
    return flattened


def _split_kv_group_block_ids(block_ids) -> tuple[list[int], Optional[list[int]]]:
    """Return latent and optional indexer block ids from vLLM block metadata."""
    if block_ids is None or len(block_ids) == 0:
        return [], None
    first = block_ids[0]
    if isinstance(first, (list, tuple)):
        latent_block_ids = _flatten_block_ids(block_ids[0])
        indexer_block_ids = (
            _flatten_block_ids(block_ids[1]) if len(block_ids) > 1 else None
        )
        return latent_block_ids, indexer_block_ids
    return _flatten_block_ids(block_ids), None


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
    # Whether to save the latent (MLA) group (kv_group=0).
    # Defaults to True for backward compat. When dsa_two_groups is enabled,
    # the indexer group (kv_group=1) can be independently gated.
    can_save_latent: bool = True
    # Whether to save the indexer (DSA) group (kv_group=1).
    can_save_indexer: bool = False


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
    allocated_block_ids_indexer: Optional[list[int]] = None

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
    cached_shared_handles: list[list[Any]] = field(default_factory=list)
    # Two-group DSA: separate sparse/prefill retrieve cache for kv_group=1 (indexer).
    cached_keys_indexer: list[list] = field(default_factory=list)
    cached_starts_indexer: list[int] = field(default_factory=list)
    cached_ends_indexer: list[int] = field(default_factory=list)
    cached_memory_objs_indexer: list[list] = field(default_factory=list)
    cached_tensors_indexer: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs_indexer: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu_indexer: list[Optional[torch.Tensor]] = field(
        default_factory=list
    )
    cached_shared_handles_indexer: list[list[Any]] = field(default_factory=list)
    # Sparse decode only: prompt token ids for retrieve keys, built once.
    sparse_token_ids: list[int] = field(default_factory=list, repr=False)
    # Sparse decode only: single-element list holding CPU then NPU slot_mapping.
    sparse_slot_mapping: list[torch.Tensor] = field(default_factory=list, repr=False)
    sparse_indexer_slot_mapping: list[torch.Tensor] = field(
        default_factory=list, repr=False
    )
    # Sparse decode only: reused across decode steps to avoid per-step allocation.
    sparse_decode_token_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    sparse_decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    # Decode window save only: independent progress for decode-window chunks.
    decode_window_save_next_start: Optional[int] = field(default=None, repr=False)
    # Decode window save only: highest token boundary confirmed readable from
    # LMCache by worker-side completion output.
    decode_window_save_committed_end: int = field(default=0, repr=False)

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

        unfolded_block_ids, indexer_block_ids = _split_kv_group_block_ids(
            new_request.block_ids
        )

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        num_tokens_to_track = min(
            len(new_request.prompt_token_ids),
            max(num_tokens_to_compute, lmcache_cached_tokens),
        )

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_track].copy(),
            allocated_block_ids=unfolded_block_ids,
            allocated_block_ids_indexer=indexer_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
            decode_window_save_committed_end=lmcache_cached_tokens,
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

        if new_block_ids is not None and not isinstance(new_block_ids, (list, tuple)):
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        if new_block_ids is None:
            new_block_ids = []
        new_block_ids, new_indexer_block_ids = _split_kv_group_block_ids(
            new_block_ids
        )

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            self.sparse_token_ids.clear()
            self.sparse_slot_mapping.clear()
            self.sparse_indexer_slot_mapping.clear()
            self.sparse_decode_token_mask = None
            self.sparse_decode_ret_mask = None
            self.cached_keys.clear()
            self.cached_starts.clear()
            self.cached_ends.clear()
            self.cached_memory_objs.clear()
            self.cached_tensors.clear()
            self.cached_chunk_dev_ptrs.clear()
            self.cached_chunk_ptrs_npu.clear()
            self.cached_shared_handles.clear()
            self.cached_keys_indexer.clear()
            self.cached_starts_indexer.clear()
            self.cached_ends_indexer.clear()
            self.cached_memory_objs_indexer.clear()
            self.cached_tensors_indexer.clear()
            self.cached_chunk_dev_ptrs_indexer.clear()
            self.cached_chunk_ptrs_npu_indexer.clear()
            self.cached_shared_handles_indexer.clear()
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            self.allocated_block_ids_indexer = new_indexer_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            self.decode_window_save_committed_end = lmcache_cached_tokens
            self.decode_window_save_next_start = None
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
            if new_indexer_block_ids is not None:
                if self.allocated_block_ids_indexer is None:
                    self.allocated_block_ids_indexer = []
                self.allocated_block_ids_indexer.extend(new_indexer_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True

    def seed_sparse_decode_tokens(
        self,
        token_ids: list[int],
        token_count: Optional[int] = None,
    ) -> None:
        """Seed token ids used to build sparse decode chunk keys."""
        target_count = self.prompt_len if token_count is None else token_count
        sparse_tokens = token_ids[:target_count]
        if len(sparse_tokens) < target_count:
            logger.warning(
                "Request %s sparse decode token source is shorter than target: "
                "source_tokens=%d target_tokens=%d prompt_len=%d",
                self.req_id,
                len(sparse_tokens),
                target_count,
                self.prompt_len,
            )
        if self.mm_hashes:
            token_ids_tensor = torch.tensor(sparse_tokens)
            assert self.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids_tensor, self.mm_hashes, self.mm_positions
            )
            sparse_tokens = token_ids_tensor.tolist()
        self.sparse_token_ids = sparse_tokens


@dataclass
class WorkerRetrieveState:
    """Worker-local retrieve cache; survives scheduler/worker IPC each decode step."""

    req_id: Optional[str] = None
    cached_keys: list[list] = field(default_factory=list)
    cached_starts: list[int] = field(default_factory=list)
    cached_ends: list[int] = field(default_factory=list)
    cached_memory_objs: list[list] = field(default_factory=list)
    cached_tensors: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = field(default_factory=list)
    cached_shared_handles: list[list[Any]] = field(default_factory=list)
    cached_keys_indexer: list[list] = field(default_factory=list)
    cached_starts_indexer: list[int] = field(default_factory=list)
    cached_ends_indexer: list[int] = field(default_factory=list)
    cached_memory_objs_indexer: list[list] = field(default_factory=list)
    cached_tensors_indexer: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs_indexer: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu_indexer: list[Optional[torch.Tensor]] = field(
        default_factory=list
    )
    cached_shared_handles_indexer: list[list[Any]] = field(default_factory=list)
    shared_handles_by_group: dict[int, list[list[Any]]] = field(
        default_factory=dict
    )
    shared_views_by_group: dict[int, list[list[Any]]] = field(default_factory=dict)
    shared_chunk_ptrs_npu_by_group: dict[int, list[Optional[torch.Tensor]]] = field(
        default_factory=dict
    )
    rank0_backing_objs_by_group: dict[int, list[list[Any]]] = field(
        default_factory=dict
    )
    shared_latent_status: str = "missing"
    shared_index_status: str = "missing"
    shared_generation: int = 0
    pointer_cache_generation: int = 0
    shared_request_active: bool = False
    request_scope_token: Optional[str] = None
    shared_validation_signature: Optional[tuple[Any, ...]] = None
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
    indexer_slot_mapping: list[torch.Tensor] = field(default_factory=list)

    # key of cached object
    cached_keys: list[list] = field(default_factory=list)
    cached_starts: list[int] = field(default_factory=list)
    cached_ends: list[int] = field(default_factory=list)
    cached_memory_objs: list[list] = field(default_factory=list)
    cached_tensors: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu: list[Optional[torch.Tensor]] = field(default_factory=list)
    cached_shared_handles: list[list[Any]] = field(default_factory=list)
    cached_keys_indexer: list[list] = field(default_factory=list)
    cached_starts_indexer: list[int] = field(default_factory=list)
    cached_ends_indexer: list[int] = field(default_factory=list)
    cached_memory_objs_indexer: list[list] = field(default_factory=list)
    cached_tensors_indexer: list[list] = field(default_factory=list)
    cached_chunk_dev_ptrs_indexer: list[list[int]] = field(default_factory=list)
    cached_chunk_ptrs_npu_indexer: list[Optional[torch.Tensor]] = field(
        default_factory=list
    )
    cached_shared_handles_indexer: list[list[Any]] = field(default_factory=list)
    # Sparse shared CPU decode only: kv_group=1 was intentionally skipped by
    # config, so hot-path validation may accept absent DSA index state.
    shared_index_skipped: bool = False
    # Sparse decode only: shared with RequestTracker, reused across decode steps.
    decode_token_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    decode_ret_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    # Decode window save metadata, separate from the regular request save progress.
    is_decode_window_save: bool = False
    decode_window_start: Optional[int] = None
    decode_window_end: Optional[int] = None
    decode_window_size: Optional[int] = None

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
    def _token_ids_with_mm_hashes(
        token_ids: list[int],
        tracker: RequestTracker,
    ) -> list[int]:
        if not tracker.mm_hashes:
            return token_ids

        token_ids_tensor = torch.tensor(token_ids)
        assert tracker.mm_positions is not None, (
            "tracker got mm_hashes but no mm_positions"
        )
        apply_mm_hashes_to_token_ids(
            token_ids_tensor, tracker.mm_hashes, tracker.mm_positions
        )
        return token_ids_tensor.tolist()

    @staticmethod
    def from_decode_window_save(
        tracker: RequestTracker,
        block_size: int,
        window_start: int,
        window_end: int,
        window_size: int,
    ) -> Optional["ReqMeta"]:
        """Create isolated metadata for a decode-window save."""
        if window_start < 0 or window_end <= window_start:
            return None
        if window_end > len(tracker.token_ids):
            return None

        token_ids = tracker.token_ids[:window_end].copy()
        token_ids = ReqMeta._token_ids_with_mm_hashes(token_ids, tracker)

        num_blocks = len(tracker.allocated_block_ids)
        if window_end > num_blocks * block_size:
            logger.warning(
                "Skipping decode window save for request %s: "
                "window_end=%d exceeds slot capacity %d",
                tracker.req_id,
                window_end,
                num_blocks * block_size,
            )
            return None

        indexer_slot_mapping: list[torch.Tensor] = []
        if tracker.allocated_block_ids_indexer:
            indexer_num_blocks = len(tracker.allocated_block_ids_indexer)
            if window_end > indexer_num_blocks * block_size:
                logger.warning(
                    "Skipping decode window indexer save for request %s: "
                    "window_end=%d exceeds indexer slot capacity %d",
                    tracker.req_id,
                    window_end,
                    indexer_num_blocks * block_size,
                )
            else:
                indexer_slot_mapping = [
                    _build_slot_mapping(
                        tracker.allocated_block_ids_indexer,
                        block_size,
                        len(token_ids),
                    )
                ]

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=[
                _build_slot_mapping(
                    tracker.allocated_block_ids, block_size, len(token_ids)
                )
            ],
            is_last_prefill=True,
            save_spec=SaveSpec(
                window_start,
                True,
                can_save_latent=True,
                can_save_indexer=bool(indexer_slot_mapping),
            ),
            indexer_slot_mapping=indexer_slot_mapping,
            load_spec=None,
            disagg_spec=None,
            request_configs=tracker.request_configs,
            is_decode_window_save=True,
            decode_window_start=window_start,
            decode_window_end=window_end,
            decode_window_size=window_size,
        )

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        is_sparse_decode: bool = False,
        save_full_chunk_in_decode: bool = False,
        dsa_two_groups: bool = False,
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
        if (
            is_sparse_decode
            and load_spec is not None
            and tracker.sparse_token_ids
        ):
            sparse_token_count = (
                load_spec.lmcache_cached_tokens
                if load_spec.can_load
                else len(tracker.sparse_token_ids)
            )
            if len(tracker.sparse_token_ids) >= sparse_token_count:
                input_token_ids = tracker.sparse_token_ids[:sparse_token_count]
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

        allow_final_prefill_partial_save = (
            is_last_prefill
            and not tracker.is_decode_phase
            and not discard_partial_chunks
            and tracker.num_saved_tokens > 0
            and input_token_len > tracker.num_saved_tokens
            and input_token_len < chunk_boundary
        )
        skip_by_tracker = bool(tracker.skip_save)
        skip_by_chunk_boundary = (
            tracker.num_saved_tokens > 0
            and input_token_len < chunk_boundary
            and not allow_final_prefill_partial_save
        )
        skip_by_decode_phase = bool(tracker.is_decode_phase and not save_decode_cache)
        skip_by_request_config = bool(request_skip)

        skip_save = tracker.disagg_spec is None and (
            skip_by_tracker
            or skip_by_chunk_boundary
            or skip_by_decode_phase
            or skip_by_request_config
        )

        # Decode-full-chunk rule: when save_full_chunk_in_decode is enabled,
        # only save during decode if a full chunk boundary is crossed.
        # This sidesteps the plane-major append cost (each store is a complete
        # tight buffer for exactly chunk_size tokens, no in-place growth).
        # Applied to BOTH latent and indexer groups.
        if (
            tracker.is_decode_phase
            and save_full_chunk_in_decode
            and not skip_save
        ):
            # Only save if we've crossed a full chunk boundary since last save
            new_boundary = (
                (tracker.num_saved_tokens + input_token_len)
                // lmcache_chunk_size * lmcache_chunk_size
            )
            if new_boundary <= tracker.num_saved_tokens:
                skip_save = True

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        if skip_save and load_spec is None:
            return None

        # If we need to save, update the number of saved tokens
        # NOTE: num_saved_tokens is advanced optimistically before the store
        # completes. If the store fails (CPU memory pressure), the scheduler
        # will skip re-storing on later steps. This is partially mitigated by
        # the lookup-miss re-store path in the async lookup client (min(hit)
        # aggregation detects missing chunks). A full fix would defer the
        # advance until wait_for_save confirms success (requires worker-to-scheduler
        # feedback channel - future work).
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save

        # Determine per-group save flags for two-group DSA mode.
        can_save_latent = not skip_save
        can_save_indexer = not skip_save and dsa_two_groups
        save_spec = SaveSpec(
            skip_leading_tokens,
            not skip_save,
            can_save_latent=can_save_latent,
            can_save_indexer=can_save_indexer,
        )

        # Calculate the token ids and slot mappings for load and save
        if is_sparse_decode and load_spec is not None and skip_save:
            sparse_token_count = min(
                load_spec.lmcache_cached_tokens,
                len(input_token_ids),
            )
            if (
                not tracker.sparse_token_ids
                or len(tracker.sparse_token_ids) != sparse_token_count
            ):
                tracker.seed_sparse_decode_tokens(
                    input_token_ids,
                    token_count=sparse_token_count,
                )
            token_ids = tracker.sparse_token_ids
            if len(token_ids) < load_spec.lmcache_cached_tokens:
                logger.warning(
                    "Request %s sparse decode token metadata is shorter than "
                    "LMCache hit: sparse_tokens=%d lmcache_cached_tokens=%d "
                    "prompt_len=%d",
                    tracker.req_id,
                    len(token_ids),
                    load_spec.lmcache_cached_tokens,
                    tracker.prompt_len,
            )
        else:
            retrieve_token_len = 0
            if load_spec is not None and load_spec.can_load:
                retrieve_token_len = load_spec.lmcache_cached_tokens
            token_len = max(num_tokens_to_save, retrieve_token_len)
            token_ids = input_token_ids[:token_len]
            if retrieve_token_len > 0 and len(token_ids) < retrieve_token_len:
                logger.warning(
                    "Request %s prefix-hit token metadata is shorter than "
                    "LMCache hit: tokens=%d lmcache_cached_tokens=%d "
                    "prompt_len=%d",
                    tracker.req_id,
                    len(token_ids),
                    retrieve_token_len,
                    tracker.prompt_len,
                )

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

        if len(token_ids) > num_blocks * block_size:
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

        if is_sparse_decode and load_spec is not None:
            num_slots = _sparse_slot_mapping_len(load_spec.lmcache_cached_tokens)
            current_slots = (
                int(tracker.sparse_slot_mapping[0].numel())
                if tracker.sparse_slot_mapping
                else -1
            )
            if current_slots != num_slots:
                tracker.sparse_slot_mapping.clear()
                tracker.sparse_slot_mapping.append(
                    _build_slot_mapping(
                        tracker.allocated_block_ids, block_size, num_slots
                    )
                )
            slot_mapping = tracker.sparse_slot_mapping
        else:
            slot_mapping = [
                _build_slot_mapping(
                    tracker.allocated_block_ids, block_size, len(token_ids)
                )
            ]

        indexer_slot_mapping: list[torch.Tensor] = []
        if dsa_two_groups and tracker.allocated_block_ids_indexer:
            indexer_num_blocks = len(tracker.allocated_block_ids_indexer)
            if len(token_ids) > indexer_num_blocks * block_size:
                logger.error(
                    "The number of tokens is more than the number of indexer "
                    "blocks for request %s. Something might be wrong in "
                    "scheduling logic!",
                    tracker.req_id,
                )
                logger.error(
                    "Num tokens: %d, num indexer blocks: %d, block size: %d",
                    len(token_ids),
                    indexer_num_blocks,
                    block_size,
                )
            if is_sparse_decode and load_spec is not None and load_spec.can_load:
                if (
                    not tracker.sparse_indexer_slot_mapping
                    or tracker.sparse_indexer_slot_mapping[0].numel()
                    < load_spec.lmcache_cached_tokens
                ):
                    tracker.sparse_indexer_slot_mapping.clear()
                    tracker.sparse_indexer_slot_mapping.append(
                        _build_slot_mapping(
                            tracker.allocated_block_ids_indexer,
                            block_size,
                            load_spec.lmcache_cached_tokens,
                        )
                    )
                indexer_slot_mapping = tracker.sparse_indexer_slot_mapping
            elif not is_sparse_decode:
                indexer_slot_mapping = [
                    _build_slot_mapping(
                        tracker.allocated_block_ids_indexer,
                        block_size,
                        len(token_ids),
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
            if load_spec.vllm_cached_tokens > 0:
                if (
                    tracker.sparse_decode_token_mask is None
                    or tracker.sparse_decode_token_mask.numel()
                    != num_retrieve_tokens
                ):
                    tracker.sparse_decode_token_mask = torch.ones(
                        num_retrieve_tokens, dtype=torch.bool
                    )
                decode_token_mask = tracker.sparse_decode_token_mask
            else:
                tracker.sparse_decode_token_mask = None
            if (
                tracker.sparse_decode_ret_mask is None
                or tracker.sparse_decode_ret_mask.numel() != num_retrieve_tokens
            ):
                tracker.sparse_decode_ret_mask = torch.zeros(
                    num_retrieve_tokens, dtype=torch.bool, device="cpu"
                )
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
        req_meta = ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            indexer_slot_mapping=indexer_slot_mapping,
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
            cached_shared_handles=tracker.cached_shared_handles,
            cached_keys_indexer=tracker.cached_keys_indexer,
            cached_starts_indexer=tracker.cached_starts_indexer,
            cached_ends_indexer=tracker.cached_ends_indexer,
            cached_memory_objs_indexer=tracker.cached_memory_objs_indexer,
            cached_tensors_indexer=tracker.cached_tensors_indexer,
            cached_chunk_dev_ptrs_indexer=tracker.cached_chunk_dev_ptrs_indexer,
            cached_chunk_ptrs_npu_indexer=tracker.cached_chunk_ptrs_npu_indexer,
            cached_shared_handles_indexer=tracker.cached_shared_handles_indexer,
            decode_token_mask=decode_token_mask,
            decode_ret_mask=decode_ret_mask,
        )
        return req_meta


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

        if config.extra_config is None:
            config.extra_config = {}

        model_config = getattr(vllm_config, "model_config", None)
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        if model_config is not None:
            max_model_len = getattr(model_config, "max_model_len", None)
            if max_model_len is not None:
                config.extra_config["vllm_max_model_len"] = max_model_len
        if scheduler_config is not None:
            max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
            if max_num_seqs is not None:
                config.extra_config["vllm_max_num_seqs"] = max_num_seqs
            max_num_batched_tokens = getattr(
                scheduler_config,
                "max_num_batched_tokens",
                None,
            )
            if max_num_batched_tokens is not None:
                config.extra_config["vllm_max_num_batched_tokens"] = (
                    max_num_batched_tokens
                )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        # Each entry is a (primary, secondary) retriever pair. primary is the
        # latent (kv_group=0) retriever; secondary is the indexer (kv_group=1)
        # retriever for two-group prefix/sparse retrieve, or None for
        # single-group. wait_for_layer_load routes latent and indexer layer
        # waits to the matching group and advances current_layer after all
        # required groups for that layer have completed.
        self.layerwise_retrievers: list[
            tuple[Optional[Generator[Optional[torch.Tensor], None, None]],
                  Optional[Generator[Optional[torch.Tensor], None, None]]]
        ] = []
        self._layerwise_requests: list[ReqMeta] = []
        self._layerwise_retriever_is_sparse: list[bool] = []
        self._layerwise_sparse_req_ids: list[str] = []
        self._layerwise_waited_groups: set[int] = set()
        self._layerwise_save_storers: dict[
            Any, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        # Under dsa_two_groups + TP>1, latent store_layer is deferred until
        # after all indexer layers in a forward to avoid interleaved latent/
        # indexer GPU transfers on store_stream (MTE OOB on chunk 2+).
        self._deferred_latent_pending: set[Any] = set()
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
        # Two-group MLA+DSA: latent (kv_group=0) and indexer (kv_group=1)
        # caches are partitioned by layer name ("indexer" in name) so that
        # per-group store/retrieve can pass the correct, group-filtered
        # kvcaches list to the connector. _kvcaches_list stays equal to the
        # latent list for backward-compatible latent-only callers.
        self._latent_layer_names: list[str] = []
        self._indexer_layer_names: list[str] = []
        self._latent_kvcaches: list[torch.Tensor] = []
        self._indexer_kvcaches: list[torch.Tensor] = []
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
        self._decode_window_save_window_size = (
            self._get_decode_window_save_window_size(config)
        )

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        base_num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        metadata_kv_shape = getattr(self.lmcache_engine_metadata, "kv_shape", None)
        metadata_num_layers = (
            metadata_kv_shape[0]
            if metadata_kv_shape is not None and len(metadata_kv_shape) > 0
            else None
        )
        self.num_layers = metadata_num_layers or (
            base_num_layers + calculate_draft_layers(vllm_config)
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()
        if role != KVConnectorRole.SCHEDULER:
            self._worker_retrieve_state: dict[str, WorkerRetrieveState] = {}
            self._completed_decode_window_saves: dict[str, int] = {}
            self._decode_window_save_completed_groups: set[Any] = set()
            self._decode_window_save_expected_start: dict[str, int] = {}
            self._warn_mla_per_rank_lookup_config(config)

    def _warn_mla_per_rank_lookup_config(self, config: LMCacheEngineConfig) -> None:
        metadata = self.lmcache_engine_metadata
        if metadata is None or not metadata.use_mla:
            return
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank:
            return
        lookup_ids = config.get_lookup_server_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        if len(lookup_ids) < metadata.world_size:
            logger.warning(
                "MLA per-rank store (save_only_first_rank=false) but lookup "
                "server runs on ranks %s only (world_size=%d). The scheduler "
                "may trust rank0 hit count while other TP ranks miss KV on "
                "retrieve_layer -> garbled generation. Remove "
                "lookup_server_worker_ids override or list all TP ranks.",
                lookup_ids,
                metadata.world_size,
            )

    def _get_decode_window_save_window_size(
        self, config: LMCacheEngineConfig
    ) -> int:
        raw_window_size = os.environ.get("LMCACHE_DECODE_WINDOW_SAVE_WINDOW_SIZE")
        if raw_window_size is None:
            raw_window_size = config.get_extra_config_value(
                "decode_window_save_window_size", 0
            )

        try:
            window_size = int(raw_window_size or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "decode_window_save_window_size must be an integer, "
                f"got {raw_window_size!r}"
            ) from exc

        if window_size == 0:
            return 0
        if window_size < self._lmcache_chunk_size:
            raise ValueError(
                "decode_window_save_window_size must be >= lmcache chunk_size "
                f"({self._lmcache_chunk_size}), got {window_size}"
            )
        if window_size % self._lmcache_chunk_size != 0:
            raise ValueError(
                "decode_window_save_window_size must be a multiple of "
                f"lmcache chunk_size ({self._lmcache_chunk_size}), got {window_size}"
            )

        logger.info(
            "Decode window save enabled: window_size=%d, lmcache_chunk_size=%d",
            window_size,
            self._lmcache_chunk_size,
        )
        return window_size

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
        manager = getattr(self, "_manager", None)
        if manager is None:
            return None
        return manager.lmcache_engine

    @lmcache_engine.setter
    def lmcache_engine(self, value: Optional[LMCacheEngine]) -> None:
        """Set the LMCache engine instance on manager-backed adapters."""
        manager = getattr(self, "_manager", None)
        if manager is None:
            self._manager = SimpleNamespace(lmcache_engine=value)
            return
        manager.lmcache_engine = value

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
            self._normalize_dsa_kv_layer_groups()

    def _normalize_dsa_kv_layer_groups(self) -> None:
        """Keep metadata group index aligned with the DSA kv_group contract."""
        if not self._is_dsa_two_groups() or self.lmcache_engine is None:
            return
        manager = self.lmcache_engine.metadata.kv_layer_groups_manager
        groups = list(manager.kv_layer_groups)
        if not groups:
            return

        latent_names = set(getattr(self, "_latent_layer_names", []))
        indexer_names = set(getattr(self, "_indexer_layer_names", []))
        if not latent_names and not indexer_names:
            self._refresh_kvcaches_list()
            latent_names = set(getattr(self, "_latent_layer_names", []))
            indexer_names = set(getattr(self, "_indexer_layer_names", []))
        if not indexer_names:
            return

        latent_groups = []
        indexer_groups = []
        for group in groups:
            names = set(group.layer_names)
            has_indexer = bool(names & indexer_names) or any(
                "indexer" in name for name in names
            )
            has_latent = bool(names & latent_names) or not has_indexer
            if has_indexer and has_latent:
                raise RuntimeError(
                    "DSA two-group KV metadata is ambiguous: one metadata "
                    "group contains both latent and indexer layers. "
                    f"layer_names={group.layer_names}"
                )
            if has_indexer:
                indexer_groups.append(group)
            else:
                latent_groups.append(group)

        if len(latent_groups) != 1 or len(indexer_groups) != 1:
            raise RuntimeError(
                "DSA two-group KV metadata requires exactly one latent "
                "metadata group and one indexer metadata group so kv_group=0 "
                "maps to latent and kv_group=1 maps to indexer. "
                f"latent_groups={len(latent_groups)}, "
                f"indexer_groups={len(indexer_groups)}, "
                f"groups={groups}"
            )

        normalized_groups = latent_groups + indexer_groups
        if manager.kv_layer_groups != normalized_groups:
            logger.info(
                "Reordered DSA KV metadata groups to latent/indexer order: "
                "latent_dtype=%s, indexer_dtype=%s",
                normalized_groups[0].dtype,
                normalized_groups[1].dtype,
            )
            manager.kv_layer_groups = normalized_groups

    def _refresh_kvcaches_list(self) -> None:
        self._latent_layer_names = []
        self._indexer_layer_names = []
        self._latent_kvcaches = []
        self._indexer_kvcaches = []
        dsa_two_groups = getattr(self.config, "dsa_two_groups", False)
        for layer_name, kv_cache in self.kv_caches.items():
            if dsa_two_groups and "indexer" in layer_name:
                self._indexer_layer_names.append(layer_name)
                self._indexer_kvcaches.append(kv_cache)
            else:
                self._latent_layer_names.append(layer_name)
                self._latent_kvcaches.append(kv_cache)
        # Backward-compatible flat list = latent group (the default group).
        self._kvcaches_list = self._latent_kvcaches
        self._kv_layer_name_to_index = {
            layer_name: idx for idx, layer_name in enumerate(self.kv_caches)
        }
        if (
            dsa_two_groups
            and len(self._indexer_kvcaches) == 0
            and len(self.kv_caches) > 0
            and getattr(self, "_role", None) != KVConnectorRole.SCHEDULER
        ):
            logger.warning(
                "dsa_two_groups is enabled but no indexer KV caches were "
                "registered with the connector (no layer name contains "
                "'indexer'). Two-group store/retrieve for the indexer group "
                "will be skipped. Ensure vLLM registers the indexer KV cache "
                "group with this connector."
            )

    def _kvcaches_for_group(self, kv_group: int) -> list[torch.Tensor]:
        """Return the per-group kv_caches list for the connector."""
        if not hasattr(self, "_latent_kvcaches"):
            if hasattr(self, "kv_caches"):
                self._refresh_kvcaches_list()
            else:
                self._latent_kvcaches = list(getattr(self, "_kvcaches_list", []))
        if not hasattr(self, "_indexer_kvcaches"):
            self._indexer_kvcaches = []
        if kv_group == 1 and getattr(
            getattr(self, "config", None), "dsa_two_groups", False
        ):
            return self._indexer_kvcaches
        return self._latent_kvcaches

    def _num_layers_for_group(self, kv_group: int) -> int:
        return len(self._kvcaches_for_group(kv_group))

    @staticmethod
    def _min_layer_cache_chunks(
        layer_cache: Optional[list], num_layers: int
    ) -> int:
        """Return the minimum chunk count across layers for diagnostics.

        Returns 0 if any expected layer is missing or empty, so the
        caller can treat 0 as 'not fully covered'.
        """
        if not layer_cache:
            return 0
        if num_layers > 0 and len(layer_cache) < num_layers:
            return 0
        counts: list[int] = []
        for entry in layer_cache:
            if entry is None:
                counts.append(0)
            elif isinstance(entry, list):
                counts.append(len(entry))
            elif hasattr(entry, "shape") and entry.shape:
                counts.append(int(entry.shape[0]))
            elif hasattr(entry, "__len__"):
                counts.append(len(entry))
            else:
                counts.append(0)
        return min(counts) if counts else 0

    def _is_dsa_two_groups(self) -> bool:
        return bool(getattr(getattr(self, "config", None), "dsa_two_groups", False))

    def _is_indexer_layer_wait(self, layer_name: str) -> bool:
        if not self._is_dsa_two_groups():
            return False
        indexer_names = getattr(self, "_indexer_layer_names", [])
        return layer_name in indexer_names or "indexer" in layer_name

    def _layerwise_wait_group(self, layer_name: str) -> int:
        return 1 if self._is_indexer_layer_wait(layer_name) else 0

    @staticmethod
    def _layerwise_layer_id_from_name(layer_name: str) -> Optional[int]:
        marker = "layers."
        marker_idx = layer_name.find(marker)
        if marker_idx < 0:
            return None
        start = marker_idx + len(marker)
        end = start
        while end < len(layer_name) and layer_name[end].isdigit():
            end += 1
        if end == start:
            return None
        return int(layer_name[start:end])

    def _layerwise_required_wait_groups(self) -> set[int]:
        cached = getattr(self, "_layerwise_required_wait_groups_cache", None)
        if cached is not None:
            return cached

        required = {0}
        if self._is_dsa_two_groups():
            for idx, (_, indexer_retriever) in enumerate(
                getattr(self, "layerwise_retrievers", [])
            ):
                is_sparse = (
                    idx < len(getattr(self, "_layerwise_retriever_is_sparse", []))
                    and self._layerwise_retriever_is_sparse[idx]
                )
                if indexer_retriever is not None and not is_sparse:
                    required.add(1)
                    break
        self._layerwise_required_wait_groups_cache = required
        return required

    def _layerwise_wait_should_advance(self, wait_group: int) -> bool:
        waited_groups = getattr(self, "_layerwise_waited_groups", None)
        if waited_groups is None:
            waited_groups = set()
            self._layerwise_waited_groups = waited_groups
        waited_groups.add(wait_group)
        if self._layerwise_required_wait_groups().issubset(waited_groups):
            waited_groups.clear()
            return True
        return False

    def _shared_cpu_config_value(self, key: str, default: Any = None) -> Any:
        missing = object()

        def config_value(config: Any) -> Any:
            extra_config = getattr(config, "extra_config", None)
            if isinstance(extra_config, dict) and key in extra_config:
                return extra_config[key]
            config_dict = getattr(config, "__dict__", None)
            if isinstance(config_dict, dict) and key in config_dict:
                return config_dict[key]
            getter = getattr(config, "get_extra_config_value", None)
            if (
                callable(getter)
                and not type(config).__module__.startswith("unittest.mock")
            ):
                return getter(key, default)
            return missing

        engine = getattr(self, "lmcache_engine", None)
        if engine is not None:
            getter = getattr(engine, "_get_shared_config_value", None)
            if callable(getter):
                return getter(key, default)
            config = getattr(engine, "config", None)
            if config is not None:
                value = config_value(config)
                if value is not missing:
                    return value
            engine_dict = getattr(engine, "__dict__", None)
            if isinstance(engine_dict, dict) and key in engine_dict:
                return engine_dict[key]

        config = getattr(self, "config", None)
        if config is not None:
            value = config_value(config)
            if value is not missing:
                return value
        return default

    def _shared_cpu_materialize_index_on_decode_cold(self) -> bool:
        return bool(
            self._shared_cpu_config_value(
                "shared_cpu_materialize_index_on_decode_cold",
                True,
            )
        )

    def _sparse_decode_requires_index_materialization(
        self,
        request: "ReqMeta",
        shared_cpu_enabled: bool,
    ) -> bool:
        """True when sparse decode must materialize DSA index from LMCache.

        In the non-shared kv_both path, prefill may populate the resident DSA
        index cache in vLLM. A shared-CPU sparse decode hit, however, can skip
        prompt prefill entirely, so it must materialize the index group from
        LMCache instead of assuming resident index state is valid.
        """
        if not self._is_dsa_two_groups():
            return False
        if not self._shared_cpu_materialize_index_on_decode_cold():
            return False
        if shared_cpu_enabled:
            return True
        kv_role = getattr(self, "kv_role", "kv_both")
        return kv_role == "kv_consumer"

    @staticmethod
    def _shared_sparse_decode_indexer_is_resident(
        request: "ReqMeta",
        bound_state: Optional[WorkerRetrieveState],
        token_count: int,
    ) -> bool:
        """Return true when the live request already has DSA index in vLLM.

        Shared CPU decode must cold-materialize the DSA index once because a
        prefix hit may skip prefill. After that, the same live request keeps
        the indexer KV resident in vLLM; reloading it from LMCache on every
        decode token is redundant. Decode-save growth resets/extends the
        request state, so the next larger prefix still materializes once.
        """
        if bound_state is None or not bound_state.shared_request_active:
            return False
        if bound_state.shared_index_status != "present":
            return False
        if request.load_spec is None:
            return False
        return (
            int(request.load_spec.lmcache_cached_tokens) <= int(bound_state.token_count)
            and int(token_count) <= int(bound_state.token_count)
        )

    @staticmethod
    def _mark_shared_index_skipped(
        state: Optional[WorkerRetrieveState],
        req_id: str,
        generation: int,
        token_count: int,
    ) -> None:
        if state is None:
            return
        state.req_id = req_id
        state.shared_index_status = "skipped"
        state.shared_generation = generation
        state.shared_validation_signature = None
        if state.shared_latent_status == "present":
            state.shared_request_active = True
            state.pointer_cache_generation = generation
            state.request_scope_token = f"{req_id}:{generation}:{token_count}"

    @staticmethod
    def _shared_request_scope_token(
        req_id: str,
        generation: int,
        token_count: int,
    ) -> str:
        return f"{req_id}:{generation}:{token_count}"

    @staticmethod
    def _shared_retrieve_token_count_for_request(
        request: ReqMeta,
    ) -> int:
        token_count = len(request.token_ids)
        if request.is_sparse_decode and request.load_spec is not None:
            token_count = int(request.load_spec.lmcache_cached_tokens)
        return token_count

    @classmethod
    def _shared_request_scope_token_for_request(
        cls,
        request: ReqMeta,
        generation: int,
    ) -> str:
        return cls._shared_request_scope_token(
            request.req_id,
            generation,
            cls._shared_retrieve_token_count_for_request(request),
        )

    def _shared_worker_validation_signature(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        *,
        current_generation: int,
        pointer_generation: int,
        materialize_index: bool,
    ) -> tuple[Any, ...]:
        return (
            state.request_scope_token,
            current_generation,
            pointer_generation,
            state.shared_latent_status,
            state.shared_index_status,
            bool(self._is_dsa_two_groups()),
            bool(materialize_index),
            int(getattr(self, "num_layers", 0) or 0),
            request.req_id,
            len(state.cached_starts or []),
            len(state.cached_ends or []),
            id(state.cached_memory_objs),
            id(state.cached_chunk_ptrs_npu),
        )

    @staticmethod
    def _clear_request_indexer_cache(request: ReqMeta) -> None:
        for field_name in (
            "cached_keys_indexer",
            "cached_starts_indexer",
            "cached_ends_indexer",
            "cached_memory_objs_indexer",
            "cached_tensors_indexer",
            "cached_chunk_dev_ptrs_indexer",
            "cached_chunk_ptrs_npu_indexer",
            "cached_shared_handles_indexer",
        ):
            _ensure_list_attr(request, field_name).clear()

    def _validate_shared_worker_retrieve_state(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> None:
        engine = getattr(self, "lmcache_engine", None)
        if (
            engine is None
            or not getattr(engine, "enable_shared_cpu_cache", False)
            or not getattr(request, "is_sparse_decode", False)
            or not state.shared_request_active
        ):
            return

        current_generation = int(
            getattr(engine, "shared_cpu_cache_generation", 0) or 0
        )
        state_generation = int(state.shared_generation or 0)
        pointer_generation = int(
            getattr(state, "pointer_cache_generation", 0) or state_generation
        )
        if state_generation != current_generation:
            raise RuntimeError(
                "Shared CPU sparse decode state generation mismatch before "
                "hot-path reuse: "
                f"req_id={request.req_id}, state_generation="
                f"{state.shared_generation}, current_generation="
                f"{current_generation}"
            )
        if pointer_generation != current_generation:
            raise RuntimeError(
                "Shared CPU sparse decode pointer-cache generation mismatch "
                "before hot-path reuse: "
                f"req_id={request.req_id}, pointer_cache_generation="
                f"{pointer_generation}, current_generation={current_generation}"
            )
        expected_scope_token = self._shared_request_scope_token_for_request(
            request,
            current_generation,
        )
        if state.request_scope_token != expected_scope_token:
            raise RuntimeError(
                "Shared CPU sparse decode request scope mismatch before "
                "hot-path reuse: "
                f"req_id={request.req_id}, request_scope_token="
                f"{state.request_scope_token!r}, expected="
                f"{expected_scope_token!r}"
            )
        retrieve_token_count = self._shared_retrieve_token_count_for_request(
            request
        )
        if state.shared_latent_status != "present":
            raise RuntimeError(
                "Shared CPU sparse decode hot path requires MLA latent "
                "state before transfer: "
                f"req_id={request.req_id}, status={state.shared_latent_status!r}"
            )
        materialize_index = False
        if self._is_dsa_two_groups():
            materialize_index = self._sparse_decode_requires_index_materialization(
                request,
                True,
            )
            allowed = ("present",) if materialize_index else ("present", "skipped")
            if state.shared_index_status not in allowed:
                raise RuntimeError(
                    "Shared CPU sparse decode hot path has invalid DSA index "
                    "state before transfer: "
                    f"req_id={request.req_id}, status="
                    f"{state.shared_index_status!r}, materialize_index="
                    f"{materialize_index}"
                )

        validation_signature = self._shared_worker_validation_signature(
            state,
            request,
            current_generation=current_generation,
            pointer_generation=pointer_generation,
            materialize_index=materialize_index,
        )
        if state.shared_validation_signature == validation_signature:
            return

        if not self._cached_ranges_cover_prefix(
            state.cached_starts,
            state.cached_ends,
            retrieve_token_count,
        ):
            cached_ranges = list(
                zip(state.cached_starts, state.cached_ends, strict=False)
            )
            raise RuntimeError(
                "Shared CPU sparse decode hot path has non-contiguous MLA "
                "latent prefix coverage before transfer: "
                f"req_id={request.req_id}, kv_group=0, "
                f"token_count={retrieve_token_count}, "
                f"cached_ranges={cached_ranges}"
            )
        expected_layers = int(getattr(self, "num_layers", 0) or 0)
        required_latent_chunks = self._shared_required_chunk_count(
            state.cached_starts,
            state.cached_ends,
            state.cached_memory_objs,
        )
        missing_latent_layers = self._missing_shared_layer_cache_coverage(
            state.cached_memory_objs,
            expected_layers,
            required_latent_chunks,
        )
        if missing_latent_layers:
            raise RuntimeError(
                "Shared CPU sparse decode hot path has incomplete MLA "
                "latent state before transfer: "
                f"req_id={request.req_id}, kv_group=0, "
                f"missing_layers={missing_latent_layers}"
            )
        missing_latent_pointer_layers = self._missing_shared_pointer_cache_layers(
            state.cached_memory_objs,
            state.cached_chunk_ptrs_npu,
            required_latent_chunks,
        )
        if missing_latent_pointer_layers:
            raise RuntimeError(
                "Shared CPU sparse decode hot path is missing MLA latent "
                "NPU pointer-cache tensors before transfer: "
                f"req_id={request.req_id}, "
                f"missing_layers={missing_latent_pointer_layers}"
            )
        # DSA index is cold-materialized and admitted into the live request
        # state before warm sparse decode reuse. Warm decode retrieves only MLA
        # latent rows, so avoid walking index metadata or pointer caches here.
        state.shared_validation_signature = validation_signature

    def _shared_worker_retrieve_state_is_current(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
        token_count: int,
    ) -> bool:
        if not state.shared_request_active or state.shared_validation_signature is None:
            return False
        if request.load_spec is not None and (
            int(request.load_spec.lmcache_cached_tokens) > int(state.token_count)
        ):
            return False
        if int(token_count) > int(state.token_count):
            return False

        engine = getattr(self, "lmcache_engine", None)
        current_generation = int(
            getattr(engine, "shared_cpu_cache_generation", 0) or 0
        )
        pointer_generation = int(
            getattr(state, "pointer_cache_generation", 0)
            or int(state.shared_generation or 0)
        )
        materialize_index = (
            self._is_dsa_two_groups()
            and self._sparse_decode_requires_index_materialization(request, True)
        )
        validation_signature = self._shared_worker_validation_signature(
            state,
            request,
            current_generation=current_generation,
            pointer_generation=pointer_generation,
            materialize_index=materialize_index,
        )
        return state.shared_validation_signature == validation_signature

    @staticmethod
    def _save_storer_key(req_id: str, kv_group: int) -> Union[str, tuple[str, int]]:
        """Latent (kv_group=0) uses dev-qzy req_id key; indexer uses (req_id, 1)."""
        if kv_group == 0:
            return req_id
        return (req_id, kv_group)

    @staticmethod
    def _latent_slot_mapping_from_attn_metadata(
        attn_metadata, layer_name: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Return MLA latent slot mapping from per-layer vLLM attention metadata."""
        if isinstance(attn_metadata, dict):
            if layer_name is not None:
                meta = attn_metadata.get(layer_name)
                if meta is not None:
                    slot_mapping = getattr(meta, "slot_mapping", None)
                    if slot_mapping is not None:
                        return slot_mapping
            for name, meta in attn_metadata.items():
                if "indexer" in name:
                    continue
                slot_mapping = getattr(meta, "slot_mapping", None)
                if slot_mapping is not None:
                    return slot_mapping
            return None
        return getattr(attn_metadata, "slot_mapping", None)

    @staticmethod
    def _indexer_slot_mapping_from_attn_metadata(
        attn_metadata, layer_name: Optional[str] = None
    ) -> Optional[torch.Tensor]:
        """Return the DSA indexer slot mapping from vLLM attention metadata.

        vLLM may pass either a single metadata object or a per-layer metadata
        dict. In the dict form, indexer layers have their own
        DeepseekV32IndexerMetadata whose slot mapping is stored as
        ``slot_mapping``. Latent/SFA metadata instead carries the group-1 slot
        mapping as ``indexer_slot_mapping``.
        """
        if isinstance(attn_metadata, dict):
            if layer_name is not None:
                meta = attn_metadata.get(layer_name)
                if meta is not None:
                    slot_mapping = getattr(meta, "slot_mapping", None)
                    if slot_mapping is not None:
                        return slot_mapping
                    slot_mapping = getattr(meta, "indexer_slot_mapping", None)
                    if slot_mapping is not None:
                        return slot_mapping

                latent_layer_name = layer_name.replace(
                    ".indexer.k_cache", ".attn"
                )
                latent_meta = attn_metadata.get(latent_layer_name)
                if latent_meta is not None:
                    slot_mapping = getattr(
                        latent_meta, "indexer_slot_mapping", None
                    )
                    if slot_mapping is not None:
                        return slot_mapping

            for name, meta in attn_metadata.items():
                if "indexer" not in name:
                    continue
                slot_mapping = getattr(meta, "slot_mapping", None)
                if slot_mapping is not None:
                    return slot_mapping

            for meta in attn_metadata.values():
                slot_mapping = getattr(meta, "indexer_slot_mapping", None)
                if slot_mapping is not None:
                    return slot_mapping
            return None

        slot_mapping = getattr(attn_metadata, "indexer_slot_mapping", None)
        if slot_mapping is not None:
            return slot_mapping
        return getattr(attn_metadata, "slot_mapping", None)

    @staticmethod
    def _pad_chunk_local_slot_mapping(
        slot_mapping: torch.Tensor,
        total_tokens: int,
        token_offset: int,
    ) -> torch.Tensor:
        """Convert a chunk-local slot mapping to token-sequence coordinates.

        LMCache store_layer returns absolute token ranges, e.g. [4096, 8192)
        for the second chunked-prefill step. vLLM's per-layer indexer metadata
        may only carry the current chunk's slot mapping of length 4096. Pad the
        leading range with dummy values so later slot_mapping[start:end] slicing
        returns the chunk-local mapping.
        """
        if token_offset <= 0 or len(slot_mapping) >= total_tokens:
            return slot_mapping

        expected_local_tokens = total_tokens - token_offset
        if len(slot_mapping) != expected_local_tokens:
            return slot_mapping

        padded = torch.empty(
            total_tokens, device=slot_mapping.device, dtype=slot_mapping.dtype
        )
        padded[:token_offset] = 0
        padded[token_offset:] = slot_mapping
        return padded

    def _indexer_retrieve_slot_mapping(
        self,
        attn_metadata,
        lmcache_cached_tokens: int,
        layer_name: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Return the indexer group's slot mapping for prefix retrieve.

        Mirrors the save path's indexer slot logic and handles both vLLM's
        per-layer metadata dict and single-object metadata forms.
        """
        attn_slot = _am_get(attn_metadata, "slot_mapping", None)
        idx_attr = _am_get(attn_metadata, "indexer_slot_mapping", None)
        candidates: list[tuple[str, torch.Tensor]] = []

        def add_candidate(source: str, slot_mapping) -> None:
            if isinstance(slot_mapping, torch.Tensor):
                candidates.append((source, slot_mapping))

        if isinstance(attn_metadata, dict):
            if layer_name is not None:
                latent_layer_name = layer_name.replace(
                    ".indexer.k_cache", ".attn"
                )
                latent_meta = attn_metadata.get(latent_layer_name)
                if latent_meta is not None:
                    add_candidate(
                        "latent_meta.indexer_slot_mapping",
                        getattr(latent_meta, "indexer_slot_mapping", None),
                    )

                meta = attn_metadata.get(layer_name)
                if meta is not None:
                    add_candidate(
                        "indexer_meta.indexer_slot_mapping",
                        getattr(meta, "indexer_slot_mapping", None),
                    )
                    add_candidate(
                        "indexer_meta.slot_mapping",
                        getattr(meta, "slot_mapping", None),
                    )

            for name, meta in attn_metadata.items():
                if "indexer" in name:
                    continue
                add_candidate(
                    "any_latent_meta.indexer_slot_mapping",
                    getattr(meta, "indexer_slot_mapping", None),
                )

            for name, meta in attn_metadata.items():
                if "indexer" not in name:
                    continue
                add_candidate(
                    "any_indexer_meta.indexer_slot_mapping",
                    getattr(meta, "indexer_slot_mapping", None),
                )
                add_candidate(
                    "any_indexer_meta.slot_mapping",
                    getattr(meta, "slot_mapping", None),
                )
        else:
            add_candidate(
                "attn_metadata.indexer_slot_mapping",
                getattr(attn_metadata, "indexer_slot_mapping", None),
            )
            add_candidate(
                "attn_metadata.slot_mapping",
                getattr(attn_metadata, "slot_mapping", None),
            )

        idx_slot = None
        for _, candidate in candidates:
            if len(candidate) >= lmcache_cached_tokens:
                idx_slot = candidate
                break

        if idx_slot is None:
            return None
        idx_slot = idx_slot.to(device=self.device, dtype=torch.long)
        if lmcache_cached_tokens < len(idx_slot):
            idx_slot = idx_slot[:lmcache_cached_tokens]
        return idx_slot

    def _indexer_save_slot_mapping(
        self,
        request: "ReqMeta",
        attn_metadata,
        layer_name: Optional[str],
        token_count: int,
    ) -> Optional[torch.Tensor]:
        """Return indexer save slots from the active layer metadata.

        DSA indexer save is reading the current indexer-layer KV buffer, so the
        source slots must come from the attention metadata for that layer. The
        request-level indexer_slot_mapping may be a full/cumulative scheduler
        destination used by retrieve; using it for save can publish cache chunks
        that differ from the cold prefill state.
        """
        if (
            self._is_decode_window_save_request(request)
            and request.indexer_slot_mapping
        ):
            return request.indexer_slot_mapping[0]
        _ = token_count
        return self._indexer_slot_mapping_from_attn_metadata(
            attn_metadata, layer_name
        )

    def _sparse_indexer_slot_mapping(
        self,
        attn_metadata,
        latent_sparse_slots: torch.Tensor,
        lmcache_cached_tokens: int,
        request_indexer_slots: Optional[torch.Tensor] = None,
        strict: bool = False,
    ) -> Optional[torch.Tensor]:
        """Indexer slots for sparse decode, covering the full LMCache-hit prefix.

        The latent group loads only selected top-k rows into a compact scratch
        window, but the DSA index group must be fully materialized before top-k
        selection. Capping index slots to the latent scratch window leaves most
        prompt index rows stale and degrades sparse decode quality.
        """
        sparse_len = len(latent_sparse_slots)
        indexer_len = int(lmcache_cached_tokens)
        if request_indexer_slots is not None and request_indexer_slots.numel() > 0:
            request_indexer_slots = request_indexer_slots.to(
                device=self.device, dtype=torch.long
            )
            if request_indexer_slots.numel() >= indexer_len:
                return request_indexer_slots[:indexer_len]

        idx_slot = self._indexer_retrieve_slot_mapping(
            attn_metadata, lmcache_cached_tokens
        )
        if idx_slot is not None and idx_slot.numel() >= indexer_len:
            return idx_slot[:indexer_len]
        if strict:
            request_len = (
                int(request_indexer_slots.numel())
                if request_indexer_slots is not None
                else 0
            )
            metadata_len = int(idx_slot.numel()) if idx_slot is not None else 0
            raise RuntimeError(
                "Shared CPU sparse decode with dsa_two_groups=true could not "
                "resolve full DSA index slot mapping. Refusing to fall back "
                "to latent slots because that can load indexer KV into the "
                "wrong cache group: "
                f"indexer_len={indexer_len}, sparse_len={sparse_len}, "
                f"request_indexer_slots={request_len}, "
                f"metadata_indexer_slots={metadata_len}, "
                f"lmcache_cached_tokens={lmcache_cached_tokens}"
            )
        if idx_slot is None or idx_slot.numel() == 0:
            return latent_sparse_slots
        return idx_slot

    def _layer_index_from_name(self, layer_name: str) -> int:
        layer_map = getattr(self, "_kv_layer_name_to_index", None)
        if layer_map is None:
            self._refresh_kvcaches_list()
            layer_map = getattr(self, "_kv_layer_name_to_index", {})
        if layer_name in layer_map:
            return int(layer_map[layer_name])
        return max(0, min(self.current_layer - 1, self.num_layers - 1))

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

        self._refresh_kvcaches_list()
        self._build_kv_layer_groups()

    ####################
    # Worker side APIs
    ####################
    @staticmethod
    def _load_tokens_for_retrieve(
        tokens: list[int], lmcache_cached_tokens: int, *, is_sparse_decode: bool
    ) -> list[int]:
        """Return token ids for retrieve without redundant list copy on decode."""
        if is_sparse_decode:
            # Sparse decode scatters into a compact scratch slot window, but
            # selected_tokens can point anywhere in the cached prefix. Retrieve
            # metadata and cached chunk pointers must therefore cover the full
            # LMCache-hit prefix, not only the scratch window length.
            if 0 < lmcache_cached_tokens < len(tokens):
                return tokens[:lmcache_cached_tokens]
            return tokens
        if lmcache_cached_tokens >= len(tokens):
            return tokens
        return tokens[:lmcache_cached_tokens]

    @staticmethod
    def _load_token_mask_for_retrieve(
        request: "ReqMeta",
        token_count: int,
        lmcache_chunk_size: int,
    ) -> Optional[torch.Tensor]:
        """Build or reuse the token mask for a retrieve call."""
        if (
            request.is_sparse_decode
            and request.load_spec is not None
            and request.load_spec.vllm_cached_tokens <= 0
        ):
            request.decode_token_mask = None
            return None

        if request.is_sparse_decode and request.decode_token_mask is not None:
            mask = request.decode_token_mask
            if mask.numel() == token_count:
                token_mask = mask.clone()
            else:
                token_mask = torch.ones(token_count, dtype=torch.bool)
        else:
            token_mask = torch.ones(token_count, dtype=torch.bool)

        if request.load_spec is not None:
            prefix_tokens = request.load_spec.vllm_cached_tokens
            # Sparse decode still needs LMCache chunks for the selected prefix
            # tokens. lmcache_cached_tokens means "available in LMCache", not
            # "already resident in vLLM", so do not mask it out here.
            prefix_tokens = min(prefix_tokens, token_count)
            masked_token_count = (
                prefix_tokens
                // lmcache_chunk_size
                * lmcache_chunk_size
            )
            if masked_token_count:
                token_mask[:masked_token_count] = False

        if request.is_sparse_decode:
            request.decode_token_mask = token_mask
        return token_mask

    @staticmethod
    def _full_hit_recalc_last_token(
        load_spec: Optional[LoadSpec],
        prompt_len: int,
        *,
        is_sparse_decode: bool,
    ) -> bool:
        """True when vLLM expects the last prompt token to be recomputed, not loaded."""
        if is_sparse_decode or load_spec is None:
            return False
        return (
            load_spec.lmcache_cached_tokens >= prompt_len
            and load_spec.lmcache_cached_tokens > load_spec.vllm_cached_tokens
        )

    @staticmethod
    def _trim_prefill_for_recalc_last(
        request: "ReqMeta",
        retrieve_tokens: list[int],
        slot_mapping: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor]:
        """Handle vLLM recalc_last=1 on a full-cache-hit prefill retrieve.

        We intentionally do NOT trim retrieve_tokens or slot_mapping. Rationale:

        Chunk keys hash the chunk's tokens (token_database._prefix_hash yields
        the hash AFTER each chunk), so the last partial chunk's key depends on
        its token count. The store saved the full prompt (partial =
        prompt_len % chunk_size tokens, e.g. 191 for a 18879-prompt with
        chunk_size=256). If we trimmed retrieve_tokens to prompt_len-1 here,
        the queried partial chunk would be 190 tokens and its key
        H(tokens[0:prompt_len-1]) would NOT match the stored key
        H(tokens[0:prompt_len]) -- the last partial chunk silently misses
        (the "missing 190" / "loaded 18688/18878" shortfall).

        By keeping retrieve_tokens and slot_mapping at prompt_len, the retrieve
        queries the same partial chunk the store saved (191 tokens, matching
        key) and scatters KV to all prompt_len slots. vLLM, on a full-hit with
        recalc_last=1, still recomputes the last prompt token's logits (and KV)
        -- overwriting whatever we scattered into that slot -- so loading it is
        harmless. This also keeps token_count == len(slot_mapping) so the dense
        prefill retrieve copies exactly match chunk sizes (no OOB).

        Note: this means num_retrieved_tokens (18879) will be 1 more than
        num_expected_load (18878 = lmcache_cached - recalc_last); the shortfall
        guard uses a strict `<`, so no false warning is emitted.
        """
        return retrieve_tokens, slot_mapping

    def _drain_layerwise_retrievers(self) -> None:
        """Finish suspended layerwise generators to avoid GC cost on reset."""
        for idx, retriever_pair in enumerate(self.layerwise_retrievers):
            is_sparse = (
                idx < len(self._layerwise_retriever_is_sparse)
                and self._layerwise_retriever_is_sparse[idx]
            )
            primary, secondary = retriever_pair
            for retriever in (primary, secondary):
                if retriever is None:
                    continue
                try:
                    if is_sparse:
                        self._drain_sparse_layerwise_retriever(retriever)
                    else:
                        while True:
                            next(retriever)
                except StopIteration:
                    pass
        self.layerwise_retrievers.clear()
        if hasattr(self, "_layerwise_requests"):
            self._layerwise_requests.clear()
        self._layerwise_retriever_is_sparse.clear()
        if hasattr(self, "_layerwise_sparse_req_ids"):
            self._layerwise_sparse_req_ids.clear()
        if hasattr(self, "_layerwise_waited_groups"):
            self._layerwise_waited_groups.clear()
        if hasattr(self, "_layerwise_sparse_indexer_sent_layers"):
            self._layerwise_sparse_indexer_sent_layers.clear()
        self._layerwise_required_wait_groups_cache = None

    def _drain_sparse_layerwise_retriever(
        self, retriever: Generator[Any, Any, Any]
    ) -> None:
        """Close sparse head-token-wise retrievers waiting on send()."""
        try:
            retriever.close()
        except (GeneratorExit, RuntimeError, ValueError):
            pass

    def _should_defer_lookup_unpin_for_sparse_decode(self, request: ReqMeta) -> bool:
        """Keep lookup pins across decode steps while sparse retrieve is active."""
        return (
            getattr(request, "is_sparse_decode", False)
            and request.load_spec is not None
            and request.load_spec.can_load
        )

    @staticmethod
    def _is_decode_window_save_request(request: ReqMeta) -> bool:
        return bool(getattr(request, "is_decode_window_save", False))

    def _layerwise_save_range(self, request: ReqMeta) -> tuple[int, int]:
        if self._is_decode_window_save_request(request):
            return (
                int(getattr(request, "decode_window_start", 0) or 0),
                int(getattr(request, "decode_window_end", len(request.token_ids))),
            )

        start = 0
        save_spec = request.save_spec
        if self.kv_role != "kv_producer" and save_spec is not None:
            start = save_spec.skip_leading_tokens
            start = start // self._lmcache_chunk_size * self._lmcache_chunk_size
        return start, len(request.token_ids)

    def _layerwise_save_storer_key(
        self, request: ReqMeta, kv_group: int = 0
    ) -> Any:
        start, end = self._layerwise_save_range(request)
        if not self._is_decode_window_save_request(request):
            return (request.req_id, "normal_save", kv_group, start, end)
        return (
            request.req_id,
            "decode_window_save",
            kv_group,
            start,
            end,
        )

    @staticmethod
    def _storer_key_matches_req_group(
        storer_key: Any, req_id: str, kv_group: int
    ) -> bool:
        if storer_key == req_id:
            return kv_group == 0
        if not isinstance(storer_key, tuple) or not storer_key:
            return False
        if storer_key[0] != req_id:
            return False
        if len(storer_key) >= 3 and storer_key[1] in (
            "normal_save",
            "decode_window_save",
        ):
            return storer_key[2] == kv_group
        if len(storer_key) >= 2 and isinstance(storer_key[1], int):
            return storer_key[1] == kv_group
        return False

    def _clear_decode_window_save_groups_for_req(self, req_id: str) -> None:
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is not None:
            for group_key in list(groups):
                if (
                    isinstance(group_key, tuple)
                    and group_key
                    and group_key[0] == req_id
                ):
                    groups.discard(group_key)
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is not None:
            expected.pop(req_id, None)

    def _clear_decode_window_save_groups_for_window(
        self,
        request: ReqMeta,
    ) -> None:
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is None:
            return
        for kv_group in self._decode_window_save_required_groups(request):
            groups.discard(self._layerwise_save_storer_key(request, kv_group))

    def _record_decode_window_save_group_completed(
        self,
        request: ReqMeta,
        kv_group: int,
    ) -> None:
        if not self._is_decode_window_save_request(request):
            return
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is None:
            return
        groups.add(self._layerwise_save_storer_key(request, kv_group))

    def _note_decode_window_save_seen(self, request: ReqMeta) -> None:
        if not self._is_decode_window_save_request(request):
            return
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is None:
            return
        window_start = getattr(request, "decode_window_start", None)
        if window_start is None:
            return
        window_start = int(window_start)
        expected.setdefault(request.req_id, window_start)

    def _decode_window_save_is_next_expected(self, request: ReqMeta) -> bool:
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is None:
            return True
        window_start = getattr(request, "decode_window_start", None)
        if window_start is None:
            return True
        self._note_decode_window_save_seen(request)
        return int(window_start) == int(expected.get(request.req_id, window_start))

    def _decode_window_save_required_groups(self, request: ReqMeta) -> set[int]:
        if not self._is_decode_window_save_request(request):
            return set()
        save_spec = request.save_spec
        if save_spec is None:
            return set()
        required: set[int] = set()
        if getattr(save_spec, "can_save_latent", getattr(save_spec, "can_save", False)):
            required.add(0)
        if (
            getattr(self.config, "dsa_two_groups", False)
            and getattr(save_spec, "can_save_indexer", False)
        ):
            required.add(1)
        return required

    def _decode_window_save_has_required_groups(self, request: ReqMeta) -> bool:
        required = self._decode_window_save_required_groups(request)
        if not required:
            return False
        groups = getattr(self, "_decode_window_save_completed_groups", None)
        if groups is None:
            return False
        return all(
            self._layerwise_save_storer_key(request, kv_group) in groups
            for kv_group in required
        )

    def _decode_window_save_uses_shared_cpu(self) -> bool:
        engine = getattr(self, "lmcache_engine", None)
        return bool(getattr(engine, "enable_shared_cpu_cache", False))

    def _decode_window_save_group_pointer_ready(
        self,
        request: ReqMeta,
        kv_group: int,
    ) -> bool:
        cache_kwargs = _retrieve_cache_kwargs(
            request,
            kv_group=kv_group,
            dsa_two_groups=self._is_dsa_two_groups(),
        )
        starts = cache_kwargs["cached_starts"]
        ends = cache_kwargs["cached_ends"]
        memory_objs = cache_kwargs["cached_memory_objs"]
        chunk_ptrs = cache_kwargs["cached_chunk_ptrs_npu"]
        window_start = getattr(request, "decode_window_start", None)
        window_end = getattr(request, "decode_window_end", None)
        if window_start is not None and window_end is not None:
            if not self._cached_ranges_cover_interval(
                starts,
                ends,
                int(window_start),
                int(window_end),
            ):
                return False
        required_chunks = self._shared_required_chunk_count(
            starts,
            ends,
            memory_objs,
        )
        if required_chunks <= 0:
            return False
        expected_layers = int(getattr(self, "num_layers", 0) or 0)
        if expected_layers <= 0:
            expected_layers = len(memory_objs or [])
        if expected_layers <= 0:
            return False
        if self._missing_shared_layer_cache_coverage(
            memory_objs,
            expected_layers,
            required_chunks,
        ):
            return False
        return not self._missing_shared_pointer_cache_layers(
            memory_objs,
            chunk_ptrs,
            required_chunks,
        )

    def _decode_window_save_store_cache_ready(self, request: ReqMeta) -> bool:
        if not self._decode_window_save_uses_shared_cpu():
            return True
        required = self._decode_window_save_required_groups(request)
        if not required:
            return False
        return all(
            self._decode_window_save_group_pointer_ready(request, kv_group)
            for kv_group in required
        )

    def _drop_layerwise_save_storers(self, req_id: str) -> None:
        if hasattr(self, "_layerwise_save_storers"):
            for storer_key in list(self._layerwise_save_storers):
                should_drop = False
                if storer_key == req_id:
                    should_drop = True
                elif (
                    isinstance(storer_key, tuple)
                    and storer_key
                    and storer_key[0] == req_id
                ):
                    should_drop = True
                if should_drop:
                    self._close_layerwise_storer(
                        self._layerwise_save_storers.pop(storer_key, None)
                    )
        self._clear_decode_window_save_groups_for_req(req_id)

        pending = getattr(self, "_deferred_latent_pending", None)
        if pending is not None:
            for pending_key in list(pending):
                if pending_key == req_id:
                    pending.discard(pending_key)
                elif (
                    isinstance(pending_key, tuple)
                    and pending_key
                    and pending_key[0] == req_id
                ):
                    pending.discard(pending_key)

    def _release_request_lookup_pins(self, req_id: str) -> None:
        manager = getattr(self, "_manager", None)
        if manager is None:
            return
        engine = manager.lmcache_engine
        if engine is not None:
            engine.lookup_unpin(req_id)

    def _maybe_lookup_unpin_for_request(self, request: ReqMeta) -> None:
        if self._is_decode_window_save_request(request):
            return
        if self._should_defer_lookup_unpin_for_sparse_decode(request):
            return
        self._release_request_lookup_pins(request.req_id)

    def _mark_decode_window_save_completed(self, request: ReqMeta) -> None:
        if not self._is_decode_window_save_request(request):
            return
        self._note_decode_window_save_seen(request)
        engine = self.lmcache_engine
        is_passive = getattr(engine, "_is_passive", None)
        if callable(is_passive) and is_passive():
            return
        if not self._decode_window_save_has_required_groups(request):
            logger.debug(
                "Decode-window save not marked complete before required "
                "store groups are seen: req_id=%s required_groups=%s",
                request.req_id,
                sorted(self._decode_window_save_required_groups(request)),
            )
            return
        if not self._decode_window_save_is_next_expected(request):
            expected = getattr(self, "_decode_window_save_expected_start", {})
            logger.debug(
                "Decode-window save not marked complete out of order: "
                "req_id=%s window_start=%s expected_start=%s",
                request.req_id,
                getattr(request, "decode_window_start", None),
                expected.get(request.req_id),
            )
            return
        if not self._decode_window_save_store_cache_ready(request):
            if _DSA_PROF:
                required = self._decode_window_save_required_groups(request)
                cache_kwargs = _retrieve_cache_kwargs(
                    request,
                    kv_group=0,
                    dsa_two_groups=self._is_dsa_two_groups(),
                )
                print(
                    f"[DECODE_WINDOW_READY] req={request.req_id} "
                    f"ready=False "
                    f"window_start={getattr(request, 'decode_window_start', None)} "
                    f"window_end={getattr(request, 'decode_window_end', None)} "
                    f"required_groups={sorted(required)} "
                    f"min_mem_chunks={self._min_layer_cache_chunks(cache_kwargs.get('cached_memory_objs'), self._num_layers_for_group(0))} "
                    f"min_ptr_chunks={self._min_layer_cache_chunks(cache_kwargs.get('cached_chunk_ptrs_npu'), self._num_layers_for_group(0))}",
                    flush=True,
                )
            return
        window_end = request.decode_window_end
        if window_end is None:
            return
        completed = getattr(self, "_completed_decode_window_saves", None)
        if completed is None:
            return
        completed[request.req_id] = max(completed.get(request.req_id, 0), window_end)
        expected = getattr(self, "_decode_window_save_expected_start", None)
        if expected is not None:
            expected[request.req_id] = max(
                int(expected.get(request.req_id, 0)),
                int(window_end),
            )
        if _DSA_PROF:
            cache_kwargs = _retrieve_cache_kwargs(
                request,
                kv_group=0,
                dsa_two_groups=self._is_dsa_two_groups(),
            )
            print(
                f"[DECODE_WINDOW_READY] req={request.req_id} "
                f"ready=True "
                f"window_start={getattr(request, 'decode_window_start', None)} "
                f"window_end={window_end} "
                f"min_mem_chunks={self._min_layer_cache_chunks(cache_kwargs.get('cached_memory_objs'), self._num_layers_for_group(0))} "
                f"min_ptr_chunks={self._min_layer_cache_chunks(cache_kwargs.get('cached_chunk_ptrs_npu'), self._num_layers_for_group(0))}",
                flush=True,
            )
        self._clear_decode_window_save_groups_for_window(request)

    def get_completed_decode_window_saves(self) -> dict[str, int]:
        completed = getattr(self, "_completed_decode_window_saves", None)
        if not completed:
            return {}
        drained = dict(completed)
        completed.clear()
        return drained

    def update_connector_output(self, connector_output: Any) -> None:
        completed = getattr(connector_output, "completed_decode_window_saves", None)
        if not completed:
            return
        for req_id, window_end in completed.items():
            tracker = self._request_trackers.get(req_id)
            if tracker is None:
                continue
            committed_end = int(window_end)
            window_size = int(getattr(self, "_decode_window_save_window_size", 0) or 0)
            if window_size > 0 and tracker.decode_window_save_next_start is None:
                logger.debug(
                    "Ignoring decode-window completion before scheduler "
                    "emitted any save window: req_id=%s window_end=%s",
                    req_id,
                    window_end,
                )
                continue
            if tracker.decode_window_save_next_start is not None:
                next_start = int(tracker.decode_window_save_next_start)
                committed_end = min(
                    committed_end,
                    next_start,
                )
                committed_end = min(committed_end, len(tracker.token_ids))
                if window_size > 0 and committed_end < next_start:
                    delta = next_start - committed_end
                    windows_back = (delta + window_size - 1) // window_size
                    committed_end = max(0, next_start - windows_back * window_size)
            else:
                committed_end = min(committed_end, len(tracker.token_ids))
                if window_size > 0:
                    committed_end = committed_end // window_size * window_size
            tracker.decode_window_save_committed_end = max(
                tracker.decode_window_save_committed_end,
                committed_end,
            )

    def _prune_worker_retrieve_state(self, active_req_ids: set[str]) -> None:
        if not hasattr(self, "_worker_retrieve_state"):
            return
        dropped_req_ids = set(self._worker_retrieve_state) - active_req_ids
        kept_warm_req_ids: list[str] = []
        for req_id in dropped_req_ids:
            state = self._worker_retrieve_state.get(req_id)
            if state is not None and state.shared_request_active:
                self._release_shared_worker_retrieve_state(
                    state,
                    getattr(self, "lmcache_engine", None),
                )
            if state is not None and (state.metadata_warm or state.cached_keys):
                kept_warm_req_ids.append(req_id)
                continue
            if state is not None:
                self._release_shared_worker_retrieve_state(
                    state,
                    getattr(self, "lmcache_engine", None),
                )
            self._release_request_lookup_pins(req_id)
        self._worker_retrieve_state = {
            req_id: state
            for req_id, state in self._worker_retrieve_state.items()
            if req_id in active_req_ids or (state.metadata_warm or state.cached_keys)
        }

    def _drop_worker_retrieve_state(self, req_id: str) -> None:
        if hasattr(self, "_worker_retrieve_state"):
            state = self._worker_retrieve_state.pop(req_id, None)
            if state is not None:
                self._release_shared_worker_retrieve_state(
                    state,
                    getattr(self, "lmcache_engine", None),
                )
        self._release_request_lookup_pins(req_id)

    @staticmethod
    def _release_shared_worker_retrieve_state(
        state: WorkerRetrieveState,
        engine: Optional[Any] = None,
    ) -> None:
        if engine is not None and state.shared_request_active:
            release_fn = getattr(engine, "release_shared_cpu_sparse_request", None)
            if callable(release_fn):
                release_fn(state.req_id)
        for layers in state.shared_views_by_group.values():
            for layer_views in layers:
                for mem_obj in layer_views:
                    try:
                        mem_obj.ref_count_down()
                    except Exception as exc:
                        logger.warning(
                            "Failed to release passive shared view: %s", exc
                        )
        for layers in state.rank0_backing_objs_by_group.values():
            for layer_objs in layers:
                for mem_obj in layer_objs:
                    try:
                        if getattr(mem_obj, "is_pinned", False):
                            mem_obj.unpin()
                        mem_obj.ref_count_down()
                    except Exception as exc:
                        logger.warning(
                            "Failed to release rank0 shared backing object: %s",
                            exc,
                        )
        state.shared_handles_by_group.clear()
        state.shared_views_by_group.clear()
        state.shared_chunk_ptrs_npu_by_group.clear()
        state.rank0_backing_objs_by_group.clear()
        state.cached_memory_objs.clear()
        state.cached_tensors.clear()
        state.cached_chunk_dev_ptrs.clear()
        state.cached_chunk_ptrs_npu.clear()
        state.cached_shared_handles.clear()
        state.cached_memory_objs_indexer.clear()
        state.cached_tensors_indexer.clear()
        state.cached_chunk_dev_ptrs_indexer.clear()
        state.cached_chunk_ptrs_npu_indexer.clear()
        state.cached_shared_handles_indexer.clear()
        state.shared_latent_status = "missing"
        state.shared_index_status = "missing"
        state.shared_generation = 0
        state.pointer_cache_generation = 0
        state.shared_request_active = False
        state.request_scope_token = None
        state.shared_validation_signature = None
        state.req_id = None

    @staticmethod
    def _release_replaced_shared_layer_objs(
        old_layers: list[list[Any]],
        new_layers: list[list[Any]],
        *,
        rank0_backing: bool,
    ) -> None:
        new_ids = {
            id(mem_obj)
            for layer_objs in (new_layers or [])
            for mem_obj in layer_objs
        }
        for layer_objs in old_layers or []:
            for mem_obj in layer_objs:
                if id(mem_obj) in new_ids:
                    continue
                try:
                    if rank0_backing and getattr(mem_obj, "is_pinned", False):
                        mem_obj.unpin()
                    mem_obj.ref_count_down()
                except Exception as exc:
                    logger.warning(
                        "Failed to release replaced shared CPU %s object: %s",
                        "rank0 backing" if rank0_backing else "passive view",
                        exc,
                    )

    @classmethod
    def _release_replaced_shared_groups(
        cls,
        old_by_group: dict[int, list[list[Any]]],
        new_by_group: dict[int, list[list[Any]]],
        *,
        rank0_backing: bool,
    ) -> None:
        for kv_group, new_layers in new_by_group.items():
            old_layers = old_by_group.get(kv_group)
            if old_layers is None:
                continue
            cls._release_replaced_shared_layer_objs(
                old_layers,
                new_layers,
                rank0_backing=rank0_backing,
            )

    @staticmethod
    def _shared_required_chunk_count(
        starts: list[int],
        ends: list[int],
        layers: list[list[Any]],
    ) -> int:
        count = max(len(starts or []), len(ends or []))
        if count > 0:
            return count
        if any(layer for layer in (layers or [])):
            return 1
        return 0

    @staticmethod
    def _cached_prefix_covered_token_count(
        starts: list[int],
        ends: list[int],
    ) -> int:
        covered = 0
        for start, end in zip(starts or [], ends or [], strict=False):
            start = int(start)
            end = int(end)
            if end <= start:
                continue
            if start > covered:
                break
            covered = max(covered, end)
        return covered

    @staticmethod
    def _cached_ranges_cover_interval(
        starts: list[int],
        ends: list[int],
        interval_start: int,
        interval_end: int,
    ) -> bool:
        interval_start = max(0, int(interval_start))
        interval_end = max(interval_start, int(interval_end))
        if interval_end == interval_start:
            return True
        covered = interval_start
        for start, end in zip(starts or [], ends or [], strict=False):
            start = int(start)
            end = int(end)
            if end <= start or end <= covered:
                continue
            if start > covered:
                break
            covered = max(covered, end)
            if covered >= interval_end:
                return True
        return False

    @classmethod
    def _cached_ranges_cover_prefix(
        cls,
        starts: list[int],
        ends: list[int],
        token_count: int,
    ) -> bool:
        token_count = max(0, int(token_count))
        if token_count == 0:
            return True
        return cls._cached_prefix_covered_token_count(starts, ends) >= token_count

    @staticmethod
    def _shared_pointer_cache_entry_covers(
        entry: Any,
        required_chunks: int,
    ) -> bool:
        if entry is None:
            return False
        required_chunks = max(1, int(required_chunks))
        if isinstance(entry, torch.Tensor):
            return int(entry.numel()) >= required_chunks
        try:
            return len(entry) >= required_chunks
        except TypeError:
            return required_chunks == 1

    @staticmethod
    def _missing_shared_layer_cache_coverage(
        layers: list[list[Any]],
        expected_layers: int,
        required_chunks: int,
    ) -> list[int]:
        if expected_layers <= 0:
            return []
        required_chunks = max(1, int(required_chunks))
        missing = []
        layers = layers or []
        for layer_id in range(expected_layers):
            if layer_id >= len(layers):
                missing.append(layer_id)
                continue
            layer_cache = layers[layer_id]
            if layer_cache is None or len(layer_cache) < required_chunks:
                missing.append(layer_id)
        return missing

    @staticmethod
    def _missing_shared_pointer_cache_layers(
        layers: list[list[Any]],
        chunk_ptrs: list[Optional[torch.Tensor]],
        required_chunks: int = 1,
    ) -> list[int]:
        missing: list[int] = []
        for layer_id, layer_entries in enumerate(layers or []):
            if not layer_entries:
                continue
            if layer_id >= len(chunk_ptrs):
                missing.append(layer_id)
                continue
            if not LMCacheConnectorV1Impl._shared_pointer_cache_entry_covers(
                chunk_ptrs[layer_id],
                required_chunks,
            ):
                missing.append(layer_id)
        return missing

    def _validate_decode_save_shared_pointer_cache(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> None:
        required_latent_chunks = self._shared_required_chunk_count(
            state.cached_starts,
            state.cached_ends,
            state.cached_memory_objs,
        )
        missing_latent_layers = self._missing_shared_pointer_cache_layers(
            state.cached_memory_objs,
            state.cached_chunk_ptrs_npu,
            required_latent_chunks,
        )
        if missing_latent_layers:
            raise RuntimeError(
                "Decode-window save merge produced incomplete shared CPU MLA "
                "latent pointer cache before refreshing the hot request "
                "scope: "
                f"req_id={request.req_id}, kv_group=0, "
                f"missing_layers={missing_latent_layers}"
            )

        if (
            self._is_dsa_two_groups()
            and self._sparse_decode_requires_index_materialization(request, True)
        ):
            expected_index_layers = int(getattr(self, "num_layers", 0) or 0)
            if expected_index_layers <= 0:
                expected_index_layers = len(state.cached_memory_objs_indexer or [])
            required_index_chunks = max(
                required_latent_chunks,
                self._shared_required_chunk_count(
                    state.cached_starts_indexer,
                    state.cached_ends_indexer,
                    state.cached_memory_objs_indexer,
                ),
            )
            missing_index_layers = self._missing_shared_layer_cache_coverage(
                state.cached_memory_objs_indexer,
                expected_index_layers,
                required_index_chunks,
            )
            if missing_index_layers:
                raise RuntimeError(
                    "Decode-window save merge produced incomplete shared CPU "
                    "DSA index state before refreshing the hot request scope: "
                    f"req_id={request.req_id}, kv_group=1, "
                    f"missing_layers={missing_index_layers}"
                )
            missing_index_layers = self._missing_shared_pointer_cache_layers(
                state.cached_memory_objs_indexer,
                state.cached_chunk_ptrs_npu_indexer,
                required_index_chunks,
            )
            if missing_index_layers:
                raise RuntimeError(
                    "Decode-window save merge produced incomplete shared CPU "
                    "DSA index pointer cache before refreshing the hot "
                    "request scope: "
                    f"req_id={request.req_id}, kv_group=1, "
                    f"missing_layers={missing_index_layers}"
                )

    @staticmethod
    def _missing_required_shared_layers(
        layers: list[list[Any]],
        expected_layers: int,
    ) -> list[int]:
        if expected_layers <= 0:
            return []
        missing = []
        for layer_id in range(expected_layers):
            if layer_id >= len(layers) or not layers[layer_id]:
                missing.append(layer_id)
        return missing

    @staticmethod
    def _copy_shared_layer_map(
        layer_map: dict[int, list[list[Any]]],
    ) -> dict[int, list[list[Any]]]:
        return {
            kv_group: [list(layer) for layer in layers]
            for kv_group, layers in layer_map.items()
        }

    @staticmethod
    def _copy_shared_ptr_map(
        ptr_map: dict[int, list[Optional[torch.Tensor]]],
    ) -> dict[int, list[Optional[torch.Tensor]]]:
        return {
            kv_group: list(ptrs)
            for kv_group, ptrs in ptr_map.items()
        }

    @staticmethod
    def _copy_layer_cache(cache: list[list[Any]]) -> list[list[Any]]:
        return [
            list(layer_cache) if isinstance(layer_cache, list) else layer_cache
            for layer_cache in (cache or [])
        ]

    def _snapshot_worker_retrieve_cache_state(
        self,
        state: WorkerRetrieveState,
    ) -> dict[str, Any]:
        return {
            "cached_starts": list(state.cached_starts),
            "cached_ends": list(state.cached_ends),
            "cached_keys": self._copy_layer_cache(state.cached_keys),
            "cached_memory_objs": self._copy_layer_cache(state.cached_memory_objs),
            "cached_tensors": self._copy_layer_cache(state.cached_tensors),
            "cached_chunk_dev_ptrs": self._copy_layer_cache(
                state.cached_chunk_dev_ptrs
            ),
            "cached_chunk_ptrs_npu": list(state.cached_chunk_ptrs_npu),
            "cached_shared_handles": self._copy_layer_cache(
                state.cached_shared_handles
            ),
            "cached_starts_indexer": list(state.cached_starts_indexer),
            "cached_ends_indexer": list(state.cached_ends_indexer),
            "cached_keys_indexer": self._copy_layer_cache(
                state.cached_keys_indexer
            ),
            "cached_memory_objs_indexer": self._copy_layer_cache(
                state.cached_memory_objs_indexer
            ),
            "cached_tensors_indexer": self._copy_layer_cache(
                state.cached_tensors_indexer
            ),
            "cached_chunk_dev_ptrs_indexer": self._copy_layer_cache(
                state.cached_chunk_dev_ptrs_indexer
            ),
            "cached_chunk_ptrs_npu_indexer": list(
                state.cached_chunk_ptrs_npu_indexer
            ),
            "cached_shared_handles_indexer": self._copy_layer_cache(
                state.cached_shared_handles_indexer
            ),
            "shared_chunk_ptrs_npu_by_group": self._copy_shared_ptr_map(
                state.shared_chunk_ptrs_npu_by_group
            ),
            "location": state.location,
            "metadata_warm": state.metadata_warm,
            "token_count": state.token_count,
            "shared_generation": state.shared_generation,
            "pointer_cache_generation": state.pointer_cache_generation,
            "request_scope_token": state.request_scope_token,
            "shared_validation_signature": state.shared_validation_signature,
        }

    @staticmethod
    def _restore_worker_retrieve_cache_state(
        state: WorkerRetrieveState,
        snapshot: dict[str, Any],
    ) -> None:
        for attr, value in snapshot.items():
            setattr(state, attr, value)

    def _release_unstored_shared_request_objects(
        self,
        request: ReqMeta,
        old_state: Optional[WorkerRetrieveState],
    ) -> None:
        engine = getattr(self, "lmcache_engine", None)
        if (
            engine is None
            or not getattr(engine, "enable_shared_cpu_cache", False)
            or not request.is_sparse_decode
        ):
            return

        _retrieve_cache_kwargs(request, kv_group=0, dsa_two_groups=False)
        _retrieve_cache_kwargs(request, kv_group=1, dsa_two_groups=True)

        inherited_ids: set[int] = set()
        if old_state is not None:
            for group_map in (
                old_state.shared_views_by_group,
                old_state.rank0_backing_objs_by_group,
            ):
                for layers in group_map.values():
                    for layer_objs in layers:
                        inherited_ids.update(id(mem_obj) for mem_obj in layer_objs)

        def filter_new(layers: list[list[Any]]) -> list[list[Any]]:
            return [
                [
                    mem_obj
                    for mem_obj in layer_objs
                    if id(mem_obj) not in inherited_ids
                ]
                for layer_objs in (layers or [])
            ]

        groups = [
            (0, filter_new(request.cached_memory_objs)),
        ]
        if request.cached_memory_objs_indexer:
            groups.append((1, filter_new(request.cached_memory_objs_indexer)))

        if not any(any(layer for layer in layers) for _, layers in groups):
            return

        metadata = getattr(engine, "metadata", None)
        is_first_rank_fn = getattr(metadata, "is_first_rank", None)
        is_rank0 = bool(is_first_rank_fn()) if callable(is_first_rank_fn) else False
        temp_state = WorkerRetrieveState(req_id=request.req_id)
        for kv_group, layers in groups:
            if not any(layer for layer in layers):
                continue
            if is_rank0:
                temp_state.rank0_backing_objs_by_group[kv_group] = layers
            else:
                temp_state.shared_views_by_group[kv_group] = layers
        self._release_shared_worker_retrieve_state(temp_state)

    def _record_shared_worker_retrieve_state(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> None:
        engine = self.lmcache_engine
        if (
            engine is None
            or not getattr(engine, "enable_shared_cpu_cache", False)
            or not request.is_sparse_decode
        ):
            return

        _retrieve_cache_kwargs(request, kv_group=0, dsa_two_groups=False)
        if self._is_dsa_two_groups():
            _retrieve_cache_kwargs(request, kv_group=1, dsa_two_groups=True)

        generation = int(getattr(engine, "shared_cpu_cache_generation", 0) or 0)
        metadata = getattr(engine, "metadata", None)
        is_first_rank_fn = getattr(metadata, "is_first_rank", None)
        is_rank0 = bool(is_first_rank_fn()) if callable(is_first_rank_fn) else False

        def layer_has_entries(layers: list[list]) -> bool:
            return bool(layers and any(layer for layer in layers))

        expected_layers = int(getattr(self, "num_layers", 0) or 0)

        pending_handles_by_group: dict[int, list[list[Any]]] = {}
        pending_views_by_group: dict[int, list[list[Any]]] = {}
        pending_backing_by_group: dict[int, list[list[Any]]] = {}
        pending_chunk_ptrs_by_group: dict[int, list[Optional[torch.Tensor]]] = {}
        materialize_index = (
            self._is_dsa_two_groups()
            and self._sparse_decode_requires_index_materialization(
                request,
                True,
            )
        )
        skip_index_hot_state = self._is_dsa_two_groups() and not materialize_index

        groups: list[tuple[int, list[list], list[list]]] = [
            (0, request.cached_memory_objs, request.cached_shared_handles),
        ]
        required_latent_chunks = self._shared_required_chunk_count(
            request.cached_starts,
            request.cached_ends,
            request.cached_memory_objs,
        )
        missing_latent_layers = self._missing_shared_layer_cache_coverage(
            request.cached_memory_objs,
            expected_layers,
            required_latent_chunks,
        )
        if missing_latent_layers:
            raise RuntimeError(
                "Shared CPU sparse decode cannot mark request state "
                "hot-reusable with incomplete MLA latent state: "
                f"req_id={request.req_id}, kv_group=0, "
                f"missing_layers={missing_latent_layers}"
            )
        required_index_chunks = max(
            required_latent_chunks,
            self._shared_required_chunk_count(
                request.cached_starts_indexer,
                request.cached_ends_indexer,
                request.cached_memory_objs_indexer,
            ),
        )
        missing_index_layers = self._missing_shared_layer_cache_coverage(
            request.cached_memory_objs_indexer,
            expected_layers,
            required_index_chunks,
        )
        if materialize_index and missing_index_layers:
            raise RuntimeError(
                "Shared CPU sparse decode cannot mark request state "
                "hot-reusable without complete materialized DSA index state: "
                f"req_id={request.req_id}, kv_group=1, "
                f"missing_layers={missing_index_layers}"
            )
        if materialize_index and request.cached_memory_objs_indexer:
            groups.append(
                (
                    1,
                    request.cached_memory_objs_indexer,
                    request.cached_shared_handles_indexer,
                )
            )

        for kv_group, layers, handles in groups:
            if not layer_has_entries(layers):
                continue
            required_token_count = (
                state.token_count
                if state.token_count > 0
                else self._shared_retrieve_token_count_for_request(request)
            )
            range_starts = (
                request.cached_starts
                if kv_group == 0
                else request.cached_starts_indexer
            )
            range_ends = (
                request.cached_ends
                if kv_group == 0
                else request.cached_ends_indexer
            )
            if (range_starts or range_ends) and not self._cached_ranges_cover_prefix(
                range_starts,
                range_ends,
                required_token_count,
            ):
                cached_ranges = list(zip(range_starts, range_ends, strict=False))
                raise RuntimeError(
                    "Shared CPU sparse decode cannot mark request state "
                    "hot-reusable with non-contiguous prefix coverage: "
                    f"req_id={request.req_id}, kv_group={kv_group}, "
                    f"token_count={required_token_count}, "
                    f"cached_ranges={cached_ranges}"
                )
            chunk_ptrs = (
                request.cached_chunk_ptrs_npu
                if kv_group == 0
                else request.cached_chunk_ptrs_npu_indexer
            )
            missing_pointer_layers = self._missing_shared_pointer_cache_layers(
                layers,
                chunk_ptrs,
                required_latent_chunks if kv_group == 0 else required_index_chunks,
            )
            if missing_pointer_layers:
                raise RuntimeError(
                    "Shared CPU sparse decode cannot mark request state "
                    "hot-reusable before NPU pointer-cache install: "
                    f"req_id={request.req_id}, kv_group={kv_group}, "
                    f"missing_layers={missing_pointer_layers}"
                )
            if layer_has_entries(handles):
                pending_handles_by_group[kv_group] = handles
            if is_rank0:
                pending_backing_by_group[kv_group] = layers
            else:
                pending_views_by_group[kv_group] = layers
            if chunk_ptrs:
                pending_chunk_ptrs_by_group[kv_group] = chunk_ptrs

        has_shared_request = bool(
            pending_views_by_group or pending_backing_by_group
        )
        if has_shared_request:
            if state.token_count <= 0:
                state.token_count = self._shared_retrieve_token_count_for_request(
                    request
                )
            replaced_views_by_group = {
                kv_group: state.shared_views_by_group[kv_group]
                for kv_group in pending_views_by_group
                if kv_group in state.shared_views_by_group
            }
            replaced_backing_by_group = {
                kv_group: state.rank0_backing_objs_by_group[kv_group]
                for kv_group in pending_backing_by_group
                if kv_group in state.rank0_backing_objs_by_group
            }
            state.shared_handles_by_group.update(pending_handles_by_group)
            state.shared_views_by_group.update(pending_views_by_group)
            state.rank0_backing_objs_by_group.update(pending_backing_by_group)
            state.shared_chunk_ptrs_npu_by_group.update(
                pending_chunk_ptrs_by_group
            )
            state.req_id = request.req_id
            state.shared_generation = generation
            state.pointer_cache_generation = generation
            state.request_scope_token = self._shared_request_scope_token(
                request.req_id,
                generation,
                state.token_count,
            )
            state.shared_latent_status = (
                "present" if layer_has_entries(request.cached_memory_objs)
                else "missing"
            )
            state.shared_index_status = (
                "present"
                if (
                    materialize_index
                    and layer_has_entries(request.cached_memory_objs_indexer)
                )
                else "skipped"
                if (
                    getattr(request, "shared_index_skipped", False)
                    or state.shared_index_status == "skipped"
                    or skip_index_hot_state
                )
                else "missing"
            )
            state.shared_request_active = True
            state.shared_validation_signature = (
                self._shared_worker_validation_signature(
                    state,
                    request,
                    current_generation=generation,
                    pointer_generation=generation,
                    materialize_index=materialize_index,
                )
            )
            if is_rank0:
                register_fn = getattr(
                    engine,
                    "register_shared_cpu_sparse_request",
                    None,
                )
                if callable(register_fn):
                    register_fn(
                        request.req_id,
                        token_count=state.token_count,
                        phase=SPARSE_DECODE_SHARED_CPU_PHASE,
                    )
            self._release_replaced_shared_groups(
                replaced_views_by_group,
                pending_views_by_group,
                rank0_backing=False,
            )
            self._release_replaced_shared_groups(
                replaced_backing_by_group,
                pending_backing_by_group,
                rank0_backing=True,
            )

    def _should_invalidate_worker_retrieve_state(
        self, request: ReqMeta, token_count: int
    ) -> bool:
        if request.resumed_from_preemption:
            if _DSA_PROF:
                print(
                    f"[WORKER_RETRIEVE_STATE] req={request.req_id} "
                    f"event=invalidate reason=preemption",
                    flush=True,
                )
            return True
        state = self._worker_retrieve_state.get(request.req_id)
        if state is None:
            return False
        if request.is_sparse_decode:
            if state.shared_request_active:
                engine = getattr(self, "lmcache_engine", None)
                generation = int(
                    getattr(engine, "shared_cpu_cache_generation", 0) or 0
                )
                expected_scope_token = self._shared_request_scope_token(
                    request.req_id,
                    generation,
                    token_count,
                )
                if state.request_scope_token != expected_scope_token:
                    if _DSA_PROF:
                        print(
                            f"[WORKER_RETRIEVE_STATE] req={request.req_id} "
                            f"event=invalidate reason=scope_mismatch "
                            f"load_cached={request.load_spec.lmcache_cached_tokens if request.load_spec else None} "
                            f"state_token_count={state.token_count} "
                            f"token_count={token_count}",
                            flush=True,
                        )
                    return True
            if state.cached_starts and state.cached_starts[0] != 0:
                if _DSA_PROF:
                    print(
                        f"[WORKER_RETRIEVE_STATE] req={request.req_id} "
                        f"event=invalidate reason=cached_starts_nonzero",
                        flush=True,
                    )
                return True
            if (
                request.load_spec is not None
                and request.load_spec.lmcache_cached_tokens > state.token_count
            ):
                if _DSA_PROF:
                    print(
                        f"[WORKER_RETRIEVE_STATE] req={request.req_id} "
                        f"event=invalidate reason=cached_grew "
                        f"load_cached={request.load_spec.lmcache_cached_tokens} "
                        f"state_token_count={state.token_count}",
                        flush=True,
                    )
                return True
            # Sparse decode metadata is keyed by the full LMCache-hit prefix.
            # A shorter current prefix means the cached request state is stale.
            if state.token_count and (
                token_count < state.token_count
                or len(request.token_ids) < state.token_count
            ):
                if _DSA_PROF:
                    print(
                        f"[WORKER_RETRIEVE_STATE] req={request.req_id} "
                        f"event=invalidate reason=prefix_shrunk "
                        f"token_count={token_count} "
                        f"state_token_count={state.token_count}",
                        flush=True,
                    )
                return True
            if _DSA_PROF:
                print(
                    f"[WORKER_RETRIEVE_STATE] req={request.req_id} "
                    f"event=reuse reason=ok "
                    f"load_cached={request.load_spec.lmcache_cached_tokens if request.load_spec else None} "
                    f"state_token_count={state.token_count} "
                    f"token_count={token_count} "
                    f"shared_active={state.shared_request_active}",
                    flush=True,
                )
            return False
        if state.cached_ends and token_count < state.cached_ends[-1]:
            return True
        if (
            request.load_spec is not None
            and request.load_spec.lmcache_cached_tokens > state.token_count
        ):
            return True
        return False

    def _bind_worker_retrieve_state_to_request(
        self, request: ReqMeta
    ) -> Optional[WorkerRetrieveState]:
        state = self._worker_retrieve_state.get(request.req_id)
        if state is None or not (state.metadata_warm or state.cached_keys):
            return None
        self._validate_shared_worker_retrieve_state(state, request)
        request.cached_keys = state.cached_keys
        request.cached_starts = state.cached_starts
        request.cached_ends = state.cached_ends
        request.cached_memory_objs = state.cached_memory_objs
        request.cached_tensors = state.cached_tensors
        request.cached_chunk_dev_ptrs = state.cached_chunk_dev_ptrs
        request.cached_chunk_ptrs_npu = state.cached_chunk_ptrs_npu
        request.cached_shared_handles = state.cached_shared_handles
        request.cached_keys_indexer = state.cached_keys_indexer
        request.cached_starts_indexer = state.cached_starts_indexer
        request.cached_ends_indexer = state.cached_ends_indexer
        request.cached_memory_objs_indexer = state.cached_memory_objs_indexer
        request.cached_tensors_indexer = state.cached_tensors_indexer
        request.cached_chunk_dev_ptrs_indexer = state.cached_chunk_dev_ptrs_indexer
        request.cached_chunk_ptrs_npu_indexer = state.cached_chunk_ptrs_npu_indexer
        request.cached_shared_handles_indexer = state.cached_shared_handles_indexer
        return state

    def _request_has_retrieve_tensor_cache(self, request: ReqMeta) -> bool:
        num_layers = self._num_layers_for_group(0)
        tensors = request.cached_tensors
        if num_layers <= 0:
            if tensors and any(tensors):
                return True
            mem = request.cached_memory_objs
            return bool(mem and any(mem))
        if tensors and len(tensors) == num_layers and any(tensors):
            return True
        mem = request.cached_memory_objs
        return bool(mem and len(mem) == num_layers and any(mem))

    def _resolve_store_retrieve_location(self, request: ReqMeta) -> Optional[str]:
        engine = self.lmcache_engine
        if engine is None or not request.cached_keys or not request.cached_keys[0]:
            return None
        storage_manager = getattr(engine, "storage_manager", None)
        if storage_manager is None:
            return getattr(engine, "store_location", None)
        return storage_manager.contains(
            request.cached_keys[0][0],
            getattr(engine, "retrieve_locations", None),
        )

    @staticmethod
    def _ensure_layer_cache_shape(dst: list, src: list) -> None:
        if not src:
            return
        if not dst:
            dst.extend([] for _ in range(len(src)))
        while len(dst) < len(src):
            dst.append([])

    @classmethod
    def _merge_cache_group_by_ranges(
        cls,
        *,
        dst_starts: list[int],
        dst_ends: list[int],
        dst_keys: list[list],
        dst_memory_objs: list[list],
        dst_tensors: list[list],
        dst_chunk_dev_ptrs: list[list[int]],
        dst_chunk_ptrs_npu: list[Optional[torch.Tensor]],
        dst_shared_handles: list[list[Any]],
        src_starts: list[int],
        src_ends: list[int],
        src_keys: list[list],
        src_memory_objs: list[list],
        src_tensors: list[list],
        src_chunk_dev_ptrs: list[list[int]],
        src_chunk_ptrs_npu: list[Optional[torch.Tensor]],
        src_shared_handles: list[list[Any]],
        require_pointer_cache: bool = False,
    ) -> int:
        if not src_starts or not src_ends:
            return 0

        existing_ranges = set(zip(dst_starts, dst_ends, strict=False))
        append_indices: list[int] = []
        for chunk_idx, chunk_range in enumerate(
            zip(src_starts, src_ends, strict=False)
        ):
            if chunk_range in existing_ranges:
                continue
            existing_ranges.add(chunk_range)
            append_indices.append(chunk_idx)

        if not append_indices:
            return 0

        def selected_ptrs_from_source(src_ptrs: torch.Tensor) -> Optional[torch.Tensor]:
            if not isinstance(src_ptrs, torch.Tensor):
                return None
            if any(chunk_idx >= int(src_ptrs.numel()) for chunk_idx in append_indices):
                return None
            if (
                append_indices[0] == 0
                and append_indices[-1] == len(append_indices) - 1
            ):
                return src_ptrs[: len(append_indices)]
            if append_indices[-1] - append_indices[0] + 1 == len(append_indices):
                return src_ptrs[append_indices[0] : append_indices[-1] + 1]
            return None

        def select_layer_ptr_tensors() -> Optional[list[torch.Tensor]]:
            if not src_chunk_ptrs_npu:
                return None
            selected_by_layer: list[torch.Tensor] = []
            for src_ptrs in src_chunk_ptrs_npu:
                selected = selected_ptrs_from_source(src_ptrs)
                if selected is None:
                    return None
                selected_by_layer.append(selected)
            return selected_by_layer

        selected_ptrs_by_layer = select_layer_ptr_tensors()
        source_layer_count = max(
            len(src_memory_objs or []),
            len(src_tensors or []),
            len(src_keys or []),
        )

        def can_append_layer_ptr_tensors() -> bool:
            if selected_ptrs_by_layer is None:
                return False
            if require_pointer_cache and len(selected_ptrs_by_layer) < source_layer_count:
                return False
            for layer_id in range(len(selected_ptrs_by_layer)):
                if layer_id >= len(dst_chunk_ptrs_npu):
                    continue
                existing = dst_chunk_ptrs_npu[layer_id]
                if existing is not None and not isinstance(existing, torch.Tensor):
                    return False
            return True

        if require_pointer_cache and not can_append_layer_ptr_tensors():
            return 0

        for chunk_idx in append_indices:
            dst_starts.append(src_starts[chunk_idx])
            dst_ends.append(src_ends[chunk_idx])

        def append_layer_values(dst: list, src: list) -> None:
            if not src:
                return
            cls._ensure_layer_cache_shape(dst, src)
            for layer_id, src_layer in enumerate(src):
                for chunk_idx in append_indices:
                    if chunk_idx < len(src_layer):
                        dst[layer_id].append(src_layer[chunk_idx])

        append_layer_values(dst_keys, src_keys)
        append_layer_values(dst_memory_objs, src_memory_objs)
        append_layer_values(dst_tensors, src_tensors)
        append_layer_values(dst_chunk_dev_ptrs, src_chunk_dev_ptrs)
        append_layer_values(dst_shared_handles, src_shared_handles)

        def append_layer_ptr_tensors() -> bool:
            if not can_append_layer_ptr_tensors():
                return False
            if not dst_chunk_ptrs_npu:
                dst_chunk_ptrs_npu.extend(
                    None for _ in range(len(src_chunk_ptrs_npu))
                )
            while len(dst_chunk_ptrs_npu) < len(src_chunk_ptrs_npu):
                dst_chunk_ptrs_npu.append(None)

            for layer_id, selected in enumerate(selected_ptrs_by_layer):
                existing = dst_chunk_ptrs_npu[layer_id]
                dst_chunk_ptrs_npu[layer_id] = (
                    selected
                    if existing is None
                    else torch.cat((existing, selected))
                )
            return True

        if not append_layer_ptr_tensors():
            if dst_chunk_ptrs_npu:
                dst_chunk_ptrs_npu.clear()
            if src_chunk_ptrs_npu and not dst_chunk_ptrs_npu:
                dst_chunk_ptrs_npu.extend(None for _ in range(len(src_chunk_ptrs_npu)))
        return len(append_indices)

    def _merge_store_cache_into_worker_state(
        self,
        state: WorkerRetrieveState,
        request: ReqMeta,
    ) -> int:
        _retrieve_cache_kwargs(request, kv_group=0, dsa_two_groups=False)
        _retrieve_cache_kwargs(request, kv_group=1, dsa_two_groups=True)
        require_pointer_cache = (
            self._is_decode_window_save_request(request)
            and bool(
                getattr(
                    getattr(self, "lmcache_engine", None),
                    "enable_shared_cpu_cache",
                    False,
                )
            )
        )
        merged_chunks = self._merge_cache_group_by_ranges(
            dst_starts=state.cached_starts,
            dst_ends=state.cached_ends,
            dst_keys=state.cached_keys,
            dst_memory_objs=state.cached_memory_objs,
            dst_tensors=state.cached_tensors,
            dst_chunk_dev_ptrs=state.cached_chunk_dev_ptrs,
            dst_chunk_ptrs_npu=state.cached_chunk_ptrs_npu,
            dst_shared_handles=state.cached_shared_handles,
            src_starts=request.cached_starts,
            src_ends=request.cached_ends,
            src_keys=request.cached_keys,
            src_memory_objs=request.cached_memory_objs,
            src_tensors=request.cached_tensors,
            src_chunk_dev_ptrs=request.cached_chunk_dev_ptrs,
            src_chunk_ptrs_npu=request.cached_chunk_ptrs_npu,
            src_shared_handles=request.cached_shared_handles,
            require_pointer_cache=require_pointer_cache,
        )
        merged_chunks += self._merge_cache_group_by_ranges(
            dst_starts=state.cached_starts_indexer,
            dst_ends=state.cached_ends_indexer,
            dst_keys=state.cached_keys_indexer,
            dst_memory_objs=state.cached_memory_objs_indexer,
            dst_tensors=state.cached_tensors_indexer,
            dst_chunk_dev_ptrs=state.cached_chunk_dev_ptrs_indexer,
            dst_chunk_ptrs_npu=state.cached_chunk_ptrs_npu_indexer,
            dst_shared_handles=state.cached_shared_handles_indexer,
            src_starts=request.cached_starts_indexer,
            src_ends=request.cached_ends_indexer,
            src_keys=request.cached_keys_indexer,
            src_memory_objs=request.cached_memory_objs_indexer,
            src_tensors=request.cached_tensors_indexer,
            src_chunk_dev_ptrs=request.cached_chunk_dev_ptrs_indexer,
            src_chunk_ptrs_npu=request.cached_chunk_ptrs_npu_indexer,
            src_shared_handles=request.cached_shared_handles_indexer,
            require_pointer_cache=require_pointer_cache,
        )
        return merged_chunks

    def _warm_request_retrieve_metadata(
        self,
        request: ReqMeta,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        kv_group: int,
        dsa_two_groups: bool,
    ) -> tuple[Optional[str], bool]:
        engine = self.lmcache_engine
        if (
            engine is not None
            and getattr(engine, "enable_shared_cpu_cache", False)
            and getattr(engine, "storage_manager", None) is None
        ):
            is_passive_fn = getattr(engine, "_is_passive", None)
            if callable(is_passive_fn) and is_passive_fn():
                return None, False

        ensure_metadata = getattr(
            engine, "_ensure_retrieve_chunk_metadata", None
        )
        if ensure_metadata is None:
            return None, False

        cache_kwargs = _retrieve_cache_kwargs(
            request, kv_group=kv_group, dsa_two_groups=dsa_two_groups
        )
        cached_keys = cache_kwargs["cached_keys"]
        cached_starts = cache_kwargs["cached_starts"]
        cached_ends = cache_kwargs["cached_ends"]
        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
        retrieve_kwargs: dict[str, Any] = {"kv_group": kv_group}

        location, _, _, _ = ensure_metadata(
            tokens=tokens,
            mask=mask,
            request_configs=request.request_configs,
            cached_keys=cached_keys,
            cached_starts=cached_starts,
            cached_ends=cached_ends,
            ret_mask=ret_mask,
            retrieve_kwargs=retrieve_kwargs,
        )
        if location is None:
            location = retrieve_kwargs.get("cached_retrieve_location")
        metadata_warm = bool(
            retrieve_kwargs.get("_retrieve_metadata_warm")
            and cached_keys
            and cached_ends
        )
        return location, metadata_warm

    def _maybe_seed_worker_retrieve_state_from_store(
        self, request: ReqMeta
    ) -> None:
        """Keep prefill store warm cache on the worker for sparse decode reload."""
        if not hasattr(self, "_worker_retrieve_state"):
            return
        if request.is_sparse_decode:
            return
        if not request.cached_keys or not request.cached_starts or not request.cached_ends:
            return
        if not self._request_has_retrieve_tensor_cache(request):
            return

        if (
            self._is_decode_window_save_request(request)
            and self._decode_window_save_uses_shared_cpu()
        ):
            # Rank0 is the only worker that owns newly saved shared CPU backing
            # objects during decode-window save. If it privately merges those
            # chunks into its hot sparse state, the next decode step can diverge:
            # rank0 takes the cached path while passive ranks wait for a fresh
            # broadcast. Leave all ranks on the old scope; the next sparse load
            # invalidates it and refreshes shared handles once, in ordered TP
            # collective flow.
            logger.debug(
                "Skipping shared CPU decode-window worker-state merge so the "
                "next sparse load refreshes handles on every TP rank: req_id=%s "
                "window=[%s,%s)",
                request.req_id,
                getattr(request, "decode_window_start", None),
                getattr(request, "decode_window_end", None),
            )
            return

        location = self._resolve_store_retrieve_location(request)
        existing_state = self._worker_retrieve_state.get(request.req_id)
        if existing_state is not None and (
            existing_state.metadata_warm or existing_state.cached_keys
        ):
            if (
                self._is_decode_window_save_request(request)
                and not self._decode_window_save_is_next_expected(request)
            ):
                logger.debug(
                    "Skipping decode-window warm-state merge for out-of-order "
                    "window: req_id=%s window_start=%s",
                    request.req_id,
                    getattr(request, "decode_window_start", None),
                )
                return
            if (
                self._is_decode_window_save_request(request)
                and not self._decode_window_save_store_cache_ready(request)
            ):
                logger.debug(
                    "Skipping decode-window warm-state merge before shared CPU "
                    "store cache is fully pointer-ready: req_id=%s",
                    request.req_id,
                )
                return
            rollback_snapshot = (
                self._snapshot_worker_retrieve_cache_state(existing_state)
                if existing_state.shared_request_active
                else None
            )
            try:
                merged_chunks = self._merge_store_cache_into_worker_state(
                    existing_state, request
                )
                existing_state.location = location or existing_state.location
                existing_state.metadata_warm = True
                next_token_count = max(
                    existing_state.token_count,
                    len(request.token_ids),
                    request.cached_ends[-1] if request.cached_ends else 0,
                )
                if self._is_decode_window_save_request(request):
                    next_token_count = min(
                        next_token_count,
                        self._cached_prefix_covered_token_count(
                            existing_state.cached_starts,
                            existing_state.cached_ends,
                        ),
                    )
                existing_state.token_count = next_token_count
                if existing_state.shared_request_active:
                    self._validate_decode_save_shared_pointer_cache(
                        existing_state,
                        request,
                    )
                    engine = getattr(self, "lmcache_engine", None)
                    generation = int(
                        getattr(engine, "shared_cpu_cache_generation", 0) or 0
                    )
                    existing_state.shared_chunk_ptrs_npu_by_group[0] = (
                        existing_state.cached_chunk_ptrs_npu
                    )
                    if existing_state.cached_chunk_ptrs_npu_indexer:
                        existing_state.shared_chunk_ptrs_npu_by_group[1] = (
                            existing_state.cached_chunk_ptrs_npu_indexer
                        )
                    existing_state.shared_generation = generation
                    existing_state.pointer_cache_generation = generation
                    existing_state.request_scope_token = (
                        self._shared_request_scope_token(
                            request.req_id,
                            generation,
                            existing_state.token_count,
                        )
                    )
                    existing_state.shared_validation_signature = None
            except Exception:
                if rollback_snapshot is not None:
                    self._restore_worker_retrieve_cache_state(
                        existing_state,
                        rollback_snapshot,
                    )
                raise
            return

        if (
            self._is_decode_window_save_request(request)
            and not self._cached_ranges_cover_prefix(
                request.cached_starts,
                request.cached_ends,
                len(request.token_ids),
            )
        ):
            logger.debug(
                "Skipping worker retrieve-state seed for decode-window save "
                "without full-prefix cache coverage: req_id=%s ranges=%s "
                "token_count=%d",
                request.req_id,
                list(zip(request.cached_starts, request.cached_ends, strict=False)),
                len(request.token_ids),
            )
            return

        self._save_worker_retrieve_state_from_request(
            request,
            location=location,
            metadata_warm=True,
            token_count=len(request.token_ids),
        )

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
        _retrieve_cache_kwargs(request, kv_group=0, dsa_two_groups=False)
        _retrieve_cache_kwargs(request, kv_group=1, dsa_two_groups=True)
        if not metadata_warm and not request.cached_keys:
            return
        old_state = self._worker_retrieve_state.get(request.req_id)
        new_state = WorkerRetrieveState(
            req_id=request.req_id,
            cached_keys=request.cached_keys,
            cached_starts=request.cached_starts,
            cached_ends=request.cached_ends,
            cached_memory_objs=request.cached_memory_objs,
            cached_tensors=request.cached_tensors,
            cached_chunk_dev_ptrs=request.cached_chunk_dev_ptrs,
            cached_chunk_ptrs_npu=request.cached_chunk_ptrs_npu,
            cached_shared_handles=request.cached_shared_handles,
            cached_keys_indexer=request.cached_keys_indexer,
            cached_starts_indexer=request.cached_starts_indexer,
            cached_ends_indexer=request.cached_ends_indexer,
            cached_memory_objs_indexer=request.cached_memory_objs_indexer,
            cached_tensors_indexer=request.cached_tensors_indexer,
            cached_chunk_dev_ptrs_indexer=request.cached_chunk_dev_ptrs_indexer,
            cached_chunk_ptrs_npu_indexer=request.cached_chunk_ptrs_npu_indexer,
            cached_shared_handles_indexer=request.cached_shared_handles_indexer,
            location=location,
            metadata_warm=metadata_warm,
            token_count=token_count,
        )
        if old_state is not None:
            if not new_state.cached_shared_handles:
                new_state.cached_shared_handles = old_state.cached_shared_handles
            if not new_state.cached_shared_handles_indexer:
                new_state.cached_shared_handles_indexer = (
                    old_state.cached_shared_handles_indexer
                )
            new_state.shared_handles_by_group = self._copy_shared_layer_map(
                old_state.shared_handles_by_group
            )
            new_state.shared_views_by_group = self._copy_shared_layer_map(
                old_state.shared_views_by_group
            )
            new_state.shared_chunk_ptrs_npu_by_group = self._copy_shared_ptr_map(
                old_state.shared_chunk_ptrs_npu_by_group
            )
            new_state.rank0_backing_objs_by_group = self._copy_shared_layer_map(
                old_state.rank0_backing_objs_by_group
            )
            new_state.shared_latent_status = old_state.shared_latent_status
            new_state.shared_index_status = old_state.shared_index_status
            new_state.shared_generation = old_state.shared_generation
            new_state.pointer_cache_generation = (
                old_state.pointer_cache_generation
            )
            new_state.shared_request_active = old_state.shared_request_active
            new_state.request_scope_token = old_state.request_scope_token
            new_state.shared_validation_signature = None
        try:
            self._record_shared_worker_retrieve_state(new_state, request)
        except Exception:
            self._release_unstored_shared_request_objects(request, old_state)
            raise
        self._worker_retrieve_state[request.req_id] = new_state

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
            token_count = len(request.token_ids)
            if request.load_spec is not None:
                token_count = int(request.load_spec.lmcache_cached_tokens)
            if (
                existing is not None
                and self._shared_worker_retrieve_state_is_current(
                    existing,
                    request,
                    token_count,
                )
            ):
                continue
            self._save_worker_retrieve_state_from_request(
                request,
                location=location,
                metadata_warm=metadata_warm or bool(request.cached_keys),
                token_count=token_count,
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
            and token_count <= bound_state.token_count
            and (not bound_state.cached_starts or bound_state.cached_starts[0] == 0)
        ):
            warm_kwargs["_retrieve_metadata_warm"] = True
        return warm_kwargs

    @staticmethod
    def _prime_dense_prefix_retrievers(
        layerwise_retriever: Generator[Optional[torch.Tensor], None, None],
        indexer_retriever: Optional[Generator[Optional[torch.Tensor], None, None]],
    ) -> None:
        """Prime dense prefix retrievers without breaking two-group ordering."""
        next(layerwise_retriever)
        if indexer_retriever is not None:
            next(indexer_retriever)
        next(layerwise_retriever)
        if indexer_retriever is not None:
            next(indexer_retriever)

    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._refresh_kvcaches_list()
        self._build_kv_layer_groups()
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

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        active_req_ids = {
            req.req_id
            for req in metadata.requests
            if not self._is_decode_window_save_request(req)
        }
        self._prune_worker_retrieve_state(active_req_ids)

        assert len(self.kv_caches) > 0
        if not self._kvcaches_list:
            self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_list

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return

        assert self.lmcache_engine is not None

        self._drain_layerwise_retrievers()
        self._layerwise_requests = []
        self._layerwise_sparse_req_ids = []
        self._layerwise_waited_groups = set()
        self._layerwise_sparse_indexer_sent_layers = set()
        self._layerwise_required_wait_groups_cache = None

        load_count = sum(
            1
            for req in metadata.requests
            if req.load_spec is not None and req.load_spec.can_load
        )
        gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
        if gpu_connector is not None and hasattr(
            gpu_connector, "set_layerwise_staging_concurrency"
        ):
            # Each loading request holds a staging buffer for the full layer
            # loop; add one slot for an overlapping layerwise store.
            gpu_connector.set_layerwise_staging_concurrency(
                max(2, load_count + 1)
            )

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
            recalc_last_applied = self._full_hit_recalc_last_token(
                request.load_spec,
                len(request.token_ids),
                is_sparse_decode=request.is_sparse_decode,
            )
            if recalc_last_applied:
                retrieve_tokens, slot_mapping = self._trim_prefill_for_recalc_last(
                    request, retrieve_tokens, slot_mapping
                )
            token_count = len(retrieve_tokens)
            token_mask = self._load_token_mask_for_retrieve(
                request, token_count, self._lmcache_chunk_size
            )
            if (
                not request.is_sparse_decode
                and token_count > len(slot_mapping)
            ):
                logger.warning(
                    "Request %s: retrieve_len=%d exceeds slot_mapping len=%d "
                    "(KV scatter will be incomplete -> garbage). "
                    "Often chunked-prefill metadata out of sync with lookup_hit.",
                    request.req_id,
                    token_count,
                    len(slot_mapping),
                )

            if self.use_layerwise or request.is_sparse_decode:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    if token_mask is None:
                        token_mask = torch.ones(token_count, dtype=torch.bool)
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
                        worker_state = self._worker_retrieve_state.get(
                            request.req_id
                        )
                    else:
                        bound_state = None

                    dsa_two_groups = self._is_dsa_two_groups()
                    shared_cpu_enabled = bool(
                        getattr(
                            self.lmcache_engine,
                            "enable_shared_cpu_cache",
                            False,
                        )
                    )
                    shared_cpu_preflight_state: Optional[dict[str, Any]] = (
                        {} if shared_cpu_enabled and dsa_two_groups else None
                    )
                    latent_cache = _retrieve_cache_kwargs(
                        request, kv_group=0, dsa_two_groups=dsa_two_groups
                    )
                    retrieve_kwargs: dict[str, Any] = {
                        "kvcaches": kvcaches,
                        "slot_mapping": slot_mapping,
                        "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                        "lmcache_cached_tokens": request.load_spec.lmcache_cached_tokens,
                        "sync": sync,
                        "kv_group": 0,
                        "req_id": request.req_id,
                        "request_configs": request.request_configs,
                        "shared_cpu_phase": SPARSE_DECODE_SHARED_CPU_PHASE,
                        "shared_cpu_request_ordinal": idx,
                        **latent_cache,
                    }
                    if shared_cpu_enabled and bound_state is not None:
                        existing_layers = (
                            bound_state.rank0_backing_objs_by_group.get(0)
                        )
                        if existing_layers:
                            retrieve_kwargs[
                                "shared_cpu_existing_rank0_backing_layers"
                            ] = existing_layers
                    if shared_cpu_preflight_state is not None:
                        retrieve_kwargs["shared_cpu_request_preflight_state"] = (
                            shared_cpu_preflight_state
                        )
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
                            len(getattr(request, "cached_keys", [])),
                            len(getattr(request, "cached_starts", [])),
                            len(getattr(request, "cached_ends", [])),
                            len(getattr(request, "cached_tensors", [])),
                            len(getattr(request, "cached_chunk_ptrs_npu", [])),
                            bool(retrieve_kwargs.get("_retrieve_metadata_warm")),
                            bound_state is not None,
                            _dsa_debug_shape(
                                getattr(request, "decode_ret_mask", None)
                            ),
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
                    kwargs_metadata_warm = bool(
                        retrieve_kwargs.get("_retrieve_metadata_warm")
                    )
                    metadata_warm = bool(kwargs_metadata_warm or request.cached_keys)

                    indexer_retriever = None
                    indexer_skipped = False
                    if dsa_two_groups:
                        indexer_kvcaches = self._kvcaches_for_group(1)
                        materialize_index = (
                            self._sparse_decode_requires_index_materialization(
                                request,
                                shared_cpu_enabled,
                            )
                        )
                        if (
                            shared_cpu_enabled
                            and not materialize_index
                        ):
                            indexer_skipped = True
                        elif (
                            shared_cpu_enabled
                            and materialize_index
                            and self._shared_sparse_decode_indexer_is_resident(
                                request,
                                bound_state,
                                token_count,
                            )
                        ):
                            logger.debug(
                                "Skipping shared CPU DSA index retrieve for "
                                "resident live sparse decode state: req_id=%s "
                                "token_count=%d",
                                request.req_id,
                                token_count,
                            )
                        elif not materialize_index:
                            indexer_skipped = True
                        elif not indexer_kvcaches:
                            if shared_cpu_enabled:
                                raise RuntimeError(
                                    "Shared CPU sparse decode with "
                                    "dsa_two_groups=true requires DSA index "
                                    "kvcaches for kv_group=1."
                                )
                        else:
                            latent_sparse_slots = (
                                slot_mapping[0]
                                if isinstance(slot_mapping, list)
                                else slot_mapping
                            )
                            request_indexer_slots = (
                                request.indexer_slot_mapping[0]
                                if request.indexer_slot_mapping
                                else None
                            )
                            if (
                                request_indexer_slots is not None
                                and request_indexer_slots.device.type
                                != torch.device(self.device).type
                            ):
                                request.indexer_slot_mapping[0] = (
                                    request_indexer_slots.to(
                                        device=self.device, dtype=torch.long
                                    )
                                )
                                request_indexer_slots = (
                                    request.indexer_slot_mapping[0]
                                )
                            idx_slot = self._sparse_indexer_slot_mapping(
                                attn_metadata,
                                latent_sparse_slots,
                                request.load_spec.lmcache_cached_tokens,
                                request_indexer_slots=request_indexer_slots,
                                strict=shared_cpu_enabled,
                            )
                            assert idx_slot is not None
                            indexer_cache = _retrieve_cache_kwargs(
                                request,
                                kv_group=1,
                                dsa_two_groups=dsa_two_groups,
                            )
                            indexer_kwargs: dict[str, Any] = {
                                "kvcaches": indexer_kvcaches,
                                "slot_mapping": idx_slot,
                                "vllm_cached_tokens": request.load_spec.vllm_cached_tokens,
                                "lmcache_cached_tokens": request.load_spec.lmcache_cached_tokens,
                                "sync": sync,
                                "kv_group": 1,
                                "req_id": request.req_id,
                                "request_configs": request.request_configs,
                                "shared_cpu_phase": SPARSE_DECODE_SHARED_CPU_PHASE,
                                "shared_cpu_request_ordinal": idx,
                                **indexer_cache,
                            }
                            if shared_cpu_enabled and bound_state is not None:
                                existing_layers = (
                                    bound_state.rank0_backing_objs_by_group.get(1)
                                )
                                if existing_layers:
                                    indexer_kwargs[
                                        "shared_cpu_existing_rank0_backing_layers"
                                    ] = existing_layers
                            if shared_cpu_preflight_state is not None:
                                indexer_kwargs[
                                    "shared_cpu_request_preflight_state"
                                ] = shared_cpu_preflight_state
                            indexer_kwargs.update(
                                self._sparse_decode_retrieve_warm_kwargs(
                                    request, token_count, bound_state
                                )
                            )
                            if request.decode_ret_mask is not None:
                                indexer_kwargs["ret_mask"] = request.decode_ret_mask
                            indexer_retriever = (
                                self.lmcache_engine.retrieve_layer_head_token_wise(
                                    retrieve_tokens,
                                    token_mask,
                                    **indexer_kwargs,
                                )
                            )
                            next(indexer_retriever)

                    if indexer_skipped:
                        request.shared_index_skipped = True
                        self._clear_request_indexer_cache(request)
                    if shared_cpu_enabled:
                        logger.debug(
                            "Deferring shared CPU sparse retrieve state save "
                            "until pointer-cache install completes: req_id=%s",
                            request.req_id,
                        )
                    else:
                        self._save_worker_retrieve_state_from_request(
                            request,
                            location=location,
                            metadata_warm=metadata_warm,
                            token_count=token_count,
                        )
                    self.layerwise_retrievers.append(
                        (layerwise_retriever, indexer_retriever)
                    )
                    self._layerwise_requests.append(request)
                    self._layerwise_retriever_is_sparse.append(True)
                    self._layerwise_sparse_req_ids.append(request.req_id)
                else:
                    retrieve_slot_mapping = slot_mapping
                    if lmcache_cached_tokens < len(slot_mapping):
                        retrieve_slot_mapping = slot_mapping[:lmcache_cached_tokens]
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        retrieve_tokens,
                        token_mask,
                        kvcaches=kvcaches,
                        slot_mapping=retrieve_slot_mapping,
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                        sync=sync,
                        kv_group=0,
                        req_id=request.req_id,
                        request_configs=request.request_configs,
                        shared_cpu_request_ordinal=idx,
                    )

                    # Two-group DSA: also retrieve the indexer group (kv_group=1)
                    # for the same latent hit token count, scattering into vLLM's
                    # indexer KV via the indexer slot mapping. Decode stays
                    # latent-only (this branch is prefill/prefix, not sparse).
                    indexer_retriever = None
                    idx_slot = None
                    if self._is_dsa_two_groups():
                        shared_cpu_enabled = bool(
                            getattr(
                                self.lmcache_engine,
                                "enable_shared_cpu_cache",
                                False,
                            )
                        )
                        indexer_kvcaches = self._kvcaches_for_group(1)
                        if shared_cpu_enabled and not indexer_kvcaches:
                            raise RuntimeError(
                                "Shared CPU dense prefix with "
                                "dsa_two_groups=true requires DSA index "
                                "kvcaches for kv_group=1."
                            )
                    if self._is_dsa_two_groups() and self._kvcaches_for_group(1):
                        indexer_layer_name = (
                            self._indexer_layer_names[0]
                            if self._indexer_layer_names
                            else None
                        )
                        if request.indexer_slot_mapping:
                            idx_slot = request.indexer_slot_mapping[0].to(
                                device=self.device, dtype=torch.long
                            )
                            if lmcache_cached_tokens < len(idx_slot):
                                idx_slot = idx_slot[:lmcache_cached_tokens]
                            if len(idx_slot) < lmcache_cached_tokens:
                                idx_slot = None
                        if idx_slot is None:
                            idx_slot = self._indexer_retrieve_slot_mapping(
                                attn_metadata,
                                lmcache_cached_tokens,
                                indexer_layer_name,
                            )
                        if (
                            idx_slot is None
                            and bool(
                                getattr(
                                    self.lmcache_engine,
                                    "enable_shared_cpu_cache",
                                    False,
                                )
                            )
                        ):
                            raise RuntimeError(
                                "Shared CPU dense prefix with "
                                "dsa_two_groups=true could not resolve DSA "
                                "index slot mapping for kv_group=1."
                            )
                        if idx_slot is not None:
                            indexer_retriever = self.lmcache_engine.retrieve_layer(
                                retrieve_tokens,
                                token_mask,
                                kvcaches=self._kvcaches_for_group(1),
                                slot_mapping=idx_slot,
                                vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                                sync=sync,
                                kv_group=1,
                                req_id=request.req_id,
                                request_configs=request.request_configs,
                                shared_cpu_request_ordinal=idx,
                            )

                    # Prime the same two-step window as the legacy dense path,
                    # but interleave groups so shared-cache collectives remain
                    # layer-major: latent L0, index L0, latent L1, index L1.
                    self._prime_dense_prefix_retrievers(
                        layerwise_retriever,
                        indexer_retriever,
                    )

                    dsa_two_groups = self._is_dsa_two_groups()
                    prefix_location, metadata_warm = (
                        self._warm_request_retrieve_metadata(
                            request,
                            retrieve_tokens,
                            token_mask,
                            kv_group=0,
                            dsa_two_groups=dsa_two_groups,
                        )
                    )
                    indexer_metadata_warm = False
                    if indexer_retriever is not None:
                        indexer_location, indexer_metadata_warm = (
                            self._warm_request_retrieve_metadata(
                                request,
                                retrieve_tokens,
                                token_mask,
                                kv_group=1,
                                dsa_two_groups=dsa_two_groups,
                            )
                        )
                        if prefix_location is None:
                            prefix_location = indexer_location
                    if prefix_location is None:
                        prefix_location = self._resolve_store_retrieve_location(
                            request
                        )
                    metadata_warm = bool(metadata_warm or indexer_metadata_warm)
                    self._save_worker_retrieve_state_from_request(
                        request,
                        location=prefix_location,
                        metadata_warm=metadata_warm,
                        token_count=lmcache_cached_tokens,
                    )


                    self.layerwise_retrievers.append(
                        (layerwise_retriever, indexer_retriever)
                    )
                    self._layerwise_requests.append(request)
                    self._layerwise_retriever_is_sparse.append(False)
            else:
                retrieve_slot_mapping = slot_mapping
                if lmcache_cached_tokens < len(slot_mapping):
                    retrieve_slot_mapping = slot_mapping[:lmcache_cached_tokens]
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
                if recalc_last_applied:
                    num_expected_tokens -= 1
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
        target_slot_mapping=None,
        payload_event=None,
    ) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
            selected_tokens: sparse token indices per decode row.
            token_start_index: legacy per-row start offset into slot_mapping.
            request_ids: req_id for each selected_tokens row (duplicates allowed).
            target_slot_mapping: optional batched physical destination slots,
                row-aligned with selected_tokens.
            payload_event: optional producer event recorded by vLLM after
                selected_tokens/target_slot_mapping were built. LMCache waits on
                this before row-selecting from those tensors.
        """
        if self.layerwise_retrievers and logger.isEnabledFor(10):
            logger.debug("Waiting for layer %d to be loaded", self.current_layer)

        if not self.layerwise_retrievers:
            return

        metadata: Optional[LMCacheConnectorMetadata] = None

        layerwise_requests = getattr(self, "_layerwise_requests", None)
        if not layerwise_requests:
            metadata = self._parent._get_connector_metadata()
            assert isinstance(metadata, LMCacheConnectorMetadata)
            layerwise_requests = [
                request
                for request in metadata.requests
                if request.load_spec is not None and request.load_spec.can_load
            ]

        rows_of_req = None
        if request_ids is not None:
            sparse_req_ids = getattr(self, "_layerwise_sparse_req_ids", None)
            if sparse_req_ids is None:
                if metadata is None:
                    metadata = self._parent._get_connector_metadata()
                    assert isinstance(metadata, LMCacheConnectorMetadata)
                sparse_req_ids = [
                    request.req_id
                    for request in metadata.requests
                    if request.load_spec is not None
                    and request.load_spec.can_load
                    and request.is_sparse_decode
                ]
            ordered_sparse_rows = (
                len(request_ids) == len(sparse_req_ids)
                and request_ids == sparse_req_ids
            )
            if not ordered_sparse_rows:
                rows_of_req = {}
                for row, rid in enumerate(request_ids):
                    rows_of_req.setdefault(rid, []).append(row)

        selected_rows = None
        if selected_tokens is not None:
            # After this wait, row selection and connector-side packing are
            # ordered by the current stream; the load stream later waits on it.
            _dsa_wait_payload_event(payload_event)
            selected_rows = (
                int(selected_tokens.shape[0])
                if hasattr(selected_tokens, "shape")
                and len(selected_tokens.shape) > 0
                else len(selected_tokens)
            )

        wait_group = self._layerwise_wait_group(layer_name)
        parsed_layer_id = None
        parsed_layer_id_loaded = False
        sparse_indexer_sent_layers = None

        idx = 0
        decode_row = 0
        for request in layerwise_requests:
            if idx >= len(self.layerwise_retrievers):
                logger.warning(
                    "wait_for_layer_load: missing retriever for request %s "
                    "(idx=%d, retrievers=%d)",
                    request.req_id,
                    idx,
                    len(self.layerwise_retrievers),
                )
                break
            layerwise_retriever, indexer_retriever = self.layerwise_retrievers[idx]
            if request.is_sparse_decode:
                payload = None
                rows = None
                row_count = 1
                if selected_tokens is None:
                    selected_tokens_per_req = None
                    token_start_index_per_req = 0
                    target_slot_mapping_per_req = None
                else:
                    assert selected_rows is not None
                    if rows_of_req is None:
                        row = decode_row
                        if row >= selected_rows:
                            raise RuntimeError(
                                "Sparse decode row out of bounds for "
                                f"layer={layer_name} req={request.req_id} "
                                f"rows={[row]} selected_rows={selected_rows}"
                            )
                        selected_tokens_per_req = _single_row_select(
                            selected_tokens, row
                        )
                    else:
                        if request.req_id not in rows_of_req:
                            raise RuntimeError(
                                "Missing sparse decode row for "
                                f"layer={layer_name} req={request.req_id} "
                                f"sparse_decode_row={decode_row}"
                            )
                        rows = rows_of_req[request.req_id]
                        row_count = len(rows)
                        if max(rows) >= selected_rows:
                            raise RuntimeError(
                                "Sparse decode row out of bounds for "
                                f"layer={layer_name} req={request.req_id} "
                                f"rows={rows} selected_rows={selected_rows}"
                            )
                        selected_tokens_per_req = _row_select(selected_tokens, rows)
                    if target_slot_mapping is not None:
                        if rows_of_req is None:
                            target_slot_mapping_per_req = _single_row_select(
                                target_slot_mapping, row
                            )
                        else:
                            target_slot_mapping_per_req = _row_select(
                                target_slot_mapping, rows
                            )
                        selected_tokens_payload = _sparse_payload_value(
                            selected_tokens_per_req
                        )
                        target_slot_mapping_payload = _sparse_payload_value(
                            target_slot_mapping_per_req
                        )
                        local_payload_event = (
                            _dsa_record_payload_event_if_needed(
                                selected_tokens_payload,
                                target_slot_mapping_payload,
                            )
                            if rows_of_req is not None
                            else None
                        )
                        if local_payload_event is not None:
                            payload = {
                                "selected_token_ids": selected_tokens_payload,
                                "target_slot_mapping": target_slot_mapping_payload,
                                "payload_event": local_payload_event,
                            }
                        else:
                            payload = (
                                selected_tokens_payload,
                                None,
                                target_slot_mapping_payload,
                            )
                        token_start_index_per_req = None
                    else:
                        token_start_index_per_req = (
                            0
                            if token_start_index is None
                            else (
                                _single_row_select(token_start_index, row)
                                if rows_of_req is None
                                else _row_select(token_start_index, rows)
                            )
                        )
                        selected_tokens_payload = _sparse_payload_value(
                            selected_tokens_per_req
                        )
                        token_start_payload = _sparse_payload_value(
                            token_start_index_per_req
                        )
                        local_payload_event = (
                            _dsa_record_payload_event_if_needed(
                                selected_tokens_payload,
                                token_start_payload,
                            )
                            if rows_of_req is not None
                            else None
                        )
                        if local_payload_event is not None:
                            payload = {
                                "selected_token_ids": selected_tokens_payload,
                                "token_start_index": token_start_payload,
                                "payload_event": local_payload_event,
                            }
                        else:
                            selected_tokens_per_req = selected_tokens_payload
                            token_start_index_per_req = token_start_payload
                sparse_payload = (
                    payload
                    if payload is not None
                    else (selected_tokens_per_req, token_start_index_per_req)
                )
                indexer_sent_key = (
                    (request.req_id, self.current_layer)
                    if indexer_retriever is not None
                    else None
                )
                if indexer_retriever is not None:
                    if not parsed_layer_id_loaded:
                        parsed_layer_id = self._layerwise_layer_id_from_name(
                            layer_name
                        )
                        parsed_layer_id_loaded = True
                    if sparse_indexer_sent_layers is None:
                        sparse_indexer_sent_layers = getattr(
                            self,
                            "_layerwise_sparse_indexer_sent_layers",
                            None,
                        )
                        if sparse_indexer_sent_layers is None:
                            sparse_indexer_sent_layers = set()
                            self._layerwise_sparse_indexer_sent_layers = (
                                sparse_indexer_sent_layers
                            )
                if wait_group == 1:
                    ret_token_mask = None
                    if (
                        indexer_retriever is not None
                        and sparse_indexer_sent_layers is not None
                        and (
                            parsed_layer_id is None
                            or parsed_layer_id == self.current_layer
                        )
                        and indexer_sent_key not in sparse_indexer_sent_layers
                    ):
                        indexer_retriever.send((None, 0))
                        sparse_indexer_sent_layers.add(indexer_sent_key)
                else:
                    ret_token_mask = layerwise_retriever.send(sparse_payload)
                    if (
                        indexer_retriever is not None
                        and sparse_indexer_sent_layers is not None
                        and indexer_sent_key not in sparse_indexer_sent_layers
                    ):
                        indexer_ret_mask = indexer_retriever.send((None, 0))
                        sparse_indexer_sent_layers.add(indexer_sent_key)
                        if ret_token_mask is None:
                            ret_token_mask = indexer_ret_mask
                decode_row += row_count
            else:
                if wait_group == 1:
                    if indexer_retriever is not None:
                        next(indexer_retriever)
                    ret_token_mask = None
                else:
                    ret_token_mask = next(layerwise_retriever)

            if (
                wait_group == 0
                and self.current_layer == self.num_layers - 1
                and not request.is_sparse_decode
            ):
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info("Retrieved %d tokens", num_retrieved_tokens)
            idx += 1

        if self.layerwise_retrievers and self._layerwise_wait_should_advance(wait_group):
            self.current_layer += 1
            if self.current_layer >= self.num_layers:
                if metadata is None:
                    metadata = self._parent._get_connector_metadata()
                    assert isinstance(metadata, LMCacheConnectorMetadata)
                self._finalize_worker_retrieve_state_from_metadata(metadata)
                self._drain_layerwise_retrievers()

        return

    def _should_defer_latent_save_under_tp(self) -> bool:
        if not getattr(self.config, "dsa_two_groups", False):
            return False
        meta = getattr(self.lmcache_engine, "metadata", None)
        world_size = getattr(meta, "world_size", 1) if meta else 1
        return world_size > 1

    @staticmethod
    def _advance_layerwise_storer_once(storer) -> None:
        if storer is None:
            return
        try:
            next(storer)
        except StopIteration:
            pass

    def _layerwise_storer_drain_limit(self) -> int:
        engine = getattr(self, "lmcache_engine", None)
        num_layers = int(getattr(engine, "num_layers", 0) or 0)
        if num_layers <= 0:
            num_layers = len(getattr(self, "_latent_layer_names", []) or [])
        if num_layers <= 0:
            num_layers = len(getattr(self, "kv_caches", {}) or {})
        return max(num_layers + 2, 2)

    def _drain_layerwise_storer_fully(self, storer) -> bool:
        if storer is None:
            return True
        for _ in range(self._layerwise_storer_drain_limit()):
            try:
                next(storer)
            except StopIteration:
                return True
        logger.warning(
            "Layerwise storer did not finish after bounded drain; closing it"
        )
        return False

    @staticmethod
    def _close_layerwise_storer(storer) -> None:
        if storer is None:
            return
        try:
            storer.close()
        except (GeneratorExit, RuntimeError, ValueError):
            pass

    def _flush_deferred_latent_store(
        self,
        request: "ReqMeta",
        save_spec: Optional["SaveSpec"],
    ) -> None:
        """Run a full latent store_layer after indexer layers finish (TP>1)."""
        pending_key = self._layerwise_save_storer_key(request, 0)
        legacy_pending_key = request.req_id
        if (
            pending_key not in self._deferred_latent_pending
            and legacy_pending_key not in self._deferred_latent_pending
        ):
            return
        self._note_decode_window_save_seen(request)
        if save_spec is None or not save_spec.can_save_latent:
            self._deferred_latent_pending.discard(pending_key)
            self._deferred_latent_pending.discard(legacy_pending_key)
            return

        self._refresh_kvcaches_list()
        kvcaches = self._kvcaches_for_group(0)
        if not kvcaches:
            self._deferred_latent_pending.discard(pending_key)
            self._deferred_latent_pending.discard(legacy_pending_key)
            return

        token_ids = request.token_ids
        assert isinstance(token_ids, list)
        assert request.slot_mapping is not None and len(request.slot_mapping) > 0
        if request.is_sparse_decode:
            if request.slot_mapping[0].device.type != torch.device(self.device).type:
                request.slot_mapping[0] = request.slot_mapping[0].to(
                    device=self.device, dtype=torch.long
                )
            slot_mapping = request.slot_mapping[0]
        else:
            slot_mapping = request.slot_mapping[0].to(
                device=self.device, dtype=torch.long
            )

        if (
            self.kv_role == "kv_producer"
            and not self._is_decode_window_save_request(request)
        ):
            skip_leading_tokens = 0
        else:
            skip_leading_tokens = save_spec.skip_leading_tokens
            if skip_leading_tokens == len(token_ids):
                self._deferred_latent_pending.discard(pending_key)
                self._deferred_latent_pending.discard(legacy_pending_key)
                return
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False

        store_kwargs: dict[str, Any] = {
            "cached_keys": request.cached_keys,
            "cached_starts": request.cached_starts,
            "cached_ends": request.cached_ends,
            "cached_memory_objs": request.cached_memory_objs,
            "cached_tensors": request.cached_tensors,
            "decode_window_save": self._is_decode_window_save_request(request),
            "decode_window_start": getattr(request, "decode_window_start", None),
            "decode_window_end": getattr(request, "decode_window_end", None),
            "decode_window_size": getattr(request, "decode_window_size", None),
            "request_configs": request.request_configs,
        }
        if request.is_sparse_decode or self._is_decode_window_save_request(request):
            store_kwargs["cached_chunk_dev_ptrs"] = _ensure_list_attr(
                request, "cached_chunk_dev_ptrs"
            )
            store_kwargs["cached_chunk_ptrs_npu"] = _ensure_list_attr(
                request, "cached_chunk_ptrs_npu"
            )

        storer = self.lmcache_engine.store_layer(
            token_ids,
            mask=store_mask,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            offset=skip_leading_tokens,
            sync=True,
            req_id=request.req_id,
            **store_kwargs,
        )
        latent_completed = self._drain_layerwise_storer_fully(storer)
        self._close_layerwise_storer(storer)
        if latent_completed:
            self._record_decode_window_save_group_completed(request, 0)
        self._deferred_latent_pending.discard(pending_key)
        self._deferred_latent_pending.discard(legacy_pending_key)
        indexer_required = (
            self._is_decode_window_save_request(request)
            and getattr(save_spec, "can_save_indexer", False)
            and getattr(self.config, "dsa_two_groups", False)
        )
        if not indexer_required:
            self._mark_decode_window_save_completed(request)


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

        assert len(self.kv_caches) > 0

        if not self._kvcaches_list:
            self._refresh_kvcaches_list()

        dsa_two_groups = getattr(self.config, "dsa_two_groups", False)
        is_indexer_layer = dsa_two_groups and "indexer" in layer_name
        kv_group = 1 if is_indexer_layer else 0
        # Latent path uses the same kv list as dev-qzy (_kvcaches_list); indexer
        # uses the partitioned indexer caches only.
        kvcaches = self._kvcaches_for_group(kv_group)
        if not kvcaches:
            # No caches registered for this group (e.g. indexer not
            # registered with the connector); nothing to store.
            return

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue
            self._note_decode_window_save_seen(request)

            # Per-group gating: in two-group mode, skip indexer save if
            # can_save_indexer is False, and skip latent save if
            # can_save_latent is False.
            if dsa_two_groups and save_spec is not None:
                if is_indexer_layer and not save_spec.can_save_indexer:
                    continue
                if not is_indexer_layer and not save_spec.can_save_latent:
                    continue

            # TP>1 + dsa_two_groups: skip interleaved latent saves during
            # forward; flush the full latent store after the last indexer
            # layer (or in wait_for_save as fallback).
            if self._should_defer_latent_save_under_tp() and not is_indexer_layer:
                if save_spec is not None and save_spec.can_save_latent:
                    self._deferred_latent_pending.add(
                        self._layerwise_save_storer_key(request, 0)
                    )
                continue

            storer_key = self._layerwise_save_storer_key(request, kv_group)
            layerwise_storer = self._layerwise_save_storers.get(storer_key)
            # Forward-boundary recovery: the store_layer generator is sized for
            # exactly one forward (num_layers layer yields + 1 drain yield). It
            # is normally drained and popped by wait_for_save between forwards.
            # Some vLLM-Ascend forward paths do not call wait_for_save between
            # consecutive forwards (e.g. chunked prefill), which would leave the
            # previous forward's storer in place and cause the next forward's
            # save_kv_layer calls to exhaust it (StopIteration). When we see the
            # group's first layer again while a storer still exists, drain the
            # old storer fully and create a fresh one for the new forward.
            _first_layer = (
                self._indexer_layer_names[0]
                if kv_group == 1 and self._indexer_layer_names
                else (
                    self._latent_layer_names[0]
                    if self._latent_layer_names
                    else None
                )
            )
            if _first_layer is not None and layer_name == _first_layer:
                active_keys = {
                    self._layerwise_save_storer_key(req, kv_group)
                    for req in connector_metadata.requests
                }
                stale_keys = [
                    key
                    for key in list(self._layerwise_save_storers)
                    if self._storer_key_matches_req_group(
                        key, request.req_id, kv_group
                    )
                    and (key == storer_key or key not in active_keys)
                ]
                for stale_key in stale_keys:
                    stale_storer = self._layerwise_save_storers.pop(stale_key)
                    self._drain_layerwise_storer_fully(stale_storer)
                    self._close_layerwise_storer(stale_storer)
                if stale_keys:
                    layerwise_storer = None
            if layerwise_storer is None:
                # Refresh from the live kv_caches dict before creating a new
                # storer. Chunked prefill may update registered buffers between
                # forwards; stale _latent_kvcaches pointers cause MTE OOB.
                self._refresh_kvcaches_list()
                kvcaches = self._kvcaches_for_group(kv_group)
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

                # Latent save matches dev-qzy: use scheduler request.slot_mapping
                # (cumulative across chunked-prefill steps). Indexer save must
                # use the active layer's attention metadata because it is the
                # source view for the indexer KV buffer. Retrieve can use the
                # request-level mapping as its destination view.

                # Two-group DSA: for indexer layers, use the indexer group's
                # slot mapping. vLLM may pass a per-layer metadata dict; the
                # indexer metadata stores this as "slot_mapping", while the
                # latent metadata stores it as "indexer_slot_mapping".
                if is_indexer_layer:
                    idx_slot = self._indexer_save_slot_mapping(
                        request,
                        attn_metadata,
                        layer_name,
                        len(token_ids),
                    )
                    if idx_slot is not None:
                        slot_mapping = idx_slot.to(
                            device=self.device, dtype=torch.long
                        )
                    if idx_slot is None:
                        logger.warning(
                            "Skipping DSA indexer save for layer %s: "
                            "indexer slot mapping is unavailable",
                            layer_name,
                        )
                        continue

                if (
                    self.kv_role == "kv_producer"
                    and not self._is_decode_window_save_request(request)
                ):
                    skip_leading_tokens = 0
                else:
                    assert save_spec is not None
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                if is_indexer_layer:
                    slot_mapping = self._pad_chunk_local_slot_mapping(
                        slot_mapping,
                        total_tokens=len(token_ids),
                        token_offset=skip_leading_tokens,
                    )
                    if len(slot_mapping) < len(token_ids):
                        logger.warning(
                            "Skipping DSA indexer save for layer %s: "
                            "slot mapping length %d does not cover token range "
                            "[%d, %d)",
                            layer_name,
                            len(slot_mapping),
                            skip_leading_tokens,
                            len(token_ids),
                        )
                        continue


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
                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                # Keep store_layer kwargs minimal: shared cached_* state plus
                # request_configs for cache-key salting. Indexer-only sparse
                # ptr fields stay on the latent/sparse path.
                store_kwargs: dict[str, Any] = {
                    "cached_keys": request.cached_keys,
                    "cached_starts": request.cached_starts,
                    "cached_ends": request.cached_ends,
                    "cached_memory_objs": request.cached_memory_objs,
                    "cached_tensors": request.cached_tensors,
                    "decode_window_save": self._is_decode_window_save_request(request),
                    "decode_window_start": getattr(
                        request, "decode_window_start", None
                    ),
                    "decode_window_end": getattr(request, "decode_window_end", None),
                    "decode_window_size": getattr(
                        request, "decode_window_size", None
                    ),
                    "request_configs": request.request_configs,
                }
                if (
                    request.is_sparse_decode
                    or self._is_decode_window_save_request(request)
                ):
                    store_kwargs["cached_chunk_dev_ptrs"] = _ensure_list_attr(
                        request, "cached_chunk_dev_ptrs"
                    )
                    store_kwargs["cached_chunk_ptrs_npu"] = _ensure_list_attr(
                        request, "cached_chunk_ptrs_npu"
                    )
                # Indexer-only extras. Latent (kv_group=0) matches dev-qzy and
                # does not pass kv_group or indexer cached_* fields.
                if dsa_two_groups and kv_group == 1:
                    store_kwargs["kv_group"] = kv_group
                    store_kwargs.update(
                        _retrieve_cache_kwargs(
                            request,
                            kv_group=1,
                            dsa_two_groups=True,
                        )
                    )
                # Match dev-qzy: sync=True when creating the latent storer.
                # Under TP>1 + dsa_two_groups, also sync indexer storers so
                # latent/indexer transfers do not overlap on store_stream.
                _meta = getattr(self.lmcache_engine, "metadata", None)
                _world_size = getattr(_meta, "world_size", 1) if _meta else 1
                sync = layerwise_storer is None and (
                    kv_group == 0
                    or (dsa_two_groups and _world_size > 1)
                )
                logger.debug(
                    "Creating layerwise save storer: req_id=%s key=%s "
                    "layer=%s kv_group=%s decode_window=%s "
                    "range=[%s,%s) tokens=%d skip=%d slot_mapping_len=%d "
                    "kvcaches=%d can_save_latent=%s can_save_indexer=%s",
                    request.req_id,
                    storer_key,
                    layer_name,
                    kv_group,
                    self._is_decode_window_save_request(request),
                    getattr(request, "decode_window_start", None),
                    getattr(request, "decode_window_end", None),
                    len(token_ids),
                    skip_leading_tokens,
                    len(slot_mapping),
                    len(kvcaches),
                    getattr(save_spec, "can_save_latent", None),
                    getattr(save_spec, "can_save_indexer", None),
                )
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=sync,
                    req_id=request.req_id,
                    **store_kwargs,
                )
                self._layerwise_save_storers[storer_key] = layerwise_storer

            indexer_group_last = (
                is_indexer_layer
                and self._indexer_layer_names
                and layer_name == self._indexer_layer_names[-1]
            )

            try:
                next(layerwise_storer)
            except StopIteration:
                logger.error(
                    "Layerwise save storer exhausted early: req_id=%s key=%s "
                    "layer=%s kv_group=%s decode_window=%s range=[%s,%s) "
                    "active_storer_keys=%s deferred_latent_pending=%s",
                    request.req_id,
                    storer_key,
                    layer_name,
                    kv_group,
                    self._is_decode_window_save_request(request),
                    getattr(request, "decode_window_start", None),
                    getattr(request, "decode_window_end", None),
                    list(self._layerwise_save_storers.keys()),
                    list(getattr(self, "_deferred_latent_pending", set())),
                )
                raise

            if indexer_group_last:
                indexer_completed = self._drain_layerwise_storer_fully(
                    layerwise_storer
                )
                self._layerwise_save_storers.pop(storer_key, None)
                self._close_layerwise_storer(layerwise_storer)
                layerwise_storer = None
                if indexer_completed:
                    self._record_decode_window_save_group_completed(
                        request,
                        kv_group,
                    )

            if (
                indexer_group_last
                and self._should_defer_latent_save_under_tp()
            ):
                self._flush_deferred_latent_store(request, save_spec)

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

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
            if self._should_defer_latent_save_under_tp():
                for request in connector_metadata.requests:
                    if (
                        self._layerwise_save_storer_key(request, 0)
                        in self._deferred_latent_pending
                    ):
                        self._flush_deferred_latent_store(
                            request, request.save_spec
                        )
            for request in connector_metadata.requests:
                # Drain both possible groups for this request. Missing storers
                # are fine; the actual save path decides which groups exist.
                storer_items = [
                    (_kv_group, self._layerwise_save_storer_key(request, _kv_group))
                    for _kv_group in (0, 1)
                ]
                for _kv_group, storer_key in storer_items:
                    layerwise_storer = self._layerwise_save_storers.pop(
                        storer_key, None
                    )
                    if layerwise_storer is not None:
                        if self._is_decode_window_save_request(request):
                            save_completed = self._drain_layerwise_storer_fully(
                                layerwise_storer
                            )
                        else:
                            self._advance_layerwise_storer_once(layerwise_storer)
                            save_completed = True
                        self._close_layerwise_storer(layerwise_storer)
                        if save_completed:
                            self._record_decode_window_save_group_completed(
                                request, _kv_group
                            )
                self._maybe_seed_worker_retrieve_state_from_store(request)
                self._mark_decode_window_save_completed(request)
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
            self._record_decode_window_save_group_completed(request, 0)
            self._mark_decode_window_save_completed(request)

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
            logger.debug(
                "Looking up cache for the first time for request %s!",
                req_id,
            )
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

    def _should_decode_window_save(self, tracker: RequestTracker) -> bool:
        window_size = getattr(self, "_decode_window_save_window_size", 0)
        if window_size <= 0:
            return False
        if self.kv_role == "kv_consumer":
            return False
        if tracker.disagg_spec is not None:
            return False
        if tracker.skip_save:
            return False
        if (tracker.request_configs or {}).get("lmcache.skip_save", False):
            return False
        if not tracker.is_decode_phase:
            return False
        return len(tracker.token_ids) > tracker.prompt_len

    def _init_decode_window_save_start(self, tracker: RequestTracker) -> int:
        if tracker.decode_window_save_next_start is not None:
            return tracker.decode_window_save_next_start
        prompt_chunk_start = (
            tracker.prompt_len // self._lmcache_chunk_size * self._lmcache_chunk_size
        )
        saved_chunk_start = (
            tracker.num_saved_tokens
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        start = max(prompt_chunk_start, saved_chunk_start)
        tracker.decode_window_save_next_start = start
        tracker.decode_window_save_committed_end = max(
            tracker.decode_window_save_committed_end,
            min(start, len(tracker.token_ids)),
        )
        return start

    def _add_decode_window_save_metas(
        self,
        meta: LMCacheConnectorMetadata,
        tracker: RequestTracker,
    ) -> None:
        if not self._should_decode_window_save(tracker):
            return

        window_size = self._decode_window_save_window_size
        next_start = self._init_decode_window_save_start(tracker)

        while len(tracker.token_ids) >= next_start + window_size:
            window_start = next_start
            window_end = window_start + window_size
            req_meta = ReqMeta.from_decode_window_save(
                tracker,
                self._block_size,
                window_start,
                window_end,
                window_size,
            )
            if req_meta is None:
                return
            if (
                self._is_dsa_two_groups()
                and bool(
                    self._shared_cpu_config_value(
                        "enable_shared_cpu_cache", False
                    )
                )
                and getattr(req_meta.save_spec, "can_save_indexer", False) is False
            ):
                logger.warning(
                    "Skipping decode-window save for request %s because "
                    "dsa_two_groups requires matching DSA index slots for "
                    "shared CPU decode-save correctness: window=[%d,%d)",
                    tracker.req_id,
                    window_start,
                    window_end,
                )
                return

            meta.add_request(req_meta)
            tracker.decode_window_save_next_start = window_end
            next_start = window_end

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
                save_full_chunk_in_decode=getattr(
                    self.config, "save_full_chunk_in_decode", False
                ),
                dsa_two_groups=getattr(self.config, "dsa_two_groups", False),
            )
            if req_meta is not None:
                meta.add_request(req_meta)
            self._add_decode_window_save_metas(meta, request_tracker)

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

                self._add_decode_window_save_metas(meta, request_tracker)
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                    save_full_chunk_in_decode=getattr(
                        self.config, "save_full_chunk_in_decode", False
                    ),
                    dsa_two_groups=getattr(self.config, "dsa_two_groups", False),
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
                request_tracker.decode_window_save_committed_end = min(
                    request_tracker.decode_window_save_committed_end,
                    tokens_to_keep,
                )
                request_tracker.decode_window_save_next_start = None

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

            self._add_decode_window_save_metas(meta, request_tracker)
            is_sparse_decode = self.enable_sparse_attention and (
                request.num_computed_tokens > request_tracker.prompt_len
            )
            if is_sparse_decode:
                if (
                    not request_tracker.sparse_token_ids
                    or len(request_tracker.sparse_token_ids)
                    < request_tracker.prompt_len
                ):
                    request_tracker.seed_sparse_decode_tokens(
                        list(request.all_token_ids)
                    )
                lmcache_cached_for_sparse = request_tracker.prompt_len
                if self._decode_window_save_window_size > 0:
                    token_len = len(request_tracker.token_ids)
                    save_frontier = request_tracker.decode_window_save_next_start
                    if save_frontier is None:
                        save_frontier = self._init_decode_window_save_start(
                            request_tracker
                        )
                    save_frontier = min(int(save_frontier), token_len)
                    lmcache_cached_for_sparse = min(
                        request_tracker.decode_window_save_committed_end,
                        save_frontier,
                        token_len,
                    )
                    if _DSA_PROF:
                        print(
                            f"[SPARSE_BOUNDARY] req={req_id} "
                            f"prompt_len={request_tracker.prompt_len} "
                            f"token_len={token_len} "
                            f"committed_end={request_tracker.decode_window_save_committed_end} "
                            f"save_frontier={save_frontier} "
                            f"lmcache_cached_for_sparse={lmcache_cached_for_sparse}",
                            flush=True,
                        )
                load_spec = LoadSpec(
                    vllm_cached_tokens=0,
                    lmcache_cached_tokens=lmcache_cached_for_sparse,
                    can_load=lmcache_cached_for_sparse > 0,
                )

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
                is_sparse_decode=is_sparse_decode,
                save_full_chunk_in_decode=getattr(
                    self.config, "save_full_chunk_in_decode", False
                ),
                dsa_two_groups=getattr(self.config, "dsa_two_groups", False),
            )
            if req_meta is not None:
                req_meta.resumed_from_preemption = preempted
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
            self._drop_layerwise_save_storers(request.request_id)

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
