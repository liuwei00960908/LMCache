# SPDX-License-Identifier: Apache-2.0
"""Gated profiler for DSA decode-offload retrieve path (LMCache adapter side).

Enable with VLLM_ASCEND_DSA_RETRIEVE_PROFILE=1. Each section accumulates
mean ms/layer-call; a summary line is logged every
VLLM_ASCEND_DSA_RETRIEVE_PROFILE_WINDOW (default 78) layer-calls.

Set VLLM_ASCEND_DSA_RETRIEVE_PROFILE_SYNC=1 to add torch.npu.synchronize()
around each section so measured time reflects real device time.
"""

# Standard
import os
import time
from collections import defaultdict
from typing import Optional

# Third Party
import torch
from lmcache.logging import init_logger

logger = init_logger(__name__)

ENABLED = bool(int(os.getenv("VLLM_ASCEND_DSA_RETRIEVE_PROFILE", "0")))
_SYNC = bool(int(os.getenv("VLLM_ASCEND_DSA_RETRIEVE_PROFILE_SYNC", "0")))
_WINDOW = max(1, int(os.getenv("VLLM_ASCEND_DSA_RETRIEVE_PROFILE_WINDOW", "78")))

_acc: dict[str, float] = defaultdict(float)
_n: dict[str, int] = defaultdict(int)
_calls = [0]


def _maybe_sync() -> None:
    if _SYNC and hasattr(torch, "npu"):
        torch.npu.synchronize()


class section:
    __slots__ = ("name", "_t")

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "section":
        if ENABLED:
            _maybe_sync()
            self._t = time.perf_counter()
        else:
            self._t = None
        return self

    def __exit__(self, *exc) -> None:
        if ENABLED and self._t is not None:
            _maybe_sync()
            _acc[self.name] += (time.perf_counter() - self._t) * 1000.0
            _n[self.name] += 1


def begin(name: str) -> Optional[tuple]:
    if not ENABLED:
        return None
    _maybe_sync()
    return (name, time.perf_counter())


def end(token: Optional[tuple]) -> None:
    if not ENABLED or token is None:
        return
    _maybe_sync()
    name, t = token
    _acc[name] += (time.perf_counter() - t) * 1000.0
    _n[name] += 1


def step() -> None:
    if not ENABLED:
        return
    _calls[0] += 1
    if _calls[0] % _WINDOW != 0:
        return
    means = {}
    for k in _acc:
        cnt = _n.get(k, 0)
        if cnt > 0:
            means[k] = _acc[k] / cnt
    parts = "  ".join(f"{k}={v:.3f}" for k, v in sorted(means.items()))
    logger.info(
        "[DSA-RETR-PROF] window=%d calls=%d sync=%d  %s",
        _WINDOW,
        _calls[0],
        int(_SYNC),
        parts,
    )
    _acc.clear()
    _n.clear()
