"""Conservative host-memory planning for streamed ERA5 workflows.

These checks do not set RLIMIT_AS (JAX reserves virtual address space), and a
host-RAM budget is not a GPU-memory limit. Check RSS between chunks as well.
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass

GIB = 1024 ** 3
MAX_HOST_BYTES = 96_000_000_000
MAX_HOST_GIB = MAX_HOST_BYTES / GIB


@dataclass(frozen=True)
class MemoryPlan:
    estimated_bytes: int
    budget_bytes: int
    usable_bytes: int
    components: dict[str, int]

    def to_dict(self):
        return asdict(self)


def memory_preflight(*, channels: int, latitude: int, longitude: int,
                     history_steps: int, future_steps: int, batch_size: int = 1,
                     chunk_steps: int = 1, dtype_bytes: int = 4,
                     model_bytes: int = 0, activation_multiplier: float = 12.0,
                     budget_gib: float = MAX_HOST_GIB, headroom: float = 0.2) -> MemoryPlan:
    """Estimate working memory, excluding on-disk memmaps and full trajectories.

    ``model_bytes`` should include backbone parameters/compiler allocations;
    ``activation_multiplier`` is a planning allowance, not a measured bound.
    """
    positive = (channels, latitude, longitude, history_steps, future_steps,
                batch_size, chunk_steps, dtype_bytes)
    if any(v <= 0 for v in positive) or not 0 < budget_gib <= MAX_HOST_GIB:
        raise ValueError("Positive dimensions and a host budget at most 96 decimal GB required")
    if not 0 < headroom < 1 or activation_multiplier < 1 or model_bytes < 0:
        raise ValueError("Invalid headroom, activation multiplier or model bytes")
    field = channels * latitude * longitude * dtype_bytes
    window = batch_size * (2 * history_steps + future_steps) * field
    parts = {"window_arrays": window,
             "training_workspace_allowance": int(window * activation_multiplier),
             "streaming_chunk": chunk_steps * field * 3,
             "backbone_and_parameters": model_bytes}
    budget = int(budget_gib * GIB)
    usable = int(budget * (1 - headroom))
    total = sum(parts.values())
    if total > usable:
        raise MemoryError(f"Estimated {total/GIB:.2f} GiB exceeds usable {usable/GIB:.2f} GiB; "
                          "reduce batch/window/grid or increase measured model allowance accuracy")
    return MemoryPlan(total, budget, usable, parts)


def check_process_memory(budget_gib: float = MAX_HOST_GIB) -> int:
    """Return current process-tree RSS, raising at the configured host limit."""
    if not 0 < budget_gib <= MAX_HOST_GIB:
        raise ValueError("Host budget must be positive and at most 96 decimal GB")
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("Install psutil to enforce runtime RSS checks") from exc
    try:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # Some container PID namespaces expose getpid() without a matching
        # /proc entry. Peak RSS is a conservative fallback for this process;
        # RUSAGE_CHILDREN covers terminated children, not live descendants.
        import resource
        unit = 1 if sys.platform == "darwin" else 1024
        rss = int((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss +
                   resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * unit)
    if rss > budget_gib * GIB:
        raise MemoryError(f"Process-tree RSS {rss/GIB:.2f} GiB exceeds {budget_gib} GiB")
    return rss
