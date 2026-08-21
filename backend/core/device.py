"""Runtime device selection for CPU/GPU deployments."""

from __future__ import annotations

import logging

import torch


logger = logging.getLogger(__name__)


def resolve_device(requested: str | None = "auto") -> str:
    """Resolve a configured device and safely fall back to CPU.

    ``auto``/``gpu`` select the first CUDA device when available. Explicit CUDA
    requests also fall back to CPU when the Space was built without a GPU.
    Numeric values such as ``0`` are accepted as CUDA device indices.
    """
    value = str(requested or "auto").strip().lower()
    if value in {"", "auto", "gpu"}:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if value.isdigit():
        value = f"cuda:{value}"

    if value == "cuda":
        value = "cuda:0"

    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("CUDA requested (%s), but CUDA is unavailable; falling back to CPU", requested)
            return "cpu"
        try:
            index = int(value.split(":", 1)[1]) if ":" in value else 0
        except ValueError:
            logger.warning("Invalid CUDA device %r; falling back to CPU", requested)
            return "cpu"
        if index < 0 or index >= torch.cuda.device_count():
            logger.warning(
                "CUDA device %s is unavailable (count=%s); falling back to CPU",
                index,
                torch.cuda.device_count(),
            )
            return "cpu"
        return value

    if value == "cpu":
        return value

    # Preserve supported torch backends (for example mps), while making an
    # unknown configuration safe for a serverless deployment.
    try:
        torch.device(value)
    except (RuntimeError, TypeError):
        logger.warning("Unknown device %r; falling back to CPU", requested)
        return "cpu"
    return value


def device_info(device: str) -> dict[str, object]:
    """Return JSON-safe runtime details for health/debug endpoints."""
    info: dict[str, object] = {
        "resolved": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        info["gpu_name"] = torch.cuda.get_device_name(index)
        info["gpu_index"] = index
    return info
