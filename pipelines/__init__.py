"""Pipeline infrastructure for training robustness improvement methods.

This module intentionally avoids importing ``pipelines.runner`` at import time
to prevent a circular import with ``improvements``.
"""

from .specs import PipelineSpec
from .signatures import compute_data_config_signature, build_signature

__all__ = [
    "PipelineSpec",
    "compute_data_config_signature",
    "build_signature",
    "PipelineRunner",
    "WrapPipelineRunner",
]


def __getattr__(name: str):
    """Lazily import runner classes to avoid circular imports."""
    if name in {"PipelineRunner", "WrapPipelineRunner"}:
        from .runner import PipelineRunner, WrapPipelineRunner

        return {
            "PipelineRunner": PipelineRunner,
            "WrapPipelineRunner": WrapPipelineRunner,
        }[name]
    raise AttributeError(f"module 'pipelines' has no attribute {name!r}")
