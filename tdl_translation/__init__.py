"""TDL translation reference pipeline implementation."""

from .artifacts import *  # noqa: F401,F403 - convenient re-exports
from .pipeline import TDLPipeline

__all__ = ["TDLPipeline"]

try:  # pragma: no cover - optional export
    from .hf_gemma import GemmaTDLCompiler, HuggingFaceConfig, HuggingFaceGemmaService

    __all__.extend(["GemmaTDLCompiler", "HuggingFaceConfig", "HuggingFaceGemmaService"])
except Exception:  # pragma: no cover - huggingface optional at import time
    pass
