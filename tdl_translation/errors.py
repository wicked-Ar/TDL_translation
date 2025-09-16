"""Domain specific exceptions used by the pipeline state machine."""

from __future__ import annotations


class PipelineStepError(RuntimeError):
    """Base exception carrying retry hints for the orchestrator."""

    def __init__(self, message: str, *, retry_state: str | None = None) -> None:
        super().__init__(message)
        self.retry_state = retry_state


class RequirementAnalysisError(PipelineStepError):
    pass


class PathPlanningError(PipelineStepError):
    pass


class MotionPlanningError(PipelineStepError):
    pass


class TDLGenerationError(PipelineStepError):
    pass


class ConstraintViolationError(PipelineStepError):
    pass


class TranslationError(PipelineStepError):
    pass


class VerificationError(PipelineStepError):
    pass


class DeploymentError(PipelineStepError):
    pass


class PipelineError(RuntimeError):
    """Raised when the orchestrator cannot make forward progress."""


__all__ = [
    "ConstraintViolationError",
    "DeploymentError",
    "MotionPlanningError",
    "PathPlanningError",
    "PipelineError",
    "PipelineStepError",
    "RequirementAnalysisError",
    "TDLGenerationError",
    "TranslationError",
    "VerificationError",
]

