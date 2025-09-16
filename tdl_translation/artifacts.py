"""Core data models used across the TDL translation pipeline.

The repository models the architecture described in :mod:`README.md`.  Each
stage of the pipeline exchanges structured artifacts instead of loosely typed
Python dictionaries.  The dataclasses in this module aim to mirror the
documents referenced in the specification (e.g., ``TaskSpec``, ``MotionPlan``,
``TDLProgram``) and provide small helpers used by the orchestration logic.

The real system would populate these models from sophisticated planners and
robot drivers.  Here we keep the implementations lightweight while still
capturing the semantics of the interface between components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class TaskSpec:
    """Structured description of a user task.

    Attributes
    ----------
    goal:
        Natural language representation of the task objective.
    constraints:
        Human readable list of constraints inferred from the prompt or domain
        policy.  Examples include payload limits or safety instructions.
    environment_context:
        Key-value metadata pulled from the environment model or knowledge
        base (e.g., map identifiers).
    success_criteria:
        Conditions that determine when the task has been completed.
    required_payload_kg:
        Estimated payload the robot is expected to manipulate.
    source_location / target_location:
        Logical labels used by the planning layer to identify start/end
        waypoints.
    priority:
        Lower numbers indicate higher urgency and can influence scheduling.
    """

    goal: str
    constraints: List[str]
    environment_context: Dict[str, str]
    success_criteria: List[str]
    required_payload_kg: float = 0.0
    source_location: Optional[str] = None
    target_location: Optional[str] = None
    priority: int = 5


@dataclass
class Waypoint:
    """A named pose in the robot workspace."""

    name: str
    position: Tuple[float, float, float]


@dataclass
class PathPlan:
    """Geometric path produced by the path planner."""

    waypoints: List[Waypoint]
    length_m: float
    metadata: Dict[str, str] = field(default_factory=dict)

    def waypoint_names(self) -> List[str]:
        return [waypoint.name for waypoint in self.waypoints]


@dataclass
class MotionStep:
    """Single motion primitive targeting a waypoint."""

    waypoint: Waypoint
    joint_targets: Dict[str, float]
    gripper: str = "hold"


@dataclass
class MotionPlan:
    """Vendor agnostic sequence of motion primitives."""

    steps: List[MotionStep]
    duration_s: float


@dataclass
class TDLProgram:
    """Intermediate Task Description Language representation."""

    text: str
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ConstraintReport:
    """Result of validating a TDL program against robot constraints."""

    valid: bool
    violations: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValueError("ConstraintReport is invalid: " + "; ".join(self.violations))


@dataclass
class VendorCode:
    """Robot specific program ready for deployment."""

    vendor: str
    content: str
    diagnostics: Dict[str, str] = field(default_factory=dict)


@dataclass
class VerificationReport:
    """Evidence produced by the verification suite."""

    approved: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class DeploymentLog:
    """Result of applying vendor code on a robot."""

    success: bool
    details: str


@dataclass
class RobotProfile:
    """Capability profile for a specific robot model."""

    name: str
    vendor: str
    payload_capacity_kg: float
    reach_radius_m: float
    joints: Iterable[str]


@dataclass
class EnvironmentModel:
    """Workspace map and safety metadata used by planners/validators."""

    name: str
    locations: Dict[str, Tuple[float, float, float]]
    adjacency: Dict[str, List[str]]
    forbidden_zones: List[str] = field(default_factory=list)
    dynamic_obstacles: Dict[str, str] = field(default_factory=dict)
    home_base: str = "home"

    def neighbors(self, location: str) -> List[str]:
        return list(self.adjacency.get(location, []))


@dataclass
class PipelineLogEntry:
    """Single entry in the orchestration event log."""

    state: str
    status: str
    message: str


@dataclass
class PipelineResult:
    """Structured output of the :class:`tdl_translation.pipeline.TDLPipeline`."""

    task_spec: Optional[TaskSpec]
    path_plan: Optional[PathPlan]
    motion_plan: Optional[MotionPlan]
    tdl_program: Optional[TDLProgram]
    vendor_code: Optional[VendorCode]
    verification: Optional[VerificationReport]
    deployment: Optional[DeploymentLog]
    logs: List[PipelineLogEntry]


__all__ = [
    "ConstraintReport",
    "DeploymentLog",
    "EnvironmentModel",
    "MotionPlan",
    "MotionStep",
    "PathPlan",
    "PipelineLogEntry",
    "PipelineResult",
    "RobotProfile",
    "TaskSpec",
    "TDLProgram",
    "VendorCode",
    "VerificationReport",
    "Waypoint",
]

