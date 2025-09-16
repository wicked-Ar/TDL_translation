"""Building blocks that implement the stages described in the README."""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from .artifacts import (
    ConstraintReport,
    EnvironmentModel,
    MotionPlan,
    MotionStep,
    PathPlan,
    RobotProfile,
    TaskSpec,
    TDLProgram,
    VendorCode,
    VerificationReport,
    Waypoint,
)
from .errors import (
    ConstraintViolationError,
    MotionPlanningError,
    PathPlanningError,
    RequirementAnalysisError,
    TDLGenerationError,
    TranslationError,
    VerificationError,
)


_WEIGHT_PATTERN = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kg|kilograms?|g|grams?)", re.I)
_SOURCE_PATTERN = re.compile(r"from\s+the\s+(?P<location>[\w\s]+?)(?:\s+to|\.|$)", re.I)
_TARGET_PATTERN = re.compile(r"to\s+the\s+(?P<location>[\w\s]+?)(?:\.|$)", re.I)
_TIME_PATTERN = re.compile(r"within\s+(?P<minutes>\d+)\s+minutes?", re.I)


def _normalize_location(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    cleaned = label.strip().lower()
    for suffix in (" carefully", " safely", " slowly", " gently"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    slug = cleaned.replace(" ", "_")
    return slug or None


class GemmaLLMService:
    """Simple heuristic stand-in for the natural-language analysis service."""

    def analyze_requirement(
        self,
        command: str,
        *,
        knowledge_base: Optional[Dict[str, str]] = None,
    ) -> TaskSpec:
        if not command.strip():
            raise RequirementAnalysisError("Empty command provided")

        text = command.strip()
        lower = text.lower()

        constraints: List[str] = []
        if "careful" in lower or "carefully" in lower:
            constraints.append("Handle payload carefully")
        if "without spilling" in lower:
            constraints.append("Maintain upright orientation to avoid spillage")
        if "avoid" in lower:
            constraints.append("Respect operator avoidance zones")

        payload_kg = 0.0
        match = _WEIGHT_PATTERN.search(lower)
        if match:
            value = float(match.group("value"))
            unit = match.group("unit").lower()
            if unit.startswith("g"):
                payload_kg = value / 1000.0
            else:
                payload_kg = value

        source_match = _SOURCE_PATTERN.search(lower)
        target_match = _TARGET_PATTERN.search(lower)
        source = _normalize_location(source_match.group("location")) if source_match else None
        target = _normalize_location(target_match.group("location")) if target_match else None

        success_criteria = ["Task executed without collisions"]
        time_match = _TIME_PATTERN.search(lower)
        if time_match:
            minutes = int(time_match.group("minutes"))
            success_criteria.append(f"Complete within {minutes} minutes")

        priority = 3 if "urgent" in lower else 5

        context = dict(knowledge_base or {})
        context.setdefault("parsed_by", "GemmaLLMService")

        return TaskSpec(
            goal=text,
            constraints=constraints,
            environment_context=context,
            success_criteria=success_criteria,
            required_payload_kg=payload_kg,
            source_location=source,
            target_location=target,
            priority=priority,
        )


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


@dataclass
class PathPlanningResult:
    plan: PathPlan
    attempts: int


class PlanningEngine:
    """Produces geometric and motion level plans from task specifications."""

    def __init__(self, *, max_replans: int = 3) -> None:
        self.max_replans = max_replans

    def plan_path(self, task: TaskSpec, environment: EnvironmentModel) -> PathPlanningResult:
        source = task.source_location or environment.home_base
        target = task.target_location
        if target is None:
            raise PathPlanningError(
                "No target location inferred from requirement",
                retry_state="User Requirement Analysis",
            )

        if source not in environment.locations:
            raise PathPlanningError(
                f"Unknown source location '{source}'",
                retry_state="User Requirement Analysis",
            )
        if target not in environment.locations:
            raise PathPlanningError(
                f"Unknown target location '{target}'",
                retry_state="User Requirement Analysis",
            )

        forbidden = set(name.lower() for name in environment.forbidden_zones)
        for obs in environment.dynamic_obstacles.values():
            if obs.lower() != (target or "").lower():
                forbidden.add(obs.lower())

        attempts = 0
        path: List[str] | None = None
        queue: deque[tuple[str, List[str]]] = deque([(source, [source])])
        visited = {source}
        while queue:
            location, current_path = queue.popleft()
            if location == target:
                path = current_path
                break
            for neighbor in environment.neighbors(location):
                if neighbor in visited or neighbor.lower() in forbidden:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, current_path + [neighbor]))
            attempts += 1
            if attempts > self.max_replans * len(environment.locations):
                break

        if path is None:
            raise PathPlanningError(
                f"No feasible path from {source} to {target}",
                retry_state="Path Planning",
            )

        waypoints: List[Waypoint] = []
        length = 0.0
        for idx, name in enumerate(path):
            position = environment.locations[name]
            waypoint = Waypoint(name=name, position=position)
            waypoints.append(waypoint)
            if idx > 0:
                prev = environment.locations[path[idx - 1]]
                length += _euclidean(prev, position)

        metadata = {
            "source": source,
            "target": target,
            "hops": str(len(path) - 1),
        }
        return PathPlanningResult(PathPlan(waypoints=waypoints, length_m=length, metadata=metadata), attempts)

    def plan_motion(self, plan: PathPlan, robot: RobotProfile) -> MotionPlan:
        if not plan.waypoints:
            raise MotionPlanningError("Path plan contained no waypoints")

        steps: List[MotionStep] = []
        duration = 0.0
        for waypoint in plan.waypoints:
            distance_from_base = _euclidean(waypoint.position, (0.0, 0.0, 0.0))
            if distance_from_base > robot.reach_radius_m:
                raise MotionPlanningError(
                    f"Waypoint {waypoint.name} ({distance_from_base:.2f}m) exceeds reach radius {robot.reach_radius_m}m",
                    retry_state="Path Planning",
                )

            joint_targets: Dict[str, float] = {}
            for idx, joint_name in enumerate(robot.joints):
                axis_value = waypoint.position[idx % len(waypoint.position)]
                joint_targets[joint_name] = axis_value / (idx + 1)
            gripper_state = "open" if waypoint == plan.waypoints[-1] else "hold"
            steps.append(MotionStep(waypoint=waypoint, joint_targets=joint_targets, gripper=gripper_state))
            duration += max(0.5, _euclidean(waypoint.position, (0.0, 0.0, 0.0)) / 0.5)

        return MotionPlan(steps=steps, duration_s=duration)


class TDLCompiler:
    """Creates a declarative TDL program from motion primitives."""

    def generate_program(self, task: TaskSpec, motion_plan: MotionPlan) -> TDLProgram:
        if not motion_plan.steps:
            raise TDLGenerationError("Motion plan did not contain any steps")

        lines = ["TDL TASK auto_generated_task {", "  precondition: robot_ready", "  steps:"]
        for step in motion_plan.steps:
            coords = ", ".join(f"{value:.3f}" for value in step.waypoint.position)
            lines.append(
                f"    - move_to name='{step.waypoint.name}' coords=({coords}) joints={step.joint_targets} gripper={step.gripper}"
            )
        lines.append("  postcondition: task_complete")
        if task.success_criteria:
            lines.append(f"  success_criteria: {task.success_criteria}")
        if task.constraints:
            lines.append(f"  constraints: {task.constraints}")
        lines.append("}")

        program_text = "\n".join(lines)
        metadata = {
            "generated_from": "TDLCompiler",
            "duration_s": f"{motion_plan.duration_s:.2f}",
        }
        return TDLProgram(text=program_text, metadata=metadata)


class ConstraintEvaluator:
    """Validates generated TDL programs against robot capability profiles."""

    def evaluate(
        self,
        task: TaskSpec,
        motion_plan: MotionPlan,
        robot: RobotProfile,
    ) -> ConstraintReport:
        violations: List[str] = []
        if task.required_payload_kg > robot.payload_capacity_kg:
            violations.append(
                f"Required payload {task.required_payload_kg:.2f}kg exceeds robot capacity {robot.payload_capacity_kg:.2f}kg"
            )

        if motion_plan.duration_s > 600:
            violations.append("Motion plan duration exceeds 10 minute safety threshold")

        for step in motion_plan.steps:
            for joint, value in step.joint_targets.items():
                if not -3.14 <= value <= 3.14:
                    violations.append(f"Joint {joint} target {value:.2f} rad outside safe envelope")

        return ConstraintReport(valid=not violations, violations=violations)


class VendorAdapter:
    """Base class for vendor specific translators."""

    vendor: str

    def translate(self, program: TDLProgram) -> VendorCode:  # pragma: no cover - interface
        raise NotImplementedError


class ROS2Adapter(VendorAdapter):
    vendor = "ros2"

    def translate(self, program: TDLProgram) -> VendorCode:
        content = ["# ROS2 job file generated from TDL", "---", program.text]
        return VendorCode(vendor=self.vendor, content="\n".join(content), diagnostics={"adapter": self.vendor})


class FanucAdapter(VendorAdapter):
    vendor = "fanuc"

    def translate(self, program: TDLProgram) -> VendorCode:
        header = "; FANUC TP program"
        body = program.text.replace("move_to", "J P[1] 100% FINE")
        return VendorCode(vendor=self.vendor, content=f"{header}\n{body}", diagnostics={"adapter": self.vendor})


class VendorAdapterLayer:
    """Dispatches to the correct adapter for the target robot family."""

    def __init__(self, adapters: Optional[Iterable[VendorAdapter]] = None) -> None:
        adapters = adapters or [ROS2Adapter(), FanucAdapter()]
        self._adapters: Dict[str, VendorAdapter] = {adapter.vendor.lower(): adapter for adapter in adapters}

    def translate(self, program: TDLProgram, robot: RobotProfile) -> VendorCode:
        vendor_key = robot.vendor.lower()
        adapter = self._adapters.get(vendor_key)
        if adapter is None:
            raise TranslationError(f"No adapter registered for vendor '{robot.vendor}'", retry_state="TDL Generation")
        return adapter.translate(program)


class VerificationSuite:
    """Runs simplified verification checks mimicking simulation/formal tools."""

    def verify(
        self,
        vendor_code: VendorCode,
        path_plan: PathPlan,
        environment: EnvironmentModel,
    ) -> VerificationReport:
        issues: List[str] = []
        forbidden = {zone.lower() for zone in environment.forbidden_zones}
        for waypoint in path_plan.waypoint_names():
            if waypoint.lower() in forbidden:
                issues.append(f"Waypoint '{waypoint}' intersects forbidden zone")

        if environment.dynamic_obstacles:
            obstacles = ", ".join(environment.dynamic_obstacles.keys())
            issues.append(f"Dynamic obstacles pending clearance: {obstacles}")

        approved = not issues
        if not approved:
            raise VerificationError(
                "Verification failed: " + "; ".join(issues),
                retry_state="Path Planning" if any("Waypoint" in issue for issue in issues) else "Target Job File Translation",
            )

        diagnostics = {"lines": str(len(vendor_code.content.splitlines()))}
        return VerificationReport(approved=True, issues=[f"Diagnostics: {diagnostics}"])


class DeploymentOrchestrator:
    """Simulates deployment to a physical robot."""

    def deploy(self, vendor_code: VendorCode) -> str:
        return (
            "Deployment succeeded for vendor="
            f"{vendor_code.vendor} (size={len(vendor_code.content.splitlines())} lines)"
        )


__all__ = [
    "ConstraintEvaluator",
    "DeploymentOrchestrator",
    "FanucAdapter",
    "GemmaLLMService",
    "PlanningEngine",
    "ROS2Adapter",
    "TDLCompiler",
    "VendorAdapterLayer",
    "VerificationSuite",
]

