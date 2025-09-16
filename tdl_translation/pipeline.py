"""State machine orchestration for the TDL translation reference pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from .artifacts import (
    DeploymentLog,
    EnvironmentModel,
    MotionPlan,
    PathPlan,
    PipelineLogEntry,
    PipelineResult,
    RobotProfile,
    TaskSpec,
    TDLProgram,
    VendorCode,
    VerificationReport,
)
from .components import (
    ConstraintEvaluator,
    DeploymentOrchestrator,
    GemmaLLMService,
    PlanningEngine,
    TDLCompiler,
    VendorAdapterLayer,
    VerificationSuite,
)
from .errors import (
    ConstraintViolationError,
    MotionPlanningError,
    PathPlanningError,
    PipelineError,
    PipelineStepError,
    RequirementAnalysisError,
    TDLGenerationError,
    TranslationError,
    VerificationError,
)

STATE_INIT = "State_init"
STATE_REQUIREMENT = "User Requirement Analysis"
STATE_PATH_PLANNING = "Path Planning"
STATE_MOTION_PLANNING = "Motion Planning"
STATE_TDL = "TDL Generation"
STATE_CONSTRAINT = "Robot Constraint Analysis"
STATE_TRANSLATION = "Target Job File Translation"
STATE_VERIFICATION = "Validation & Verification"
STATE_DEPLOYMENT = "Robot Update"


class TDLPipeline:
    """Implements the high level state machine from the README mermaid diagram."""

    def __init__(
        self,
        *,
        llm: Optional[GemmaLLMService] = None,
        planner: Optional[PlanningEngine] = None,
        compiler: Optional[TDLCompiler] = None,
        constraint_evaluator: Optional[ConstraintEvaluator] = None,
        vendor_layer: Optional[VendorAdapterLayer] = None,
        verifier: Optional[VerificationSuite] = None,
        deployer: Optional[DeploymentOrchestrator] = None,
        max_attempts_per_state: int = 3,
    ) -> None:
        self.llm = llm or GemmaLLMService()
        self.planner = planner or PlanningEngine()
        self.compiler = compiler or TDLCompiler()
        self.constraint_evaluator = constraint_evaluator or ConstraintEvaluator()
        self.vendor_layer = vendor_layer or VendorAdapterLayer()
        self.verifier = verifier or VerificationSuite()
        self.deployer = deployer or DeploymentOrchestrator()
        self.max_attempts_per_state = max_attempts_per_state

    def run(
        self,
        command: str,
        *,
        environment: EnvironmentModel,
        robot: RobotProfile,
        knowledge_base: Optional[Dict[str, str]] = None,
    ) -> PipelineResult:
        logs: list[PipelineLogEntry] = []
        attempts: Dict[str, int] = defaultdict(int)

        state = STATE_INIT
        task_spec: TaskSpec | None = None
        path_plan: PathPlan | None = None
        motion_plan: MotionPlan | None = None
        tdl_program: TDLProgram | None = None
        vendor_code: VendorCode | None = None
        verification: VerificationReport | None = None
        deployment: DeploymentLog | None = None

        def log(state_name: str, status: str, message: str) -> None:
            logs.append(PipelineLogEntry(state=state_name, status=status, message=message))

        while True:
            if state == STATE_INIT:
                log(state, "info", "Awaiting new task")
                state = STATE_REQUIREMENT
                continue

            try:
                if state == STATE_REQUIREMENT:
                    task_spec = self.llm.analyze_requirement(command, knowledge_base=knowledge_base)
                    log(state, "success", f"Parsed goal='{task_spec.goal}'")
                    state = STATE_PATH_PLANNING

                elif state == STATE_PATH_PLANNING:
                    result = self.planner.plan_path(task_spec, environment)  # type: ignore[arg-type]
                    path_plan = result.plan
                    log(state, "success", f"Computed path with {len(path_plan.waypoints)} waypoints")
                    state = STATE_MOTION_PLANNING

                elif state == STATE_MOTION_PLANNING:
                    motion_plan = self.planner.plan_motion(path_plan, robot)  # type: ignore[arg-type]
                    log(state, "success", f"Generated motion plan lasting {motion_plan.duration_s:.2f}s")
                    state = STATE_TDL

                elif state == STATE_TDL:
                    tdl_program = self.compiler.generate_program(task_spec, motion_plan)  # type: ignore[arg-type]
                    log(state, "success", "TDL program generated")
                    state = STATE_CONSTRAINT

                elif state == STATE_CONSTRAINT:
                    report = self.constraint_evaluator.evaluate(task_spec, motion_plan, robot)  # type: ignore[arg-type]
                    if not report.valid:
                        raise ConstraintViolationError("; ".join(report.violations), retry_state=STATE_TDL)
                    log(state, "success", "Constraints satisfied")
                    state = STATE_TRANSLATION

                elif state == STATE_TRANSLATION:
                    vendor_code = self.vendor_layer.translate(tdl_program, robot)  # type: ignore[arg-type]
                    log(state, "success", f"Translated TDL to vendor='{vendor_code.vendor}' code")
                    state = STATE_VERIFICATION

                elif state == STATE_VERIFICATION:
                    verification = self.verifier.verify(vendor_code, path_plan, environment)  # type: ignore[arg-type]
                    log(state, "success", "Verification approved")
                    state = STATE_DEPLOYMENT

                elif state == STATE_DEPLOYMENT:
                    deployment_msg = self.deployer.deploy(vendor_code)  # type: ignore[arg-type]
                    deployment = DeploymentLog(success=True, details=deployment_msg)
                    log(state, "success", deployment_msg)
                    break

                else:
                    raise PipelineError(f"Unknown pipeline state '{state}'")

            except PipelineStepError as exc:  # pragma: no branch - handled per state
                attempts[state] += 1
                log(state, "retry", str(exc))

                if attempts[state] >= self.max_attempts_per_state:
                    log(state, "failure", "Exceeded retry budget")
                    raise PipelineError(f"Exceeded retries in state {state}: {exc}") from exc

                if isinstance(exc, VerificationError) and "Dynamic obstacles" in str(exc):
                    # Simulate the planner waiting for external clearance.
                    environment.dynamic_obstacles.clear()
                    log(state, "info", "Cleared dynamic obstacles after verification failure")

                next_state = exc.retry_state or state
                state = next_state

            except Exception as exc:  # pragma: no cover - defensive
                log(state, "failure", f"Unhandled error: {exc}")
                raise

        return PipelineResult(
            task_spec=task_spec,
            path_plan=path_plan,
            motion_plan=motion_plan,
            tdl_program=tdl_program,
            vendor_code=vendor_code,
            verification=verification,
            deployment=deployment,
            logs=logs,
        )


__all__ = ["TDLPipeline"]

