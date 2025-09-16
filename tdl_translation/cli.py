"""Command line entry point for the demo pipeline implementation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Dict

from .artifacts import EnvironmentModel, PipelineResult, RobotProfile
from .pipeline import TDLPipeline


def build_demo_environment() -> EnvironmentModel:
    """Create a small environment graph used by the CLI demo."""

    locations = {
        "home": (0.0, 0.0, 0.0),
        "storage_rack": (1.0, 0.0, 0.0),
        "inspection_table": (1.5, 0.5, 0.0),
        "loading_dock": (2.0, 1.0, 0.0),
        "waste_bin": (0.5, -0.5, 0.0),
    }
    adjacency = {
        "home": ["storage_rack", "waste_bin"],
        "storage_rack": ["home", "inspection_table", "loading_dock"],
        "inspection_table": ["storage_rack"],
        "loading_dock": ["storage_rack"],
        "waste_bin": ["home"],
    }
    return EnvironmentModel(
        name="demo_cell",
        locations=locations,
        adjacency=adjacency,
        forbidden_zones=[],
        dynamic_obstacles={"forklift": "loading_dock"},
        home_base="home",
    )


def build_robot_profile(vendor: str, payload: float, reach: float) -> RobotProfile:
    joints = [f"j{i}" for i in range(1, 7)]
    return RobotProfile(
        name=f"{vendor}_arm",
        vendor=vendor,
        payload_capacity_kg=payload,
        reach_radius_m=reach,
        joints=joints,
    )


def _result_to_dict(result: PipelineResult) -> Dict[str, Dict]:
    def maybe(dataclass_obj):
        return asdict(dataclass_obj) if dataclass_obj is not None else None

    return {
        "task_spec": maybe(result.task_spec),
        "path_plan": maybe(result.path_plan),
        "motion_plan": maybe(result.motion_plan),
        "tdl_program": maybe(result.tdl_program),
        "vendor_code": maybe(result.vendor_code),
        "verification": maybe(result.verification),
        "deployment": maybe(result.deployment),
        "logs": [asdict(log) for log in result.logs],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the TDL translation pipeline demo")
    parser.add_argument("command", help="Natural language instruction for the robot")
    parser.add_argument("--vendor", default="ros2", help="Target robot vendor (default: ros2)")
    parser.add_argument("--payload", type=float, default=5.0, help="Robot payload capacity in kg")
    parser.add_argument("--reach", type=float, default=2.5, help="Robot reach radius in meters")
    parser.add_argument("--json", action="store_true", help="Emit full JSON description of pipeline artifacts")
    parser.add_argument(
        "--show-logs",
        action="store_true",
        help="Print state machine logs after execution",
    )

    args = parser.parse_args(argv)

    environment = build_demo_environment()
    robot = build_robot_profile(args.vendor, payload=args.payload, reach=args.reach)

    pipeline = TDLPipeline()
    result = pipeline.run(
        args.command,
        environment=environment,
        robot=robot,
        knowledge_base={"environment": environment.name},
    )

    if args.json:
        print(json.dumps(_result_to_dict(result), indent=2))
    else:
        if result.deployment:
            print(result.deployment.details)
        if result.tdl_program:
            print("\nTDL Program:\n" + result.tdl_program.text)

    if args.show_logs:
        print("\nPipeline Logs:")
        for entry in result.logs:
            print(f" - [{entry.state}] {entry.status}: {entry.message}")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())

