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
    parser.add_argument(
        "--use-hf-gemma",
        action="store_true",
        help="Use Hugging Face Gemma for requirement analysis and TDL generation",
    )
    parser.add_argument(
        "--hf-model",
        default="google/gemma-2-9b-it",
        help="Hugging Face model identifier to use when --use-hf-gemma is enabled",
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face token (falls back to HUGGINGFACE_API_TOKEN env var)",
    )
    parser.add_argument(
        "--hf-temperature",
        type=float,
        default=0.1,
        help="Sampling temperature for Gemma completions",
    )
    parser.add_argument(
        "--hf-max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate per completion",
    )
    parser.add_argument(
        "--hf-top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling parameter for Gemma completions",
    )

    args = parser.parse_args(argv)

    environment = build_demo_environment()
    robot = build_robot_profile(args.vendor, payload=args.payload, reach=args.reach)

    pipeline_kwargs = {}
    if args.use_hf_gemma:
        from .hf_gemma import GemmaTDLCompiler, HuggingFaceConfig, HuggingFaceGemmaService

        config = HuggingFaceConfig(
            model=args.hf_model,
            token=args.hf_token,
            temperature=args.hf_temperature,
            max_tokens=args.hf_max_tokens,
            top_p=args.hf_top_p,
        )
        hf_service = HuggingFaceGemmaService(config=config)
        hf_compiler = GemmaTDLCompiler(config=config)
        pipeline_kwargs.update({"llm": hf_service, "compiler": hf_compiler})

    pipeline = TDLPipeline(**pipeline_kwargs)
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

