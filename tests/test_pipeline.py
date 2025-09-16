"""Integration tests for the demo pipeline implementation."""

from __future__ import annotations

import pytest

from tdl_translation.artifacts import EnvironmentModel, RobotProfile
from tdl_translation.pipeline import TDLPipeline
from tdl_translation.errors import PipelineError


def _demo_environment(include_dynamic_obstacle: bool = False) -> EnvironmentModel:
    dynamic = {"forklift": "loading_dock"} if include_dynamic_obstacle else {}
    return EnvironmentModel(
        name="unit_test_cell",
        locations={
            "home": (0.0, 0.0, 0.0),
            "storage_rack": (1.0, 0.0, 0.0),
            "inspection_table": (1.5, 0.5, 0.0),
            "loading_dock": (2.0, 1.0, 0.0),
        },
        adjacency={
            "home": ["storage_rack"],
            "storage_rack": ["home", "inspection_table", "loading_dock"],
            "inspection_table": ["storage_rack"],
            "loading_dock": ["storage_rack"],
        },
        forbidden_zones=[],
        dynamic_obstacles=dynamic,
        home_base="home",
    )


def _robot(vendor: str = "ros2", payload: float = 10.0, reach: float = 3.0) -> RobotProfile:
    return RobotProfile(
        name="test_robot",
        vendor=vendor,
        payload_capacity_kg=payload,
        reach_radius_m=reach,
        joints=[f"j{i}" for i in range(1, 7)],
    )


def test_pipeline_succeeds_and_clears_dynamic_obstacles():
    pipeline = TDLPipeline()
    environment = _demo_environment(include_dynamic_obstacle=True)
    robot = _robot()

    result = pipeline.run(
        "Move the 2kg box from the storage rack to the loading dock carefully.",
        environment=environment,
        robot=robot,
    )

    assert result.deployment and result.deployment.success
    assert result.path_plan is not None
    assert result.path_plan.waypoint_names()[-1] == "loading_dock"
    assert any("Cleared dynamic obstacles" in entry.message for entry in result.logs)


def test_pipeline_fails_when_constraints_are_violated():
    pipeline = TDLPipeline()
    environment = _demo_environment()
    robot = _robot(payload=5.0)

    with pytest.raises(PipelineError):
        pipeline.run(
            "Move the 50kg crate from the storage rack to the loading dock.",
            environment=environment,
            robot=robot,
        )

