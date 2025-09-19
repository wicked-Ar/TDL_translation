from __future__ import annotations

from tdl_translation.artifacts import MotionPlan, MotionStep, TaskSpec, Waypoint
from tdl_translation.hf_gemma import GemmaTDLCompiler, HuggingFaceConfig, HuggingFaceGemmaService


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = {"content": content}


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls = []

    def chat_completion(self, *, messages, **_: object):  # type: ignore[override]
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("No responses configured for fake client")
        return _FakeResponse(self._responses.pop(0))


def test_huggingface_service_parses_structured_task_spec() -> None:
    fake_response = """
    {"goal": "Move the crate", "constraints": ["Be careful"],
     "environment_context": {"env": "demo"}, "success_criteria": ["done"],
     "required_payload_kg": 3.5, "source_location": "storage_rack",
     "target_location": "loading_dock", "priority": 2}
    """
    client = _FakeClient([fake_response])
    service = HuggingFaceGemmaService(config=HuggingFaceConfig(), client=client)

    spec = service.analyze_requirement("Move the crate", knowledge_base={"env": "demo"})

    assert spec.goal == "Move the crate"
    assert spec.constraints == ["Be careful"]
    assert spec.environment_context["env"] == "demo"
    assert spec.target_location == "loading_dock"
    assert spec.priority == 2


def test_huggingface_compiler_extracts_tdl_block() -> None:
    tdl_block = """```tdl\nTDL TASK auto_generated_task {\n  precondition: robot_ready\n  steps:\n    - move_to name='storage_rack' coords=(1.000, 0.000, 0.000) joints={'j1': 1.0} gripper=hold\n  postcondition: task_complete\n}\n```"""
    client = _FakeClient([tdl_block])
    compiler = GemmaTDLCompiler(config=HuggingFaceConfig(), client=client)
    task = TaskSpec(
        goal="Move the crate",
        constraints=[],
        environment_context={},
        success_criteria=[],
        required_payload_kg=0.0,
    )
    plan = MotionPlan(
        steps=[
            MotionStep(
                waypoint=Waypoint(name="storage_rack", position=(1.0, 0.0, 0.0)),
                joint_targets={"j1": 1.0},
                gripper="hold",
            )
        ],
        duration_s=12.3,
    )

    program = compiler.generate_program(task, plan)

    assert "TDL TASK auto_generated_task" in program.text
    assert program.metadata["generated_from"] == "GemmaTDLCompiler"
    assert program.metadata["model"] == HuggingFaceConfig().model
