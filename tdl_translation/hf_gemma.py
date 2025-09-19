"""Integration helpers for calling Gemma models via the Hugging Face Inference API."""

from __future__ import annotations

import inspect
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .artifacts import MotionPlan, TaskSpec, TDLProgram
from .components import GemmaLLMService, TDLCompiler
from .errors import RequirementAnalysisError, TDLGenerationError

try:  # pragma: no cover - optional dependency import guard
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - handled at runtime
    InferenceClient = None  # type: ignore[assignment]


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_BLOCK_RE = re.compile(r"```(?:tdl)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _preferred_stop_keywords(callable_obj: Any) -> Tuple[str, ...]:
    """Return the preferred stop keyword order supported by the callable."""

    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):  # pragma: no cover - signature may be unsupported
        return ("stop", "stop_sequences")

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    preferred: List[str] = []

    for candidate in ("stop", "stop_sequences"):
        parameter = signature.parameters.get(candidate)
        if parameter is not None and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            preferred.append(candidate)

    if not preferred and has_var_kwargs:
        preferred = ["stop", "stop_sequences"]

    if not preferred:
        preferred = ["stop_sequences", "stop"]

    # Deduplicate while preserving order and ensure both fallbacks are present for robustness.
    ordered = []
    for candidate in preferred + ["stop", "stop_sequences"]:
        if candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


def _default_token(token: Optional[str]) -> str:
    resolved = token or os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN")
    if not resolved:
        raise ValueError(
            "A Hugging Face access token must be supplied either via the constructor "
            "or the HUGGINGFACE_API_TOKEN/HF_API_TOKEN environment variables."
        )
    return resolved


def _extract_json_object(payload: str) -> Dict[str, Any]:
    match = _JSON_BLOCK_RE.search(payload)
    if not match:
        raise RequirementAnalysisError("LLM response did not contain a JSON object")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise RequirementAnalysisError(f"Failed to parse JSON from LLM response: {exc}") from exc


def _extract_tdl_block(payload: str) -> str:
    match = _CODE_BLOCK_RE.search(payload)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = payload.strip()
    if "TDL TASK" not in candidate:
        raise TDLGenerationError("LLM response did not contain a TDL TASK block")
    return candidate


@dataclass
class HuggingFaceConfig:
    """Configuration for the Hugging Face Gemma integration."""

    model: str = "google/gemma-2-9b-it"
    token: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 1024
    top_p: float = 0.9
    stop_sequences: Optional[Iterable[str]] = None


class _ChatMixin:
    """Helper mixin that wraps the Hugging Face chat completion API."""

    def __init__(self, *, config: HuggingFaceConfig, client: Optional[Any] = None) -> None:
        if client is not None:
            self._client = client
        else:
            if InferenceClient is None:  # pragma: no cover - import guard
                raise ImportError(
                    "huggingface_hub is required to use HuggingFaceGemmaService. Install the extra dependency"
                )
            token = _default_token(config.token)
            self._client = InferenceClient(model=config.model, token=token)
        self._config = config

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        base_kwargs = {
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "top_p": self._config.top_p,
        }
        stop_values: Tuple[str, ...] = tuple(self._config.stop_sequences or ("</s>",))

        response = None
        last_type_error: Optional[TypeError] = None
        for stop_kw in _preferred_stop_keywords(self._client.chat_completion):
            kwargs = dict(base_kwargs)
            if stop_values:
                kwargs[stop_kw] = list(stop_values)
            try:
                response = self._client.chat_completion(  # type: ignore[operator]
                    **kwargs
                )
                break
            except TypeError as exc:
                message = str(exc)
                if "unexpected keyword" in message and stop_kw in message:
                    last_type_error = exc
                    continue
                raise

        if response is None:
            raise last_type_error or TypeError(
                "Failed to call Hugging Face chat completion API due to incompatible parameters"
            )
        try:
            choice = response.choices[0]
            content = choice.message["content"] if isinstance(choice.message, Mapping) else choice.message.content
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected response format from Hugging Face API: {response}") from exc
        if not isinstance(content, str):
            raise RuntimeError(f"Unexpected message content from Hugging Face API: {content!r}")
        return content


class HuggingFaceGemmaService(GemmaLLMService, _ChatMixin):
    """LLM-backed requirement analysis using Hugging Face hosted Gemma models."""

    def __init__(
        self,
        *,
        config: Optional[HuggingFaceConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config or HuggingFaceConfig()
        _ChatMixin.__init__(self, config=self._config, client=client)

    def analyze_requirement(
        self,
        command: str,
        *,
        knowledge_base: Optional[Dict[str, str]] = None,
    ) -> TaskSpec:
        if not command.strip():
            raise RequirementAnalysisError("Empty command provided")

        kb_display = json.dumps(knowledge_base or {}, indent=2, ensure_ascii=False)
        system_prompt = (
            "You are an expert industrial robotics planner. "
            "Extract structured task specifications from operator requests. "
            "Always respond with strict JSON matching the provided schema."
        )
        user_prompt = f"""
Instruction:
{command.strip()}

KnowledgeBase:
{kb_display}

Respond with a JSON object containing the following keys:
- goal (string, echo the operator instruction)
- constraints (array of strings)
- environment_context (object mapping string keys to string values)
- success_criteria (array of strings)
- required_payload_kg (number)
- source_location (string or null)
- target_location (string or null)
- priority (integer from 1-5, 1 is highest urgency)
Ensure the JSON is valid and does not include trailing commentary.
"""
        content = self._chat(system_prompt, user_prompt)
        data = _extract_json_object(content)

        try:
            return TaskSpec(
                goal=str(data.get("goal", command.strip())),
                constraints=list(data.get("constraints", [])),
                environment_context={k: str(v) for k, v in dict(data.get("environment_context", {})).items()},
                success_criteria=list(data.get("success_criteria", [])),
                required_payload_kg=float(data.get("required_payload_kg", 0.0)),
                source_location=(data.get("source_location") or None),
                target_location=(data.get("target_location") or None),
                priority=int(data.get("priority", 5)),
            )
        except (TypeError, ValueError) as exc:
            raise RequirementAnalysisError(f"LLM returned malformed TaskSpec fields: {exc}") from exc


class GemmaTDLCompiler(TDLCompiler, _ChatMixin):
    """TDL compiler that asks a Gemma model to format the motion plan."""

    def __init__(
        self,
        *,
        config: Optional[HuggingFaceConfig] = None,
        client: Optional[Any] = None,
    ) -> None:
        self._config = config or HuggingFaceConfig()
        _ChatMixin.__init__(self, config=self._config, client=client)

    def generate_program(self, task: TaskSpec, motion_plan: MotionPlan) -> TDLProgram:
        if not motion_plan.steps:
            raise TDLGenerationError("Motion plan did not contain any steps")

        steps_payload: List[Dict[str, Any]] = []
        for step in motion_plan.steps:
            steps_payload.append(
                {
                    "name": step.waypoint.name,
                    "position": [round(float(coord), 4) for coord in step.waypoint.position],
                    "joints": step.joint_targets,
                    "gripper": step.gripper,
                }
            )

        system_prompt = (
            "You are a compiler that formats motion plans into valid Task Description Language (TDL). "
            "Return only the TDL program."
        )
        user_prompt = f"""
Create a TDL program for the following task specification and motion steps.
TaskSpec:
{json.dumps(asdict(task), indent=2, ensure_ascii=False)}

MotionPlan Steps (ordered):
{json.dumps(steps_payload, indent=2, ensure_ascii=False)}

Requirements:
- Emit a single TDL TASK block named 'auto_generated_task'.
- Include a precondition of robot_ready and postcondition of task_complete.
- For each step emit a move_to entry matching this schema:
    - move_to name='waypoint_name' coords=(x.xxx, y.yyy, z.zzz) joints={{...}} gripper=state
- Include success_criteria and constraints arrays from the task spec if they are not empty.
Respond only with the TDL code. Do not add explanations.
"""
        content = self._chat(system_prompt, user_prompt)
        tdl_text = _extract_tdl_block(content)

        metadata = {
            "generated_from": "GemmaTDLCompiler",
            "duration_s": f"{motion_plan.duration_s:.2f}",
            "model": self._config.model,
        }
        return TDLProgram(text=tdl_text, metadata=metadata)


__all__ = [
    "GemmaTDLCompiler",
    "HuggingFaceConfig",
    "HuggingFaceGemmaService",
]
