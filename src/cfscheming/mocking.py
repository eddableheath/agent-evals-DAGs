"""Offline model doubles, so the whole solver chain can be exercised without API spend.

`mockllm` accepts a callable `(input, tools, tool_choice, config) -> ModelOutput`,
which is what makes deterministic counterfactual tests possible: the double can
branch on the *content of the conversation it is handed*, so a branch whose
`think()` call has been rewritten can be made to answer differently from the
baseline branch. That is the mechanism the real eval measures, reproduced with
zero randomness and zero cost.

Use these helpers in tests; never in a scored run.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo

MOCK_MODEL = "mockllm/model"

#: Signature `mockllm` calls on every generation.
Responder = Callable[
    [list[ChatMessage], list[ToolInfo], ToolChoice, GenerateConfig], ModelOutput
]


def think_output(thought: str, *, tool_call_id: str | None = None) -> ModelOutput:
    """A model turn that calls `think()` with `thought`."""
    return ModelOutput.for_tool_call(
        model="mockllm",
        tool_name="think",
        tool_arguments={"thought": thought},
        tool_call_id=tool_call_id,
    )


def tool_output(
    name: str, tool_call_id: str | None = None, **arguments: object
) -> ModelOutput:
    """A model turn that calls tool `name` with `arguments`."""
    return ModelOutput.for_tool_call(
        model="mockllm",
        tool_name=name,
        tool_arguments=dict(arguments),
        tool_call_id=tool_call_id,
    )


def text_output(content: str) -> ModelOutput:
    """A model turn that ends with plain text (no tool call)."""
    return ModelOutput.from_content(model="mockllm", content=content)


def scripted(outputs: Iterable[ModelOutput]) -> Model:
    """A model that returns `outputs` in order, regardless of what it is asked.

    Use for a fixed trajectory. Use `reactive` when a branch must respond to an
    edited history — a fixed script cannot demonstrate a counterfactual, because
    it returns the same thing whatever the intervention did.
    """
    return get_model(MOCK_MODEL, custom_outputs=list(outputs))


def reactive(responder: Callable[[list[ChatMessage]], ModelOutput]) -> Model:
    """A model that decides each turn from the conversation it is handed."""

    def _respond(
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        return responder(input)

    return get_model(MOCK_MODEL, custom_outputs=_respond)


def think_texts(messages: Sequence[ChatMessage]) -> list[str]:
    """Every `thought` argument passed to `think()` so far, in order.

    The observable reasoning channel. Interventions rewrite these strings; the
    classifier reads them; this helper is how tests see what the model 'thought'.
    """
    thoughts: list[str] = []
    for message in messages:
        if not isinstance(message, ChatMessageAssistant) or not message.tool_calls:
            continue
        for call in message.tool_calls:
            if call.function == "think":
                thought = call.arguments.get("thought")
                if isinstance(thought, str):
                    thoughts.append(thought)
    return thoughts
