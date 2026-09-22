"""M0 spike: can we turn Claude's native thinking off through Inspect?

Why this matters: the eval intervenes on reasoning held in `think()` tool-call
arguments. If the model *also* reasons in a native thinking block we cannot see
or edit, that block is an unmeasured mediator and the intervention is incomplete.

This script builds the Anthropic request parameters Inspect would send, without
making any API call, and asserts what lands in the `thinking` field. Run it after
any `inspect-ai` upgrade: it is the regression guard for the primary run's
central assumption.

    uv run python scripts/spike_thinking_config.py
"""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-spike-no-request-is-made")

from inspect_ai.model import GenerateConfig, get_model

SUBJECT = "anthropic/claude-sonnet-5"
CLASSIFIER = "anthropic/claude-opus-5"


def params_for(model_name: str, config: GenerateConfig) -> dict[str, Any]:
    api = get_model(model_name, config=config).api
    params, _extra_body, _headers, _betas = api.completion_config(config)
    return params


def describe(label: str, model_name: str, config: GenerateConfig) -> dict[str, Any]:
    params = params_for(model_name, config)
    thinking = params.get("thinking", "<field absent>")
    output_config = params.get("output_config", "<field absent>")
    print(f"{label:<34} thinking={thinking}  output_config={output_config}")
    return params


def main() -> None:
    print(f"subject model: {SUBJECT}\n")

    off = describe(
        "reasoning_effort='none'",
        SUBJECT,
        GenerateConfig(max_tokens=4096, reasoning_effort="none"),
    )
    on = describe(
        "reasoning_effort='high'",
        SUBJECT,
        GenerateConfig(max_tokens=4096, reasoning_effort="high"),
    )
    default = describe(
        "unset (provider default)",
        SUBJECT,
        GenerateConfig(max_tokens=4096),
    )

    assert off.get("thinking") == {"type": "disabled"}, (
        "PRIMARY RUN ASSUMPTION BROKEN: reasoning_effort='none' no longer disables "
        f"native thinking on {SUBJECT}. Got {off.get('thinking')!r}. The think() "
        "intervention would be incomplete — see README limitations before running."
    )
    assert (
        on.get("thinking", {}).get("type") == "adaptive"
    ), f"expected adaptive thinking at reasoning_effort='high', got {on.get('thinking')!r}"
    assert "thinking" not in default, (
        "provider default now sets `thinking` explicitly; the robustness arm's "
        f"configuration needs revisiting. Got {default.get('thinking')!r}"
    )

    print("\nRESULT")
    print(
        "  Primary arm     : reasoning_effort='none' -> thinking {'type': 'disabled'}."
    )
    print("                    think() is the whole observable reasoning channel.")
    print(
        "  Robustness arm  : reasoning_effort='high' -> adaptive thinking, summarized."
    )
    print("                    Effect direction must survive here too.")
    print(
        "\nNote: reasoning_tokens is rejected outright on Claude 4.7+ — do not use it."
    )
    try:
        params_for(SUBJECT, GenerateConfig(max_tokens=4096, reasoning_tokens=8192))
    except Exception as exc:  # noqa: BLE001 - we are demonstrating the failure mode
        print(f"  confirmed: {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")
    else:
        raise AssertionError(
            "reasoning_tokens was accepted; provider behaviour changed"
        )

    print("\nAll spike assertions passed.")


if __name__ == "__main__":
    main()
