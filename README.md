# Counterfactual interventions on agent reasoning

Does a coding agent's *evaluation-awareness* reasoning causally change whether it takes covert
actions? This repo measures that with counterfactual interventions on an agent's reasoning trace,
built on [Inspect AI](https://inspect.aisi.org.uk/).

The method follows Apollo Research and OpenAI's
[Stress Testing Deliberative Alignment](https://www.apolloresearch.ai/science/stress-testing-deliberative-alignment-for-anti-scheming-training),
which found that removing evaluation-awareness from o3's chain of thought raised covert-action rate
from 13.2% to 24.2%, while injecting it lowered the rate from 24.5% to 18.6%.

**Status: under construction (M0 — scaffold).** Results, limitations, and method details land here as
the milestones complete.

## Why the intervention happens at message boundaries

Apollo had raw chain-of-thought access under collaboration with OpenAI. This repo does not, and
assistant prefill is rejected by current Claude models — so there is no way to edit a partial
reasoning trace and resume generation from inside it.

Instead the agent is required to reason in `think()` **tool-call arguments**. A tool call can be
rewritten and generation resumed normally, because the message that follows it is a tool *result*.
The intervention is therefore complete with respect to the observable reasoning channel, and needs no
unsupported API behaviour.

## Layout

| Path | Contents |
|---|---|
| `src/cfscheming/` | The eval: environment, agent, classifier, interventions, scorers, analysis |
| `src/cfdag/` | Deferred — causal identification / assumption checker, no dependency on the eval |
| `scenarios/` | Scenario template and repo fixtures |
| `tests/` | Offline tests; no API spend |

## Development

```bash
uv sync --all-extras                          # install
uv run pytest                                 # offline tests, no API spend
uv run python scripts/spike_thinking_config.py  # re-check the native-thinking assumption
```

Once the task exists (M1):

```bash
uv run inspect eval src/cfscheming/task.py --model mockllm/model --limit 2
```

`tests/test_m0_harness.py` is the guard on the central mechanism — that `think()`
reasoning is readable, rewritable and resumable, and that `fork()` isolates branches.
Run it after any `inspect-ai` upgrade.
