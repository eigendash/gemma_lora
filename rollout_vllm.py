"""vLLM generation backend for tool-interleaved rollouts (for CUDA cloud GPUs).

Same semantics as :func:`rollout.run_rollouts_batched`: every rollout is driven by the
shared ``rollout._Rollout`` state machine, so tool calls, the round cap, hallucinated
``<output>`` handling, the token budget and the loss mask are identical. Only the
token source differs.

One *turn* = one ``llm.generate`` call over every pending row, each prompted with its
exact token ids (``prompt_ids + completion_ids``; vLLM's prefix cache makes the
re-prefill cheap). vLLM stops a row at ``</python>``, ``</answer>`` or ``<output>``
(stop strings are checked only in text generated in this turn, which is exactly the
state machine's current segment). The returned tokens are fed to ``_Rollout.add_token``
one by one; the state machine decides where the segment really ends (anything vLLM
returned after that point is dropped). Rows that paused for a tool get their output
injected and go to the next turn.

Log-probs: create the engine with ``logprobs_mode="processed_logprobs"`` so the sampled
token's log-prob is under softmax(logits / T), matching the trainer's tempered log-probs.

The policy's current LoRA weights reach vLLM as a ``LoRARequest`` (adapter saved to
disk each step with a new integer id).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from rollout import (ANSWER_CLOSE, CONTINUE, OUTPUT_OPEN, PYTHON_CLOSE, TOOL, RolloutResult, _Rollout,
                     encode_text)

STOPS = [PYTHON_CLOSE, ANSWER_CLOSE, OUTPUT_OPEN]


def _default_params(temperature: float, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(n=1, temperature=temperature, max_tokens=max_tokens, stop=STOPS,
                          include_stop_str_in_output=True, logprobs=0, skip_special_tokens=False)


def _token_logprob(logprobs, k: int, tid: int) -> float:
    if not logprobs:
        return 0.0
    entry = logprobs[k]
    v = entry.get(tid) if isinstance(entry, dict) else None
    if v is None:
        return 0.0
    return float(getattr(v, "logprob", v))


def run_rollouts_vllm(
    llm,
    tokenizer,
    prompt_texts: list[str],
    sandbox,
    *,
    max_rounds: int = 2,
    max_new_tokens: int = 768,
    temperature: float = 1.0,
    on_round_cap: str = "truncate",
    output_transform=None,
    lora_request=None,
    params_factory=None,
) -> list[RolloutResult]:
    make = params_factory or _default_params
    states = [_Rollout(tokenizer, encode_text(tokenizer, t, add_special_tokens=True), max_rounds=max_rounds,
                       max_new_tokens=max_new_tokens, on_round_cap=on_round_cap,
                       output_transform=output_transform) for t in prompt_texts]
    pending = list(range(len(states)))
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        while pending:
            prompts = [{"prompt_token_ids": states[i].context_ids} for i in pending]
            params = [make(temperature, max(1, max_new_tokens - len(states[i].completion_ids))) for i in pending]
            kw = {"lora_request": lora_request} if lora_request is not None else {}
            outs = llm.generate(prompts, params, use_tqdm=False, **kw)
            tools, again = [], []
            for i, o in zip(pending, outs):
                st, c = states[i], o.outputs[0]
                ev = CONTINUE
                for k, tid in enumerate(c.token_ids):
                    ev = st.add_token(int(tid), _token_logprob(c.logprobs, k, int(tid)))
                    if ev != CONTINUE:
                        break
                if ev == TOOL:
                    tools.append(i)
                elif ev == CONTINUE:
                    if not c.token_ids:          # engine produced nothing: end the rollout
                        st._finish("empty", True)
                    else:                        # ended without a terminal event: keep going
                        again.append(i)
            list(pool.map(lambda i: states[i].run_tool(sandbox), tools))
            pending = sorted(again + [i for i in tools if not states[i].done])
    finally:
        pool.shutdown(wait=True)
    return [st.result() for st in states]
