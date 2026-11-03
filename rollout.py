"""Custom rollout engine: generation with tool interleaving + env-token masking.

This is a hand-written autoregressive loop (TRL's GRPOTrainer cannot pause
generation for tool execution). The hard constraint from the audit (M2) is baked
in: **we carry token IDs end to end and never re-tokenize concatenated text.**

Key rules
---------
- Stop strings ``</python>`` and ``</answer>`` are detected on the *decoded*
  text (not on a fixed token-ID suffix), because edge tokens merge with
  neighbours (Qwen merges ``)</``, ``><`` and ``>\n``; gemma merges ``)</`` and
  ``><``). A merged trailing token (e.g. ``>\n``) is kept as one token; the
  injected ``<output>`` segment is tokenized standalone and appended.
- Injected tokens (and the prompt) get loss_mask 0 (label -100). Model-generated
  tokens get loss_mask 1.
- Max 2 code-execution rounds. No ``<answer>`` => truncated (zero correctness).
- A model-generated ``<output>`` is truncated, flagged, and replaced with the
  real sandbox output.

``run_rollout`` is model-agnostic: it calls ``model(input_ids=..., past_key_values=..., use_cache=True)``
and expects ``(.logits, .past_key_values)``. Log-probs of the sampled token are
gathered from the step logits (single-token gather; no full-vocab materialization
is ever needed here — full-batch chunked log-probs belong to train_rl.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

PYTHON_OPEN = "<python>"
PYTHON_CLOSE = "</python>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
OUTPUT_OPEN = "<output>"
OUTPUT_CLOSE = "</output>"


@dataclass
class RolloutResult:
    prompt_ids: list[int]
    completion_ids: list[int]           # model-generated + injected, interleaved
    loss_mask: list[int]                # 1 = model-generated (trainable), 0 = env
    old_logprobs: list[float]           # per completion token; 0.0 for env tokens
    transcript: str                     # decoded prompt + completion
    completion_text: str = ""           # decoded completion only (gen + injected)
    codes: list[str] = field(default_factory=list)   # executed code per round
    code_oks: list[bool] = field(default_factory=list)  # whether each ran cleanly
    code_rounds: int = 0
    hallucinated_output: bool = False
    has_answer: bool = False
    answer_text: str = ""
    truncated: bool = False
    terminal_reason: str = ""

    # convenience: full token ids (prompt + completion)
    @property
    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.completion_ids


def format_prompt(nums: list[int], target: int) -> str:
    nums_s = ", ".join(str(n) for n in nums)
    return (
        f"Using the numbers {nums_s}, make {target} with +, -, *, / "
        f"(each number at most once). Reply in exactly this format:\n"
        f"<think>\n...\n</think>\n<python>\n...\n</python>\n<answer>\n...\n</answer>\n"
    )


# ---------------------------------------------------------------------------
# Pure helpers (tested against both tokenizers without a model)
# ---------------------------------------------------------------------------
def find_stop(text: str, stop: str) -> bool:
    """True if ``stop`` has been completed in ``text`` (trailing whitespace from a
    merged boundary token such as ``>\\n`` is allowed)."""
    idx = text.rfind(stop)
    if idx == -1:
        return False
    tail = text[idx + len(stop):]
    return tail == "" or tail.isspace()


def extract_between(text: str, open_tag: str, close_tag: str) -> str:
    i = text.rfind(open_tag)
    j = text.rfind(close_tag)
    if i == -1 or j == -1 or j < i:
        return ""
    return text[i + len(open_tag):j].strip()


def assemble(
    tokenizer,
    prompt_ids: list[int],
    segments: list[tuple[bool, object]],
) -> tuple[list[int], list[int]]:
    """Build (ids, loss_mask) from segments.

    ``segments`` is a list of ``(is_env, payload)`` where payload is a list of
    token ids for model-generated segments (``is_env=False``) or a text string
    for injected segments (``is_env=True``, tokenized separately, never joined
    with neighbours). Prompt tokens are env (loss_mask 0).
    """
    ids = list(prompt_ids)
    mask = [0] * len(prompt_ids)
    for is_env, payload in segments:
        if is_env:
            seg_ids = tokenizer.encode(payload, add_special_tokens=False)
            ids.extend(seg_ids)
            mask.extend([0] * len(seg_ids))
        else:
            ids.extend(payload)
            mask.extend([1] * len(payload))
    return ids, mask


# ---------------------------------------------------------------------------
# Model loop
# ---------------------------------------------------------------------------
def _forward(model, input_ids: torch.Tensor, past, device):
    out = model(input_ids=input_ids.to(device), past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


def _sample(logits: torch.Tensor, temperature: float, rng: torch.Generator):
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=rng)
    else:
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
    logp = torch.log_softmax(logits, dim=-1).gather(1, nxt).item()
    return int(nxt.item()), logp


def run_rollout(
    model,
    tokenizer,
    prompt_text: str,
    sandbox,
    *,
    max_rounds: int = 2,
    max_new_tokens: int = 768,
    temperature: float = 1.0,
    rng: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> RolloutResult:
    device = device or next(model.parameters()).device
    rng = rng or torch.Generator(device=device)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    eos_id = tokenizer.eos_token_id

    completion_ids: list[int] = []
    loss_mask: list[int] = []
    old_logprobs: list[float] = []
    pieces: list[str] = []
    codes: list[str] = []
    code_oks: list[bool] = []
    code_rounds = 0
    hallucinated = False
    truncated = False
    answer_text = ""
    has_answer = False
    reason = ""

    # seed cache + first logits from the prompt
    prompt_t = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        out = model(input_ids=prompt_t, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    while len(completion_ids) < max_new_tokens:
        next_id, logp = _sample(logits, temperature, rng)
        completion_ids.append(next_id)
        loss_mask.append(1)
        old_logprobs.append(logp)
        piece = tokenizer.decode([next_id], skip_special_tokens=False)
        pieces.append(piece)
        text = "".join(pieces)

        # 1) hallucinated <output>? truncate + rebuild, then let stop-detection run
        rolled_back = False
        if OUTPUT_OPEN in text:
            hallucinated = True
            start = text.find(OUTPUT_OPEN)
            keep = _truncate_tokens(pieces, start)
            n_keep = len(keep)
            completion_ids = completion_ids[:n_keep]
            loss_mask = loss_mask[:n_keep]
            old_logprobs = old_logprobs[:n_keep]
            pieces = pieces[:n_keep]
            text = "".join(pieces)
            past, logits = _rebuild(model, prompt_ids + completion_ids, device)
            rolled_back = True

        # 2) stop detection
        if find_stop(text, PYTHON_CLOSE):
            if code_rounds >= max_rounds:
                reason, truncated = "max_rounds", True
                break
            code = extract_between(text, PYTHON_OPEN, PYTHON_CLOSE)
            ok, stdout = sandbox.run(code)
            code_rounds += 1
            codes.append(code)
            code_oks.append(ok)
            injected = f"\n{OUTPUT_OPEN}\n{stdout}\n{OUTPUT_CLOSE}\n"
            inj_ids = tokenizer.encode(injected, add_special_tokens=False)
            completion_ids.extend(inj_ids)
            loss_mask.extend([0] * len(inj_ids))
            old_logprobs.extend([0.0] * len(inj_ids))
            # advance cache + logits through the injected tokens
            with torch.no_grad():
                for tid in inj_ids:
                    past, logits = _forward(model, torch.tensor([[tid]]), past, device)
            continue

        if find_stop(text, ANSWER_CLOSE):
            answer_text = extract_between(text, ANSWER_OPEN, ANSWER_CLOSE)
            has_answer = True
            reason = "answer"
            break

        if rolled_back:
            # logits already rebuilt for the truncated prefix; re-sample next iteration
            continue

        if eos_id is not None and next_id == eos_id:
            reason, truncated = "eos", True
            break

        # normal step: advance cache through the token just generated
        with torch.no_grad():
            past, logits = _forward(model, torch.tensor([[next_id]]), past, device)
    else:
        reason, truncated = "max_length", True

    return RolloutResult(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        loss_mask=loss_mask,
        old_logprobs=old_logprobs,
        transcript=tokenizer.decode(prompt_ids + completion_ids, skip_special_tokens=False),
        completion_text="".join(pieces),
        codes=codes,
        code_oks=code_oks,
        code_rounds=code_rounds,
        hallucinated_output=hallucinated,
        has_answer=has_answer,
        answer_text=answer_text,
        truncated=truncated,
        terminal_reason=reason,
    )


def _truncate_tokens(pieces: list[str], start_char: int) -> list[str]:
    """Keep only leading tokens whose decoded text ends at or before ``start_char``."""
    keep: list[str] = []
    cum = 0
    for p in pieces:
        if cum + len(p) <= start_char:
            keep.append(p)
            cum += len(p)
        else:
            break
    return keep


def _rebuild(model, ids: list[int], device):
    """Re-forward a prefix of ids to restore (past, next_logits). Used only on
    hallucinated-output rollback (rare), so O(T) is acceptable."""
    if not ids:
        raise ValueError("empty prefix")
    input_ids = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]
