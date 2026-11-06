"""Custom rollout engine: generation with tool interleaving + env-token masking.

A hand-written autoregressive loop (TRL's GRPOTrainer cannot pause generation
for tool execution). Invariant: **the model is fed exactly the token ids we later
train on** (prompt + completion, in order), and concatenated text is never
re-tokenized.

Key rules
---------
- Stop strings ``</python>`` and ``</answer>`` are detected on the *decoded*
  text generated since the last injection (never on injected text, never on a
  fixed token-ID suffix: edge tokens merge with neighbours, e.g. Qwen ``>\n``,
  ``)</``, ``><``). A merged trailing token (e.g. ``>\n``) is kept as one token.
- On ``</python>`` the sampled stop token and the injected
  ``\n<output>\n{stdout}\n</output>\n`` tokens are fed to the model together,
  so the KV cache matches the training sequence.
- Injected tokens (and the prompt) get loss_mask 0 (label -100). Model-generated
  tokens get loss_mask 1.
- Max ``max_rounds`` code executions. A further ``</python>`` either ends the
  rollout (``on_round_cap="truncate"``, training default) or gets an
  ``ExecutionLimit`` notice instead of running (``"notice"``, used by eval's
  rounds sweep so the model can still answer).
- A model-generated ``<output>`` (hallucinated environment output) ends the
  rollout immediately: it is flagged (``hallucinated_output``; rewards.py zeroes
  exec + format) and its tokens are KEPT with loss_mask 1, so the negative
  advantage is applied to exactly the tokens that hallucinated. Nothing is
  rolled back, so generation always terminates (also under greedy decoding).

``run_rollout`` is model-agnostic: it calls ``model(input_ids=..., past_key_values=..., use_cache=True)``
and expects ``(.logits, .past_key_values)``. Recorded log-probs are under the
sampling distribution ``softmax(logits / T)``; train_rl.py computes training
log-probs at the same temperature.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

PYTHON_OPEN = "<python>"
PYTHON_CLOSE = "</python>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
OUTPUT_OPEN = "<output>"
OUTPUT_CLOSE = "</output>"
ROUND_CAP_NOTICE = "ExecutionLimit: no more code executions are allowed; give your final answer."


@dataclass
class RolloutResult:
    prompt_ids: list[int]
    completion_ids: list[int]           # model-generated + injected, interleaved
    loss_mask: list[int]                # 1 = model-generated (trainable), 0 = env
    old_logprobs: list[float]           # per completion token; 0.0 for env tokens
    transcript: str                     # decoded prompt + completion
    completion_text: str = ""           # decoded completion: generated + injected text, in order
    codes: list[str] = field(default_factory=list)   # executed code per round
    code_oks: list[bool] = field(default_factory=list)  # whether each ran cleanly
    outputs: list[str] = field(default_factory=list)    # real (post-transform) stdout injected per round
    generated_text: str = ""            # decoded model-generated tokens only
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


RETRY_HINT = "If the check prints False, try a different expression and check again before answering."
MULTI_HINT = ("Check several candidate expressions in one <python> block; if none prints True, "
              "check new candidates before answering.")
WIDE_HINT = ("Check about 8 candidate expressions in one <python> block; if none prints True, use the printed "
             "values closest to the target to pick 8 new candidates and check them before answering.")


def format_prompt(nums: list[int], target: int, retry_hint: bool = False, multi: bool = False,
                  wide: bool = False, solution_hint: str = "") -> str:
    """Problem prompt. ``retry_hint`` adds one instruction sentence (no demonstration);
    ``multi`` uses MULTI_HINT instead (several candidates per check); ``wide`` uses WIDE_HINT
    (~8 candidates per check, refine from the closest printed values). ``solution_hint`` (training only,
    P10): a sub-expression of a known solution, for problems the policy cannot solve yet."""
    nums_s = ", ".join(str(n) for n in nums)
    if retry_hint and wide:
        hint = f" {WIDE_HINT}"
    else:
        hint = f" {MULTI_HINT}" if (retry_hint and multi) else (f" {RETRY_HINT}" if retry_hint else "")
    return (
        f"Using the numbers {nums_s}, make {target} with +, -, *, / "
        f"(each number at most once).{hint}{solution_hint and f' Hint: one solution contains {solution_hint}.'}"
        f" Reply in exactly this format:\n"
        f"<think>\n...\n</think>\n<python>\n...\n</python>\n<answer>\n...\n</answer>\n"
    )


# ---------------------------------------------------------------------------
# Tokenization of protocol text
# ---------------------------------------------------------------------------
def encode_text(tokenizer, text: str, add_special_tokens: bool = False) -> list[int]:
    """Encode prompt / trace / injected text so protocol tags stay ORDINARY tokens.

    Some tokenizers register tag strings as single added tokens (Qwen3: ``<think>``
    and ``</think>`` are reserved ids that its base model never learned to emit).
    Any added-token string found in ``text`` is encoded as ``s[0]`` + the rest, so
    it can never collapse into the reserved id. For tokenizers without such added
    tokens (gemma-2, Qwen2.5) this equals ``tokenizer.encode``. Tokenization is
    segment-wise by design (we never re-tokenize joined text).
    """
    added = [t for t in getattr(tokenizer, "get_added_vocab", dict)() if len(t) > 1 and t in text]
    if not added:
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)
    ids = tokenizer.encode("", add_special_tokens=True) if add_special_tokens else []
    i = 0
    while i < len(text):
        hits = [(text.find(t, i), t) for t in added if text.find(t, i) != -1]
        if not hits:
            ids += tokenizer.encode(text[i:], add_special_tokens=False)
            break
        j, tag = min(hits, key=lambda h: (h[0], -len(h[1])))
        if j > i:
            ids += tokenizer.encode(text[i:j], add_special_tokens=False)
        ids += tokenizer.encode(tag[0], add_special_tokens=False) + encode_text(tokenizer, tag[1:])
        i = j + len(tag)
    return ids


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
            seg_ids = encode_text(tokenizer, payload)
            ids.extend(seg_ids)
            mask.extend([0] * len(seg_ids))
        else:
            ids.extend(payload)
            mask.extend([1] * len(payload))
    return ids, mask


# ---------------------------------------------------------------------------
# Per-sequence state machine (shared by the sequential and batched loops, so
# both have exactly the same stop / tool / hallucination / budget semantics)
# ---------------------------------------------------------------------------
CONTINUE, TOOL, DONE = "continue", "tool", "done"


class _Rollout:
    def __init__(self, tokenizer, prompt_ids: list[int], *, max_rounds: int, max_new_tokens: int,
                 on_round_cap: str, output_transform=None):
        if on_round_cap not in ("truncate", "notice"):
            raise ValueError(f"on_round_cap must be 'truncate' or 'notice', got {on_round_cap!r}")
        self.tok = tokenizer
        self.prompt_ids = list(prompt_ids)
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens
        self.on_round_cap = on_round_cap
        self.output_transform = output_transform
        self.eos_id = tokenizer.eos_token_id
        self.completion_ids: list[int] = []
        self.loss_mask: list[int] = []
        self.old_logprobs: list[float] = []
        # decoded text per completion token; an injected segment's full text sits on
        # its first token (the rest are "") so "".join(pieces) == completion text
        self.pieces: list[str] = []
        self.seg_start = 0  # completion index where the current model-generated segment starts
        self.codes: list[str] = []
        self.code_oks: list[bool] = []
        self.outputs: list[str] = []
        self.code_rounds = 0
        self.hallucinated = False
        self.truncated = False
        self.answer_text = ""
        self.has_answer = False
        self.reason = ""
        self.done = False

    @property
    def context_ids(self) -> list[int]:
        return self.prompt_ids + self.completion_ids

    def _finish(self, reason: str, truncated: bool) -> str:
        self.reason, self.truncated, self.done = reason, truncated, True
        return DONE

    def add_token(self, tid: int, logp: float) -> str:
        """Record one sampled token; return CONTINUE, TOOL (pause for execution) or DONE."""
        self.completion_ids.append(tid)
        self.loss_mask.append(1)
        self.old_logprobs.append(logp)
        self.pieces.append(self.tok.decode([tid], skip_special_tokens=False))
        seg_text = "".join(self.pieces[self.seg_start:])
        # 1) hallucinated environment output: stop, keep the tokens (penalized by rewards)
        if OUTPUT_OPEN in seg_text:
            self.hallucinated = True
            return self._finish("hallucinated_output", True)
        # 2) tool call
        if find_stop(seg_text, PYTHON_CLOSE):
            if self.code_rounds >= self.max_rounds and self.on_round_cap == "truncate":
                return self._finish("max_rounds", True)
            return TOOL
        # 3) final answer
        if find_stop(seg_text, ANSWER_CLOSE):
            self.answer_text = extract_between("".join(self.pieces), ANSWER_OPEN, ANSWER_CLOSE)
            self.has_answer = True
            return self._finish("answer", False)
        if self.eos_id is not None and tid == self.eos_id:
            return self._finish("eos", True)
        if len(self.completion_ids) >= self.max_new_tokens:
            return self._finish("max_length", True)
        return CONTINUE

    def run_tool(self, sandbox) -> list[int]:
        """Execute the pending code block (or emit the round-cap notice), append the
        injected tokens (loss_mask 0) and return their ids."""
        seg_text = "".join(self.pieces[self.seg_start:])
        if self.code_rounds >= self.max_rounds:
            stdout = ROUND_CAP_NOTICE
        else:
            code = extract_between(seg_text, PYTHON_OPEN, PYTHON_CLOSE)
            ok, stdout = sandbox.run(code)
            if self.output_transform is not None:
                stdout = self.output_transform(stdout)
            self.code_rounds += 1
            self.codes.append(code)
            self.code_oks.append(ok)
            self.outputs.append(stdout)
        injected = f"\n{OUTPUT_OPEN}\n{stdout}\n{OUTPUT_CLOSE}\n"
        inj_ids = encode_text(self.tok, injected)
        self.completion_ids.extend(inj_ids)
        self.loss_mask.extend([0] * len(inj_ids))
        self.old_logprobs.extend([0.0] * len(inj_ids))
        self.pieces.extend([injected] + [""] * (len(inj_ids) - 1))
        self.seg_start = len(self.completion_ids)
        if len(self.completion_ids) >= self.max_new_tokens:
            self._finish("max_length", True)
        return inj_ids

    def result(self) -> RolloutResult:
        return RolloutResult(
            prompt_ids=self.prompt_ids,
            completion_ids=self.completion_ids,
            loss_mask=self.loss_mask,
            old_logprobs=self.old_logprobs,
            transcript=self.tok.decode(self.prompt_ids + self.completion_ids, skip_special_tokens=False),
            completion_text="".join(self.pieces),
            codes=self.codes,
            code_oks=self.code_oks,
            outputs=self.outputs,
            generated_text="".join(p for p, m in zip(self.pieces, self.loss_mask) if m == 1),
            code_rounds=self.code_rounds,
            hallucinated_output=self.hallucinated,
            has_answer=self.has_answer,
            answer_text=self.answer_text,
            truncated=self.truncated,
            terminal_reason=self.reason,
        )


# ---------------------------------------------------------------------------
# KV-cache surgery (HF DynamicCache; test doubles implement select/crop_left/merge_rows)
# ---------------------------------------------------------------------------
def _plain_layers(past):
    layers = getattr(past, "layers", None)
    if layers is None:
        return None
    return layers if all(type(l).__name__ in ("DynamicLayer", "_PreallocLayer") for l in layers) else None


def _kv_buffer(t, need: int = 0):
    """A [B, H, cap, D] buffer holding ``t`` in its first L positions, with room to grow."""
    if t is None:
        return None
    L = t.shape[-2]
    cap = L + max(need, 64, L // 4)
    buf = t.new_empty(t.shape[:-2] + (cap,) + t.shape[-1:])
    buf[..., :L, :] = t
    return buf


_PREALLOC_CLS = {}


def _prealloc_class(base):
    """Subclass of HF ``DynamicLayer`` whose ``update`` writes new tokens into a preallocated buffer instead of
    ``torch.cat``-ing the whole cache every step. ``DynamicLayer.update`` copies the full KV cache per token per
    layer; at 128 rows x ~1.2k tokens (Qwen3-4B: 147 KB/token) that was ~30 GB of copying per decode step on
    soos, more than half the step time. ``keys``/``values`` are views of the buffers, so in-place writes
    (``dst[slots] = src``) go straight into the cache; assigning a new tensor (crop, pad, select) re-buffers it."""
    if base not in _PREALLOC_CLS:
        class _PreallocLayer(base):
            @property
            def keys(self):
                b = self.__dict__.get("_kb")
                return None if b is None else b[..., :self._kn, :]

            @keys.setter
            def keys(self, t):
                self._kb, self._kn = _kv_buffer(t), (0 if t is None else t.shape[-2])

            @property
            def values(self):
                b = self.__dict__.get("_vb")
                return None if b is None else b[..., :self._vn, :]

            @values.setter
            def values(self, t):
                self._vb, self._vn = _kv_buffer(t), (0 if t is None else t.shape[-2])

            def update(self, key_states, value_states, *args, **kwargs):
                if not self.is_initialized:
                    self.lazy_initialization(key_states, value_states)
                n = key_states.shape[-2]
                if self._kb is None or self._kn + n > self._kb.shape[-2]:
                    self._kb = _kv_buffer(self.keys, n)
                if self._vb is None or self._vn + n > self._vb.shape[-2]:
                    self._vb = _kv_buffer(self.values, n)
                self._kb[..., self._kn:self._kn + n, :] = key_states
                self._vb[..., self._vn:self._vn + n, :] = value_states
                self._kn += n
                self._vn += n
                return self.keys, self.values

        _PREALLOC_CLS[base] = _PreallocLayer
    return _PREALLOC_CLS[base]


def _cache_prealloc(past) -> None:
    """Switch a plain HF DynamicCache to preallocated in-place layers (no-op for anything else)."""
    if os.environ.get("ROLLOUT_KV_PREALLOC", "1") == "0":
        return
    layers = getattr(past, "layers", None)
    if not layers or not all(type(l).__name__ == "DynamicLayer" for l in layers):
        return
    for l in layers:
        k, v = l.__dict__.pop("keys", None), l.__dict__.pop("values", None)
        l.__class__ = _prealloc_class(type(l))
        l.keys, l.values = k, v


def _cache_select(past, idx: torch.Tensor) -> None:
    if hasattr(past, "batch_select_indices"):
        past.batch_select_indices(idx)
    else:
        past.select(idx.tolist())


def _cache_crop_left(past, c: int) -> None:
    layers = _plain_layers(past)
    if layers is None:
        past.crop_left(c)
        return
    for l in layers:
        l.keys, l.values = l.keys[:, :, c:], l.values[:, :, c:]


def _cache_pad_left(past, c: int) -> None:
    layers = _plain_layers(past)
    if layers is None:
        past.pad_left(c)
        return
    for l in layers:
        z = l.keys.new_zeros(l.keys.shape[:2] + (c,) + l.keys.shape[3:])
        l.keys, l.values = torch.cat([z, l.keys], 2), torch.cat([z.clone(), l.values], 2)


def _cache_merge_rows(past, slots: list[int], new, pad: int) -> None:
    """Write the rows of cache ``new`` (length Ln) into ``slots`` of ``past`` (length Ln + pad),
    left-padded with zeros (masked out by the attention mask)."""
    layers = _plain_layers(past)
    if layers is None:
        past.merge_rows(slots, new, pad)
        return
    st = torch.tensor(slots, device=layers[0].keys.device)
    for l, nl in zip(layers, new.layers):
        for src, name in ((nl.keys, "keys"), (nl.values, "values")):
            dst = getattr(l, name)
            if pad:
                src = torch.cat([src.new_zeros(src.shape[:2] + (pad,) + src.shape[3:]), src], 2)
            dst[st] = src


def _prefill(model, ctxs: list[list[int]], pad_id: int, device):
    n, L = len(ctxs), max(len(c) for c in ctxs)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for b, c in enumerate(ctxs):
        input_ids[b, L - len(c):] = torch.tensor(c, dtype=torch.long)
        attn[b, L - len(c):] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    input_ids, attn, pos = input_ids.to(device), attn.to(device), pos.to(device)
    with torch.no_grad():
        out = _model_call(model, input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :], attn, pos[:, -1] + 1


def _run_continuous(model, states: list[_Rollout], sandbox, pad_id: int, temperature, rng, device,
                    slots: int) -> bool:
    """Continuous-batching engine (see run_rollouts_batched). Returns False, before generating
    anything, if the model's cache does not support the needed surgery."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    waiting = deque(range(len(states)))
    S = min(slots, len(states))
    first = [waiting.popleft() for _ in range(S)]
    past, logits, attn, next_pos = _prefill(model, [states[r].context_ids for r in first], pad_id, device)
    if _plain_layers(past) is None and not hasattr(past, "merge_rows"):
        return False
    _cache_prealloc(past)
    slot_row = list(first)
    forced: list[list[int]] = [[] for _ in range(S)]
    admit_min = max(1, S // 8)
    step = 0
    pool = ThreadPoolExecutor(max_workers=8)
    try:
        while True:
            # 1) admit waiting prompts into free slots (prefilled as a mini-batch, then merged)
            free = [s for s in range(S) if slot_row[s] < 0]
            if waiting and free and (len(free) >= admit_min or len(free) == S):
                take = free[:len(waiting)]
                rows = [waiting.popleft() for _ in take]
                npast, nlogits, nattn, npos = _prefill(model, [states[r].context_ids for r in rows], pad_id, device)
                Ln, Lb = nattn.shape[1], attn.shape[1]
                if Ln > Lb:
                    _cache_pad_left(past, Ln - Lb)
                    attn = torch.cat([attn.new_zeros((S, Ln - Lb)), attn], 1)
                    Lb = Ln
                _cache_merge_rows(past, take, npast, Lb - Ln)
                ts = torch.tensor(take, device=attn.device)
                attn[ts] = torch.cat([nattn.new_zeros((len(take), Lb - Ln)), nattn], 1)
                logits[ts] = nlogits.to(logits.dtype)
                next_pos[ts] = npos
                for s, r in zip(take, rows):
                    slot_row[s], forced[s] = r, []
                del npast, nlogits
            active = [s for s in range(S) if slot_row[s] >= 0]
            if not active:
                return True
            # 2) shrink the batch once nothing is waiting and a quarter of the slots are idle
            #    (decode cost falls with batch size: ~196 ms at 128 rows vs ~117 ms at 64 on soos)
            if not waiting and len(active) * 4 <= S * 3 and S > 1:
                kt = torch.tensor(active, device=attn.device)
                _cache_select(past, kt)
                attn, logits, next_pos = attn[kt], logits[kt], next_pos[kt]
                slot_row = [slot_row[s] for s in active]
                forced = [forced[s] for s in active]
                S = len(active)
                active = list(range(S))
            # 3) choose this step's token per slot: forced (stop token / tool output) or sampled
            need = [s for s in active if not forced[s]]
            sampled = {}
            if need:
                ids, lps = _sample_batch(logits[torch.tensor(need, device=logits.device)], temperature, rng)
                sampled = {s: (i, lp) for s, i, lp in zip(need, ids, lps)}
            feed, mask = [pad_id] * S, [0] * S
            tools = []
            for s in active:
                if forced[s]:
                    feed[s], mask[s] = forced[s].pop(0), 1
                    continue
                r = slot_row[s]
                tid, lp = sampled[s]
                ev = states[r].add_token(tid, lp)
                if ev == CONTINUE:
                    feed[s], mask[s] = tid, 1
                elif ev == TOOL:
                    feed[s], mask[s] = tid, 1  # the stop token is part of the context
                    tools.append(s)
                else:
                    slot_row[s] = -1
            if tools:
                injected = list(pool.map(lambda s: states[slot_row[s]].run_tool(sandbox), tools))
                for s, inj in zip(tools, injected):
                    if states[slot_row[s]].done:   # budget exhausted by the injection
                        slot_row[s], feed[s], mask[s] = -1, pad_id, 0
                    else:
                        forced[s] = list(inj)
            for s in range(S):
                if slot_row[s] < 0:
                    attn[s] = 0   # idle slot: no real columns, so it never blocks trimming
            if not any(mask):
                continue
            # 4) one decode step for all slots
            attn = torch.cat([attn, torch.tensor(mask, dtype=torch.long, device=attn.device).unsqueeze(1)], 1)
            with torch.no_grad():
                out = _model_call(model, input_ids=torch.tensor(feed, device=attn.device).unsqueeze(1),
                                  attention_mask=attn, position_ids=next_pos.unsqueeze(1),
                                  past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            next_pos = next_pos + torch.tensor(mask, dtype=next_pos.dtype, device=next_pos.device)
            step += 1
            # 5) trim leading columns that no slot attends to
            if step % 32 == 0:
                used = attn.any(0).nonzero()
                c = int(used[0]) if used.numel() else attn.shape[1] - 1
                if c >= 64:
                    _cache_crop_left(past, c)
                    attn = attn[:, c:]
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Model calls
# ---------------------------------------------------------------------------
def _model_call(model, **kw):
    """Forward that only materializes last-position logits when the model supports it
    (full prefill logits for a 256k vocab would be (B, T, 256000))."""
    try:
        return model(**kw, logits_to_keep=1)
    except TypeError:
        return model(**kw)


def _forward(model, input_ids: torch.Tensor, past, device):
    out = _model_call(model, input_ids=input_ids.to(device), past_key_values=past, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


def _sample(logits: torch.Tensor, temperature: float, rng: torch.Generator):
    logits = logits.float()  # fp32 for numerical stability (fp16 logits can overflow softmax)
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=rng)
    else:
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
    lp_logits = logits / temperature if temperature and temperature > 0 else logits
    logp = torch.log_softmax(lp_logits, dim=-1).gather(1, nxt).item()
    return int(nxt.item()), logp


def _sample_batch(logits: torch.Tensor, temperature: float, rng: torch.Generator):
    """Row-wise sampling for a (B, V) logits batch -> (ids, log-probs under the sampling
    distribution softmax(logits / T); greedy (T=0) reports untempered log-probs)."""
    logits = logits.float()
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=rng)
    else:
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
    lp_logits = logits / temperature if temperature and temperature > 0 else logits
    logp = torch.log_softmax(lp_logits, dim=-1).gather(1, nxt).squeeze(1)
    return nxt.squeeze(1).tolist(), logp.tolist()


# ---------------------------------------------------------------------------
# Sequential loop (reference implementation; one sequence, KV cache)
# ---------------------------------------------------------------------------
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
    output_transform=None,
    on_round_cap: str = "truncate",
) -> RolloutResult:
    device = device or next(model.parameters()).device
    rng = rng or torch.Generator(device=device)
    st = _Rollout(tokenizer, encode_text(tokenizer, prompt_text, add_special_tokens=True), max_rounds=max_rounds,
                  max_new_tokens=max_new_tokens, on_round_cap=on_round_cap, output_transform=output_transform)
    with torch.no_grad():
        out = _model_call(model, input_ids=torch.tensor([st.prompt_ids], device=device), use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    while not st.done:
        next_id, logp = _sample(logits, temperature, rng)
        event = st.add_token(next_id, logp)
        if event == DONE:
            break
        feed = [next_id]
        if event == TOOL:
            # feed the sampled stop token AND the injected tokens, so the context the
            # model continues from is exactly prompt_ids + completion_ids
            feed += st.run_tool(sandbox)
            if st.done:
                break
        with torch.no_grad():
            past, logits = _forward(model, torch.tensor([feed]), past, device)
    return st.result()


# ---------------------------------------------------------------------------
# Batched loop (default for training and eval)
# ---------------------------------------------------------------------------
def run_rollouts_batched(
    model,
    tokenizer,
    prompt_texts: list[str],
    sandbox,
    *,
    max_rounds: int = 2,
    max_new_tokens: int = 768,
    temperature: float = 1.0,
    rng: torch.Generator | None = None,
    device: torch.device | str | None = None,
    output_transform=None,
    on_round_cap: str = "truncate",
    batch_size: int = 0,
    impl: str = "continuous",
) -> list[RolloutResult]:
    """Generate many rollouts at once with per-sequence tool pauses.

    ``impl`` selects the batching engine (all share the ``_Rollout`` state machine, so the
    rollouts are the same up to floating-point rounding and the RNG stream):

    - ``"lockstep"``: the reference engine described below. Each round runs until its
      slowest row stops, so finished rows ride along as padding.
    - ``"compact"``: lockstep, but finished/paused rows are dropped from the batch (and the
      KV cache) as soon as a quarter of the batch is dead.
    - ``"continuous"`` (default): fixed slots, one token per slot per step. A row that calls
      the tool has its stop token and the injected output teacher-forced through the same
      decode loop (it never leaves the batch); finished rows free their slot for a waiting
      prompt (prefilled separately, then merged into the cache, left-padded); the cache is
      trimmed on the left as padding columns become unused, and the batch shrinks once
      no prompts are waiting. Requires a cache whose layers expose ``keys``/``values``
      (HF ``DynamicCache``); otherwise it falls back to ``"compact"``.

    Lockstep engine:

    Each *round* left-pads the current contexts (prompt + completion so far, as
    ids) into one batch, prefills it, then decodes all rows in lockstep. A row
    that emits ``</python>`` pauses (its stop token is part of its context); a
    row that finishes stops being fed (pad token, attention 0). When every row
    has paused or finished, the paused rows' code is executed, the real output
    is appended as env tokens, and the next round re-prefills exactly
    ``prompt_ids + completion_ids`` for them. Positions are explicit
    (``position_ids`` = index among real tokens), so a padded row sees the same
    positions and the same context as it would alone.

    ``batch_size`` caps rows per forward (0 = all); memory scales with it.
    Semantics are identical to :func:`run_rollout` (same state machine); only
    floating-point rounding and the RNG stream differ.
    """
    device = device or next(model.parameters()).device
    rng = rng or torch.Generator(device=device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    states = [_Rollout(tokenizer, encode_text(tokenizer, t, add_special_tokens=True), max_rounds=max_rounds,
                       max_new_tokens=max_new_tokens, on_round_cap=on_round_cap,
                       output_transform=output_transform) for t in prompt_texts]
    bs = batch_size if batch_size and batch_size > 0 else max(1, len(states))
    if impl not in ("lockstep", "compact", "continuous"):
        raise ValueError(f"impl must be lockstep|compact|continuous, got {impl!r}")
    if impl == "continuous":
        if _run_continuous(model, states, sandbox, pad_id, temperature, rng, device, bs):
            return [st.result() for st in states]
        impl = "compact"  # cache type without plain keys/values: nothing was generated yet
    pending = list(range(len(states)))
    while pending:
        paused: list[int] = []
        for c in range(0, len(pending), bs):
            paused += _decode_round(model, [states[i] for i in pending[c:c + bs]], pending[c:c + bs],
                                    pad_id, temperature, rng, device, compact=(impl == "compact"))
        for i in paused:
            states[i].run_tool(sandbox)
        pending = [i for i in paused if not states[i].done]
    return [st.result() for st in states]


def _decode_round(model, rows: list[_Rollout], idx: list[int], pad_id: int, temperature, rng, device,
                  compact: bool = False) -> list[int]:
    """Prefill + lockstep decode for one chunk of rows. Returns indices that paused for a tool.
    ``compact``: drop dead rows from the batch and the cache once a quarter of it is dead."""
    ctx = [r.context_ids for r in rows]
    n, L = len(ctx), max(len(c) for c in ctx)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for b, c in enumerate(ctx):  # left padding: the last column is real for every row
        input_ids[b, L - len(c):] = torch.tensor(c, dtype=torch.long)
        attn[b, L - len(c):] = 1
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    input_ids, attn, pos = input_ids.to(device), attn.to(device), pos.to(device)
    with torch.no_grad():
        out = _model_call(model, input_ids=input_ids, attention_mask=attn, position_ids=pos, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    next_pos = pos[:, -1] + 1
    live = [True] * n
    paused = []
    while True:
        ids, logps = _sample_batch(logits, temperature, rng)
        feed, feed_mask = [], []
        for b in range(n):
            if not live[b]:
                feed.append(pad_id)
                feed_mask.append(0)
                continue
            event = rows[b].add_token(ids[b], logps[b])
            if event == CONTINUE:
                feed.append(ids[b])
                feed_mask.append(1)
            else:
                live[b] = False
                if event == TOOL:
                    paused.append(idx[b])
                feed.append(pad_id)
                feed_mask.append(0)
        if not any(live):
            return paused
        if compact and live.count(False) * 4 >= n and n > 1:
            keep = [b for b in range(n) if live[b]]
            kt = torch.tensor(keep, device=device)
            _cache_select(past, kt)
            attn, next_pos = attn[kt], next_pos[kt]
            rows, idx = [rows[b] for b in keep], [idx[b] for b in keep]
            feed, feed_mask = [feed[b] for b in keep], [feed_mask[b] for b in keep]
            n, live = len(keep), [True] * len(keep)
        attn = torch.cat([attn, torch.tensor(feed_mask, dtype=torch.long, device=device).unsqueeze(1)], dim=1)
        with torch.no_grad():
            out = _model_call(model, input_ids=torch.tensor(feed, device=device).unsqueeze(1), attention_mask=attn,
                              position_ids=next_pos.unsqueeze(1), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        next_pos = next_pos + 1
