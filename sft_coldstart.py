"""SFT cold start: programmatic protocol traces + LoRA fine-tuning.

Generates templated Countdown traces from the reference solver (no human-written
CoT anywhere): a wrong attempt -> real sandbox ``False`` -> correction -> real
sandbox ``True`` -> correct ``<answer>``. The injected ``<output>`` segments are
masked (label -100) in SFT too, reusing ``rollout.assemble`` so the token
sequence is built segment-wise and never re-tokenized.

Uses train-split problems only (never the eval split), to avoid leakage.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import torch

import data_generator
from rollout import assemble, format_prompt
from sandbox import Sandbox


def make_wrong_attempt(nums: list[int], target: int, rng: random.Random, max_attempts: int = 200):
    """A wrong (but integer-intermediate) expression over the inputs, != target."""
    for _ in range(max_attempts):
        res = data_generator.build_problem(list(nums), rng, data_generator.OP_WEIGHTS["easy"])
        if res is not None:
            w, expr = res
            if w != target:
                return w, expr
    a, b = nums[0], nums[1]
    if a + b != target:
        return a + b, f"({a}+{b})"
    return abs(a - b), f"({a}-{b})" if a >= b else f"({b}-{a})"


def _check_code(expr: str, target: int) -> str:
    return f"res = {expr}\nprint(res == {target}, res)"


def generate_trace(problem: dict, rng: random.Random, sandbox: Sandbox) -> list[tuple[bool, str]]:
    """Return a list of ``(is_env, text)`` segments for one trace.

    ``is_env=True`` marks the injected ``<output>`` segments (real sandbox stdout),
    which must be masked (label -100) during SFT.
    """
    nums = problem["nums"]
    target = problem["target"]
    solution = problem["solution"]

    wrong_value, wrong_expr = make_wrong_attempt(nums, target, rng)
    code1 = _check_code(wrong_expr, target)
    _, out1 = sandbox.run(code1)

    code2 = _check_code(solution, target)
    _, out2 = sandbox.run(code2)

    return [
        (False, f"<think>\nTry to reach {target}. First attempt: {wrong_expr}.\n</think>\n"),
        (False, f"<python>\n{code1}\n</python>\n"),
        (True, f"\n<output>\n{out1}\n</output>\n"),
        (False, f"<think>\n{out1}; that is wrong. The right expression is {solution} = {target}.\n</think>\n"),
        (False, f"<python>\n{code2}\n</python>\n"),
        (True, f"\n<output>\n{out2}\n</output>\n"),
        (False, f"<answer>\n{solution}\n</answer>\n"),
    ]


def build_example(tokenizer, prompt_text: str, segments: list[tuple[bool, str]], max_length: int = 768):
    """Tokenize prompt + segments into (input_ids, labels) with env tokens masked.

    Each segment is tokenized separately (never re-tokenized joined). Generated
    segments (is_env=False) are text -> ids; injected ``<output>`` segments
    (is_env=True) stay text and are tokenized + masked inside ``assemble``.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    norm_segments = [
        (is_env, text if is_env else tokenizer.encode(text, add_special_tokens=False))
        for is_env, text in segments
    ]
    ids, mask = assemble(tokenizer, prompt_ids, norm_segments)
    if len(ids) > max_length:
        ids, mask = ids[:max_length], mask[:max_length]
    labels = [i if m else -100 for i, m in zip(ids, mask)]
    return {"input_ids": ids, "labels": labels}


def make_traces(problems: list[dict], n: int, seed: int, sandbox: Sandbox):
    """Return (chosen_problems, traces) for ``n`` sampled problems."""
    rng = random.Random(seed)
    if n <= len(problems):
        chosen = rng.sample(problems, n)
    else:
        chosen = [problems[i] for i in rng.choices(range(len(problems)), k=n)]
    traces = [generate_trace(p, rng, sandbox) for p in chosen]
    return chosen, traces


def train_sft(
    model,
    tokenizer,
    examples: list[dict],
    *,
    epochs: int = 1,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    device: torch.device | str | None = None,
    output_dir: str | None = None,
    log_every: int = 10,
):
    """Minimal LoRA SFT loop (batch=1, no padding, gradient step per example)."""
    from peft import LoraConfig, get_peft_model

    device = device or next(model.parameters()).device
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    step = 0
    for epoch in range(epochs):
        for ex in examples:
            input_ids = torch.tensor([ex["input_ids"]], device=device)
            labels = torch.tensor([ex["labels"]], device=device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            losses.append(loss.item())
            if step % log_every == 0:
                print(f"step {step} loss {sum(losses[-log_every:]) / len(losses[-log_every:]):.4f}")

    if losses:
        print(f"done {step} steps | final loss {losses[-1]:.4f} | mean {sum(losses) / len(losses):.4f}")
    else:
        print(f"done {step} steps | no losses recorded")

    if output_dir:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"saved adapter -> {output_dir}")
    return model, losses


def load_model_and_tokenizer(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, attn_implementation="eager")
    model.to(device)
    model.eval()
    return model, tokenizer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Generate SFT cold-start traces and train LoRA.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--n-traces", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--output-dir", default="outputs/sft")
    ap.add_argument("--device", default="auto", help="cuda|mps|cpu|auto")
    ap.add_argument("--traces-out", default=None, help="dump traces as JSONL for inspection")
    args = ap.parse_args(argv)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    with open(args.train_file) as f:
        problems = [json.loads(line) for line in f]

    sandbox = Sandbox(n_workers=2, timeout=0.050)
    try:
        chosen, traces = make_traces(problems, args.n_traces, args.seed, sandbox)
    finally:
        sandbox.close()

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    examples = [build_example(tokenizer, format_prompt(p["nums"], p["target"]), segs, args.max_length)
                for p, segs in zip(chosen, traces)]

    if args.traces_out:
        os.makedirs(os.path.dirname(args.traces_out) or ".", exist_ok=True)
        with open(args.traces_out, "w") as f:
            for segs in traces:
                f.write(json.dumps({"text": "".join(t for _, t in segs)}) + "\n")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    train_sft(model, tokenizer, examples, epochs=args.epochs, lr=args.lr, lora_r=args.lora_r,
              device=device, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
