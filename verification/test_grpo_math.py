"""Independent checks of train_rl's GRPO math (no project helper used as oracle).

Run: python -m pytest -q verification/test_grpo_math.py
"""

from __future__ import annotations

import random
import statistics

import pytest
import torch

from countdown_rl.rl import grpo
from countdown_rl.rl.rollout import RolloutResult


# --- advantages -------------------------------------------------------------
@pytest.mark.parametrize("rewards", [[1.2, 0.2, 0.2, 0.0], [0.1, 0.1, 0.1, 1.1, 0.0, 0.2, 0.2, 0.2], [0.0, 1.0]])
def test_group_advantage_matches_formula(rewards):
    a, zv = grpo.group_advantage(rewards)
    mu, sd = statistics.mean(rewards), statistics.stdev(rewards)  # unbiased, like torch.std
    exp = [(r - mu) / (sd + 1e-4) for r in rewards]
    assert not zv
    assert a.tolist() == pytest.approx(exp, abs=1e-5)
    assert abs(sum(a.tolist())) < 1e-4


def test_zero_variance_group_flagged():
    a, zv = grpo.group_advantage([0.2, 0.2, 0.2, 0.2])
    assert zv and a.abs().sum().item() == 0


# --- collate alignment ------------------------------------------------------
def _rr(prompt, comp, mask, old):
    return RolloutResult(prompt_ids=prompt, completion_ids=comp, loss_mask=mask, old_logprobs=old, transcript="")


def test_collate_alignment_independent():
    r1 = _rr([5, 6, 7], [10, 11, 90, 91, 12], [1, 1, 0, 0, 1], [-1.0, -1.1, 0.0, 0.0, -1.2])
    r2 = _rr([5], [20, 21], [1, 1], [-2.0, -2.1])
    b = grpo.collate_rollouts([r1, r2], pad_token_id=0, device="cpu")
    for i, r in enumerate([r1, r2]):
        ids = r.prompt_ids + r.completion_ids
        P = len(r.prompt_ids)
        for p in range(b["input_ids"].shape[1]):
            trainable = p + 1 < len(ids) and p + 1 >= P and r.loss_mask[p + 1 - P] == 1
            assert bool(b["pred_mask"][i, p]) == trainable, (i, p)
            if trainable:
                assert b["target_ids"][i, p].item() == ids[p + 1]
                assert b["old_logp"][i, p].item() == pytest.approx(r.old_logprobs[p + 1 - P])
        assert b["attention_mask"][i].sum().item() == len(ids)
    # right padding: real tokens are a prefix
    assert b["attention_mask"][1].tolist() == [1, 1, 1] + [0] * 5


# --- chunked log-probs ------------------------------------------------------
def test_chunked_logprobs_equals_log_softmax():
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 50_000) * 15
    tgt = torch.randint(0, 50_000, (2, 7))
    ref = torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    got = grpo.chunked_logprobs(logits, tgt, chunk_size=4096)
    assert torch.allclose(got, ref, atol=1e-4)


def test_logprobs_from_hidden_equals_full_no_softcap():
    torch.manual_seed(0)
    head = torch.nn.Linear(64, 30_000, bias=False)
    h = torch.randn(2, 5, 64)
    tgt = torch.randint(0, 30_000, (2, 5))
    ref = torch.log_softmax(head(h), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    got = grpo._logprobs_from_hidden(h, head, tgt, vocab_chunk=4096)
    assert torch.allclose(got, ref, atol=1e-4)


# --- KL estimator (grpo.kl_k3 is what the loss uses) ---------------------
def test_kl_estimator_nonnegative_and_has_gradient():
    """(a) never negative; (b) differentiated on policy samples (as in the loss), the
    k3 estimate's expected gradient equals grad KL(ref || policy) -- the standard GRPO
    regularizer -- and points toward the reference like grad KL(policy || ref).
    The previous k1 log-ratio had ~0 expected gradient."""
    torch.manual_seed(0)
    V = 6
    theta = torch.tensor([2.0, 0.5, 0.0, -1.0, 0.3, 1.0], requires_grad=True)
    lref = torch.log_softmax(torch.zeros(V), -1)
    logp = torch.log_softmax(theta, -1)
    rev_kl = (logp.exp() * (logp - lref)).sum()             # KL(policy || ref)
    fwd_kl = (lref.exp() * (lref - logp)).sum()             # KL(ref || policy)
    g_rev, = torch.autograd.grad(rev_kl, theta, retain_graph=True)
    g_fwd, = torch.autograd.grad(fwd_kl, theta, retain_graph=True)
    x = torch.multinomial(logp.exp().detach(), 400_000, replacement=True)
    k = grpo.kl_k3(logp[x], lref[x])
    g_k, = torch.autograd.grad(k.mean(), theta)
    small = [grpo.kl_k3(logp[x[i:i + 8]], lref[x[i:i + 8]]).mean().item() for i in range(0, 8000, 8)]
    cos = torch.nn.functional.cosine_similarity
    print(f"\nKL(p||ref)={rev_kl.item():.4f} |grad k3|={g_k.norm():.4f} |grad KL(ref||p)|={g_fwd.norm():.4f} "
          f"cos(k3, KL(ref||p))={cos(g_k, g_fwd, 0).item():.4f} cos(k3, KL(p||ref))={cos(g_k, g_rev, 0).item():.4f} "
          f"min 8-token est={min(small):.4f}")
    assert (k >= 0).all() and min(small) >= 0
    assert cos(g_k, g_fwd, 0) > 0.99 and abs(g_k.norm() - g_fwd.norm()) / g_fwd.norm() < 0.05
    assert cos(g_k, g_rev, 0) > 0.9


def test_clipped_surrogate_sign_and_clipping():
    old = torch.zeros(4)
    new = torch.log(torch.tensor([1.0, 1.5, 0.5, 1.5]))
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0])
    surr, ratio = grpo.clipped_surrogate(new, old, adv, 0.2)
    # positive advantage: ratio 1.5 clipped to 1.2; negative advantage: 0.5 clipped to 0.8, 1.5 kept (pessimistic)
    assert surr.tolist() == pytest.approx([1.0, 1.2, -0.8, -1.5])
    # maximizing surrogate raises log-prob of positive-advantage tokens
    lp = torch.zeros(1, requires_grad=True)
    s, _ = grpo.clipped_surrogate(lp, torch.zeros(1), torch.ones(1), 0.2)
    (-s).sum().backward()
    assert lp.grad.item() < 0  # loss gradient negative -> gradient descent increases lp


def test_softcap_chunked_matches_capped_log_softmax_and_grads():
    torch.manual_seed(0)
    head = torch.nn.Linear(32, 5000, bias=False)
    with torch.no_grad():
        head.weight.mul_(30)  # large logits so the softcap matters
    h = torch.randn(2, 5, 32, requires_grad=True)
    tgt = torch.randint(0, 5000, (2, 5))
    cap = 30.0
    full = torch.tanh(head(h) / cap) * cap
    ref = torch.log_softmax(full, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    got = grpo._logprobs_from_hidden(h, head, tgt, vocab_chunk=1024, softcap=cap)
    assert (head(h).abs().max() > 60)
    assert torch.allclose(got, ref, atol=1e-4)
    g_ref, = torch.autograd.grad(ref.sum(), h)
    g_got, = torch.autograd.grad(got.sum(), h)
    assert torch.allclose(g_got, g_ref, atol=1e-4)


def test_config_precedence_cli_over_yaml(tmp_path):
    y = tmp_path / "p.yaml"
    y.write_text("G: 8\ntotal_steps: 10\nn_problems: 10000\nchunked_vocab: true\n")
    cfg = grpo.build_config(["--config", str(y), "--total-steps", "3"])
    assert (cfg.G, cfg.total_steps, cfg.n_problems, cfg.chunked_vocab) == (8, 3, 10000, True)
    y.write_text("not_a_field: 1\n")
    with pytest.raises(SystemExit):
        grpo.build_config(["--config", str(y)])


# --- LoRA-only training -----------------------------------------------------
@pytest.mark.skipif(not __import__("os").environ.get("RUN_MODEL_TESTS"), reason="needs Qwen weights")
def test_only_lora_trainable_and_base_frozen_after_step():
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32)
    model = grpo._apply_lora(base, 16, 32)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"\ntrainable={n_tr:,} / total={n_all:,} ({100*n_tr/n_all:.2f}%) in {len(trainable)} tensors")
    w = dict(model.named_parameters())
    base_name = next(n for n in w if "q_proj.base_layer.weight" in n)
    lora_b = next(n for n in w if "q_proj.lora_B" in n)
    before_base, before_b = w[base_name].detach().clone(), w[lora_b].detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    model(input_ids=ids, labels=ids).loss.backward()
    opt.step()
    assert torch.equal(w[base_name], before_base)
    assert not torch.equal(w[lora_b], before_b)


# --- curriculum -------------------------------------------------------------
def test_curriculum_shifts_toward_easier_when_hard_unsolved():
    """Spec: curriculum 'preventing gradient starvation' / difficulty scaling.
    With easy mastered-ish and hard unsolved, a curriculum that avoids zero-variance
    groups should NOT put most samples on the tier with 0% pass rate."""
    c = grpo.Curriculum()
    for t, (n, k) in {"easy": (100, 60), "medium": (100, 10), "hard": (100, 0)}.items():
        for i in range(n):
            c.update(t, i < k)
    rng = random.Random(0)
    draws = [c.sample_tier(rng) for _ in range(20_000)]
    share = {t: draws.count(t) / len(draws) for t in ("easy", "medium", "hard")}
    print(f"\npass rates easy .6 / medium .1 / hard 0 -> sampling share {share}")
    assert share["hard"] < share["easy"], f"curriculum puts most mass on the unsolved tier: {share}"
