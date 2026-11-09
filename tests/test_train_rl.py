import pytest
import torch

from countdown_rl.rl import grpo
from countdown_rl.rl.rollout import RolloutResult


def test_group_advantage_normalizes():
    a, zv = grpo.group_advantage([1.0, 1.1, 0.9, 1.0])
    assert not zv
    assert abs(a.sum().item()) < 1e-5  # zero-mean
    assert a[1] > a[2]  # higher reward -> higher advantage


def test_group_advantage_zero_variance():
    a, zv = grpo.group_advantage([1.0, 1.0, 1.0, 1.0])
    assert zv and bool((a == 0).all())


def test_chunked_logprobs_matches_direct():
    torch.manual_seed(0)
    B, T, V = 2, 6, 128
    logits = torch.randn(B, T, V)
    target = torch.randint(0, V, (B, T))
    got = grpo.chunked_logprobs(logits, target, chunk_size=13)
    direct = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, direct, atol=1e-5)


def test_collate_rollouts_alignment():
    r1 = RolloutResult(prompt_ids=[10, 11], completion_ids=[20, 21, 22],
                       loss_mask=[1, 0, 1], old_logprobs=[-0.1, -0.2, -0.3], transcript="")
    r2 = RolloutResult(prompt_ids=[10], completion_ids=[30, 31],
                       loss_mask=[1, 1], old_logprobs=[-0.4, -0.5], transcript="")
    b = grpo.collate_rollouts([r1, r2], pad_token_id=0, device="cpu")

    assert b["input_ids"].shape == (2, 5)
    # shifted targets
    assert torch.equal(b["target_ids"][:, :-1], b["input_ids"][:, 1:])
    # r1: gen_mask=[0,0,1,0,1]; pred_mask[p]=gen_mask[p+1] => [0,1,0,1,0]
    assert b["pred_mask"][0].tolist() == [0, 1, 0, 1, 0]
    # old logp aligned: old_full=[0,0,-0.1,-0.2,-0.3]; pred[p]=old[p+1] => [0,-0.1,-0.2,-0.3,0]
    assert abs(b["old_logp"][0][1].item() - (-0.1)) < 1e-6
    assert abs(b["old_logp"][0][3].item() - (-0.3)) < 1e-6


def test_curriculum_state_roundtrip():
    c = grpo.Curriculum()
    c.update("easy", True)
    c.update("easy", False)
    c.update("hard", True)
    assert c.pass_rate("easy") == 0.5
    c2 = grpo.Curriculum()
    c2.load_state(c.state())
    assert c2.pass_rate("easy") == 0.5 and c2.pass_rate("hard") == 1.0


def test_logprobs_from_hidden_matches_direct():
    torch.manual_seed(0)
    B, T, D, V = 2, 4, 16, 64
    hidden = torch.randn(B, T, D)
    lm_head = torch.nn.Linear(D, V)
    target = torch.randint(0, V, (B, T))
    got = grpo._logprobs_from_hidden(hidden, lm_head, target, vocab_chunk=13)
    direct_logits = torch.nn.functional.linear(hidden, lm_head.weight, lm_head.bias)
    direct = torch.log_softmax(direct_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, direct, atol=1e-5)


def test_config_hash_deterministic():
    c1 = grpo.GRPOConfig()
    c2 = grpo.GRPOConfig()
    assert c1.hash() == c2.hash()
    c2.G = 8
    assert c1.hash() != c2.hash()


def test_tempered_chunked_logprobs_match_scaled_log_softmax():
    torch.manual_seed(0)
    B, T, D, V = 2, 4, 16, 300
    hidden = torch.randn(B, T, D)
    head = torch.nn.Linear(D, V, bias=False)
    target = torch.randint(0, V, (B, T))
    for temp in (0.7, 1.0, 1.3):
        ref = torch.log_softmax(head(hidden) / temp, -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        got = grpo._logprobs_from_hidden(hidden, head, target, vocab_chunk=64, temperature=temp)
        assert torch.allclose(got, ref, atol=1e-5), temp


def test_sampler_logprob_is_under_tempered_distribution():
    from countdown_rl.rl import rollout

    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    g = torch.Generator().manual_seed(0)
    ids, lps = rollout._sample_batch(logits, 0.7, g)
    assert abs(lps[0] - torch.log_softmax(logits / 0.7, -1)[0, ids[0]].item()) < 1e-6


def test_clipped_surrogate_clip_higher_is_asymmetric():
    import torch
    lp_old = torch.zeros(1, 4)
    ratio = torch.tensor([[0.5, 0.9, 1.25, 1.5]])
    adv = torch.ones(1, 4)
    surr, r = grpo.clipped_surrogate(torch.log(ratio), lp_old, adv, 0.2, 0.28)
    assert torch.allclose(surr, torch.tensor([[0.5, 0.9, 1.25, 1.28]]))  # A>0: only the upper bound binds
    surr_neg, _ = grpo.clipped_surrogate(torch.log(ratio), lp_old, -adv, 0.2, 0.28)
    assert torch.allclose(surr_neg, torch.tensor([[-0.8, -0.9, -1.25, -1.5]]))  # A<0: lower bound 0.8 binds
    sym, _ = grpo.clipped_surrogate(torch.log(ratio), lp_old, adv, 0.2)
    assert torch.allclose(sym, torch.tensor([[0.5, 0.9, 1.2, 1.2]]))


def test_tis_weights_truncated_and_detached():
    import torch
    old = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
    gen = torch.tensor([0.0, -0.1, -5.0])
    w = grpo.tis_weights(old, gen, 2.0)
    assert not w.requires_grad
    assert torch.allclose(w, torch.tensor([1.0, float(torch.exp(torch.tensor(0.1))), 2.0]))


def test_config_rejects_updates_with_grad_accum(tmp_path):
    import pytest
    cfgf = tmp_path / "c.yaml"
    cfgf.write_text("updates_per_batch: 2\ngrad_accum: 2\n")
    with pytest.raises(SystemExit):
        grpo.build_config(["--config", str(cfgf)])


def test_sync_gen_model_matches_peft_forward():
    """A plain copy with weights = base + LoRA delta must reproduce the PEFT model's logits,
    and repeated syncs must not drift (always recomputed from the pristine base)."""
    import copy

    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    base = Qwen2ForCausalLM(cfg).eval()
    gen = copy.deepcopy(base)
    model = grpo._apply_lora(base, 4, 8)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.2)  # non-zero delta
    x = torch.randint(0, 64, (2, 7))
    for _ in range(3):
        grpo.sync_gen_model(model, gen)
    with torch.no_grad():
        a = model(input_ids=x).logits
        b = gen(input_ids=x).logits
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())


def test_guard_trips_on_sustained_contradiction_only():
    cfg = grpo.GRPOConfig(guard_window=3, guard_contradicted=0.5, guard_max_length=0.4)
    ok = [{"contradicted": 0.2, "frac_max_length": 0.1}] * 5
    assert grpo.guard_tripped(ok, cfg) == ""
    spike = ok + [{"contradicted": 0.9, "frac_max_length": 0.1}]
    assert grpo.guard_tripped(spike, cfg) == ""          # one bad step is not enough
    bad = ok + [{"contradicted": 0.8, "frac_max_length": 0.1}] * 3
    assert "contradicted" in grpo.guard_tripped(bad, cfg)
    long_ = ok + [{"contradicted": 0.1, "frac_max_length": 0.6}] * 3
    assert "frac_max_length" in grpo.guard_tripped(long_, cfg)
    assert grpo.guard_tripped(bad, grpo.GRPOConfig(guard_window=0)) == ""


def test_behaviour_metrics_retry_after_false():
    import types
    R = lambda outs, reason="answer": types.SimpleNamespace(outputs=outs, terminal_reason=reason)
    m = grpo.behaviour_metrics([R(["False 432"]), R(["False 432", "True 348"]), R(["True 5"]),
                                    R([], "max_length")])
    assert m["n_first_false"] == 2 and m["retry_after_false"] == 0.5 and m["frac_max_length"] == 0.25


def test_strict_retry_metric_and_sampled_nll():
    import types
    R = lambda outs: types.SimpleNamespace(outputs=outs, terminal_reason="answer")
    # multi-candidate checks: "False 50\nTrue 15" is not a failure for the strict metric
    m = grpo.behaviour_metrics([R(["False 50\nTrue 15"]), R(["False 1\nFalse 2", "True 3"]),
                                    R(["False 1\nFalse 2"]), R(["True 9"])])
    assert m["n_first_false"] == 3 and m["n_first_allfalse"] == 2 and m["retry_after_allfalse"] == 0.5
    S = lambda lps, mask: types.SimpleNamespace(old_logprobs=lps, loss_mask=mask)
    # env tokens (mask 0, logprob 0.0) are excluded: mean of 1.0, 2.0, 3.0
    assert grpo.sampled_nll([S([-1.0, 0.0, -2.0], [1, 0, 1]), S([-3.0], [1])])["entropy_est"] == 2.0
    assert grpo.sampled_nll([])["entropy_est"] != grpo.sampled_nll([])["entropy_est"]  # nan when empty


# --- P7: signal curriculum, validation split ---------------------------------------------------
def test_curriculum_signal_mode_uses_mixed_group_rate():
    import random
    from countdown_rl.rl.grpo import Curriculum
    c = Curriculum(tiers=["easy", "hard"], G=8, mode="signal", group_window=16, floor=0.05)
    for _ in range(16):
        c.update_group("easy", 8, 8)      # mastered: never mixed
        c.update_group("hard", 3, 8)      # mixed every time
    assert c.signal_rate("easy") == 0.0 and c.signal_rate("hard") == 1.0
    assert c.weight("easy") == 0.05 and c.weight("hard") == 1.0
    picks = [c.sample_tier(random.Random(i)) for i in range(400)]
    assert picks.count("hard") > 330


def test_curriculum_formula_mode_unchanged_and_state_roundtrip():
    from countdown_rl.rl.grpo import Curriculum
    c = Curriculum(tiers=["easy"], G=8)
    for _ in range(10):
        c.update("easy", True)
        c.update("easy", False)
    assert abs(c.weight("easy") - (1 - 0.5 ** 8 * 2)) < 1e-9
    c.update_group("easy", 4, 8)
    d = Curriculum(tiers=["easy"], G=8, mode="signal")
    d.load_state(c.state())
    assert d.signal_rate("easy") == 1.0 and d.pass_rate("easy") == 0.5


def test_split_validation_is_stratified_disjoint_deterministic():
    from countdown_rl.rl.grpo import split_validation
    probs = [{"id": i, "tier": t} for i, t in enumerate(["easy", "medium", "hard"] * 50)]
    tr, va = split_validation(probs, 30)
    assert len(va) == 30 and len(tr) == 120
    assert {p["tier"] for p in va} == {"easy", "medium", "hard"}
    assert sum(p["tier"] == "hard" for p in va) == 10
    assert not ({p["id"] for p in va} & {p["id"] for p in tr})
    assert split_validation(probs, 30) == (tr, va)
    assert split_validation(probs, 0) == (probs, [])


def test_build_config_rejects_bad_val_and_curriculum():
    import pytest
    from countdown_rl.rl.grpo import build_config
    with pytest.raises(SystemExit):
        build_config(["--curriculum-mode", "nope"])
    with pytest.raises(SystemExit):
        build_config(["--val-every", "10", "--val-n", "30"])   # generation is not vllm


def test_run_validation_end_to_end_with_stub_engine(monkeypatch, tmp_path):
    """Runs the real run_validation (prompt building, reward, per-tier stats, eval_compare-style
    dump) with the vLLM call stubbed: catches import/name errors before a GPU run."""
    import json
    from countdown_rl.rl import rollout_vllm
    from countdown_rl.rl.rewards import RewardConfig
    from countdown_rl.rl.rollout import RolloutResult
    from countdown_rl.rl.grpo import GRPOConfig, run_validation

    probs = [{"nums": [4, 2, 50, 6, 8, 25], "target": 150, "tier": "hard"},
             {"nums": [3, 5], "target": 8, "tier": "easy"}]
    answers = ["50 + 25 * 4", "3 * 5"]   # first correct, second wrong

    def fake(llm, tokenizer, prompts, sandbox, **kw):
        assert len(prompts) == 2 and kw["temperature"] == 0.0 and kw["lora_request"] == "LORA"
        assert all("150" in prompts[0] for _ in [0]) and "Using the numbers 3, 5" in prompts[1]
        out = []
        for a in answers:
            t = f"<think>\nx\n</think>\n<answer>\n{a}\n</answer>"
            out.append(RolloutResult(prompt_ids=[], completion_ids=[0] * 10, loss_mask=[1] * 10,
                                     old_logprobs=[0.0] * 10, transcript=t, completion_text=t,
                                     has_answer=True, answer_text=a, terminal_reason="answer"))
        return out

    monkeypatch.setattr(rollout_vllm, "run_rollouts_vllm", fake)
    cfg = GRPOConfig(prompt_style="fewshot_direct_retry")
    dump = tmp_path / "val" / "step0.jsonl"
    res = run_validation(None, None, None, probs, cfg, "LORA", RewardConfig(), "", str(dump))
    assert res["val_correct"] == 0.5 and res["val_n"] == 2
    assert res["val_tier"] == {"easy": 0.0, "hard": 1.0}
    rows = [json.loads(line) for line in open(dump)]
    assert [r["correct"] for r in rows] == [1.0, 0.0] and {"nums", "target", "tier", "completion"} <= set(rows[0])


def test_kl_ref_init_is_frozen_warm_start_and_hash_stable(tmp_path):
    """kl_ref=init: the reference is a frozen copy of the init_lora weights. At the start it equals the
    policy (KL 0) and differs from the base; a training step moves only the policy; the reference
    context restores the trainable adapter; saving writes only the trainable adapter; and
    kl_ref=base leaves the config hash unchanged (resumable runs from before the field existed)."""
    import os

    import torch
    from peft import PeftModel
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    qc = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    base = Qwen2ForCausalLM(qc).eval()
    m0 = grpo._apply_lora(base, 4, 8)
    for n, p in m0.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.2)
    m0.save_pretrained(tmp_path / "init")
    base = m0.unload()  # pristine base again
    model = PeftModel.from_pretrained(base, str(tmp_path / "init"), is_trainable=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.load_adapter(str(tmp_path / "init"), adapter_name="ref", is_trainable=False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == n_train
    x = torch.randint(0, 64, (2, 7))
    ref_lp = lambda: model(input_ids=x).logits.log_softmax(-1)  # noqa: E731
    with torch.no_grad():
        pol0 = ref_lp()
        with grpo._reference(model, "init"):
            ref0 = ref_lp()
        with grpo._reference(model, "base"):
            base0 = ref_lp()
    assert torch.allclose(pol0, ref0, atol=1e-6)                 # KL starts at 0
    assert not torch.allclose(pol0, base0, atol=1e-3)            # and the reference is not the base
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1.0)
    lp = ref_lp()                                                # policy graph built first, as in the loop
    with torch.no_grad(), grpo._reference(model, "init"):
        r = ref_lp()
    (grpo.kl_k3(lp[..., 0], r[..., 0]).sum() - lp[..., 1].sum()).backward()
    assert all(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)
    opt.step()
    with torch.no_grad():
        pol1 = ref_lp()
        with grpo._reference(model, "init"):
            ref1 = ref_lp()
    assert torch.allclose(ref1, ref0, atol=1e-6)                 # reference frozen
    assert not torch.allclose(pol1, pol0, atol=1e-4)             # policy moved
    model.save_pretrained(tmp_path / "out", selected_adapters=["default"])
    assert not os.path.exists(tmp_path / "out" / "ref")

    c = grpo.GRPOConfig()
    h = c.hash()
    c.kl_ref = "init"
    assert c.hash() != h
    c.kl_ref = "base"
    assert c.hash() == h
    with pytest.raises(SystemExit):
        grpo.build_config(["--kl-ref", "init"])              # needs init_lora


def test_hint_schedule_and_solution_hints():
    """P10 hinted rollouts: schedule parsing, per-step fraction, hint = proper sub-expression of the
    stored solution that evaluates consistently, and an empty hint leaves the prompt byte-identical."""
    import ast

    from countdown_rl.rl import prompting

    sched = grpo.parse_hint_schedule("0.25@40,0.5@0,0@80")
    assert sched == [(0, 0.5), (40, 0.25), (80, 0.0)]
    assert [grpo.hint_frac_at(sched, s) for s in (0, 39, 40, 79, 80, 500)] == [0.5, 0.5, 0.25, 0.25, 0.0, 0.0]
    assert grpo.parse_hint_schedule("") == [] and grpo.hint_frac_at([], 10) == 0.0
    for bad in ("0.5@10", "1.5@0"):
        with pytest.raises(ValueError):
            grpo.parse_hint_schedule(bad)
    with pytest.raises(SystemExit):
        grpo.build_config(["--hint-schedule", "0.5@3"])

    sol = "((100-3)*8)+(7*(6+5))"                       # 5 operations
    for frac, ops in ((0.25, 1), (0.5, 2), (0.9, 2), (1.0, 2)):   # largest PROPER subtree (each half has 2 ops)
        h = prompting.solution_hint(sol, frac)
        tree = ast.parse(h, mode="eval").body
        assert sum(isinstance(n, ast.BinOp) for n in ast.walk(tree)) == ops, (frac, h)
        assert ast.dump(tree) in {ast.dump(n) for n in ast.walk(ast.parse(sol, mode="eval").body)}
    assert prompting.solution_hint("(25*3)", 0.5) == ""      # nothing proper to reveal
    assert prompting.solution_hint(sol, 0.0) == ""

    plain = prompting.problem_prompt([4, 2, 9, 7], 27, "fewshot_multi_retry")
    assert prompting.problem_prompt([4, 2, 9, 7], 27, "fewshot_multi_retry", solution_hint="") == plain
    hinted = prompting.problem_prompt([4, 2, 9, 7], 27, "fewshot_multi_retry", solution_hint="2 * 7")
    assert hinted.replace(" Hint: one solution contains 2 * 7.", "") == plain


def test_hint_fields_do_not_change_existing_hashes():
    c = grpo.GRPOConfig()
    h = c.hash()
    c.hint_schedule, c.hint_max_groups = "", 16
    assert c.hash() == h
    c.hint_schedule = "0.5@0"
    assert c.hash() != h



def test_lr_warmup_schedule_and_hash_stability():
    c = grpo.GRPOConfig()
    h = c.hash()
    c.lr = 2e-5
    h2 = c.hash()
    assert [grpo.lr_at(c, s) for s in (0, 10)] == [2e-5, 2e-5]   # off: constant
    c.lr_warmup_steps = 0
    assert c.hash() == h2 and h2 != h                                   # default 0 is hash-neutral
    c.lr_warmup_steps = 5
    assert c.hash() != h2
    lrs = [grpo.lr_at(c, s) for s in range(7)]
    assert lrs == pytest.approx([2e-5 * k / 5 for k in range(1, 6)] + [2e-5, 2e-5])

def test_solution_hint_on_chain_reveals_up_to_all_but_last_step():
    from countdown_rl.rl import prompting

    sol = "(4+(9+(2*7)))"                               # chain of 3 operations
    assert [prompting.solution_hint(sol, f) for f in (0.25, 0.5, 1.0)] == ["2 * 7", "9 + 2 * 7", "9 + 2 * 7"]


# --- recipe v2 options (EXPERIMENTS.md §2.7p) --------------------------------------------------
def test_recipe_v2_options_are_hash_neutral_at_default():
    c = grpo.GRPOConfig()
    h = c.hash()
    c.train_nocorrect_groups, c.max_grad_norm, c.shuffle_minibatches = True, 0.0, False
    assert c.hash() == h
    for field, val in (("train_nocorrect_groups", False), ("max_grad_norm", 1.0), ("shuffle_minibatches", True)):
        d = grpo.GRPOConfig(**{field: val})
        assert d.hash() != h, field


def test_clip_and_norm_reports_preclip_norm_and_clips():
    p = torch.nn.Parameter(torch.zeros(4))
    q = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.SGD([p, q], lr=0.1)
    p.grad = torch.tensor([3.0, 0.0, 0.0, 0.0])
    q.grad = torch.tensor([0.0, 4.0, 0.0])
    assert grpo._clip_and_norm(opt, 0.0) == pytest.approx(5.0)      # off: measure only
    assert p.grad[0].item() == pytest.approx(3.0)
    assert grpo._clip_and_norm(opt, 1.0) == pytest.approx(5.0)      # returns the pre-clip norm
    total = torch.sqrt(p.grad.pow(2).sum() + q.grad.pow(2).sum()).item()
    assert total == pytest.approx(1.0, rel=1e-4)
