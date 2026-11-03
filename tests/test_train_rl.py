import torch

import train_rl
from rollout import RolloutResult


def test_group_advantage_normalizes():
    a, zv = train_rl.group_advantage([1.0, 1.1, 0.9, 1.0])
    assert not zv
    assert abs(a.sum().item()) < 1e-5  # zero-mean
    assert a[1] > a[2]  # higher reward -> higher advantage


def test_group_advantage_zero_variance():
    a, zv = train_rl.group_advantage([1.0, 1.0, 1.0, 1.0])
    assert zv and bool((a == 0).all())


def test_chunked_logprobs_matches_direct():
    torch.manual_seed(0)
    B, T, V = 2, 6, 128
    logits = torch.randn(B, T, V)
    target = torch.randint(0, V, (B, T))
    got = train_rl.chunked_logprobs(logits, target, chunk_size=13)
    direct = torch.log_softmax(logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, direct, atol=1e-5)


def test_collate_rollouts_alignment():
    r1 = RolloutResult(prompt_ids=[10, 11], completion_ids=[20, 21, 22],
                       loss_mask=[1, 0, 1], old_logprobs=[-0.1, -0.2, -0.3], transcript="")
    r2 = RolloutResult(prompt_ids=[10], completion_ids=[30, 31],
                       loss_mask=[1, 1], old_logprobs=[-0.4, -0.5], transcript="")
    b = train_rl.collate_rollouts([r1, r2], pad_token_id=0, device="cpu")

    assert b["input_ids"].shape == (2, 5)
    # shifted targets
    assert torch.equal(b["target_ids"][:, :-1], b["input_ids"][:, 1:])
    # r1: gen_mask=[0,0,1,0,1]; pred_mask[p]=gen_mask[p+1] => [0,1,0,1,0]
    assert b["pred_mask"][0].tolist() == [0, 1, 0, 1, 0]
    # old logp aligned: old_full=[0,0,-0.1,-0.2,-0.3]; pred[p]=old[p+1] => [0,-0.1,-0.2,-0.3,0]
    assert abs(b["old_logp"][0][1].item() - (-0.1)) < 1e-6
    assert abs(b["old_logp"][0][3].item() - (-0.3)) < 1e-6


def test_curriculum_state_roundtrip():
    c = train_rl.Curriculum()
    c.update("easy", True)
    c.update("easy", False)
    c.update("hard", True)
    assert c.pass_rate("easy") == 0.5
    c2 = train_rl.Curriculum()
    c2.load_state(c.state())
    assert c2.pass_rate("easy") == 0.5 and c2.pass_rate("hard") == 1.0


def test_logprobs_from_hidden_matches_direct():
    torch.manual_seed(0)
    B, T, D, V = 2, 4, 16, 64
    hidden = torch.randn(B, T, D)
    lm_head = torch.nn.Linear(D, V)
    target = torch.randint(0, V, (B, T))
    got = train_rl._logprobs_from_hidden(hidden, lm_head, target, vocab_chunk=13)
    direct_logits = torch.nn.functional.linear(hidden, lm_head.weight, lm_head.bias)
    direct = torch.log_softmax(direct_logits, dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, direct, atol=1e-5)


def test_config_hash_deterministic():
    c1 = train_rl.GRPOConfig()
    c2 = train_rl.GRPOConfig()
    assert c1.hash() == c2.hash()
    c2.G = 8
    assert c1.hash() != c2.hash()
