import json

import checkpoint as ck


def _files():
    return {"adapter/weights.safetensors": b"A" * 100, "state.json": b'{"step": 1}'}


def test_commit_verify_and_find_latest(tmp_path):
    b = ck.LocalBucket(str(tmp_path))
    ck.commit_checkpoint(b, "run1", 5, _files(), config_hash="cfg", dataset_hash="ds")
    assert ck.verify_checkpoint(b, "run1", 5, "cfg", "ds")
    assert not ck.verify_checkpoint(b, "run1", 5, "wrongcfg", "ds")  # config mismatch
    assert ck.find_latest_valid(b, "run1", "cfg", "ds") == 5


def test_corrupt_checkpoint_falls_back_to_previous(tmp_path):
    b = ck.LocalBucket(str(tmp_path))
    ck.commit_checkpoint(b, "run1", 3, _files(), "cfg", "ds")
    ck.commit_checkpoint(b, "run1", 4, _files(), "cfg", "ds")
    # corrupt step 4's weights (the newest)
    key = "runs/run1/ckpt/step_4/adapter/weights.safetensors"
    b.put(key, b"X" * 100)
    assert not ck.verify_checkpoint(b, "run1", 4, "cfg", "ds")
    assert ck.find_latest_valid(b, "run1", "cfg", "ds") == 3  # fallback


def test_download_verifies_and_writes(tmp_path):
    b = ck.LocalBucket(str(tmp_path))
    ck.commit_checkpoint(b, "run1", 7, _files(), "cfg", "ds")
    dest = str(tmp_path / "out")
    out = ck.download_checkpoint(b, "run1", 7, dest, "cfg", "ds")
    assert out == dest
    with open(out + "/state.json", "rb") as f:
        assert json.loads(f.read())["step"] == 1


def test_dual_write(tmp_path):
    b1 = ck.LocalBucket(str(tmp_path / "b1"))
    b2 = ck.LocalBucket(str(tmp_path / "b2"))
    ck.commit_checkpoint_dual(b1, b2, "run1", 1, _files(), "cfg", "ds")
    assert ck.find_latest_valid(b1, "run1", "cfg", "ds") == 1
    assert ck.find_latest_valid(b2, "run1", "cfg", "ds") == 1


def test_heartbeat_and_segment(tmp_path):
    b = ck.LocalBucket(str(tmp_path))
    ck.write_heartbeat(b, "run1", {"step": 5, "tokens_per_s": 10})
    ck.write_segment(b, "run1", 0, {"start_step": 0, "gpu": "mps"})
    keys = b.list("runs/run1/hb/")
    assert any(k.endswith(".json") for k in keys)
    seg = b.get("runs/run1/history/segment_0.json")
    assert json.loads(seg)["gpu"] == "mps"
