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
    ck.write_heartbeat(b, "run1", {"step": 5, "segment": 0, "segment_elapsed_s": 7200})
    ck.write_segment(b, "run1", ck.next_segment_index(b, "run1"), {"start_step": 0, "gpu": "mps"})
    assert any(k.endswith(".json") for k in b.list("hb/run1/"))  # top-level hb/ prefix (lifecycle)
    assert ck.next_segment_index(b, "run1") == 1
    seg = b.get("runs/run1/history/segment_0.json")
    assert json.loads(seg)["gpu"] == "mps"


def test_accumulated_hours_from_heartbeats(tmp_path):
    b = ck.LocalBucket(str(tmp_path))
    for i, (seg, el) in enumerate([(0, 1800), (0, 3600), (1, 900)]):
        b.put(f"hb/r/{1000 + i}.json", json.dumps({"segment": seg, "segment_elapsed_s": el}).encode())
    total, per = ck.accumulated_hours(b, "r")
    assert per == {0: 1.0, 1: 0.25} and abs(total - 1.25) < 1e-9


def test_commit_is_create_only(tmp_path):
    b = ck.LocalBucket(str(tmp_path))
    assert ck.commit_checkpoint(b, "r", 2, _files(), "cfg", "ds")
    assert not ck.commit_checkpoint(b, "r", 2, {"state.json": b"different"}, "cfg", "ds")
    assert ck.verify_checkpoint(b, "r", 2, "cfg", "ds")  # original commit untouched


def test_cli_push_pull_roundtrip(tmp_path, monkeypatch):
    src = tmp_path / "sft"
    (src / "sub").mkdir(parents=True)
    (src / "adapter_model.safetensors").write_bytes(b"W" * 10)
    (src / "sub" / "x.json").write_text("{}")
    data = tmp_path / "train.jsonl"
    data.write_text("{}\n")
    root = str(tmp_path / "bucket")
    common = ["--kind", "sft", "--run-id", "r", "--backend", "local", "--root", root, "--dataset", str(data)]
    assert ck.main(["push", "--dir", str(src)] + common) == 0
    dest = tmp_path / "dest"
    assert ck.main(["pull", "--dir", str(dest)] + common) == 0
    assert (dest / "adapter_model.safetensors").read_bytes() == b"W" * 10
    assert (dest / "sub" / "x.json").read_text() == "{}"
    # a different dataset must not restore this artifact
    data.write_text("changed\n")
    assert ck.main(["pull", "--dir", str(tmp_path / "d2")] + common) == 1
