"""Preemption-safe checkpointing over an object store.

The training host is untrusted and can be preempted with little warning, so the
protocol is:

- Local-first: the trainer writes a checkpoint to local disk, then commits it
  to the bucket (``train_rl.py`` calls :func:`commit_dir` after every save).
- Bucket layout (immutable, create-only):
    ``runs/<run_id>/<kind>/step_<N>/...`` + ``COMMIT.json`` written *last*
    (file list + sha256s + config/dataset hash). ``kind`` is ``ckpt`` (GRPO) or
    ``sft`` (cold-start adapter) or ``eval`` (results). "Latest" = highest step with a valid COMMIT;
    a step that already has a COMMIT is never rewritten.
- Restore (:func:`find_latest_valid` + :func:`download_checkpoint`) verifies
  every sha256 and falls back to the previous step if the newest is corrupt.
- Heartbeats at ``hb/<run_id>/<unix_ts>.json`` (top-level ``hb/`` prefix so a
  bucket lifecycle rule can expire them without touching checkpoints) and a
  segment history at ``runs/<run_id>/history/segment_<n>.json``. GPU-hours per
  run are derived from heartbeats (:func:`accumulated_hours`).

Backends: ``local`` (directory; also the fake bucket in tests) and ``gcs``
(google-cloud-storage; **written against the documented client API but not run
against real GCS here**). Credentials come from the environment only
(``GOOGLE_APPLICATION_CREDENTIALS``).

CLI (used by scripts/launch_anywhere.sh)::

    python checkpoint.py push --kind sft --run-id R --dir outputs/sft_gemma
    python checkpoint.py pull --kind sft --run-id R --dir outputs/sft_gemma
    python checkpoint.py pull --kind ckpt --run-id R --dir outputs/rl_latest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from abc import ABC, abstractmethod


class Bucket(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...

    def exists(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except Exception:
            return False


class LocalBucket(Bucket):
    """Directory-backed bucket (also the fake bucket used in tests)."""

    def __init__(self, root: str):
        self.root = root

    def _path(self, key: str) -> str:
        p = os.path.join(self.root, key)
        if not os.path.abspath(p).startswith(os.path.abspath(self.root)):
            raise ValueError(f"key escapes root: {key}")
        return p

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return os.path.isfile(self._path(key))

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not os.path.isdir(base):
            return []
        out = []
        for dirpath, _, files in os.walk(base):
            for fn in files:
                if fn.endswith(".tmp"):
                    continue
                out.append(os.path.relpath(os.path.join(dirpath, fn), self.root))
        return out


class GCSBucket(Bucket):
    """Google Cloud Storage bucket. Create-only writes (``if_generation_match=0``),
    matching a service account with objectCreator + objectViewer only.

    UNVERIFIED against real GCS (no cloud access while building); the calls are
    the documented google-cloud-storage client API.
    """

    def __init__(self, name: str):
        from google.cloud import storage  # imported lazily: not needed on the Mac

        self._bucket = storage.Client().bucket(name.removeprefix("gs://"))

    def put(self, key: str, data: bytes) -> None:
        from google.api_core.exceptions import PreconditionFailed

        try:
            self._bucket.blob(key).upload_from_string(data, if_generation_match=0)
        except PreconditionFailed as e:
            raise FileExistsError(key) from e

    def get(self, key: str) -> bytes:
        return self._bucket.blob(key).download_as_bytes()

    def exists(self, key: str) -> bool:
        return self._bucket.blob(key).exists()

    def list(self, prefix: str) -> list[str]:
        return [b.name for b in self._bucket.client.list_blobs(self._bucket, prefix=prefix)]


def get_bucket(name: str, **kwargs) -> Bucket | None:
    """``local`` -> LocalBucket(root), ``gcs`` -> GCSBucket(GCS_BUCKET), ``none`` -> None."""
    if name == "local":
        return LocalBucket(kwargs["root"])
    if name == "gcs":
        bucket = kwargs.get("bucket") or os.environ.get("GCS_BUCKET", "")
        if not bucket:
            raise SystemExit("CKPT backend 'gcs' needs GCS_BUCKET")
        return GCSBucket(bucket)
    if name == "none":
        return None
    raise SystemExit(f"unsupported checkpoint backend {name!r} (supported: local, gcs, none)")


# ---------------------------------------------------------------------------
# Checkpoint protocol
# ---------------------------------------------------------------------------
def _ckpt_prefix(run_id: str, step: int, kind: str = "ckpt") -> str:
    return f"runs/{run_id}/{kind}/step_{step}/"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_from_dir(local_dir: str) -> dict[str, bytes]:
    out = {}
    for dirpath, _, files in os.walk(local_dir):
        for fn in files:
            full = os.path.join(dirpath, fn)
            out[os.path.relpath(full, local_dir)] = open(full, "rb").read()
    return out


def commit_checkpoint(bucket: Bucket, run_id: str, step: int, files: dict[str, bytes],
                      config_hash: str, dataset_hash: str, kind: str = "ckpt") -> bool:
    """Upload a checkpoint, then write COMMIT.json last. Returns False (no-op) if
    this step is already committed (create-only store: first commit wins)."""
    prefix = _ckpt_prefix(run_id, step, kind)
    if bucket.exists(prefix + "COMMIT.json"):
        return False
    commit = {"step": step, "config_hash": config_hash, "dataset_hash": dataset_hash, "files": {}}
    for name, data in sorted(files.items()):
        try:
            bucket.put(prefix + name, data)
        except FileExistsError:
            pass  # left over from an interrupted upload; verified via sha256 at restore
        commit["files"][name] = hashlib.sha256(data).hexdigest()
    bucket.put(prefix + "COMMIT.json", json.dumps(commit, sort_keys=True).encode())
    return True


def commit_dir(bucket: Bucket, run_id: str, step: int, local_dir: str, config_hash: str,
               dataset_hash: str, kind: str = "ckpt") -> bool:
    return commit_checkpoint(bucket, run_id, step, files_from_dir(local_dir), config_hash,
                             dataset_hash, kind)


def _list_steps(bucket: Bucket, run_id: str, kind: str = "ckpt") -> list[int]:
    steps = set()
    for key in bucket.list(f"runs/{run_id}/{kind}/"):
        for p in key.split("/"):
            if p.startswith("step_"):
                try:
                    steps.add(int(p[5:]))
                except ValueError:
                    pass
    return sorted(steps)


def verify_checkpoint(bucket: Bucket, run_id: str, step: int, config_hash: str | None,
                      dataset_hash: str | None, kind: str = "ckpt") -> bool:
    """Valid = COMMIT present, every file's sha256 matches, and (unless None) the
    config/dataset hashes match."""
    prefix = _ckpt_prefix(run_id, step, kind)
    try:
        commit = json.loads(bucket.get(prefix + "COMMIT.json").decode())
    except Exception:
        return False
    if config_hash is not None and commit.get("config_hash") != config_hash:
        return False
    if dataset_hash is not None and commit.get("dataset_hash") != dataset_hash:
        return False
    for name, sha in commit["files"].items():
        try:
            data = bucket.get(prefix + name)
        except Exception:
            return False
        if hashlib.sha256(data).hexdigest() != sha:
            return False
    return True


def find_latest_valid(bucket: Bucket, run_id: str, config_hash: str | None, dataset_hash: str | None,
                      kind: str = "ckpt") -> int | None:
    """Highest step with a valid COMMIT; a corrupt newest falls back to the previous."""
    for step in reversed(_list_steps(bucket, run_id, kind)):
        if verify_checkpoint(bucket, run_id, step, config_hash, dataset_hash, kind):
            return step
    return None


def download_checkpoint(bucket: Bucket, run_id: str, step: int, dest_dir: str,
                        config_hash: str | None, dataset_hash: str | None, kind: str = "ckpt") -> str | None:
    """Download a verified checkpoint into ``dest_dir``. Returns the path or None."""
    if not verify_checkpoint(bucket, run_id, step, config_hash, dataset_hash, kind):
        return None
    prefix = _ckpt_prefix(run_id, step, kind)
    commit = json.loads(bucket.get(prefix + "COMMIT.json").decode())
    os.makedirs(dest_dir, exist_ok=True)
    for name, sha in commit["files"].items():
        data = bucket.get(prefix + name)
        if hashlib.sha256(data).hexdigest() != sha:  # changed between verify and download
            return None
        p = os.path.join(dest_dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(data)
    return dest_dir


# ---------------------------------------------------------------------------
# Heartbeat + segment history + cost
# ---------------------------------------------------------------------------
def write_heartbeat(bucket: Bucket, run_id: str, data: dict) -> None:
    payload = dict(data, ts=int(time.time()))
    try:
        bucket.put(f"hb/{run_id}/{payload['ts']}.json", json.dumps(payload).encode())
    except FileExistsError:
        pass  # two heartbeats in the same second: the first is enough


def next_segment_index(bucket: Bucket, run_id: str) -> int:
    return len([k for k in bucket.list(f"runs/{run_id}/history/") if os.path.basename(k).startswith("segment_")])


def write_segment(bucket: Bucket, run_id: str, n: int, data: dict) -> None:
    bucket.put(f"runs/{run_id}/history/segment_{n}.json", json.dumps(data, sort_keys=True).encode())


def accumulated_hours(bucket: Bucket, run_id: str) -> tuple[float, dict[int, float]]:
    """GPU-hours per segment = the largest ``segment_elapsed_s`` heartbeat of that segment."""
    per_seg: dict[int, float] = {}
    for key in bucket.list(f"hb/{run_id}/"):
        try:
            hb = json.loads(bucket.get(key).decode())
        except Exception:
            continue
        seg = int(hb.get("segment", 0))
        per_seg[seg] = max(per_seg.get(seg, 0.0), float(hb.get("segment_elapsed_s", 0)) / 3600)
    return sum(per_seg.values()), per_seg


def commit_checkpoint_dual(primary: Bucket, secondary: Bucket | None, *args, **kwargs) -> None:
    commit_checkpoint(primary, *args, **kwargs)
    if secondary is not None:
        try:
            commit_checkpoint(secondary, *args, **kwargs)
        except Exception as e:  # primary already committed; secondary is best-effort
            print(f"warning: secondary checkpoint commit failed: {type(e).__name__}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="push/pull a verified artifact to/from the run bucket")
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--kind", choices=["sft", "ckpt", "eval"], required=True)
    ap.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    ap.add_argument("--dir", required=True)
    ap.add_argument("--step", type=int, default=0, help="push: step to record (sft: 0)")
    ap.add_argument("--backend", default=os.environ.get("CKPT_BACKEND", "gcs"))
    ap.add_argument("--root", default=os.environ.get("CKPT_ROOT", "outputs/_remote"), help="local backend root")
    ap.add_argument("--dataset", default="data/train.jsonl")
    a = ap.parse_args(argv)
    if not a.run_id:
        raise SystemExit("--run-id or RUN_ID required")
    bucket = get_bucket(a.backend, root=a.root)
    if bucket is None:
        raise SystemExit("backend 'none' has no bucket")
    ds_hash = sha256_file(a.dataset)
    if a.action == "push":
        done = commit_dir(bucket, a.run_id, a.step, a.dir, a.kind, ds_hash, kind=a.kind)
        print(f"{'committed' if done else 'already committed'}: {a.kind} step {a.step} <- {a.dir}")
        return 0
    step = find_latest_valid(bucket, a.run_id, None, ds_hash, kind=a.kind)
    if step is None:
        print(f"no valid {a.kind} checkpoint for run {a.run_id}")
        return 1
    if download_checkpoint(bucket, a.run_id, step, a.dir, None, ds_hash, kind=a.kind) is None:
        print(f"{a.kind} step {step} failed verification during download")
        return 1
    print(f"pulled {a.kind} step {step} -> {a.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
