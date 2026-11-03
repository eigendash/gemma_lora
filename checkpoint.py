"""Preemption-safe checkpointing over an object store.

The training host is untrusted and can be preempted with little warning, so the
protocol is:

- Local-first: the host writes a checkpoint to local disk, then uploads it.
- Bucket layout (immutable): ``runs/<run_id>/ckpt/step_<N>/...`` with a ``COMMIT``
  object written *last* containing the file list + sha256s + config/dataset hash.
  "Latest" = the highest step with a valid COMMIT. The host never overwrites or
  deletes objects; pruning is a lifecycle rule (not implemented here).
- Download verifies sha256s; a corrupt/incomplete newest checkpoint falls back to
  the previous one.
- Heartbeats at ``runs/<run_id>/hb/<unix_ts>.json`` and a segment history at
  ``runs/<run_id>/history/segment_<n>.json``.

Backends: ``local`` (directory, used as the fake bucket in tests) is implemented;
``gcs``/``s3``/``hf-hub`` are untested interfaces (Phase 8). Credentials come
from environment variables only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod


class Bucket(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...


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

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not os.path.isdir(base):
            return []
        out = []
        for dirpath, _, files in os.walk(base):
            for fn in files:
                full = os.path.join(dirpath, fn)
                out.append(os.path.relpath(full, self.root))
        return out


class _UnsupportedBucket(Bucket):
    def __init__(self, name):
        self.name = name

    def put(self, *a, **k):
        raise NotImplementedError(f"{self.name} bucket is an untested interface (Phase 8)")

    def get(self, *a, **k):
        raise NotImplementedError(f"{self.name} bucket is an untested interface (Phase 8)")

    def list(self, *a, **k):
        raise NotImplementedError(f"{self.name} bucket is an untested interface (Phase 8)")


def get_bucket(name: str, **kwargs) -> Bucket:
    if name == "local":
        return LocalBucket(kwargs["root"])
    return _UnsupportedBucket(name)  # gcs / s3 / hf-hub implemented in Phase 8


# ---------------------------------------------------------------------------
# Checkpoint protocol
# ---------------------------------------------------------------------------
def _ckpt_prefix(run_id: str, step: int) -> str:
    return f"runs/{run_id}/ckpt/step_{step}/"


def commit_checkpoint(bucket: Bucket, run_id: str, step: int, files: dict[str, bytes],
                      config_hash: str, dataset_hash: str) -> None:
    """Upload a checkpoint then write the COMMIT object last."""
    prefix = _ckpt_prefix(run_id, step)
    commit = {"step": step, "config_hash": config_hash, "dataset_hash": dataset_hash, "files": {}}
    for name, data in sorted(files.items()):
        bucket.put(prefix + name, data)
        commit["files"][name] = hashlib.sha256(data).hexdigest()
    bucket.put(prefix + "COMMIT.json", json.dumps(commit, sort_keys=True).encode())


def _list_steps(bucket: Bucket, run_id: str) -> list[int]:
    prefix = f"runs/{run_id}/ckpt/"
    steps = set()
    for key in bucket.list(prefix):
        # key like runs/<id>/ckpt/step_<N>/COMMIT.json
        parts = key.split("/")
        for p in parts:
            if p.startswith("step_"):
                try:
                    steps.add(int(p[5:]))
                except ValueError:
                    pass
    return sorted(steps)


def verify_checkpoint(bucket: Bucket, run_id: str, step: int,
                      config_hash: str, dataset_hash: str) -> bool:
    prefix = _ckpt_prefix(run_id, step)
    try:
        commit = json.loads(bucket.get(prefix + "COMMIT.json").decode())
    except Exception:
        return False
    if commit.get("config_hash") != config_hash or commit.get("dataset_hash") != dataset_hash:
        return False
    for name, sha in commit["files"].items():
        try:
            data = bucket.get(prefix + name)
        except Exception:
            return False
        if hashlib.sha256(data).hexdigest() != sha:
            return False
    return True


def find_latest_valid(bucket: Bucket, run_id: str, config_hash: str, dataset_hash: str) -> int | None:
    """Highest step with a valid COMMIT; a corrupt newest falls back to the previous."""
    for step in reversed(_list_steps(bucket, run_id)):
        if verify_checkpoint(bucket, run_id, step, config_hash, dataset_hash):
            return step
    return None


def download_checkpoint(bucket: Bucket, run_id: str, step: int, dest_dir: str,
                        config_hash: str, dataset_hash: str) -> str | None:
    """Download a verified checkpoint to ``dest_dir`` (adapter/ subdir). Returns path or None."""
    if not verify_checkpoint(bucket, run_id, step, config_hash, dataset_hash):
        return None
    prefix = _ckpt_prefix(run_id, step)
    commit = json.loads(bucket.get(prefix + "COMMIT.json").decode())
    os.makedirs(dest_dir, exist_ok=True)
    for name in commit["files"]:
        p = os.path.join(dest_dir, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f:
            f.write(bucket.get(prefix + name))
    return dest_dir


# ---------------------------------------------------------------------------
# Heartbeat + segment history
# ---------------------------------------------------------------------------
def write_heartbeat(bucket: Bucket, run_id: str, data: dict) -> None:
    bucket.put(f"runs/{run_id}/hb/{int(time.time())}.json", json.dumps(data).encode())


def write_segment(bucket: Bucket, run_id: str, n: int, data: dict) -> None:
    bucket.put(f"runs/{run_id}/history/segment_{n}.json", json.dumps(data, sort_keys=True).encode())


def commit_checkpoint_dual(primary: Bucket, secondary: Bucket | None, *args, **kwargs) -> None:
    commit_checkpoint(primary, *args, **kwargs)
    if secondary is not None:
        try:
            commit_checkpoint(secondary, *args, **kwargs)
        except Exception:
            pass  # primary already committed; secondary is best-effort
