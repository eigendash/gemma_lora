"""Reading per-problem eval dumps (one JSON object per line, optionally gzip-compressed)."""
import gzip
import json


def read_dump(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]
