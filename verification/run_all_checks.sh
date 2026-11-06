#!/usr/bin/env bash
# Re-run every verification check. Read-only on project source.
#
# Usage (from repo root):
#   verification/run_all_checks.sh                 # offline checks
#   NETWORK=1 verification/run_all_checks.sh       # + external dataset cross-check, CUDA pin resolve
#   PY=/path/to/python verification/run_all_checks.sh
#
# Writes logs to verification/logs/run_<timestamp>/ and prints a summary.
# Exit code = number of failed checks.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="$ROOT/verification/logs/run_$TS"
TMP="$(mktemp -d)"
mkdir -p "$OUT"
trap 'rm -rf "$TMP"' EXIT
export PYTHONPATH="$ROOT"
export MPLCONFIGDIR="$TMP/mpl"
# keep caches inside the workspace (sandboxed shells can't write ~/.cache)
export HF_HOME="${HF_HOME:-$ROOT/.hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

FAILS=0
declare -a SUMMARY

run() {  # run <name> <cmd...>
  local name="$1"; shift
  local log="$OUT/$name.txt"
  { echo "\$ $*"; "$@"; } >"$log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then SUMMARY+=("PASS  $name"); else SUMMARY+=("FAIL  $name (rc=$rc, see $log)"); FAILS=$((FAILS+1)); fi
}

# 0. environment
run env "$PY" -c "import platform,sys,torch;print(platform.machine(),sys.version.split()[0],'torch',torch.__version__,'mps',torch.backends.mps.is_available(),'cuda',torch.cuda.is_available())"

# 1. project test suite, 3x (flakiness)
for i in 1 2 3; do run "project_pytest_$i" "$PY" -m pytest -q -p no:cacheprovider tests; done

# 2. determinism: generate twice, byte-compare
gen() { "$PY" data_generator.py --verify --train-out "$1/train.jsonl" --eval-out "$1/eval.jsonl"; }
mkdir -p "$TMP/a" "$TMP/b"
run determinism bash -c "$(declare -f gen); PY='$PY'; gen '$TMP/a' && gen '$TMP/b' && cmp '$TMP/a/train.jsonl' '$TMP/b/train.jsonl' && cmp '$TMP/a/eval.jsonl' '$TMP/b/eval.jsonl' && shasum -a 256 '$TMP'/a/*.jsonl"

# 3. independent dataset audit (counts, tiers, overlap, dupes, ref solver, stored solutions)
run audit_dataset "$PY" verification/audit_dataset.py "$TMP/a/train.jsonl" "$TMP/a/eval.jsonl" --per-tier 100

# 4. verifier fuzz vs independent parser + edge cases
run verifier_fuzz "$PY" -m pytest -q -s -p no:cacheprovider verification/test_verifier_fuzz.py

# 5. hygiene: secrets in history, gitignore coverage
run secrets_history bash -c "! git log --all -p | grep -nE 'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'"
run gitignore_coverage bash -c 'rc=0; for f in data/train.jsonl outputs/x runs/x adapter_model.safetensors checkpoints/x .env; do if git check-ignore -q "$f"; then echo "ignored     $f"; else echo "NOT ignored $f"; rc=1; fi; done; exit $rc'
run no_build_artifacts_in_history bash -c "! git log --all --name-only --format= | sort -u | grep -E '^\.mpl-cache/|\.safetensors$|\.bin$|\.pt$'"

# 6. spec deliverables present (fails until PART 1 phases 2-7 exist)
run deliverables_present bash -c 'rc=0; for f in sandbox.py rollout.py rewards.py sft_coldstart.py train_rl.py eval.py HANDOFF.md REPORT.md scripts/setup_cuda.sh scripts/run_all.sh; do if [ -e "$f" ]; then echo "present $f"; else echo "MISSING $f"; rc=1; fi; done; ls configs/*.y*ml >/dev/null 2>&1 && echo "present configs/*.yaml" || { echo "MISSING configs/*.yaml"; rc=1; }; exit $rc'

# 6b. Round-2 module audits (independent; project code is never the oracle)
run sandbox_adversarial "$PY" -m pytest -q -s -p no:cacheprovider verification/test_sandbox_adversarial.py
run sandbox_stress "$PY" verification/stress_sandbox_independent.py --n 10000
run rollout_independent "$PY" -m pytest -q -p no:cacheprovider -rA verification/test_rollout_independent.py
run grpo_math "$PY" -m pytest -q -s -p no:cacheprovider -rA verification/test_grpo_math.py
run reward_hacks "$PY" -m pytest -q -s -p no:cacheprovider verification/test_reward_hacks.py
run eval_metrics "$PY" -m pytest -q -s -p no:cacheprovider verification/test_eval_metrics.py
mkdir -p data && [ -f data/train.jsonl ] || "$PY" data_generator.py --seed 0 >/dev/null
run sft_traces "$PY" verification/check_sft_traces.py --tokenizer Qwen/Qwen2.5-0.5B
run publish_tarball_excludes_env bash -c 'grep -q "git archive" scripts/publish_code.sh && ! grep -qE "tar c[a-z]*f .*\.$" scripts/publish_code.sh && git archive --format=tar --prefix=repo/ HEAD | tar t > "$0/ls.txt" && ! grep -qE "(^|/)\.env$" "$0/ls.txt" && ! grep -qv "^repo/" "$0/ls.txt" && echo "git archive: $(wc -l < "$0/ls.txt") entries, all under repo/, no .env"' "$TMP"
run launch_applies_preset bash -c 'grep -q -- "--config" scripts/launch_anywhere.sh'
# model-dependent (needs cached weights; gemma needs HF_TOKEN): RUN_MODEL_CHECKS=1
if [ "${RUN_MODEL_CHECKS:-0}" = "1" ]; then
  run logprob_consistency_qwen "$PY" verification/check_logprob_consistency.py --model Qwen/Qwen2.5-0.5B
  run logprob_consistency_gemma "$PY" verification/check_logprob_consistency.py --model google/gemma-2-2b --attn eager
fi

# 7. shell scripts lint (only if present)
if ls scripts/*.sh >/dev/null 2>&1; then
  run bash_n bash -c 'for f in scripts/*.sh; do bash -n "$f" || exit 1; done'
  command -v shellcheck >/dev/null && run shellcheck shellcheck scripts/*.sh
fi

# 8. network checks (opt-in)
if [ "${NETWORK:-0}" = "1" ]; then
  run crosscheck_external "$PY" verification/crosscheck_external.py
  run tokenizer_tags "$PY" verification/check_tokenizer_tags.py   # informational; needs HF_TOKEN for gemma
  if command -v uv >/dev/null; then
    run cuda_requirements_resolve uv pip compile requirements-cuda.txt --python-platform x86_64-manylinux_2_28 --python-version 3.14 -o "$TMP/cuda_resolved.txt"
  fi
fi

echo "== verification summary ($OUT)" | tee "$OUT/SUMMARY.txt"
printf '%s\n' "${SUMMARY[@]}" | tee -a "$OUT/SUMMARY.txt"
echo "failed checks: $FAILS" | tee -a "$OUT/SUMMARY.txt"
exit "$FAILS"
