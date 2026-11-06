#!/usr/bin/env bash
# auto-resume P6 after transient GPU faults (max 5 restarts); stops on success or GUARD_STOP
cd ~/gemma_lora_run
for i in 1 2 3 4 5 6; do
  env PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python -u train_rl.py --config configs/pilot_qwen3_lit4.yaml --output-dir outputs/pilot_qwen3_lit4 --run-id pilot-qwen3-lit4 --device cuda --resume auto
  rc=$?
  echo "SUPERVISOR: attempt $i exit $rc at $(date)"
  [ $rc = 0 ] && exit 0
  [ -f outputs/pilot_qwen3_lit4/GUARD_STOP ] && exit 0
  sleep 30
done
exit 1
