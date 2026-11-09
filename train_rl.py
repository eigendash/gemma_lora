"""SPEC deliverable C under its spec name. The implementation is countdown_rl.rl.grpo (also `countdown-train`).

    uv run python train_rl.py --config configs/experiments/qwen3-4b-v2-zeroshot-s2/phase_a.yaml --output_dir runs/a
"""
from countdown_rl.rl.grpo import main

if __name__ == "__main__":
    main()
