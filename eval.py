"""SPEC deliverable D under its spec name. The implementation is countdown_rl.evaluation.evaluate (also
`countdown-eval`); see README.md for the exact flags behind every reported number.

    uv run python eval.py --model Qwen/Qwen3-4B-Base --adapter <dir> --eval-file data/eval.jsonl \
        --prompt-style r1zero_tool --max-new-tokens 768 --max-rounds 2
"""
from countdown_rl.evaluation.evaluate import main

if __name__ == "__main__":
    main()
