"""SPEC deliverable A under its spec name. The implementation is countdown_rl.data.generator (also `countdown-data`).

    uv run python data_generator.py --n-train 10000 --n-eval 1000 --seed 0 --verify
"""
from countdown_rl.data.generator import main

if __name__ == "__main__":
    main()
