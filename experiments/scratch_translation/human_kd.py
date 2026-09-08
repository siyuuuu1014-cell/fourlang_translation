"""Arm B: random-initialized Transformer, human parallel + approved Teacher KD."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.scratch_translation.cli import main

    main(group="human_kd")
