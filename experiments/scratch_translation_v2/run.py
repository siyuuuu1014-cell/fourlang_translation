"""Single entry point for the isolated v2 experiment."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments.scratch_translation_v2.cli import main

    main()
