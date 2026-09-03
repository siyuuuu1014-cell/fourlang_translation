from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PROJECT_ROOT / "configs" / "runtime_profiles.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Stage:
    stage_id: str
    name: str
    runtime: str
    command: tuple[str, ...]
    requires: tuple[str, ...]
    produces: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Stage":
        stage_id = str(raw["id"]).strip()
        command = tuple(str(part) for part in raw.get("command", []))
        if not stage_id or not command:
            raise ValueError("Every pipeline stage needs a non-empty id and command.")
        return cls(
            stage_id=stage_id,
            name=str(raw.get("name", stage_id)),
            runtime=str(raw.get("runtime", "student")),
            command=command,
            requires=tuple(str(item) for item in raw.get("requires", [])),
            produces=tuple(str(item) for item in raw.get("produces", [])),
        )


class DirectionPipeline:
    def __init__(
        self,
        manifest_path: Path,
        *,
        profile_name: str,
        profiles_path: Path = DEFAULT_PROFILES,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.runtime_config_path = profiles_path.resolve()
        self.manifest = _read_toml(self.manifest_path)
        self.direction = str(self.manifest["pipeline"]["direction"])
        raw_stages = self.manifest.get("stages", [])
        self.stages = [Stage.from_dict(item) for item in raw_stages]
        if not self.stages:
            raise ValueError(f"Pipeline has no stages: {self.manifest_path}")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("Pipeline stage ids must be unique.")

        profiles = _read_toml(profiles_path).get("profiles", {})
        if profile_name not in profiles:
            raise KeyError(f"Unknown runtime profile: {profile_name}")
        self.profile_name = profile_name
        self.profile = profiles[profile_name]
        self.state_path = (
            PROJECT_ROOT / ".fourlang" / "pipeline_state" / f"{self.direction}.json"
        )

    def _python_for(self, runtime: str) -> str:
        key = f"{runtime}_python"
        value = self.profile.get(key)
        if value is None:
            raise KeyError(
                f"Runtime profile {self.profile_name!r} does not define {key!r}."
            )
        raw = str(value)
        # Keep POSIX server paths intact when a server profile is inspected
        # from Windows. Path.resolve() would otherwise turn /root into D:\root.
        if raw.startswith("/"):
            return raw
        return str(_resolve(PROJECT_ROOT, raw))

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        for key, value in self.profile.get("env", {}).items():
            path_value = str(value)
            if not Path(path_value).is_absolute():
                path_value = str((PROJECT_ROOT / path_value).resolve())
            env[str(key)] = path_value
        env["FOURLANG_PROJECT_ROOT"] = str(PROJECT_ROOT)
        env["FOURLANG_PIPELINE_DIRECTION"] = self.direction
        return env

    def _render_command(self, stage: Stage) -> list[str]:
        replacements = {
            "{project_root}": str(PROJECT_ROOT),
            "{direction}": self.direction,
            "{manifest}": str(self.manifest_path),
        }
        rendered: list[str] = []
        for part in stage.command:
            for token, value in replacements.items():
                part = part.replace(token, value)
            rendered.append(part)
        first = Path(rendered[0])
        if first.suffix.lower() == ".py":
            script = _resolve(PROJECT_ROOT, rendered[0])
            return [self._python_for(stage.runtime), str(script), *rendered[1:]]
        return rendered

    def _fingerprint(self, stage: Stage) -> str:
        def signature(path: Path) -> dict[str, Any]:
            if not path.exists():
                return {"path": str(path), "missing": True}
            stat = path.stat()
            item: dict[str, Any] = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if path.is_file() and stat.st_size <= 1_000_000:
                item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            return item

        command = self._render_command(stage)
        code_files = sorted((PROJECT_ROOT / "scripts" / "pipeline_v2").glob("*.py"))
        code_files.extend(
            sorted((PROJECT_ROOT / "scripts" / "pipeline_v3").glob("*.py"))
        )
        command_config_files = []
        for value in command[1:]:
            candidate = _resolve(PROJECT_ROOT, value)
            if candidate.suffix.lower() == ".toml" and candidate.is_file():
                command_config_files.append(candidate)
        code_files.extend(
            [
                PROJECT_ROOT / "scripts" / "pipeline" / "run_direction.py",
                self.manifest_path,
                self.runtime_config_path,
                *command_config_files,
            ]
        )
        payload = {
            "profile": self.profile_name,
            "runtime": stage.runtime,
            "command": command,
            "requires": [
                signature(_resolve(PROJECT_ROOT, value)) for value in stage.requires
            ],
            "produces": list(stage.produces),
            "pipeline_code": [signature(path) for path in code_files],
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "direction": self.direction, "stages": {}}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _missing(self, paths: tuple[str, ...]) -> list[Path]:
        invalid = []
        for value in paths:
            path = _resolve(PROJECT_ROOT, value)
            if not path.exists() or (path.is_file() and path.stat().st_size == 0):
                invalid.append(path)
                continue
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict) and "artifacts" in payload:
                        artifacts = payload["artifacts"]
                        if not isinstance(artifacts, list) or any(
                            not _resolve(PROJECT_ROOT, str(item)).is_file()
                            for item in artifacts
                        ):
                            invalid.append(path)
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    invalid.append(path)
        return invalid

    def select(
        self, start: str | None, end: str | None, only: str | None
    ) -> list[Stage]:
        if only:
            selected = [stage for stage in self.stages if stage.stage_id == only]
            if not selected:
                raise KeyError(f"Unknown stage: {only}")
            return selected
        ids = [stage.stage_id for stage in self.stages]
        start_index = ids.index(start) if start else 0
        end_index = ids.index(end) + 1 if end else len(ids)
        if start_index >= end_index:
            raise ValueError("--from must appear before or equal to --until.")
        return self.stages[start_index:end_index]

    def describe(self) -> None:
        print(f"Pipeline: {self.direction}")
        print(f"Profile:  {self.profile_name}")
        print(f"Manifest: {self.manifest_path}")
        for index, stage in enumerate(self.stages, start=1):
            print(f"  {index:02d}. {stage.stage_id:<22} {stage.name} [{stage.runtime}]")

    def run(
        self,
        stages: list[Stage],
        *,
        dry_run: bool,
        force: bool,
    ) -> None:
        state = self._load_state()
        stage_state = state.setdefault("stages", {})
        env = self._environment()
        if not dry_run:
            for runtime in sorted({stage.runtime for stage in stages}):
                executable = Path(self._python_for(runtime))
                if not executable.is_file():
                    raise FileNotFoundError(
                        f"Runtime {runtime!r} Python does not exist: {executable}"
                    )

        for stage in stages:
            command = self._render_command(stage)
            fingerprint = self._fingerprint(stage)
            previous = stage_state.get(stage.stage_id, {})
            outputs_missing = self._missing(stage.produces)
            complete = (
                previous.get("status") == "completed"
                and previous.get("fingerprint") == fingerprint
                and not outputs_missing
            )
            if complete and not force:
                print(f"SKIP {stage.stage_id}: completed and outputs exist")
                continue

            missing_inputs = self._missing(stage.requires)
            if missing_inputs:
                joined = "\n  ".join(str(path) for path in missing_inputs)
                if dry_run:
                    print(f"\nWAIT {stage.stage_id}: missing inputs\n  {joined}")
                    print("     " + subprocess.list2cmdline(command))
                    continue
                raise FileNotFoundError(
                    f"Stage {stage.stage_id!r} is missing required inputs:\n  {joined}"
                )

            print(f"\nRUN  {stage.stage_id}: {stage.name}")
            print("     " + subprocess.list2cmdline(command))
            if dry_run:
                continue

            stage_state[stage.stage_id] = {
                "status": "running",
                "started_at_utc": _utc_now(),
                "fingerprint": fingerprint,
                "command": command,
            }
            self._write_state(state)
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
            if completed.returncode != 0:
                stage_state[stage.stage_id].update(
                    {"status": "failed", "finished_at_utc": _utc_now()}
                )
                self._write_state(state)
                raise SystemExit(completed.returncode)

            outputs_missing = self._missing(stage.produces)
            if outputs_missing:
                joined = "\n  ".join(str(path) for path in outputs_missing)
                stage_state[stage.stage_id].update(
                    {"status": "failed", "finished_at_utc": _utc_now()}
                )
                self._write_state(state)
                raise RuntimeError(
                    f"Stage {stage.stage_id!r} exited successfully but did not create:\n  {joined}"
                )

            stage_state[stage.stage_id].update(
                {"status": "completed", "finished_at_utc": _utc_now()}
            )
            self._write_state(state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one reproducible FourLang language-direction pipeline."
    )
    parser.add_argument("direction", help="Pipeline name, for example en_ru")
    parser.add_argument("--profile", choices=("local", "server"), default="local")
    parser.add_argument(
        "--manifest", help="Override configs/pipelines/<direction>.toml"
    )
    parser.add_argument(
        "--list", action="store_true", help="Show stages without running"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from", dest="start")
    parser.add_argument("--until", dest="end")
    parser.add_argument("--only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else PROJECT_ROOT / "configs" / "pipelines" / f"{args.direction}.toml"
    )
    pipeline = DirectionPipeline(manifest_path, profile_name=args.profile)
    pipeline.describe()
    if args.list:
        return
    selected = pipeline.select(args.start, args.end, args.only)
    pipeline.run(selected, dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
