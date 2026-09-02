from __future__ import annotations

import argparse
import hashlib
import io
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

try:
    from .common import PROJECT_ROOT, load_config, pair_info, project_path, write_json
except ImportError:
    from common import PROJECT_ROOT, load_config, pair_info, project_path, write_json


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "fourlang-translation/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def safe_tar_member(member: tarfile.TarInfo) -> bool:
    path = Path(member.name)
    return not path.is_absolute() and ".." not in path.parts


def read_flores(payload: bytes, language: str) -> list[str]:
    expected = f"flores200_dataset/devtest/{language}.devtest"
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        member = next((item for item in archive.getmembers() if item.name.endswith(expected)), None)
        if member is None or not safe_tar_member(member):
            raise RuntimeError(f"FLORES archive does not contain safe member {expected}")
        extracted = archive.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Cannot read {expected}")
        return extracted.read().decode("utf-8").splitlines()


def find_zip_lines(archive: zipfile.ZipFile, suffix: str) -> list[str]:
    name = next((item for item in archive.namelist() if item.endswith(suffix)), None)
    if name is None or Path(name).is_absolute() or ".." in Path(name).parts:
        raise RuntimeError(f"Tatoeba archive is missing safe *{suffix}")
    return archive.read(name).decode("utf-8").splitlines()


def stable_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["_order"] = [hashlib.sha256(f"{left}\n{right}".encode()).hexdigest() for left, right in frame.itertuples(index=False, name=None)]
    return frame.sort_values("_order").head(limit).drop(columns="_order").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protected FLORES and Tatoeba benchmarks.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    pair, source, target, _ = pair_info(config)
    benchmark = config["benchmarks"]
    flores_path, tatoeba_path = project_path(benchmark["flores"]), project_path(benchmark["tatoeba"])
    flores_payload = download(benchmark["flores_url"])
    source_lines = read_flores(flores_payload, benchmark["flores_source_code"])
    target_lines = read_flores(flores_payload, benchmark["flores_target_code"])
    if len(source_lines) != len(target_lines):
        raise RuntimeError("FLORES source/target rows are not aligned.")
    flores = pd.DataFrame({source: source_lines, target: target_lines})
    tatoeba_payload = download(benchmark["tatoeba_url"])
    with zipfile.ZipFile(io.BytesIO(tatoeba_payload)) as archive:
        source_lines = find_zip_lines(archive, benchmark["tatoeba_source_suffix"])
        target_lines = find_zip_lines(archive, benchmark["tatoeba_target_suffix"])
    if len(source_lines) != len(target_lines):
        raise RuntimeError("Tatoeba source/target rows are not aligned.")
    tatoeba = stable_sample(pd.DataFrame({source: source_lines, target: target_lines}), int(benchmark["tatoeba_pairs"]))
    flores_path.parent.mkdir(parents=True, exist_ok=True)
    flores.to_parquet(flores_path, index=False)
    tatoeba.to_parquet(tatoeba_path, index=False)
    write_json(PROJECT_ROOT / "reports" / "pipeline" / pair / "benchmarks.json", {
        "schema_version": 1,
        "protected_from_training": True,
        "flores": {"rows": len(flores), "path": str(flores_path.relative_to(PROJECT_ROOT)), "source": benchmark["flores_url"]},
        "tatoeba": {"rows": len(tatoeba), "path": str(tatoeba_path.relative_to(PROJECT_ROOT)), "source": benchmark["tatoeba_url"]},
    })


if __name__ == "__main__":
    main()
