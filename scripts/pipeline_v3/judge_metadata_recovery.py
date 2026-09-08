"""Recover missing audit metadata from explicit judged artifacts, never training text."""

from __future__ import annotations

from collections import Counter, defaultdict

from scripts.pipeline_v3 import diagnose_zh_uz as diag
from scripts.pipeline_v3.language_normalization import normalize_language_text

DEFAULT_JUDGE_PATHS = tuple(
    f"data/pipeline_v2/{namespace}/teacher_judged.parquet"
    for namespace in ("zh_uz", "zh_uz_v2", "zh_uz_v3")
)
FIELDS = ("judge_label", "teacher_id", "teacher_usefulness")
MISSING = {"", "UNKNOWN", "NONE", "NAN", "NULL"}


def value(row, key):
    result = diag.metadata_value(row, key)
    if result.upper() in MISSING:
        return "UNKNOWN"
    return result.upper() if key != "teacher_id" else result


def read_judgments(path):
    """Explicit Teacher output column required; never use human reference_text."""
    frame = diag._read_table(path)
    aliases = {
        "source_lang": "src_lang",
        "target_lang": "tgt_lang",
        "source_text": "src_text",
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if v not in frame})
    required = {
        "src_lang",
        "tgt_lang",
        "src_text",
        "teacher_text",
        "judge_label",
        "judge_parse_ok",
    }
    if missing := required - set(frame.columns):
        raise ValueError(
            f"Judged artifact {path} lacks required columns: {sorted(missing)}"
        )
    records = []
    for ordinal, raw in enumerate(frame.to_dict("records")):
        src, tgt = (
            str(raw["src_lang"]).strip().lower(),
            str(raw["tgt_lang"]).strip().lower(),
        )
        if f"{src}-{tgt}" not in diag.DIRECTIONS:
            continue
        if not isinstance(raw["src_text"], str) or not isinstance(
            raw["teacher_text"], str
        ):
            continue
        source = normalize_language_text(src, raw["src_text"])
        target = normalize_language_text(tgt, raw["teacher_text"])
        if not source or not target:
            continue
        # bool("False") is True: never use truthiness to authenticate parsing.
        parse_ok = raw["judge_parse_ok"] is True
        records.append(
            {
                "src_lang": src,
                "tgt_lang": tgt,
                "src_text": source,
                "tgt_text": target,
                **{key: value(raw, key) for key in FIELDS},
                "parse_ok": parse_ok,
                "evidence_path": str(path),
                "evidence_row": ordinal,
            }
        )
    return records


def recover_metadata(original, judgments):
    """Return an audit-side copy. Conflicts block recovery; never infer from weight."""
    index = defaultdict(list)
    for record in judgments:
        index[diag.identity(record)].append(record)
    recovered, traces = [], []
    for row_number, original_row in enumerate(original):
        row = dict(original_row)
        if not diag.is_teacher(row):
            recovered.append(row)
            continue
        matches = index.get(diag.identity(row), [])
        before = {key: value(row, key) for key in FIELDS}
        choices = {
            key: sorted({value(item, key) for item in matches} - {"UNKNOWN"})
            for key in FIELDS
        }
        conflicts = [
            key
            for key, options in choices.items()
            if len(options) > 1
            or (options and before[key] != "UNKNOWN" and before[key] not in options)
        ]
        missing = [key for key in FIELDS if before[key] == "UNKNOWN"]
        status, filled = "NO_MATCH", []
        if matches:
            if conflicts or "AMBIGUOUS" in before.values():
                status = "CONFLICT"
            elif not all(item["parse_ok"] for item in matches):
                status = "PARSE_UNVERIFIED"
            elif choices["judge_label"] not in (["PASS"], ["MINOR"], ["FAIL"]):
                status = "LABEL_UNVERIFIED"
            elif choices["teacher_usefulness"] and choices[
                "teacher_usefulness"
            ] not in (["HIGH"], ["MEDIUM"], ["REJECT"]):
                status = "USEFULNESS_UNVERIFIED"
            else:
                for key in missing:
                    if len(choices[key]) == 1:
                        row[key] = choices[key][0]
                        filled.append(key)
                status = "RECOVERED" if filled else "CONSISTENT_NO_FILL"
        recovered.append(row)
        traces.append(
            {
                "original_row": row_number,
                "sample_id": diag.fingerprint(diag.identity(row)),
                "training_source": row["training_source"],
                "weight": float(row["weight"]),
                "status": status,
                "before": before,
                "after": {key: value(row, key) for key in FIELDS},
                "filled_fields": filled,
                "conflicting_fields": conflicts,
                "evidence": sorted(
                    [
                        {
                            "path": item["evidence_path"],
                            "row": item["evidence_row"],
                            "parse_ok": item["parse_ok"],
                            **{k: item[k] for k in FIELDS},
                        }
                        for item in matches
                    ],
                    key=lambda x: (x["path"], x["row"]),
                ),
            }
        )
    return recovered, {
        "status": "AUDIT_SIDE_ONLY_NOT_DATA_APPROVAL",
        "matching": "direction + exact source and Teacher text after existing language normalization; "
        "no fuzzy, source-only, reference-text or weight-derived labeling",
        "counts_current_pool_teacher_rows": dict(
            sorted(Counter(t["status"] for t in traces).items())
        ),
        "records": traces,
    }
