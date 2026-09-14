"""Compare original report evidence with current paths; recover only outside project."""
import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Recovery output must be outside source project')
    output.mkdir(parents=True, exist_ok=False)
    text = args.record.read_text(encoding='utf-8')
    baseline = {name: json.loads(body) for name, body in re.findall(r'^### ([^\n]+)\n\n```json\n(.*?)\n```', text, re.M | re.S)}
    if len(baseline) != 62:
        raise ValueError(f'Expected 62 historical reports, got {len(baseline)}')
    archive = root / 'reports/experiment_archive/20260914T053818Z/evidence.tar.gz'
    results = []
    absent_files = []
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith('evidence/'):
                continue
            rel = Path(member.name[len('evidence/'):])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archive member')
            src = root / rel
            if not src.exists():
                absent_files.append({'path': rel.as_posix(), 'archived_bytes': member.size})
            name = rel.as_posix()
            if name not in baseline:
                continue
            raw = tf.extractfile(member).read()
            archived = json.loads(raw)
            current = json.loads(src.read_text(encoding='utf-8')) if src.is_file() else None
            row = {'path': name, 'present_now': src.is_file(),
                   'archive_matches_record': archived == baseline[name],
                   'current_matches_record': current == baseline[name] if src.is_file() else None,
                   'archive_sha256': hashlib.sha256(raw).hexdigest()}
            if not src.exists():
                dest = output / 'recovered_reports' / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
                row['recovered_path'] = str(dest)
            results.append(row)
    found = {r['path'] for r in results}
    report = {'historical_report_count': len(baseline), 'archive_report_count': len(results),
              'not_in_archive': sorted(set(baseline) - found), 'reports': results,
              'archived_files_absent_at_original_path': absent_files,
              'source_project_modified': False,
              'note': 'Absent paths may have been moved; not proof of permanent deletion. Weight files were not included in old evidence archive.'}
    (output / 'reconciliation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    missing = [r for r in results if not r['present_now']]
    lines = ['# 历史实验报告差异核对', '',
             f'原记录 {len(baseline)} 份；压缩包找到 {len(results)} 份；当前原路径缺失 {len(missing)} 份。',
             f"归档与原记录 JSON 内容全部一致：{all(r['archive_matches_record'] for r in results)}。",
             '缺失报告已按原始字节恢复到本独立目录 recovered_reports 下，没有覆盖或修改源项目。', '',
             '| 报告原路径 | 当前存在 | 与历史记录一致 |', '|---|---|---|']
    lines += [f"| `{root / r['path']}` | {r['present_now']} | {r['current_matches_record']} |" for r in results]
    lines += ['', f'原归档另有 {len(absent_files)} 个文件当前不在原路径；详见 JSON。该清单涉及共同依赖，仅记录、不批量恢复或删除。',
              '旧 file_inventory.json、archive_checksums.json 不属于 evidence 压缩包中的源文件，不宣称已恢复；新清单须使用重建快照。']
    (output / 'HISTORICAL_REPORT_RECONCILIATION.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(output), 'reports_found': len(results), 'missing_reports_recovered': len(missing),
                      'not_in_archive': report['not_in_archive'], 'archive_record_mismatches': [r['path'] for r in results if not r['archive_matches_record']],
                      'current_changed': [r['path'] for r in results if r['present_now'] and not r['current_matches_record']],
                      'absent_archived_files': len(absent_files)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
