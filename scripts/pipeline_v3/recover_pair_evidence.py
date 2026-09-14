"""Recover absent pair-only evidence from archive, never overwrite existing files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--reconciliation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.is_relative_to(root):
        raise ValueError('Backup must be outside project')
    out.mkdir(parents=True, exist_ok=False)
    absent = json.loads(args.reconciliation.read_text())['archived_files_absent_at_original_path']
    selected = {x['path'] for x in absent if not x['path'].startswith('reports/diagnostics/fourlang/')}
    prefixes = ('data/experiments/weak_pair_ablation/zh_uz/', 'data/specialists/en_ru/',
                'data/specialists/en_zh/', 'data/specialists/uz_ru/', 'data/specialists/zh_ru/',
                'data/specialists/zh_uz/', 'reports/experiments/zh_uz_',
                'reports/experiments/six_pair_baseline_v1/', 'reports/diagnostics/zh_uz_')
    exact = {'docs/SIX_PAIR_SPECIALISTS.md', 'scripts/pipeline_v3/build_clean_zh_uz_v4.py'}
    records = []
    archive = root / 'reports/experiment_archive/20260914T053818Z/evidence.tar.gz'
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            if not member.isfile() or not member.name.startswith('evidence/'):
                continue
            name = member.name[len('evidence/'):]
            if name not in selected:
                continue
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe member')
            raw = tf.extractfile(member).read()
            copy = out / 'evidence' / rel
            copy.parent.mkdir(parents=True, exist_ok=True)
            with copy.open('xb') as f:
                f.write(raw)
            digest = hashlib.sha256(raw).hexdigest()
            if hashlib.sha256(copy.read_bytes()).hexdigest() != digest:
                raise ValueError('Backup verification failed')
            status = 'backup_only_shared_or_review_sensitive'
            if name.startswith(prefixes) or name in exact:
                dest = root / rel
                if not dest.resolve().is_relative_to(root):
                    raise ValueError('Destination escapes project')
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with dest.open('xb') as f:
                        f.write(raw)
                    if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                        raise ValueError('Restoration verification failed')
                    status = 'restored_missing_only'
                except FileExistsError:
                    status = 'existing_preserved'
            records.append({'path': name, 'bytes': len(raw), 'sha256': digest, 'status': status})
            (out / 'recovery_log.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'backed_up': len(records),
                      'restored': sum(r['status'] == 'restored_missing_only' for r in records),
                      'backup_only': [r['path'] for r in records if r['status'].startswith('backup_only')],
                      'shared_diagnostics_excluded': len(absent) - len(selected), 'deleted': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
