"""One reviewed four-target cleanup. Prepare separately; execute requires exact token."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/root/autodl-tmp/fourlang_translation')
BASE = Path('/root/autodl-tmp/fourlang_unified_organization')
REVIEW = BASE / '20260914T065733Z/unified_review.json'
TARGETS = [
    ROOT / 'results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints',
    ROOT / 'results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints',
    ROOT / 'results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-500',
    ROOT / 'results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1000',
]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def stash():
    return subprocess.check_output(['git', 'stash', 'list', '--format=%H'], cwd=ROOT, text=True).splitlines()


def snapshot(path):
    if path.is_symlink() or path.resolve() != path or not path.is_dir():
        raise RuntimeError(f'Unsafe or missing target: {path}')
    result = []
    for f in sorted(path.rglob('*')):
        if f.is_symlink() or not f.resolve().is_relative_to(path):
            raise RuntimeError(f'Unsafe child: {f}')
        if f.is_file():
            s = f.stat()
            result.append({'path': str(f), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns})
    return result


def no_jobs():
    # Fail closed on training/copy/delete processes. Do not print command arguments.
    bad = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            comm = (p/'comm').read_text().strip()
            if comm.lower().startswith(('python', 'torchrun', 'accelerate', 'rsync')) or comm in ('rm', 'mv', 'cp', 'tar'):
                bad.append({'pid': p.name, 'process': comm})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if bad:
        raise RuntimeError(f'Concurrent jobs require review: {bad}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'execute'))
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--confirm')
    args = parser.parse_args()
    no_jobs()
    review = json.loads(REVIEW.read_text())
    if [x['path'] for x in review['candidates']] != [str(x) for x in TARGETS]:
        raise RuntimeError('Review scope mismatch')
    if args.phase == 'prepare':
        out = BASE / ('approved_cleanup_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
        out.mkdir(exist_ok=False)
        receipt = {'status': 'PREPARING', 'review_sha256': sha(REVIEW), 'targets': [], 'backups': [],
                   'preserved': [], 'stash_before': stash(), 'deleted': [],
                   'authorization': 'User explicitly confirmed P1 P2 S1 S2 in current conversation; historical review stays unapproved.'}
        save(out/'receipt.json', receipt)
        for target, original in zip(TARGETS, review['candidates']):
            files = snapshot(target)
            prior = [{k:f[k] for k in ('path','bytes','mtime_ns')} for f in original['files']]
            if files != prior:
                raise RuntimeError(f'Target changed since review: {target}')
            receipt['targets'].append({'path': str(target), 'files': files})
            for item in files:
                source = Path(item['path'])
                if source.suffix == '.safetensors' or source.name.startswith(('pytorch_model', 'optimizer')):
                    continue
                dest = out/'checkpoint_metadata'/source.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                digest = sha(source)
                if sha(dest) != digest:
                    raise RuntimeError('Metadata backup mismatch')
                receipt['backups'].append({'path': str(dest), 'sha256': digest, 'source': str(source)})
        data = json.loads((Path('/root/autodl-tmp/fourlang_shared_organization/inventory_20260914_143314/actual_data_verification.json')).read_text())
        for entry in data:
            source = Path(entry['path'])
            if not source.is_relative_to(ROOT/'data/multilingual/fourlang'):
                raise RuntimeError('Unexpected dataset backup source')
            dest = out/'stash_dataset_backup'/source.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open('xb') as f:
                subprocess.run(['git','cat-file','blob',entry['stash_object']], cwd=ROOT, stdout=f, check=True)
            if sha(dest) != entry['byte_sha256'] or dest.stat().st_size != entry['bytes']:
                raise RuntimeError('Stash dataset backup mismatch')
            receipt['backups'].append({'path': str(dest), 'sha256': sha(dest), 'source_blob': entry['stash_object']})
        protected_dirs = [Path(x) for x in review['preserve_explicit_paths'] if Path(x).is_dir() and ('best_model' in x or 'checkpoint-1438' in x or '/models/' in x)]
        for directory in protected_dirs:
            files = snapshot(directory)
            for entry in files:
                entry['sha256'] = sha(Path(entry['path']))
            receipt['preserved'].extend(files)
            print('Protected hash verified: ' + str(directory), flush=True)
        if stash() != receipt['stash_before']:
            raise RuntimeError('Stash changed')
        for row in receipt['targets']:
            if snapshot(Path(row['path'])) != row['files']:
                raise RuntimeError('Target changed while preparing')
        receipt['status'] = 'BACKUP_VERIFIED_READY_FOR_EXECUTION'
        receipt['free_bytes_after_backup'] = shutil.disk_usage(ROOT).free
        save(out/'receipt.json', receipt)
        print('READY ' + str(out/'receipt.json'), flush=True)
    else:
        if args.confirm != 'DELETE_APPROVED_P1_P2_S1_S2' or args.receipt is None:
            raise RuntimeError('Explicit execution token required')
        file = args.receipt.resolve()
        if not file.is_relative_to(BASE) or file.name != 'receipt.json':
            raise RuntimeError('Unexpected receipt location')
        receipt = json.loads(file.read_text())
        if receipt['status'] != 'BACKUP_VERIFIED_READY_FOR_EXECUTION' or sha(REVIEW) != receipt['review_sha256']:
            raise RuntimeError('Receipt not ready or review changed')
        if [x['path'] for x in receipt['targets']] != [str(x) for x in TARGETS]:
            raise RuntimeError('Receipt scope mismatch')
        if stash() != receipt['stash_before']:
            raise RuntimeError('Stash changed')
        for item in receipt['backups']:
            if sha(Path(item['path'])) != item['sha256']:
                raise RuntimeError('Backup changed')
        for item in receipt['preserved']:
            if sha(Path(item['path'])) != item['sha256']:
                raise RuntimeError('Protected model changed')
        for row in receipt['targets']:
            if snapshot(Path(row['path'])) != row['files']:
                raise RuntimeError('Target changed')
        no_jobs()
        receipt['free_bytes_before_delete'] = shutil.disk_usage(ROOT).free
        receipt['status'] = 'EXECUTING'
        save(file, receipt)
        for target in TARGETS:
            # Exact Linux paths validated above; never delete any computed parent.
            shutil.rmtree(target)
            receipt['deleted'].append(str(target))
            save(file, receipt)
            print('DELETED ' + str(target), flush=True)
        receipt['preserved_hashes_match_after'] = all(sha(Path(x['path'])) == x['sha256'] for x in receipt['preserved'])
        receipt['stash_unchanged'] = stash() == receipt['stash_before']
        receipt['free_bytes_after_delete'] = shutil.disk_usage(ROOT).free
        receipt['observed_free_increase_bytes'] = receipt['free_bytes_after_delete'] - receipt['free_bytes_before_delete']
        receipt['status'] = 'COMPLETE' if receipt['preserved_hashes_match_after'] and receipt['stash_unchanged'] else 'POSTCHECK_REQUIRES_ATTENTION'
        save(file, receipt)
        print(json.dumps({k: receipt[k] for k in ('status','deleted','preserved_hashes_match_after','stash_unchanged','observed_free_increase_bytes')}), flush=True)


if __name__ == '__main__':
    main()
