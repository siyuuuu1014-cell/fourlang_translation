"""Read-only live reconciliation into an unapproved combined review package."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path('/root/autodl-tmp/fourlang_translation')
SHARED = Path('/root/autodl-tmp/fourlang_shared_organization')


def main():
    out = Path('/root/autodl-tmp/fourlang_unified_organization') / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    current = SHARED / 'handoff_20260914_145040'
    historical = SHARED / 'inventory_20260914_143314'
    inputs = [current / n for n in ('current_changes.md', 'current_existence_recheck.json', 'code_inventory.json', 'static_import_edges.json')]
    inputs += [historical / n for n in ('HANDOFF.md', 'experiment_history.md', 'data_lineage_and_verification.md', 'missing_files.json', 'cleanup_candidates.json')]
    inputs += [ROOT / 'configs/specialists/current_pair_models.json']
    sources = [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]
    old_recheck = json.loads((current / 'current_existence_recheck.json').read_text())
    pairs = json.loads((ROOT / 'configs/specialists/current_pair_models.json').read_text())
    code = json.loads((current / 'code_inventory.json').read_text())
    edges = json.loads((current / 'static_import_edges.json').read_text())
    preserved = set()
    for pair in pairs['pairs']:
        for key in ('model_path', 'training_entrypoint', 'training_config', 'training_data', 'validation_data', 'evaluation_evidence', 'source_data_to_preserve'):
            vals = pair.get(key, [])
            for val in ([vals] if isinstance(vals, str) else vals):
                preserved.add(str(ROOT / val))
    for exp in ('exp1', 'exp2', 'exp3_v2'):
        preserved.add(str(ROOT / f'results/student/fourlang/{exp}/best_model/shared'))
    preserved.add(str(ROOT / 'results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1438'))
    protected_prefixes = [str(ROOT / p) for p in ('data', 'scripts', 'configs', 'inference', 'src', 'tests', 'docs', 'reports', 'acceptance', 'models', 'results/evaluation')]
    protected_prefixes += ['/root/autodl-tmp/models', '/root/autodl-tmp/venvs', str(SHARED), '/root/autodl-tmp/fourlang_diagnostics_recovery_20260914']
    observed = []
    paths = {x['path'] for x in old_recheck} | preserved | {x['path'] for x in code}
    paths |= {str(ROOT / 'scripts/pipeline_v3/preview_manual_zh_uz_isolation.py')}
    previous = {x['path']: x['current_exists'] for x in old_recheck}
    for value in sorted(paths):
        p = Path(value)
        item = {'path': value, 'exists': p.exists(), 'previous_handoff_exists': previous.get(value)}
        if p.is_file():
            item['bytes'] = p.stat().st_size
        observed.append(item)
    candidates = []
    for key, owner, rel in (
        ('P1', 'pair', 'results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k/checkpoints'),
        ('P2', 'pair', 'results/experiments/weak_pair_ablation/zh_uz/flores_relaxed_8k_ep3/checkpoints'),
        ('S1', 'shared', 'results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-500'),
        ('S2', 'shared', 'results/student/fourlang/exp3_v2/checkpoints/shared/checkpoint-1000'),
    ):
        path = ROOT / rel
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Unexpected candidate resolution')
        files, links = [], []
        for p in sorted(path.rglob('*')):
            if p.is_symlink():
                links.append(str(p))
            elif p.is_file():
                stat = p.stat()
                entry = {'path': str(p), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                         'allocated_bytes': stat.st_blocks * 512, 'hardlink_count': stat.st_nlink}
                if p.suffix in ('.json', '.toml', '.md', '.txt', '.py') and stat.st_size < 8*1024*1024:
                    entry['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
                files.append(entry)
        candidates.append({'id': key, 'owner': owner, 'path': str(path), 'exists': path.is_dir(),
                           'logical_bytes': sum(x['bytes'] for x in files),
                           'allocated_bytes': sum(x['allocated_bytes'] for x in files),
                           'files': files, 'symlinks': links, 'status': 'PENDING_USER_APPROVAL',
                           'approved': False, 'loss': 'Exact model/optimizer/scheduler/RNG resume and checkpoint comparisons for these steps'})
    for c in candidates:
        if any(Path(p).is_relative_to(Path(c['path'])) for p in preserved):
            raise ValueError('Protected item nested under candidate')
    stash = subprocess.run(['git', 'stash', 'list', '--format=%H'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    report = {'status': 'REVIEW_ONLY_NO_DELETION_AUTHORIZED', 'utc': out.name, 'source_project_modified': False,
              'source_evidence': sources, 'candidates': candidates, 'preserve_explicit_paths': sorted(preserved),
              'preserve_prefixes': protected_prefixes, 'preserve_all_other_model_weights': True,
              'stash_commits_preserve': stash, 'current_existence': observed,
              'code_inventory': code, 'static_import_edges': edges,
              'historical_missing_files': json.loads((historical / 'missing_files.json').read_text()),
              'total_logical_bytes': sum(x['logical_bytes'] for x in candidates),
              'execution_permitted': False,
              'required_before_execution': ['Separate explicit approval of IDs and exact paths', 'No concurrent writers or cleanup tasks',
                 'Verified checkpoint metadata backup outside deleted paths', 'Confirm loss of exact resume is acceptable',
                 'Resolve shared dataset backup plan without applying stash', 'Recheck live paths, files and sizes',
                 'Single executor, no force reset, no git clean, no stash/route/SSH authorization changes'],
              'limitations': 'Metadata/existence check, not model loading, complete weight hashing, production readiness, or precise recoverable disk accounting.'}
    (out / 'unified_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(out), 'total_logical_bytes': report['total_logical_bytes'],
                      'candidates': [{k:x[k] for k in ('id','path','logical_bytes','exists')} for x in candidates],
                      'handoff_changes': [x for x in observed if x['previous_handoff_exists'] is not None and x['exists'] != x['previous_handoff_exists']],
                      'stash_count': len(stash), 'deleted': 0}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
