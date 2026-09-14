"""Read-only post-cleanup reconciliation; write indexes outside the project."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / 'configs/specialists/current_pair_models.json').read_text())
    archive = root / 'reports/experiment_archive/20260914T053818Z'
    archive_inventory_available = (archive / 'file_inventory.json').is_file()
    old = json.loads((archive / 'file_inventory.json').read_text()) if archive_inventory_available else []
    missing = []
    for entry in old:
        path = root / entry['path']
        if not path.exists():
            missing.append({'path': entry['path'], 'previous_bytes': entry['bytes'],
                            'evidence_copy_exists': (archive / 'evidence' / entry['path']).is_file()})
    pairs = []
    fields = ['model_path', 'training_entrypoint', 'training_config', 'training_data',
              'validation_data', 'evaluation_evidence', 'source_data_to_preserve']
    for p in manifest['pairs']:
        row = {'id': p['id'], 'references': []}
        for key in fields:
            items = p.get(key, [])
            if isinstance(items, str):
                items = [items]
            for item in items:
                row['references'].append({'role': key, 'path': item, 'exists': (root / item).exists()})
        pairs.append(row)
    checkpoints = []
    for variant in ('flores_relaxed_8k', 'flores_relaxed_8k_ep3'):
        rel = f'results/experiments/weak_pair_ablation/zh_uz/{variant}/checkpoints'
        p = root / rel
        checkpoints.append({'path': rel, 'exists': p.exists(), 'logical_bytes': sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0})
    report = {'root': str(root), 'mutations_to_project': False, 'pairs': pairs,
              'historical_inventory_available': archive_inventory_available,
              'missing_since_snapshot': missing, 'snapshot_files': len(old),
              'checkpoint_candidates': checkpoints,
              'caveat': 'Missing means absent at original path; may have been moved elsewhere. Evidence-copy presence is not a fresh checksum verification.'}
    (out / 'inventory.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 独立模型整理总索引（清理后复核）', '', f'项目：`{root}`。', '',
             '本次只读盘点，无移动、删除或模型质量测试。四语言集合模型不纳入独立模型的清理范围。', '',
             '## 六组模型及复现依赖', '']
    for p in pairs:
        lines += [f"### {p['id']}", '', '| 用途 | 绝对路径 | 存在 |', '|---|---|---|']
        lines += [f"| {r['role']} | `{root / r['path']}` | {'是' if r['exists'] else '否，需核查'} |" for r in p['references']]
        lines += ['']
    lines += ['## 历史与协调', '',
              f'历史证据快照：`{archive}`。',
              '实验经过见该目录 EXPERIMENT_RECORD.md；归档是元数据/小文件证据，不是完整模型权重备份。',
              (f'历史清单 {len(old)} 项中，有 {len(missing)} 项在原路径已不存在；详见 inventory.json，不能仅据此认定已永久删除。' if archive_inventory_available else '警告：历史 file_inventory.json 已不在原位置；无法计算清理前后差异。缺失数组为空不代表没有文件丢失。旧归档不能继续认定完整；删除暂停。'),
              '共享代码、原始双语数据和基础模型均按共同依赖保留，不由任一任务单方面清理。',
              '四语任务仅写自己的独立整理目录；两个任务均不得执行全项目清理、同步删除或覆盖原归档。', '',
              '## 原检查点候选当前状态', '']
    for c in checkpoints:
        lines += [f"- `{root / c['path']}`：exists={c['exists']}，逻辑字节={c['logical_bytes']}。"]
    lines += ['', '## 使用入口', '',
              f'统一入口：`{root}/scripts/pipeline_v3/translate_current_models.py`。',
              f'调用说明：`{root}/docs/CURRENT_MODEL_USAGE.md`。',
              '保持原路径，使用索引整理逻辑关系，避免移动目录破坏脚本。待双方依赖盘点完成，再合并一份删除清单执行。']
    (out / 'PAIR_ORGANIZATION_INDEX.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'output': str(out), 'missing': len(missing), 'missing_manifest_references': [r for p in pairs for r in p['references'] if not r['exists']], 'checkpoints': checkpoints}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
