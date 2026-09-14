"""Read-only cleanup verification plus sequential bidirectional smoke tests."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    manifest = json.loads((ROOT / 'configs/specialists/current_pair_models.json').read_text())
    archive = ROOT / 'reports/experiment_archive/20260914T053818Z'
    output = ROOT / 'reports/cleanup_verification' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=False)
    report = {'output': str(output), 'deleted': 0, 'models': [], 'candidates': []}

    def save():
        (output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    references = []
    for directory in ('configs', 'scripts', 'inference', 'service'):
        for f in (ROOT / directory).rglob('*'):
            if f.is_file() and f.suffix in ('.py', '.json', '.toml', '.yaml', '.yml'):
                references.append((str(f.relative_to(ROOT)), f.read_text(encoding='utf-8', errors='replace')))
    source = ROOT / 'results/student/pair_specialists/en_ru/exp2/best_model/shared'
    frozen = ROOT / 'models/final_pair_specialists/en_ru_v1'
    comparison = []
    for f in sorted(source.rglob('*')):
        if f.is_file():
            dest = frozen / f.relative_to(source)
            comparison.append({'file': str(f.relative_to(source)), 'same': dest.is_file() and sha(f) == sha(dest)})
    report['en_ru_duplicate_comparison'] = comparison
    report['en_ru_source_files_all_identical'] = bool(comparison) and all(r['same'] for r in comparison)
    save()
    print('Duplicate comparison complete', flush=True)
    for target in manifest['safe_delete']:
        path = ROOT / target
        if not path.exists():
            continue
        files = [f for f in path.rglob('*') if f.is_file()]
        gaps = []
        archived = 0
        for f in files:
            if f.suffix in ('.json', '.jsonl', '.toml', '.md', '.txt', '.log', '.csv', '.parquet'):
                old = archive / 'evidence' / f.relative_to(ROOT)
                if old.is_file() and sha(f) == sha(old):
                    archived += 1
                else:
                    gaps.append(str(f.relative_to(ROOT)))
        hits = [name for name, content in references if target in content and name not in (
            'configs/specialists/current_pair_models.json', 'scripts/pipeline_v3/verify_pair_cleanup.py')]
        report['candidates'].append({'path': target, 'bytes': sum(f.stat().st_size for f in files),
            'archived_metadata_files': archived, 'metadata_archive_gaps': gaps, 'literal_references': hits,
            'decision': 'HOLD_FOR_REVIEW',
            'impact': 'Removing checkpoints loses exact optimizer/RNG resume; removing old weights loses direct reruns. Dynamic path construction also requires review.'})
    save()
    print('Archive and literal-reference checks complete', flush=True)
    import torch
    from inference.loader import load_translation_model
    from inference.engine import TranslationEngine
    texts = {'en': 'The meeting starts at nine.', 'zh': '会议九点开始。', 'ru': 'Встреча начинается в девять часов.', 'uz': "Uchrashuv soat to'qqizda boshlanadi."}
    for pair in manifest['pairs']:
        row = {'pair': pair['id'], 'path': pair['model_path'], 'smoke_tests': []}
        loaded = engine = None
        try:
            path = ROOT / pair['model_path']
            row['required_tokenizer_files'] = {n: (path / n).is_file() for n in (
                'config.json', 'tokenization_small100.py', 'vocab.json', 'sentencepiece.bpe.model')}
            if not all(row['required_tokenizer_files'].values()):
                raise RuntimeError('Missing required tokenizer/model file')
            loaded = load_translation_model(path, device='cuda', dtype='float16')
            for direction in pair['directions']:
                engine = TranslationEngine(loaded, direction=direction, num_beams=5, max_source_length=256, max_new_tokens=256)
                result = engine.translate(texts[direction.split('-')[0]])
                row['smoke_tests'].append({'direction': direction, 'input': result['input'], 'translation': result['translation'], 'nonempty': bool(result['translation'].strip())})
            row['status'] = 'PASS' if all(r['nonempty'] for r in row['smoke_tests']) else 'EMPTY_OUTPUT'
        except Exception as exc:
            row['status'] = 'FAIL'
            row['error'] = str(exc)
        finally:
            del engine, loaded
            gc.collect()
            torch.cuda.empty_cache()
        report['models'].append(row)
        save()
        print(json.dumps(row, ensure_ascii=False), flush=True)
    lines = ['# 删除前核验记录', '', '本次没有删除任何文件。冒烟测试仅证明实际加载及解码成功，不代表翻译准确率。', '',
        f"EN-RU 源文件与冻结副本全部相同：{report['en_ru_source_files_all_identical']}。", '',
        '## 六模型双向加载与推理', '', '| 模型 | 状态 |', '|---|---|']
    for model in report['models']:
        lines.append(f"| {model['pair']} | {model['status']} |")
    lines += ['', '## 候选目录（全部等待复核，不执行删除）', '', '| 路径 | GiB | 元数据归档缺口 | 字面引用文件数 |', '|---|---:|---:|---:|']
    for item in report['candidates']:
        lines.append(f"| `{item['path']}` | {item['bytes']/1024**3:.3f} | {len(item['metadata_archive_gaps'])} | {len(item['literal_references'])} |")
    lines += ['', '动态拼接路径无法靠字面搜索排除。旧 best_model 删除影响旧实验重评，checkpoint 删除影响精确断点恢复；评估报告不会自动失效，但保留报告不等于保留复现能力。数据目录本轮建议全部保留。', '', '详细文件比较、引用、归档缺口和测试译文见 verification.json。']
    (output / 'CLEANUP_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print('FINISHED ' + str(output), flush=True)


if __name__ == '__main__':
    main()
