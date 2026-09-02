from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


def norm_text(s: str) -> str:
    s = unicodedata.normalize('NFKC', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def basic_ok(en: str, ru: str) -> bool:
    if not en or not ru or en == ru:
        return False
    if len(en) < 2 or len(ru) < 2 or len(en) > 1000 or len(ru) > 1000:
        return False
    en_alpha = sum(ch.isalpha() for ch in en)
    ru_cyr = sum('\u0400' <= ch <= '\u04FF' for ch in ru)
    if en_alpha < 2 or ru_cyr < 2:
        return False
    ratio = max(len(en) / max(len(ru), 1), len(ru) / max(len(en), 1))
    if ratio > 4.0:
        return False
    bad_patterns = [r'<[^>]+>', r'https?://\S+', r'\{\{.*?\}\}']
    if any(re.search(p, en) or re.search(p, ru) for p in bad_patterns):
        return False
    return True


def pair_hash(en: str, ru: str) -> str:
    return hashlib.sha256(f'{en}\n{ru}'.encode('utf-8')).hexdigest()


def read_parallel(en_path: Path, ru_path: Path) -> pd.DataFrame:
    rows = []
    with en_path.open('r', encoding='utf-8', errors='replace') as fe, ru_path.open('r', encoding='utf-8', errors='replace') as fr:
        for i, (en, ru) in enumerate(zip(fe, fr), start=1):
            en = norm_text(en)
            ru = norm_text(ru)
            if basic_ok(en, ru):
                rows.append({
                    'pair_id': pair_hash(en, ru),
                    'en': en,
                    'ru': ru,
                    'source_corpus': 'News-Commentary',
                    'source_row': i,
                })
    if not rows:
        raise RuntimeError('No usable parallel rows found.')
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--en-file', default='data/raw/en_ru/news_commentary.en')
    ap.add_argument('--ru-file', default='data/raw/en_ru/news_commentary.ru')
    ap.add_argument('--approved-size', type=int, default=70000)
    ap.add_argument('--validation-size', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=2026)
    args = ap.parse_args()

    root = Path.cwd()
    en_path = root / args.en_file
    ru_path = root / args.ru_file
    if not en_path.exists():
        raise FileNotFoundError(en_path)
    if not ru_path.exists():
        raise FileNotFoundError(ru_path)

    clean_dir = root / 'data/clean/en_ru'
    approved_dir = root / 'data/approved/en_ru'
    split_dir = root / 'data/splits/en_ru/v1'
    report_dir = root / 'reports/data/en_ru'
    for d in (clean_dir, approved_dir, split_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    print('Reading and cleaning parallel data...')
    df = read_parallel(en_path, ru_path)
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['en', 'ru']).copy()

    en_counts = df.groupby('en')['ru'].nunique()
    ru_counts = df.groupby('ru')['en'].nunique()
    bad_en = set(en_counts[en_counts > 1].index)
    bad_ru = set(ru_counts[ru_counts > 1].index)
    df = df[~df['en'].isin(bad_en) & ~df['ru'].isin(bad_ru)].reset_index(drop=True)

    clean_path = clean_dir / 'news_commentary_en_ru_clean_v1.parquet'
    df.to_parquet(clean_path, index=False)

    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    approved_n = min(args.approved_size, len(df))
    approved = df.iloc[:approved_n].copy()
    if len(approved) <= args.validation_size:
        raise RuntimeError(f'Only {len(approved)} approved pairs; validation_size={args.validation_size} is too large.')

    approved_path = approved_dir / 'en_ru_human_approved_v1.parquet'
    approved.to_parquet(approved_path, index=False)
    validation = approved.iloc[:args.validation_size].copy()
    train = approved.iloc[args.validation_size:].copy()

    train_path = split_dir / 'train_pairs_v1.parquet'
    val_path = split_dir / 'validation_pairs_v1.parquet'
    train.to_parquet(train_path, index=False)
    validation.to_parquet(val_path, index=False)

    report = {
        'pair': 'en_ru',
        'version': 'v1',
        'source_corpus': 'News-Commentary',
        'seed': args.seed,
        'raw_usable_before_dedup': before_dedup,
        'clean_rows': len(df),
        'approved_rows': len(approved),
        'train_rows': len(train),
        'validation_rows': len(validation),
        'paths': {
            'clean': str(clean_path.relative_to(root)),
            'approved': str(approved_path.relative_to(root)),
            'train': str(train_path.relative_to(root)),
            'validation': str(val_path.relative_to(root)),
        },
        'notes': [
            'FLORES and Tatoeba are not included in this training split.',
            'Commercial/release use requires provenance and license review of each source corpus.',
        ],
    }
    report_path = report_dir / 'en_ru_human_v1_report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nDONE')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
