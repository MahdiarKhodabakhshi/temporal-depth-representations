#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_root', type=str, required=True, help='Root like experiments/results/Pythia/70m')
    parser.add_argument('--expected_revisions', type=str, default='', help='Optional comma-separated revision subset')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def safe_main_score(payload: dict) -> float:
    return float(payload['scores']['test'][0]['main_score'])


def build_for_revision(rev_root: Path, overwrite: bool) -> Path | None:
    mteb_root = rev_root / 'mteb'
    if not mteb_root.exists():
        return None

    rows = []
    for layer_dir in sorted([p for p in mteb_root.iterdir() if p.is_dir() and p.name.startswith('layer_')], key=lambda p: int(p.name.split('_')[1])):
        scores = []
        for json_path in sorted(layer_dir.rglob('*.json')):
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            try:
                scores.append(safe_main_score(payload))
            except Exception:
                continue
        if not scores:
            continue
        layer_idx = int(layer_dir.name.split('_')[1])
        rows.append({
            'layer': f'layer_{layer_idx}',
            'layer_idx': layer_idx,
            'avg_main_score': sum(scores) / len(scores),
            'num_tasks': len(scores),
        })

    if not rows:
        return None

    out_dir = rev_root / 'average_main_score'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'avg_main_score_by_layer.csv'
    if out_path.exists() and not overwrite:
        return out_path

    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'layer_idx', 'avg_main_score', 'num_tasks'])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    revisions = [x.strip() for x in args.expected_revisions.split(',') if x.strip()]
    rev_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if revisions:
        wanted = set(revisions)
        rev_dirs = [p for p in rev_dirs if p.name in wanted]

    created = []
    for rev_dir in rev_dirs:
        out = build_for_revision(rev_dir, overwrite=args.overwrite)
        if out is not None:
            created.append(str(out))

    for path in created:
        print(path)


if __name__ == '__main__':
    main()
