#!/usr/bin/env python3
"""
Prepare local files required by some cached MTEB dataset scripts in offline mode.

Why this exists:
- Some trust_remote_code dataset modules (e.g., amazon_counterfactual, amazon_reviews_multi)
  use relative local file paths (like "data/EN-ext_train.tsv" or "en/train.jsonl").
- In offline mode, these paths can fail unless the files exist under the current project root.

This script materializes those files from already-cached Arrow datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc


REPO_ROOT = Path(__file__).resolve().parents[2]


def _latest_cache_leaf(root: Path, required_files: list[str] | None = None) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Cache path not found: {root}")

    required_files = required_files or []
    leaves = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not leaves:
        raise FileNotFoundError(f"No cache leaf directories under: {root}")

    skipped: list[str] = []
    for leaf in leaves:
        if leaf.name.endswith(".incomplete"):
            skipped.append(f"{leaf} (incomplete)")
            continue
        missing = [name for name in required_files if not (leaf / name).exists()]
        if missing:
            skipped.append(f"{leaf} (missing: {', '.join(missing)})")
            continue
        return leaf

    detail = "\n".join(skipped[:20])
    raise FileNotFoundError(
        f"No complete cache leaf directories under: {root}\nCandidates checked:\n{detail}"
    )


def _read_arrow_table(path: Path) -> pa.Table:
    # HF datasets cache commonly stores Arrow IPC streams.
    try:
        with pa.memory_map(str(path), "r") as source:
            return ipc.open_stream(source).read_all()
    except Exception:
        with pa.memory_map(str(path), "r") as source:
            return ipc.open_file(source).read_all()


def _materialize_amazon_counterfactual(project_root: Path, datasets_cache: Path) -> list[Path]:
    # Dataset module expects: data/{lang}_{split}.tsv where split in {train,valid,test}
    dataset_root = datasets_cache / "mteb___amazon_counterfactual"
    config_to_prefix = {"en": "EN", "en-ext": "EN-ext"}
    split_map = {"train": "train", "validation": "valid", "test": "test"}
    out_dir = project_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for config, prefix in config_to_prefix.items():
        leaf = _latest_cache_leaf(
            dataset_root / config / "1.0.0",
            required_files=[
                "amazon_counterfactual-train.arrow",
                "amazon_counterfactual-validation.arrow",
                "amazon_counterfactual-test.arrow",
            ],
        )
        for src_split, dst_split in split_map.items():
            in_file = leaf / f"amazon_counterfactual-{src_split}.arrow"
            if not in_file.exists():
                raise FileNotFoundError(f"Missing cached file: {in_file}")
            table = _read_arrow_table(in_file)
            texts = table.column("text").to_pylist()
            labels = table.column("label").to_pylist()

            out_file = out_dir / f"{prefix}_{dst_split}.tsv"
            with out_file.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["text", "label"])
                for text, label in zip(texts, labels):
                    w.writerow([text, int(label)])
            written.append(out_file)
            print(f"Wrote {out_file} rows={len(texts)}")
    return written


def _materialize_amazon_reviews_multi(project_root: Path, datasets_cache: Path) -> list[Path]:
    # Dataset module expects: en/{train,validation,test}.jsonl
    dataset_root = datasets_cache / "mteb___amazon_reviews_multi" / "en" / "1.0.0"
    leaf = _latest_cache_leaf(
        dataset_root,
        required_files=[
            "amazon_reviews_multi-train.arrow",
            "amazon_reviews_multi-validation.arrow",
            "amazon_reviews_multi-test.arrow",
        ],
    )
    out_dir = project_root / "en"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train", "validation": "validation", "test": "test"}
    written: list[Path] = []
    for src_split, dst_split in split_map.items():
        in_file = leaf / f"amazon_reviews_multi-{src_split}.arrow"
        if not in_file.exists():
            raise FileNotFoundError(f"Missing cached file: {in_file}")
        table = _read_arrow_table(in_file)

        ids = table.column("id").to_pylist()
        texts = table.column("text").to_pylist()
        labels = table.column("label").to_pylist()
        label_texts = table.column("label_text").to_pylist()

        out_file = out_dir / f"{dst_split}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for i in range(len(ids)):
                row = {
                    "id": str(ids[i]),
                    "text": texts[i],
                    "label": int(labels[i]),
                    "label_text": str(label_texts[i]),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written.append(out_file)
        print(f"Wrote {out_file} rows={len(ids)}")
    return written


def _verify_offline_loads():
    import datasets

    checks = [
        ("mteb/amazon_counterfactual", "en"),
        ("mteb/amazon_counterfactual", "en-ext"),
        ("mteb/amazon_reviews_multi", "en"),
    ]
    for name, config in checks:
        ds = datasets.load_dataset(name, config, trust_remote_code=True)
        for split in ("train", "validation", "test"):
            if split not in ds:
                raise RuntimeError(f"{name}:{config} missing split {split}")
        print(f"Verified offline load: {name}:{config}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Project root where local relative files must exist.",
    )
    parser.add_argument(
        "--datasets-cache",
        type=Path,
        default=REPO_ROOT / ".hf_cache" / "datasets",
        help="HF datasets cache root containing mteb___* folders.",
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Also run offline datasets.load_dataset checks for the prepared datasets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    datasets_cache = args.datasets_cache.resolve()
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)

    written = []
    written.extend(_materialize_amazon_counterfactual(project_root, datasets_cache))
    written.extend(_materialize_amazon_reviews_multi(project_root, datasets_cache))
    print(f"Prepared {len(written)} local dataset files.")

    if args.verify_load:
        _verify_offline_loads()


if __name__ == "__main__":
    main()
