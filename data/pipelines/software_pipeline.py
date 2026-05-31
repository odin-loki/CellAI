"""
data.pipelines.software_pipeline
=================================
Software / code training data pipeline.

Downloads and preprocesses code corpora from HuggingFace:
    - bigcode/the-stack           ~3 TB (filter to selected languages)
    - code_search_net             ~3 GB
    - codeparrot/codeparrot-clean ~50 GB

Usage:
    cell-ai data --pipeline software --download --languages python,javascript
    cell-ai data --pipeline software --preprocess
    cell-ai data --pipeline software --stats
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SOURCES = {
    "the_stack": {
        "path": "bigcode/the-stack",
        "split": "train",
        "content_field": "content",
        "lang_field": "lang",
    },
    "code_search_net": {
        "path": "code_search_net",
        "split": "train",
        "content_field": "func_code_string",
        "lang_field": "language",
    },
    "codeparrot": {
        "path": "codeparrot/codeparrot-clean",
        "split": "train",
        "content_field": "content",
        "lang_field": None,
    },
}

DEFAULT_LANGUAGES = ["python", "javascript", "rust", "go"]
DEFAULT_SOURCES   = ["code_search_net", "codeparrot"]  # the-stack opt-in (very large)
SHARD_SIZE_MB     = 256
MIN_CODE_LEN      = 30
MAX_CODE_LEN      = 50_000
MAX_AVG_LINE_LEN  = 200  # filter minified files


def _is_quality_code(code: str) -> bool:
    """Heuristic quality filter."""
    if len(code) < MIN_CODE_LEN:
        return False
    lines = code.split("\n")
    if not lines:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    if avg_len > MAX_AVG_LINE_LEN:
        return False
    return True


def download(
    sources: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    raw_dir: Optional[Path] = None,
    max_samples_per_source: Optional[int] = None,
) -> Dict[str, int]:
    """Stream HuggingFace code datasets to raw_dir."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    from data.config import CODE_RAW, ensure_dirs
    ensure_dirs()
    raw_dir = raw_dir or CODE_RAW
    raw_dir.mkdir(parents=True, exist_ok=True)

    sources   = sources   or DEFAULT_SOURCES
    languages = [l.lower() for l in (languages or DEFAULT_LANGUAGES)]

    stats: Dict[str, int] = {}

    for src_name in sources:
        if src_name not in SOURCES:
            logger.warning(f"Unknown source: {src_name}")
            continue

        cfg = SOURCES[src_name]
        logger.info(f"Downloading {src_name} (langs={languages}) ...")

        src_dir = raw_dir / src_name
        src_dir.mkdir(exist_ok=True)

        try:
            ds_kwargs: dict = {"split": cfg["split"], "streaming": True, "trust_remote_code": True}
            dataset = load_dataset(cfg["path"], **ds_kwargs)
        except Exception as e:
            logger.error(f"Failed to load {src_name}: {e}")
            stats[src_name] = 0
            continue

        shard_idx = 0
        count = 0
        shard_path = src_dir / f"shard_{shard_idx:05d}.jsonl"
        shard_file = open(shard_path, "w", encoding="utf-8")
        shard_bytes = 0

        try:
            for sample in dataset:
                code = sample.get(cfg["content_field"], "")
                lang = (sample.get(cfg["lang_field"], "python") or "python").lower()

                if cfg["lang_field"] and lang not in languages:
                    continue
                if not isinstance(code, str) or not _is_quality_code(code):
                    continue

                code = code[:MAX_CODE_LEN]
                record = json.dumps({"content": code, "language": lang, "source": src_name}, ensure_ascii=False)
                shard_file.write(record + "\n")
                shard_bytes += len(record.encode())
                count += 1

                if shard_bytes >= SHARD_SIZE_MB * 1024 * 1024:
                    shard_file.close()
                    shard_idx += 1
                    shard_path = src_dir / f"shard_{shard_idx:05d}.jsonl"
                    shard_file = open(shard_path, "w", encoding="utf-8")
                    shard_bytes = 0
                    logger.info(f"  {src_name}: {count:,} samples, shard {shard_idx}")

                if max_samples_per_source and count >= max_samples_per_source:
                    break
        finally:
            shard_file.close()

        logger.info(f"  {src_name}: {count:,} samples → {src_dir}")
        stats[src_name] = count

    return stats


def preprocess(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    languages: Optional[List[str]] = None,
) -> int:
    """Filter, deduplicate, and write per-language processed JSONL."""
    from data.config import CODE_RAW, CODE_PROCESSED, ensure_dirs
    ensure_dirs()
    raw_dir = raw_dir or CODE_RAW
    processed_dir = processed_dir or CODE_PROCESSED
    processed_dir.mkdir(parents=True, exist_ok=True)

    languages = [l.lower() for l in (languages or DEFAULT_LANGUAGES)]
    lang_files = {lang: open(processed_dir / f"{lang}.jsonl", "w", encoding="utf-8") for lang in languages}
    seen: Dict[str, set] = {lang: set() for lang in languages}
    counts: Dict[str, int] = {lang: 0 for lang in languages}

    try:
        for shard in sorted(raw_dir.rglob("*.jsonl")):
            with open(shard, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    lang = record.get("language", "python").lower()
                    if lang not in languages:
                        continue
                    code = record.get("content", "")
                    if not _is_quality_code(code):
                        continue
                    h = hash(code[:300])
                    if h in seen[lang]:
                        continue
                    seen[lang].add(h)
                    lang_files[lang].write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[lang] += 1
    finally:
        for f in lang_files.values():
            f.close()

    total = sum(counts.values())
    for lang, n in counts.items():
        logger.info(f"  {lang}: {n:,} samples")
    return total


def stats(raw_dir: Optional[Path] = None, processed_dir: Optional[Path] = None) -> Dict:
    from data.config import CODE_RAW, CODE_PROCESSED
    raw_dir = raw_dir or CODE_RAW
    processed_dir = processed_dir or CODE_PROCESSED
    raw_files = list(raw_dir.rglob("*.jsonl")) if raw_dir.exists() else []
    proc_files = list(processed_dir.glob("*.jsonl")) if processed_dir.exists() else []

    def count_lines(p: Path) -> int:
        try:
            return sum(1 for _ in open(p, encoding="utf-8"))
        except Exception:
            return 0

    return {
        "raw_shards": len(raw_files),
        "raw_total_mb": sum(f.stat().st_size for f in raw_files) / 1e6,
        "processed_by_language": {f.stem: count_lines(f) for f in proc_files},
    }
