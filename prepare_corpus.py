#!/usr/bin/env python3
"""prepare_corpus.py — 从 AISHELL-3 parquet 导出结构化语料供 prepare_align.py 使用

输出目录结构（与 preprocessor/aishell3.py 的 prepare_align 期望一致）:
    {out_dir}/{split}/content.txt                  # "SSB00050001.wav\t0 guang3 0 zhou1 ..."
    {out_dir}/{split}/wav/{speaker}/{name}.wav     # 原始 44.1kHz wav 二进制

拼音来源：以 data_refs/AISHELL3/train.txt（音素级拼音，声母/韵母分开，与
text/symbols.py 词表一致）为权威，按 basename 查表。parquet 自带的音节级
pinyin 字段不直接使用（格式与词表不符）。

content.txt 第二列写成 "0 <py> 0 <py> ..." 交错格式，是因为 aishell3.py 用
text.split(" ")[1::2] 取拼音，奇数位放占位符 "0" 即可完整取出拼音序列。

Usage:
    python prepare_corpus.py --split train --limit 500   # 小规模冒烟
    python prepare_corpus.py                             # 全量
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

PARQUET_DIR = Path("/data/datasets/AISHELL-3/data")
DATA_REF_DIR = Path(__file__).resolve().parent / "data_refs" / "AISHELL3"
DEFAULT_OUT = Path("/data/datasets/AISHELL-3-structured")


def load_pinyin_lookup() -> dict[str, str]:
    """basename -> 音素级拼音序列（空格分隔），来自 data_refs 基准数据"""
    lookup = {}
    for name in ("train.txt", "val.txt"):
        path = DATA_REF_DIR / name
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip("\n").split("|")
                if len(parts) < 3:
                    continue
                basename, _, phones = parts[0], parts[1], parts[2]
                lookup[basename] = phones.strip("{}").strip()
    return lookup


def export_split(split: str, out_dir: Path, lookup: dict, limit: int | None):
    files = sorted(glob.glob(str(PARQUET_DIR / f"{split}-*.parquet")))
    if not files:
        print(f"[WARN] no parquet files for split={split}")
        return

    wav_dir = out_dir / split / "wav"
    content_path = out_dir / split / "content.txt"
    content_path.parent.mkdir(parents=True, exist_ok=True)
    n_written, n_skipped = 0, 0

    with open(content_path, "w", encoding="utf-8") as content_f:
        for fp in tqdm(files, desc=f"export {split}"):
            table = pq.read_table(fp)
            for item, text in zip(
                table["audio"].to_pylist(), table["text"].to_pylist()
            ):
                wav_name = item["path"]  # e.g. SSB00050001.wav
                basename = wav_name.rsplit(".", 1)[0]
                speaker = basename[:7]

                phones = lookup.get(basename)
                if phones is None:
                    n_skipped += 1
                    continue

                # 交错占位格式，配合 aishell3.py 的 [1::2] 解析
                interleave = " ".join(
                    tok for p in phones.split() for tok in ("0", p)
                )
                content_f.write(f"{wav_name}\t{interleave}\n")

                speaker_dir = wav_dir / speaker
                if not (speaker_dir / wav_name).exists():
                    speaker_dir.mkdir(parents=True, exist_ok=True)
                    (speaker_dir / wav_name).write_bytes(item["bytes"])
                n_written += 1

                if limit and n_written >= limit:
                    break
            if limit and n_written >= limit:
                break

    print(f"[{split}] written={n_written}, skipped(no pinyin lookup)={n_skipped}")
    print(f"  content: {content_path}")
    print(f"  wav dir: {wav_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=None,
                        help="每个 split 最多导出多少条（冒烟测试用）")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    lookup = load_pinyin_lookup()
    print(f"pinyin lookup entries: {len(lookup)}")

    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        export_split(split, out_dir, lookup, args.limit)


if __name__ == "__main__":
    main()
