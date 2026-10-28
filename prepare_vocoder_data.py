#!/usr/bin/env python3
"""prepare_vocoder_data.py — 生成 vocoder QAT 蒸馏数据

从 preprocessed_data/AISHELL3/mel/ 抽样 ground-truth mel（与 FastSpeech2
训练目标同域），用 FP32 teacher HiFi-GAN 前向生成目标波形，切成随机训练
片段缓存为 pt 文件。训练时 train_vocoder_qat.py 只需加载该文件。

输出格式: {"segments": [(mel [1,80,32], wav [1,1,8192]), ...],
           "segment_size": 8192, "segment_mel": 32, "num_samples": N}

Usage:
    python prepare_vocoder_data.py --max-samples 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

MEL_DIR = ROOT / "preprocessed_data" / "AISHELL3" / "mel"
VOCODER_DATA_PATH = Path("/data/datasets/FastSpeech2-QAT/vocoder_data.pt")
VOCODER_CONFIG_PATH = ROOT / "hifigan" / "config.json"
VOCODER_CKPT_PATH = ROOT / "hifigan" / "generator_universal.pth.tar"

HOP = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-samples", type=int, default=2000,
                   help="抽样的 mel 条数")
    p.add_argument("--segment-size", type=int, default=8192,
                   help="训练片段的音频采样点数（=32 mel frames @hop=256）")
    p.add_argument("--segments-per-sample", type=int, default=4,
                   help="每条 mel 切出的训练片段数")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_teacher_vocoder(device):
    from hifigan import AttrDict as HifiConfig
    from hifigan.models import Generator

    with open(VOCODER_CONFIG_PATH) as f:
        config = HifiConfig(json.load(f))
    vocoder = Generator(config).to(device)
    ckpt = torch.load(VOCODER_CKPT_PATH, map_location="cpu", weights_only=False)
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.requires_grad_(False)
    print(f"Teacher vocoder loaded from {VOCODER_CKPT_PATH.name}")
    return vocoder


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if VOCODER_DATA_PATH.exists() and not args.force:
        print(f"Data already exists: {VOCODER_DATA_PATH} (use --force to regenerate)")
        return

    mel_files = sorted(MEL_DIR.glob("*.npy"))
    if not mel_files:
        raise FileNotFoundError(
            f"No mel features under {MEL_DIR}. Run preprocess.py first.")
    random.seed(0)
    random.shuffle(mel_files)
    mel_files = mel_files[: args.max_samples]
    print(f"Sampling {len(mel_files)} mel files from {MEL_DIR}")

    vocoder = load_teacher_vocoder(device)

    seg_audio = args.segment_size
    seg_mel = seg_audio // HOP
    all_segments = []

    for mel_path in tqdm(mel_files, desc="vocoder data"):
        mel = np.load(mel_path)  # [T, 80], log-mel 域（模型训练目标同域）
        T = mel.shape[0]
        if T < seg_mel + 4:
            continue
        mel = torch.from_numpy(mel).float()
        starts = torch.randperm(T - seg_mel)[: args.segments_per_sample]
        for s in starts:
            mel_seg = mel[s : s + seg_mel].t()  # [80, 32]（无 batch 维，训练时 stack）
            with torch.no_grad():
                wav = vocoder(mel_seg.unsqueeze(0).to(device))  # [1, 1, 8192]
            all_segments.append((mel_seg, wav.squeeze(0).cpu()))  # wav [1, 8192]

    VOCODER_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "segments": all_segments,
            "segment_size": seg_audio,
            "segment_mel": seg_mel,
            "num_samples": len(mel_files),
        },
        VOCODER_DATA_PATH,
    )
    print(f"Saved {len(all_segments)} segments "
          f"({len(mel_files)} mels) -> {VOCODER_DATA_PATH}")


if __name__ == "__main__":
    main()
