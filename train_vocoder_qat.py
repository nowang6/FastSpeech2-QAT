#!/usr/bin/env python3
"""train_vocoder_qat.py — HiFi-GAN vocoder 端到端量化（QAT 蒸馏）

teacher = FP32 HiFi-GAN universal（hifigan/generator_universal.pth.tar）
student = QuantGenerator（vox / vox_input，默认 8/8 bit）

流程:
    1. teacher 去除 weight_norm，拷贝 plain 权重到 student
    2. p99 分位标定各层 scale_in
    3. 波形 L1 蒸馏微调（scale_in 可学习）
    4. 保存 plain state_dict（加载侧需先 remove_weight_norm）

Usage:
    python prepare_vocoder_data.py                     # 先生成蒸馏数据
    python train_vocoder_qat.py --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils_quant import QuantizeConv1d, QuantizeConvTranspose1d

ROOT = Path(__file__).resolve().parent

VOCODER_CONFIG_PATH = ROOT / "hifigan" / "config.json"
VOCODER_CKPT_PATH = ROOT / "hifigan" / "generator_universal.pth.tar"
VOCODER_DATA_PATH = Path("/data/datasets/FastSpeech2-QAT/vocoder_data.pt")
OUTPUT_DIR = ROOT / "output" / "qat_vocoder"
CHECKPOINT_PATH = OUTPUT_DIR / "vocoder_quantized.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weight-bits", type=int, default=8)
    p.add_argument("--input-bits", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--train-count", type=int, default=100000)
    p.add_argument("--val-count", type=int, default=256)
    p.add_argument("--data-path", type=str, default=str(VOCODER_DATA_PATH))
    p.add_argument("--force-init", action="store_true",
                   help="忽略已有 checkpoint 从头训练")
    return p.parse_args()


# ============================================================
# Model loading (teacher plain weights -> student)
# ============================================================

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


def create_student_vocoder(weight_bits, input_bits, device):
    from hifigan import AttrDict as HifiConfig
    from hifigan.models_quant import QuantGenerator

    with open(VOCODER_CONFIG_PATH) as f:
        config = HifiConfig(json.load(f))
    student = QuantGenerator(config).to(device)
    # QuantizeConv1d 类默认 4-bit，必须显式覆盖
    for mod in student.modules():
        if isinstance(mod, (QuantizeConv1d, QuantizeConvTranspose1d)):
            mod.weight_bits = weight_bits
            mod.input_bits = input_bits
    student.eval()
    student.requires_grad_(False)
    return student


def init_student_from_teacher(teacher, student):
    """teacher 已去 weight_norm；student 先去 norm 再 load plain 权重"""
    student.remove_weight_norm()

    teacher_sd = teacher.state_dict()
    new_sd = {}
    for k, v in student.state_dict().items():
        if k in teacher_sd and teacher_sd[k].shape == v.shape:
            new_sd[k] = teacher_sd[k].clone()
        elif k.endswith("scale_in"):
            new_sd[k] = v  # 保留 student 的 scale_in（随后标定）
        else:
            print(f"  Warning: '{k}' not in teacher, keeping student init")
            new_sd[k] = v
    student.load_state_dict(new_sd, strict=False)


def quant_modules_of(model):
    return [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (QuantizeConv1d, QuantizeConvTranspose1d))
    ]


def calibrate_scale_in(student, segments, device):
    student.eval()
    quant_mods = quant_modules_of(student)
    input_stats = {name: [] for name, _ in quant_mods}

    def make_hook(name):
        def hook(module, inp, out):
            input_stats[name].append(inp[0].detach().abs().max().item())
        return hook

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in quant_mods]
    n_calib = min(200, len(segments))
    print(f"  Calibrating scale_in with {n_calib} segments...")
    with torch.no_grad():
        for i in range(n_calib):
            mel_seg, _ = segments[i]
            student(mel_seg.to(device))
    for h in hooks:
        h.remove()

    for name, mod in quant_mods:
        stats = input_stats[name]
        if not stats:
            continue
        max_abs = float(torch.quantile(torch.tensor(stats), 0.99))
        maxq = 2 ** (mod.input_bits - 1) - 1
        if max_abs > 0:
            mod.scale_in.data.fill_(max_abs / maxq)
    print(f"  scale_in calibrated for {len(quant_mods)} layers, "
          f"e.g. {quant_mods[0][0]}={quant_mods[0][1].scale_in.item():.6f}")


# ============================================================
# Loss / train
# ============================================================

def compute_loss(student, segments, device, batch_size):
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            mels = torch.stack([s[0] for s in batch]).to(device)
            targets = torch.stack([s[1] for s in batch]).to(device)
            preds = student(mels).squeeze(1)
            targets = targets.squeeze(1)
            min_t = min(preds.shape[-1], targets.shape[-1])
            total += F.l1_loss(preds[..., :min_t], targets[..., :min_t]).item()
            n += 1
    return total / max(n, 1)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    segments = data["segments"]
    n_train = min(args.train_count, len(segments) - args.val_count)
    train_segs, val_segs = segments[:n_train], segments[n_train : n_train + args.val_count]
    print(f"Segments: train={len(train_segs)}, val={len(val_segs)}")

    teacher = load_teacher_vocoder(device)
    student = create_student_vocoder(args.weight_bits, args.input_bits, device)

    if CHECKPOINT_PATH.exists() and not args.force_init:
        sd = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        student.remove_weight_norm()
        student.load_state_dict(sd, strict=True)
        print(f"Resumed from {CHECKPOINT_PATH.name}")
    else:
        init_student_from_teacher(teacher, student)
        calibrate_scale_in(student, train_segs, device)

    student.train()
    student.requires_grad_(True)
    params = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0)

    best = compute_loss(student, val_segs, device, args.batch_size)
    print(f"Initial val L1: {best:.6f}")
    save(student)

    for epoch in range(args.epochs):
        indices = list(range(len(train_segs)))
        random.shuffle(indices)
        pbar = tqdm(range(0, len(indices) - args.batch_size, args.batch_size),
                    desc=f"epoch {epoch+1}/{args.epochs}")
        for i in pbar:
            batch = [train_segs[j] for j in indices[i : i + args.batch_size]]
            mels = torch.stack([s[0] for s in batch]).to(device)
            targets = torch.stack([s[1] for s in batch]).squeeze(1).to(device)

            optimizer.zero_grad(set_to_none=True)
            preds = student(mels).squeeze(1)
            min_t = min(preds.shape[-1], targets.shape[-1])
            loss = F.l1_loss(preds[..., :min_t], targets[..., :min_t])
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.5f}")

        val = compute_loss(student, val_segs, device, args.batch_size)
        print(f"epoch {epoch+1}: val L1 = {val:.6f} (best {best:.6f})")
        if val < best:
            best = val
            save(student)
            print(f"  saved (new best) -> {CHECKPOINT_PATH.name}")

    print(f"Done. best val L1 = {best:.6f}, ckpt: {CHECKPOINT_PATH}")


def save(student):
    """保存 plain state_dict（无 weight_norm；加载侧需先 remove_weight_norm）"""
    sd = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    torch.save(sd, tmp)
    tmp.replace(CHECKPOINT_PATH)


if __name__ == "__main__":
    main()
