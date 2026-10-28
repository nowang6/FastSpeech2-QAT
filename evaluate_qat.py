#!/usr/bin/env python3
"""evaluate_qat.py — FP32 vs QAT 对比评估

在 val 集上对比：
    1. 六项 FastSpeech2Loss（各自模型）
    2. 逐模块输出差异（encoder / variance_adaptor / decoder / mel_linear /
       postnet），基于带真值 targets 的 forward 保证逐点可比
    3. mel 域 LSD（log-mel RMS 距离）

Usage:
    python evaluate_qat.py \
        --acoustic-ckpt output/qat_ckpt/AISHELL3/50000.pth.tar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

from dataset import Dataset
from model import FastSpeech2Loss
from model.fastspeech2 import FastSpeech2
from model.fastspeech2_quant import QuanFastSpeech2
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODULES = ("encoder", "variance_adaptor", "decoder", "mel_linear", "postnet")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-p", "--preprocess_config", default=str(ROOT / "config/AISHELL3/preprocess.yaml"))
    p.add_argument("-m", "--model_config", default=str(ROOT / "config/AISHELL3/model.yaml"))
    p.add_argument("-t", "--train_config", default=str(ROOT / "config/AISHELL3/train_qat.yaml"))
    p.add_argument("--acoustic-ckpt", type=str, required=True, help="QAT ckpt")
    p.add_argument("--teacher-ckpt", type=str, default=str(ROOT / "pretrain/AISHELL3_600000.pth.tar"))
    p.add_argument("--max-batches", type=int, default=16, help="评估的 val batch 组数")
    return p.parse_args()


def load_models(preprocess_config, model_config, args):
    quant_cfg = yaml.load(open(ROOT / "config/AISHELL3/train_qat.yaml"), Loader=yaml.FullLoader)["quant"]

    teacher = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    teacher.eval()

    student = QuanFastSpeech2(
        preprocess_config, model_config,
        weight_bits=quant_cfg["weight_bits"], input_bits=quant_cfg["input_bits"],
        weight_quant_method=quant_cfg["weight_quant_method"],
        input_quant_method=quant_cfg["input_quant_method"],
        quant_modules=tuple(quant_cfg["quant_modules"]),
    ).to(device)
    ckpt = torch.load(args.acoustic_ckpt, map_location="cpu", weights_only=False)
    student.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    student.eval()
    return teacher, student


def collect_module_outputs(model, batch):
    """带真值 targets 的 forward，hook 收集各模块输出（保证逐点可比）"""
    captured = {}

    def grab(name, pick_first=False):
        def hook(m, inp, out):
            captured[name] = (out[0] if isinstance(out, tuple) and pick_first else out).detach()
        return hook

    hooks = [
        model.encoder.register_forward_hook(grab("encoder")),
        model.variance_adaptor.register_forward_hook(grab("variance_adaptor", pick_first=True)),
        model.decoder.register_forward_hook(grab("decoder", pick_first=True)),
        model.mel_linear.register_forward_hook(grab("mel_linear")),
        model.postnet.register_forward_hook(grab("postnet")),
    ]
    with torch.no_grad():
        output = model(*(batch[2:]))
    for h in hooks:
        h.remove()
    return captured, output


def tensor_diff(a, b):
    """L1 相对误差 + 余弦相似度"""
    l1 = (a - b).abs().mean().item()
    denom = b.abs().mean().item()
    rel = l1 / max(denom, 1e-8)
    va, vb = a.flatten().float(), b.flatten().float()
    cos = torch.nn.functional.cosine_similarity(va, vb, dim=0).item()
    return l1, rel, cos


def main():
    args = parse_args()
    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config), Loader=yaml.FullLoader)

    teacher, student = load_models(preprocess_config, model_config, args)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
    loader = DataLoader(dataset, batch_size=train_config["optimizer"]["batch_size"],
                        shuffle=False, collate_fn=dataset.collate_fn)

    loss_names = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]
    loss_sums = {"fp32": [0.0] * 6, "qat": [0.0] * 6}
    mod_stats = {m: [0.0, 0.0, 0.0] for m in MODULES}  # l1, rel, cos 累加
    n_items, n_batches = 0, 0
    lsd_sum = 0.0

    for batchs in tqdm(loader, desc="evaluate"):
        for batch in batchs:
            batch = to_device(batch, device)
            cap_t, out_t = collect_module_outputs(teacher, batch)
            cap_s, out_s = collect_module_outputs(student, batch)

            for name, outs in (("fp32", out_t), ("qat", out_s)):
                losses = Loss(batch, outs)
                loss_sums[name] = [a + l.item() * len(batch[0]) for a, l in zip(loss_sums[name], losses)]

            for m in MODULES:
                a, b = cap_t[m], cap_s[m]
                if a.shape != b.shape:  # 理论上带真值 targets 时一致
                    continue
                l1, rel, cos = tensor_diff(a, b)
                mod_stats[m][0] += l1 * len(batch[0])
                mod_stats[m][1] += rel * len(batch[0])
                mod_stats[m][2] += cos * len(batch[0])

            mel_t = out_t[1]  # postnet 输出
            mel_s = out_s[1]
            mel_len = mel_t.shape[1]
            lsd = ((mel_t - mel_s) ** 2).mean().sqrt().item()
            lsd_sum += lsd * len(batch[0])

            n_items += len(batch[0])
            n_batches += 1
            if n_batches >= args.max_batches:
                break
        if n_batches >= args.max_batches:
            break

    # 汇总
    report = {"n_samples": n_items}
    for name in ("fp32", "qat"):
        report[name + "_loss"] = {n: s / n_items for n, s in zip(loss_names, loss_sums[name])}
    report["module_diff"] = {
        m: {"l1": v[0] / n_items, "rel_l1": v[1] / n_items, "cosine": v[2] / n_items}
        for m, v in mod_stats.items()
    }
    report["mel_lsd"] = lsd_sum / n_items

    print("\n===== FP32 vs QAT ({}) =====".format(report["n_samples"]))
    print(f"{'loss':<12}{'FP32':>12}{'QAT':>12}{'Δ':>12}")
    for n in loss_names:
        a = report["fp32_loss"][n]
        b = report["qat_loss"][n]
        print(f"{n:<12}{a:>12.4f}{b:>12.4f}{b - a:>+12.4f}")
    print(f"\nmel LSD (fp32 vs qat): {report['mel_lsd']:.4f}")
    print(f"{'module':<18}{'rel_L1':>12}{'cosine':>12}")
    for m in MODULES:
        d = report["module_diff"][m]
        print(f"{m:<18}{d['rel_l1']:>12.4f}{d['cosine']:>12.6f}")

    out_path = ROOT / "output" / "qat_eval" / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
