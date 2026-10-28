#!/usr/bin/env python3
"""train_qat.py — FastSpeech2 整模型端到端量化感知训练（QAT）

对整个模型（encoder / variance_adaptor / decoder / mel_linear / postnet /
speaker_emb）同时量化，用标准 FastSpeech2Loss 在真实数据上联合微调。

流程:
    1. 加载 FP32 预训练 FastSpeech2（teacher，提供初始权重）
    2. 构建全模块量化的 QuanFastSpeech2 并拷贝 teacher 权重
    3. calibrate 阶段: 用训练集文本前向，p99 分位初始化各层 scale_in
    4. qat 阶段: FastSpeech2Loss + Adam 恒定学习率微调（scale_in 可学习）

Usage:
    python train_qat.py -p config/AISHELL3/preprocess.yaml \
                        -m config/AISHELL3/model.yaml \
                        -t config/AISHELL3/train_qat.yaml \
                        --stage all --total-steps 50          # 冒烟
    python train_qat.py ... --stage all                      # 全量
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent

from dataset import Dataset
from evaluate import evaluate
from model import FastSpeech2Loss
from model.fastspeech2 import FastSpeech2
from model.fastspeech2_quant import QuanFastSpeech2
from utils.model import get_vocoder, get_param_num
from utils.tools import log, synth_one_sample, to_device
from utils.utils_quant import QuantizeConv1d, QuantizeLinear, QuantizeConvTranspose1d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_TEACHER_CKPT = ROOT / "pretrain" / "AISHELL3_600000.pth.tar"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-p", "--preprocess_config", required=True)
    parser.add_argument("-m", "--model_config", required=True)
    parser.add_argument("-t", "--train_config", required=True)
    parser.add_argument("--stage", choices=("calibrate", "train", "all"), default="all")
    parser.add_argument("--teacher-ckpt", type=str, default=str(DEFAULT_TEACHER_CKPT),
                        help="FP32 预训练 checkpoint（提供初始权重）")
    parser.add_argument("--restore-qat", type=str, default=None,
                        help="已有 QAT checkpoint（{'model': state_dict}），跳过 calibrate/teacher 加载")
    parser.add_argument("--weight-bits", type=int, default=None)
    parser.add_argument("--input-bits", type=int, default=None)
    parser.add_argument("--weight-quant-method", type=str, default=None,
                        choices=("vox", "symmetric"))
    parser.add_argument("--input-quant-method", type=str, default=None,
                        choices=("vox_input", "symmetric"))
    parser.add_argument("--quant-modules", type=str, default=None,
                        help="逗号分隔，默认 yaml quant.quant_modules 全模块")
    parser.add_argument("--calib-samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--log-steps", type=int, default=None)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--no-vocoder-synth", action="store_true",
                        help="训练中不加载 vocoder 出音频样例（省显存）")
    return parser.parse_args()


def pick(override, yaml_val, default):
    """CLI 参数 > yaml > 默认值"""
    return override if override is not None else (yaml_val if yaml_val is not None else default)


def build_quant_model(preprocess_config, model_config, args, quant_cfg):
    quant_modules = (
        tuple(args.quant_modules.split(",")) if args.quant_modules
        else tuple(quant_cfg["quant_modules"])
    )
    kwargs = dict(
        weight_bits=pick(args.weight_bits, quant_cfg.get("weight_bits"), 8),
        input_bits=pick(args.input_bits, quant_cfg.get("input_bits"), 8),
        weight_quant_method=pick(args.weight_quant_method, quant_cfg.get("weight_quant_method"), "vox"),
        input_quant_method=pick(args.input_quant_method, quant_cfg.get("input_quant_method"), "vox_input"),
        quant_modules=quant_modules,
    )
    print("Quant config:", kwargs)

    teacher = FastSpeech2(preprocess_config, model_config).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    print(f"Loaded FP32 teacher from {args.teacher_ckpt}")

    quant = QuanFastSpeech2(preprocess_config, model_config, **kwargs).to(device)
    qdict = quant.state_dict()
    tdict = {k: v for k, v in teacher.state_dict().items() if k in qdict}
    quant.load_state_dict(tdict, strict=False)
    non_scale = [k for k in qdict if k not in tdict and not k.endswith("scale_in")]
    assert not non_scale, f"unexpected missing keys: {non_scale[:8]}"
    print(f"Copied {len(tdict)}/{len(qdict)} keys from teacher "
          f"({len(qdict) - len(tdict)} new scale_in params)")
    return quant


def calibrate_scale_in(model, loader, calib_samples):
    """p99 分位标定所有量化层 scale_in"""
    quant_layers = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (QuantizeLinear, QuantizeConv1d, QuantizeConvTranspose1d))
    ]
    stats = {name: [] for name, _ in quant_layers}

    def make_hook(name):
        def hook(module, inp, out):
            stats[name].append(inp[0].detach().abs().max().item())
        return hook

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in quant_layers]

    model.eval()
    n_done = 0
    with torch.no_grad():
        for batchs in tqdm(loader, desc="calibrate"):
            for batch in batchs:
                batch = to_device(batch, device)
                speakers, texts, src_lens = batch[2], batch[3], batch[4]
                model(speakers, texts, src_lens, int(batch[5]))
                n_done += len(batch[0])
                if n_done >= calib_samples:
                    break
            if n_done >= calib_samples:
                break
    for h in hooks:
        h.remove()

    print(f"Calibrated {len(quant_layers)} layers with {n_done} samples (p99):")
    for name, module in quant_layers:
        if module.input_bits >= 32:
            # 32-bit 激活不量化；写 max/2^31 会得到毁灭性小的 scale_in
            continue
        if not stats[name]:
            continue
        max_abs = float(torch.quantile(torch.tensor(stats[name], dtype=torch.float32), 0.99))
        maxq = 2 ** (module.input_bits - 1) - 1
        if max_abs > 0:
            module.scale_in.data.fill_(max_abs / maxq)
    print("  first 3 layers:", [(n, round(m.scale_in.item(), 5)) for n, m in quant_layers[:3]])


def main(args, configs):
    preprocess_config, model_config, train_config = configs
    quant_cfg = train_config.get("quant", {})

    # Dataset
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size * group_size,
                        shuffle=True, collate_fn=dataset.collate_fn)

    # Model
    if args.restore_qat:
        quant = QuanFastSpeech2(
            preprocess_config, model_config,
            weight_bits=pick(args.weight_bits, quant_cfg.get("weight_bits"), 8),
            input_bits=pick(args.input_bits, quant_cfg.get("input_bits"), 8),
            weight_quant_method=pick(args.weight_quant_method, quant_cfg.get("weight_quant_method"), "vox"),
            input_quant_method=pick(args.input_quant_method, quant_cfg.get("input_quant_method"), "vox_input"),
            quant_modules=tuple(args.quant_modules.split(",")) if args.quant_modules
            else tuple(quant_cfg["quant_modules"]),
        ).to(device)
        ckpt = torch.load(args.restore_qat, map_location="cpu", weights_only=False)
        quant.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Restored QAT model from {args.restore_qat}")
    else:
        quant = build_quant_model(preprocess_config, model_config, args, quant_cfg)
    quant.train()
    print("Number of FastSpeech2 Parameters:", get_param_num(quant))

    # Calibrate
    if args.stage in ("calibrate", "all") and not args.restore_qat:
        calibrate_scale_in(quant, loader, pick(args.calib_samples, quant_cfg.get("calib_samples"), 256))

    if args.stage == "calibrate":
        out = train_config["path"]["ckpt_path"]
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, "calibrated.pth.tar")
        torch.save({"model": quant.state_dict()}, path)
        print(f"Saved calibrated model to {path}")
        return

    # Loss / vocoder / logger
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = None
    if not args.no_vocoder_synth:
        vocoder = get_vocoder(model_config, device)

    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Optimizer: QAT 微调用恒定学习率 Adam（预训练已完成 Noam 预热阶段）
    opt_cfg = train_config["optimizer"]
    optimizer = torch.optim.Adam(
        quant.parameters(),
        lr=pick(args.lr, opt_cfg.get("lr"), 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.98])),
        eps=opt_cfg.get("eps", 1e-8),
        weight_decay=opt_cfg.get("weight_decay", 0.0),
    )
    grad_clip_thresh = opt_cfg.get("grad_clip_thresh", 1.0)

    total_step = pick(args.total_steps, train_config["step"].get("total_step"), 50000)
    log_step = pick(args.log_steps, train_config["step"].get("log_step"), 10)
    save_step = pick(args.save_steps, train_config["step"].get("save_step"), 5000)
    val_step = pick(args.val_steps, train_config["step"].get("val_step"), 1000)
    synth_step = train_config["step"].get("synth_step", 1000)

    step = 1
    epoch = 1
    outer_bar = tqdm(total=total_step, desc="QAT Training", position=0)
    while step <= total_step:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                output = quant(*(batch[2:]))
                losses = Loss(batch, output)
                total_loss = losses[0]

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(quant.parameters(), grad_clip_thresh)
                optimizer.step()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(*losses)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")
                    outer_bar.write(message1 + message2)
                    log(train_logger, step, losses=losses)

                if step % synth_step == 0 and vocoder is not None:
                    fig, wav_recon, wav_pred, tag = synth_one_sample(
                        batch, output, vocoder, model_config, preprocess_config)
                    log(train_logger, step, fig=fig, tag=f"Training/step_{step}_{tag}")
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    log(train_logger, audio=wav_recon, sampling_rate=sampling_rate,
                        tag=f"Training/step_{step}_{tag}_reconstructed")
                    log(train_logger, audio=wav_pred, sampling_rate=sampling_rate,
                        tag=f"Training/step_{step}_{tag}_synthesized")

                if step % val_step == 0:
                    quant.eval()
                    message = evaluate(quant, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)
                    quant.train()

                if step % save_step == 0 or step == total_step:
                    torch.save(
                        {"model": quant.state_dict(), "step": step},
                        os.path.join(train_config["path"]["ckpt_path"], f"{step}.pth.tar"),
                    )

                if step == total_step:
                    outer_bar.close()
                    return
                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    args = parse_args()
    configs = tuple(
        yaml.load(open(path, "r"), Loader=yaml.FullLoader)
        for path in (args.preprocess_config, args.model_config, args.train_config)
    )
    main(args, configs)
