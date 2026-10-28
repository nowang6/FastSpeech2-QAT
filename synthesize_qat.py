#!/usr/bin/env python3
"""synthesize_qat.py — 量化模型端到端推理（text → wav）

支持自由组合：
    --acoustic-ckpt  QAT FastSpeech2 checkpoint（缺省用 FP32 预训练权重）
    --vocoder-ckpt   量化 vocoder checkpoint（缺省用 FP32 universal）
即: FP32/FP32 | QAT/FP32 | QAT/QAT 三种对照。

Usage:
    python synthesize_qat.py --text "欢迎使用语音合成" \
        --acoustic-ckpt output/qat_ckpt/AISHELL3/50000.pth.tar \
        --vocoder-ckpt output/qat_vocoder/vocoder_quantized.pt \
        --output-dir output/qat_result/demo
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from pypinyin import pinyin, Style

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", type=str, required=True, help="中文文本")
    p.add_argument("-p", "--preprocess_config", default=str(ROOT / "config/AISHELL3/preprocess.yaml"))
    p.add_argument("-m", "--model_config", default=str(ROOT / "config/AISHELL3/model.yaml"))
    p.add_argument("--acoustic-ckpt", type=str, default=None,
                   help="QAT FastSpeech2 ckpt；缺省加载 FP32 预训练权重")
    p.add_argument("--vocoder-ckpt", type=str, default=None,
                   help="量化 vocoder ckpt；缺省用 FP32 universal")
    p.add_argument("--teacher-ckpt", type=str,
                   default=str(ROOT / "pretrain/AISHELL3_600000.pth.tar"))
    p.add_argument("--speaker", type=int, default=0, help="speaker id（speakers.json）")
    p.add_argument("--output-dir", type=str, default=str(ROOT / "output/qat_result"))
    p.add_argument("--tag", type=str, default="qat")
    return p.parse_args()


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            if not temp or not temp[0]:
                continue
            lexicon.setdefault(temp[0].lower(), temp[1:])
    return lexicon


def preprocess_mandarin(text, preprocess_config):
    """中文文本 -> 音素 ID 序列（pypinyin + lexicon）"""
    from text import text_to_sequence

    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    phones = []
    pinyins = [
        p[0] for p in pinyin(text, style=Style.TONE3, strict=False,
                             neutral_tone_with_five=True)
    ]
    for p in pinyins:
        phones += lexicon.get(p, ["sp"])

    phones = "{" + " ".join(phones) + "}"
    print("  Raw Text: {}".format(text))
    print("  Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return sequence


def load_acoustic(preprocess_config, model_config, args, device):
    from model.fastspeech2 import FastSpeech2
    from model.fastspeech2_quant import QuanFastSpeech2

    quant_cfg = yaml.load(open(ROOT / "config/AISHELL3/train_qat.yaml"), Loader=yaml.FullLoader)["quant"]

    if args.acoustic_ckpt:
        model = QuanFastSpeech2(
            preprocess_config, model_config,
            weight_bits=quant_cfg["weight_bits"],
            input_bits=quant_cfg["input_bits"],
            weight_quant_method=quant_cfg["weight_quant_method"],
            input_quant_method=quant_cfg["input_quant_method"],
            quant_modules=tuple(quant_cfg["quant_modules"]),
        ).to(device)
        ckpt = torch.load(args.acoustic_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Acoustic: QAT ckpt {args.acoustic_ckpt}")
    else:
        model = FastSpeech2(preprocess_config, model_config).to(device)
        ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        print(f"Acoustic: FP32 ckpt {args.teacher_ckpt}")
    model.eval()
    return model


def load_vocoder(model_config, args, device):
    import hifigan
    from hifigan.models_quant import QuantGenerator

    with open(ROOT / "hifigan/config.json") as f:
        vconfig = hifigan.AttrDict(json.load(f))

    if args.vocoder_ckpt:
        vocoder = QuantGenerator(vconfig).to(device)
        for mod in vocoder.modules():
            if hasattr(mod, "weight_bits") and hasattr(mod, "input_bits"):
                mod.weight_bits = 8
                mod.input_bits = 8
        vocoder.remove_weight_norm()  # ckpt 存的是 plain 权重，必须先去 norm
        sd = torch.load(args.vocoder_ckpt, map_location="cpu", weights_only=False)
        vocoder.load_state_dict(sd if "model" not in sd else sd["model"], strict=True)
        print(f"Vocoder: quantized ckpt {args.vocoder_ckpt}")
    else:
        vocoder = hifigan.Generator(vconfig).to(device)
        ckpt = torch.load(ROOT / "hifigan/generator_universal.pth.tar",
                          map_location="cpu", weights_only=False)
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.remove_weight_norm()
        print("Vocoder: FP32 universal")
    vocoder.eval()
    return vocoder


def mel_to_wav(mel, vocoder, preprocess_config, mel_len):
    """mel [T, 80] -> wav int16 np（lengths 语义是采样点数 = mel 帧数 × hop）"""
    from utils.model import vocoder_infer

    lengths = [mel_len * preprocess_config["preprocessing"]["stft"]["hop_length"]]
    with torch.no_grad():
        wav = vocoder_infer(
            mel.transpose(0, 1).unsqueeze(0), vocoder,
            {"vocoder": {"model": "HiFi-GAN"}}, preprocess_config, lengths=lengths,
        )[0]
    return wav


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(open(args.preprocess_config), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config), Loader=yaml.FullLoader)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_acoustic(preprocess_config, model_config, args, device)
    vocoder = load_vocoder(model_config, args, device)

    ids = preprocess_mandarin(args.text, preprocess_config)
    texts = torch.from_numpy(ids).long().unsqueeze(0).to(device)  # [1, T]
    src_lens = torch.tensor([texts.shape[1]], device=device)
    speakers = torch.tensor([args.speaker], device=device)

    with torch.no_grad():
        output = model(speakers, texts, src_lens, texts.shape[1],
                       p_control=1.0, e_control=1.0, d_control=1.0)
    mel_pred = output[1][0].cpu().numpy()  # postnet 输出 [T, 80]
    mel_len = output[9][0].item()
    mel_pred = mel_pred[:mel_len]
    print(f"  mel: {mel_pred.shape}")

    wav = mel_to_wav(torch.from_numpy(mel_pred).float().to(device),
                     vocoder, preprocess_config, mel_len)

    from scipy.io import wavfile
    out_path = output_dir / f"{args.tag}.wav"
    wavfile.write(out_path, preprocess_config["preprocessing"]["audio"]["sampling_rate"], wav)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
