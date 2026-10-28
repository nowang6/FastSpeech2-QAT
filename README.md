# FastSpeech2-QAT

FastSpeech2 + HiFi-GAN vocoder 的**端到端量化感知训练**项目：整个模型
（encoder / variance_adaptor / decoder / mel_linear / postnet / speaker_emb）
同时量化，用标准 FastSpeech2Loss 在真实数据上联合微调，vocoder 同步量化，
实现 text→wav 全链路量化推理。

- 底模：AISHELL3 多说话人中文（`pretrain/AISHELL3_600000.pth.tar`，600k 步 FP32）
- 量化方案：`vox`（权重 per-tensor 对称）/ `vox_input`（激活可学习 `scale_in`），默认 8/8 bit
- 训练数据：AISHELL3，77,437 train / 512 val（本机流水线生成）

> 📄 完整文档（技术方案 / 数据流水线 / 验证结果 / 已知问题）：
> **[docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)**

## 环境

```bash
uv venv --python 3.10 --system-site-packages .venv
source .venv/bin/activate
# RTX 50 系 (sm_120) 需要 cu128 torch：
uv pip install "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

## 数据准备（一次性，约 4-6 小时）

本仓库不含语料与声学特征，需要现场生成：

```bash
# 1. parquet -> 结构化语料（wav + content.txt，拼音查表自 data_refs/AISHELL3/train.txt）
python prepare_corpus.py

# 2. 生成 raw wav(22050) 与 .lab
python prepare_align.py config/AISHELL3/preprocess.yaml

# 3. MFA 分批对齐 + 特征提取 + speakers.json 对齐（micromamba 环境见文档末尾）
bash scripts/prepare_data_batched.sh

# 4. vocoder 蒸馏数据（mel -> teacher wav 片段）
python prepare_vocoder_data.py --max-samples 2000
```

大体积中间产物（raw wav / 特征 / TextGrid / MFA 模型）放共享存储
`/data/datasets/FastSpeech2-QAT/`，`preprocessed_data` 是指向它的 symlink。

## 端到端 QAT

```bash
# FastSpeech2 整模型量化微调（标定 + 训练一条龙）
python train_qat.py -p config/AISHELL3/preprocess.yaml \
                    -m config/AISHELL3/model.yaml \
                    -t config/AISHELL3/train_qat.yaml \
                    --stage all --total-steps 50000

# 分阶段：--stage calibrate 仅标定；--stage train --restore-qat <ckpt> 续训

# vocoder QAT（波形 L1 蒸馏）
python train_vocoder_qat.py --epochs 30
```

常用参数：`--weight-bits/--input-bits`（位数）、`--quant-modules`（子集量化，
如 `encoder,decoder`）、`--calib-samples`、`--lr`、`--total-steps`。

## 评估与合成

```bash
# FP32 vs QAT 对比（六项 loss / 逐模块相似度 / mel LSD）
python evaluate_qat.py --acoustic-ckpt output/qat_ckpt/AISHELL3/50000.pth.tar

# 全量化合成（QAT 声学模型 + 量化 vocoder）
python synthesize_qat.py --text "欢迎使用端到端量化语音合成" \
    --acoustic-ckpt output/qat_ckpt/AISHELL3/50000.pth.tar \
    --vocoder-ckpt output/qat_vocoder/vocoder_quantized.pt \
    --tag qat_full

# FP32 对照（缺省即 FP32 路径）
python synthesize_qat.py --text "欢迎使用端到端量化语音合成" --tag fp32
```

## 关键说明

- **scale_in 是 `nn.Parameter`**：随 state_dict 保存/恢复；`--stage calibrate` 单独
  产出 `calibrated.pth.tar`（仅标定不训练）。
- **量化 vocoder checkpoint 是去 weight_norm 的 plain 权重**：加载前必须先
  `remove_weight_norm()`（`synthesize_qat.py` 已处理）。
- **`QuantizeConv1d` 类默认 4-bit**：构建 vocoder student 时显式传 8-bit
  （`train_vocoder_qat.py` 已处理）。
- 32-bit 激活层（`input_bits>=32`）在标定时跳过 `scale_in`——写入
  `max/2^31` 会得到毁灭性小步长。
- `data_refs/AISHELL3/` 是预训练模型配套的基准数据（音素级拼音 train.txt、
  speaker 行序绑定的 speakers.json）：`prepare_corpus.py` 的拼音查表和流水线的
  speakers.json 对齐都以此为准，不可重新生成替换。

## 目录结构

```
train_qat.py            # 整模型端到端 QAT（核心）
train_vocoder_qat.py    # vocoder QAT（标定 + 波形 L1 蒸馏）
evaluate_qat.py         # FP32 vs QAT 对比评估
synthesize_qat.py       # text -> wav 端到端推理（QAT/FP32 任意组合）
prepare_corpus.py       # AISHELL-3 parquet -> 结构化语料
prepare_align.py        # 语料 -> raw wav + .lab
prepare_vocoder_data.py # mel -> teacher wav 蒸馏对
model/fastspeech2_quant.py  # QuanFastSpeech2 全模块量化模型
utils/utils_quant.py        # 量化器（vox / vox_input / symmetric）
config/AISHELL3/            # preprocess / model / train / train_qat(量化配置)
data_refs/AISHELL3/         # 预训练配套基准数据（拼音查表 + speaker 行序）
```

## MFA 环境（独立于项目 .venv）

```bash
# micromamba + conda-forge（MFA 3.x 的 kaldi 绑定只在 conda-forge 提供）
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/bin --strip-components=1 bin/micromamba
MAMBA_ROOT_PREFIX=~/micromamba ~/bin/micromamba create -y -n mfa -c conda-forge montreal-forced-aligner
```
