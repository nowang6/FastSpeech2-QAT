# FastSpeech2-QAT 项目文档

> FastSpeech2 + HiFi-GAN 端到端量化感知训练 · 2026-10-28

---

## 1. 项目概述

### 1.1 目标

实现 **FastSpeech2 全模块 + HiFi-GAN vocoder 的端到端量化**（text→wav 全链路）：
整个声学模型同时量化，用标准 FastSpeech2Loss 在真实数据上联合微调，
vocoder 同步量化，最终由量化模型直接完成文本到语音的合成。

### 1.2 技术决策

| 决策项 | 选择 | 说明 |
|---|---|---|
| 量化范围 | FastSpeech2 全 6 模块 + HiFi-GAN vocoder | encoder / variance_adaptor / decoder / mel_linear / postnet / speaker_emb |
| 底模 | AISHELL3 多说话人中文 | `pretrain/AISHELL3_600000.pth.tar`（600k 步 FP32） |
| 量化方案 | vox / vox_input，默认 8/8 bit | 权重 per-tensor 对称；激活带可学习 `scale_in`（硬件对齐） |
| 训练范式 | 全模块联合 QAT | 标准 FastSpeech2Loss 真值监督，非蒸馏 |

### 1.3 设计要点

`QuanFastSpeech2.forward` 输出与 FP32 模型完全同构（10 元组），因此
`FastSpeech2Loss`、`dataset.py`、`evaluate.py`、`synth_one_sample` 可直接复用，
QAT 训练循环只需替换模型类。

---

## 2. 项目结构

```
FastSpeech2-QAT/
├── train_qat.py                  # ★ 核心：整模型端到端 QAT（标定 + 联合微调）
├── train_vocoder_qat.py          # vocoder QAT（去 norm 拷贝 + p99 标定 + 波形 L1 蒸馏）
├── evaluate_qat.py               # FP32 vs QAT 对比（六项 loss / 逐模块相似度 / mel LSD）
├── synthesize_qat.py             # text→wav 推理（QAT/FP32 声学 × QAT/FP32 vocoder 任意组合）
│
├── prepare_corpus.py             # AISHELL-3 parquet → 结构化语料（content.txt + wav）
├── prepare_align.py              # 语料 → raw wav(22050) + .lab
├── prepare_vocoder_data.py       # mel → teacher wav 蒸馏片段
├── preprocess.py                 # 特征提取（mel/pitch/energy/duration）
│
├── model/
│   ├── fastspeech2.py            # FP32 模型
│   ├── fastspeech2_quant.py      # QuanFastSpeech2 全模块量化模型
│   ├── modules.py / loss.py / optimizer.py
├── transformer/                  # Encoder / Decoder / PostNet
├── audio/                        # STFT（librosa>=0.10 keyword-only API）
├── text/                         # 中文音素词表 + text_to_sequence
├── preprocessor/                 # preprocessor.py + aishell3.py
├── utils/
│   ├── utils_quant.py            # ★ 量化器：vox / vox_input / symmetric + 4 个量化层
│   ├── tools.py                  # to_device / synth / pad 等工具
│   └── model.py                  # get_model / get_vocoder / vocoder_infer
├── hifigan/                      # models.py + models_quant.py + universal 权重
├── config/AISHELL3/
│   ├── preprocess.yaml           # corpus/特征路径指向共享存储
│   ├── model.yaml                # 模型结构
│   ├── train.yaml                # FP32 训练配置（保留）
│   ├── train_qat.yaml            # ★ QAT 专用配置（恒定 lr、量化位数、模块列表）
│   └── preprocess_smoke.yaml / train_qat_smoke.yaml   # 冒烟配置
├── scripts/
│   └── prepare_data_batched.sh   # ★ 分批数据流水线（含 MFA 分批对齐）
├── lexicon/
│   ├── pinyin-lexicon-r.txt      # 音节词典（synthesize 文本前端用）
│   └── pinyin-phoneme-lexicon.txt# ★ 恒等词典（211 token，MFA 对齐用）
├── data_refs/AISHELL3/           # ★ 预训练模型配套基准数据（不可重新生成）
│   ├── train.txt / val.txt       #   音素级拼音（prepare_corpus.py 查表来源）
│   ├── speakers.json             #   speaker 行序（speaker_emb 行序绑定）
│   └── stats.json                #   pitch/energy 统计基准
├── pretrain/AISHELL3_600000.pth.tar   # FP32 预训练 checkpoint (400MB)
├── preprocessed_data → /data/datasets/FastSpeech2-QAT/preprocessed_data
└── output/
    ├── qat_ckpt/AISHELL3/        # QAT checkpoint（{"model": state_dict, "step": N}）
    ├── qat_vocoder/              # vocoder_quantized.pt（plain state_dict，无 weight_norm）
    ├── qat_eval/report.json      # 对比评估结果
    └── qat_result/               # 合成 wav
```

**存储策略**：特征、TextGrid、raw wav、MFA 模型等大体积中间产物放共享存储
`/data/datasets/FastSpeech2-QAT/`，项目内 symlink；预训练 ckpt（400MB）与
HiFi-GAN 权重（54MB）为项目内实际文件。

**data_refs 说明**：`data_refs/AISHELL3/` 是预训练模型的配套基准——
`speakers.json` 的行序决定 speaker_emb 查表，`train.txt` 的音素级拼音是
`prepare_corpus.py` 的拼音权威来源。两者必须与预训练 ckpt 配套使用，
不可用重新生成的产物替换。

---

## 3. 快速开始

### 3.1 环境

```bash
cd /home/niwang/code/FastSpeech2-QAT
uv venv --python 3.10 --system-site-packages .venv
source .venv/bin/activate

# RTX 50 系 (sm_120) 必须用 cu128 torch（cu126 无 Blackwell kernel）
uv pip install "torch==2.7.1+cu128" --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
```

MFA 在独立环境（conda-forge，见 §4.3）。

### 3.2 数据准备（一次性，约 4-6 小时）

```bash
python prepare_corpus.py                    # parquet → 结构化语料
python prepare_align.py config/AISHELL3/preprocess.yaml
bash scripts/prepare_data_batched.sh        # MFA 分批对齐 + 特征提取 + speakers.json 对齐
python prepare_vocoder_data.py --max-samples 2000
```

当前数据规模：**train 77,437 条 / val 512 条**（MFA 对齐成功率 88.6%），
mel/pitch/energy/duration 特征齐全，vocoder 蒸馏片段 7,988 个。

### 3.3 端到端 QAT 训练

```bash
# FastSpeech2 整模型量化微调（标定 + 训练一条龙，2-5 万步收敛）
python train_qat.py -p config/AISHELL3/preprocess.yaml \
                    -m config/AISHELL3/model.yaml \
                    -t config/AISHELL3/train_qat.yaml \
                    --stage all --total-steps 50000

# 分阶段
python train_qat.py ... --stage calibrate        # 仅标定 → output/qat_ckpt/AISHELL3/calibrated.pth.tar
python train_qat.py ... --stage train --restore-qat output/qat_ckpt/AISHELL3/calibrated.pth.tar

# vocoder QAT（30 epochs 约 1-2 小时）
python train_vocoder_qat.py --epochs 30
```

常用参数：`--weight-bits/--input-bits`（位数）、`--quant-modules`（逗号分隔子集量化，
如 `encoder,decoder`）、`--calib-samples`（标定样本数）、`--lr`、`--total-steps`。

### 3.4 评估与合成

```bash
# FP32 vs QAT 对比：六项 loss / 逐模块 L1+余弦 / mel LSD
python evaluate_qat.py --acoustic-ckpt output/qat_ckpt/AISHELL3/50000.pth.tar

# 全量化合成（QAT 声学 + 量化 vocoder）
python synthesize_qat.py --text "任意中文文本" \
    --acoustic-ckpt output/qat_ckpt/AISHELL3/50000.pth.tar \
    --vocoder-ckpt output/qat_vocoder/vocoder_quantized.pt \
    --tag qat_full

# FP32 对照（不加 --acoustic-ckpt/--vocoder-ckpt 即走 FP32）
python synthesize_qat.py --text "任意中文文本" --tag fp32
```

---

## 4. 数据准备流水线

### 4.1 数据源

| 来源 | 内容 | 用途 |
|---|---|---|
| `/data/datasets/AISHELL-3/data/*.parquet` | 63,262 train + 24,773 test 条（audio bytes + text + 音节拼音） | wav 音频 |
| `data_refs/AISHELL3/train.txt` | 87,995 条 `basename\|speaker\|{音素级拼音}\|raw_text` | **拼音权威来源**（与模型词表一致） |

拼音采用查表而非 parquet 自带字段：parquet 的 pinyin 是音节级（"guang3"），
而模型词表是声母/韵母级（"g uang3"），直接用会造成 OOV。

### 4.2 流水线步骤

```
parquet ──prepare_corpus.py──> {split}/content.txt（"wav\t0 py1 0 py2 ..." 交错格式，
                                配合 aishell3.py 的 split(" ")[1::2] 解析）
       ──prepare_align.py───> raw_data/{speaker}/{name}.wav(22050 int16) + .lab
       ──MFA train/align───> TextGrid/{speaker}/{name}.TextGrid（phones tier = 声母/韵母）
       ──preprocess.py─────> mel(T,80) / pitch / energy / duration npy + train/val.txt + stats.json
```

MFA 词典 `lexicon/pinyin-phoneme-lexicon.txt`：从 data_refs 的 train.txt 提取
全部 211 个 token 做恒等映射（`t t`），保证 TextGrid phones 与词表 100% 一致
（含 sp 静音符）。

### 4.3 MFA 环境

MFA 3.4.3 的 kaldi 绑定（kalpy）只在 conda-forge 提供，PyPI 安装会缺 `_kalpy`。
用 micromamba 独立环境：

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ~/bin --strip-components=1 bin/micromamba
MAMBA_ROOT_PREFIX=~/micromamba ~/bin/micromamba create -y -n mfa -c conda-forge montreal-forced-aligner
```

### 4.4 分批对齐（内存问题的解法）

MFA 3.x 一次性导入 88k 语料时内存峰值超过 25G 可用量（会触发系统 OOM kill）。
解法（`scripts/prepare_data_batched.sh`）：

1. **10k 条子集** `mfa train` 训练声学模型（约 40 分钟，产出 mfa_model.zip 61MB）
2. **全量分 6 批**（15k 条/批）`mfa align` 复用模型（每批约 7 分钟，峰值内存 ~5G）

对齐结果：77,949 / 87,995（88.6%），失败条目多为音频质量问题，属正常范围。

### 4.5 speakers.json 对齐

预处理时 `os.listdir` 顺序不稳定，而预训练 speaker_emb 的行序绑定
`data_refs/AISHELL3/speakers.json`。流水线在特征提取后会把该基准文件覆盖到
特征目录（`prepare_data_batched.sh` 步骤 [D]），保证 speaker 查表与预训练一致。

---

## 5. QAT 技术方案

### 5.1 量化器（utils/utils_quant.py）

| 组件 | 方案 |
|---|---|
| 权重 | `VoxQuantizer`：per-tensor 对称，max 定标（2-bit 时 L1 均值×1.4 特化）；STE 反传 |
| 激活 | `VoxQuantizer_input`：`scale_in` 为 nn.Parameter（可学习、随 ckpt 保存）；STE 反传 |
| 备选 | `SymmetricQuantizer`（无 scale_in 版本） |

保留的量化层：`QuantizeLinear / QuantizeConv1d / QuantizeConvTranspose1d / QuantizeEmbedding`。

### 5.2 训练流程（train_qat.py）

```
[加载] FP32 teacher (600k ckpt) ──逐键拷贝──> QuanFastSpeech2（235/310 keys，
                                               缺的 75 个 = 新增 scale_in 参数）
[标定] 训练集前向 256 条，hook 收集每个量化层输入 abs-max，p99 分位 → scale_in
       （input_bits>=32 的层跳过——写 max/2^31 会得到毁灭性小步长）
[训练] FastSpeech2Loss（mel L1 + postnet L1 + pitch/energy/duration MSE）
       Adam 恒定 lr=1e-4（预训练已完成 Noam 预热阶段，不再调度）
       grad_clip 1.0；周期 log/synth/val/save
[保存] {"model": state_dict, "step": N}，含全部 scale_in 与 pitch/energy bins
```

### 5.3 vocoder QAT（train_vocoder_qat.py）

关键顺序：
1. teacher（FP32 universal HiFi-GAN）`remove_weight_norm()` → 拷 plain 权重
2. student 同样去 norm 后 load（**ckpt 存的是 plain 权重，加载前必须先去 norm**）
3. QuantizeConv1d 类默认 4-bit，**必须显式覆盖为 8-bit**
4. p99 标定 → 波形 L1 蒸馏（AdamW，lr 1e-4）→ best-val 保存

---

## 6. 验证结果

### 6.1 冒烟验证（500 条小数据，全链路）

- ✅ 全模块量化模型构造与权重拷贝（235/310）
- ✅ 标定 75 层 scale_in（值合理，~0.04 量级）
- ✅ QAT 训练 30 步跑通、验证集评估正常
- ✅ ckpt 保存/重载（310 keys 完整）
- ✅ vocoder QAT：标定后初始 val L1 = 0.040，训练循环与 best 保存正常
- ✅ 三种组合合成：FP32/FP32、QAT/FP32、QAT/量化 vocoder

### 6.2 真实数据验证（全量数据 100 步）

训练 loss（batch=16，恒定 lr 1e-4）：

| step | total | mel | postnet | pitch | energy | duration |
|---|---|---|---|---|---|---|
| 25  | 2.225 | 0.761 | 0.761 | 0.212 | 0.391 | 0.100 |
| 100 | 1.672 | 0.599 | 0.599 | 0.146 | 0.203 | 0.125 |

FP32 vs QAT(100 步) 对比（val 96 条）：

```
loss           FP32       QAT        Δ
total         2.2310    1.9297   -0.3013
mel           0.7633    0.6655   -0.0978
pitch         0.2171    0.1936   -0.0236
energy        0.3646    0.3129   -0.0517
duration      0.1229    0.0931   -0.0298

mel LSD (fp32 vs qat): 0.4167

module              rel_L1     cosine
encoder             0.1807   0.984360
variance_adaptor    0.0881   0.995283
decoder             0.2331   0.984542
mel_linear          0.0758   0.992494
postnet             0.6242   0.679939   ← 量化最敏感模块
```

**解读**：QAT 各项 loss 低于 FP32 零样本基线，是因为 QAT 在同数据上微调了
100 步；这说明量化模型可正常训练收敛。量化净损失的严谨度量需要「FP32 同步数
微调」对照（见 §7 待办）。postnet（5 层 Conv1d 栈）对 8-bit 量化最敏感，
是后续微调重点观察对象。

---

## 7. 实现要点与已知问题

### 关键修复（开发过程中发现并解决）

| 问题 | 位置 | 修复 |
|---|---|---|
| `librosa.load(wav, sr)` 位置参数 | preprocessor/aishell3.py | 改 keyword `sr=` |
| `pad_center` / `librosa_mel_fn` 位置参数 | audio/stft.py | 改 keyword-only |
| `energies` 缺 `.float()`（float64 特征 → backward dtype 报错） | utils/tools.py | 补 `.float()` |
| TextGrid 缺失时 `pitch` 未定义（UnboundLocalError） | preprocessor/preprocessor.py | 加 `else: continue` |
| `vocoder_infer` lengths 语义（采样点 ≠ mel 帧） | synthesize_qat.py | `mel_len * hop_length` |
| `QuantizeEmbedding` 缺 `input_quant_method` 形参 | utils/utils_quant.py | 补兼容形参 |
| 系统 cu126 torch 无 sm_120 kernel | 环境 | venv 内装 2.7.1+cu128 |
| 系统 pandas 与 numpy 2.x 二进制不兼容 | 环境 | venv 内覆盖安装 pandas 2.3.3 |
| MFA 全量导入 88k 语料 OOM | 流水线 | 10k 训练 + 6 批×15k 对齐 |

### 待办 / 后续工作

1. **全量 QAT 训练**：2-5 万步（`train_qat.yaml` 已配好），观察 val loss 与
   postnet cosine 收敛
2. **FP32 对照实验**：同数据同步数微调 FP32，量化净损失 = QAT(final) − FP32(final)
3. **vocoder GAN 微调**（可选）：在 L1 蒸馏后增加 LSGAN + feature matching +
   多分辨率 STFT 的 GAN-QAT 阶段，可显著改善听感
4. **onnx 导出 / 硬件对齐 dump**：按部署需求扩展

### 注意事项

- 训练/推理必须以项目根为 CWD 运行（yaml 内 `lexicon_path` 等是相对路径）
- 32-bit 激活层（`--input-bits 32`）标定时自动跳过 scale_in，防止毁灭性小步长
- vocoder ckpt 是 plain 权重：**任何加载处都必须先 `remove_weight_norm()`**
  （`synthesize_qat.py` / `train_vocoder_qat.py` 已处理）
- MFA 全量对齐切勿一次导入 88k 语料（会 OOM），用 `scripts/prepare_data_batched.sh`
