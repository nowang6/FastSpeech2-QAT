#!/bin/bash
# 分批 MFA 对齐（绕开 MFA 3.x 全量导入 88k 语料的内存峰值）
#   Step A: 10k 条子集训练声学模型
#   Step B: 全量分 6 批（~15k/批）复用模型对齐
#   Step C-D: 特征提取 + speakers.json 覆盖 + vocoder 数据
set -e
cd /home/niwang/code/FastSpeech2-QAT
source .venv/bin/activate
export MAMBA_ROOT_PREFIX=$HOME/micromamba
MFA=~/bin/micromamba
RAW=/data/datasets/FastSpeech2-QAT/raw_data/AISHELL3
BATCHES=/data/datasets/FastSpeech2-QAT/mfa_batches
TG_OUT=/data/datasets/FastSpeech2-QAT/preprocessed_data/AISHELL3/TextGrid
LEX=lexicon/pinyin-phoneme-lexicon.txt

echo "=== 构建分批目录（symlink，不复制数据） ==="
rm -rf "$BATCHES"
mkdir -p "$BATCHES/train_subset" "$TG_OUT"

# 一次性索引：按 basename 排序的 lab 路径列表
find "$RAW" -name "*.lab" | sort -t/ -k9 > /tmp/aishell3_lab_paths.txt
TOTAL=$(wc -l < /tmp/aishell3_lab_paths.txt)
echo "total utterances: $TOTAL"

TRAIN_N=10000
BATCH_N=15000
idx=0
while read -r lab_path; do
    if [ $idx -lt $TRAIN_N ]; then
        d="$BATCHES/train_subset"
    else
        b=$(( (idx - TRAIN_N) / BATCH_N ))
        d="$BATCHES/batch_$b"
    fi
    mkdir -p "$d"
    ln -sf "$lab_path" "$d/$(basename "$lab_path")"
    ln -sf "${lab_path%.lab}.wav" "$d/$(basename "${lab_path%.lab}").wav"
    idx=$((idx+1))
done < /tmp/aishell3_lab_paths.txt
echo "batches:"; ls -d "$BATCHES"/batch_* | while read -r b; do echo "  $b: $(ls "$b" | wc -l) files"; done

echo "=== [A] MFA train on ${TRAIN_N}-utterance subset ==="
$MFA run -n mfa mfa train "$BATCHES/train_subset" "$LEX" \
    /data/datasets/FastSpeech2-QAT/mfa_model.zip \
    --speaker_characters 7 --clean --overwrite -j 4

echo "=== [B] align all batches ==="
for b in "$BATCHES"/batch_*; do
    echo "--- aligning $(basename "$b") ---"
    $MFA run -n mfa mfa align "$b" "$LEX" /data/datasets/FastSpeech2-QAT/mfa_model.zip \
        /tmp/mfa_align_out --speaker_characters 7 --clean --overwrite -j 4 --single_speaker false
    cp -r /tmp/mfa_align_out/* "$TG_OUT"/
    rm -rf /tmp/mfa_align_out
done
echo "TextGrid total: $(find "$TG_OUT" -name '*.TextGrid' | wc -l)"

echo "=== [C] preprocess ==="
python preprocess.py config/AISHELL3/preprocess.yaml

echo "=== [D] 对齐基准 speakers.json（speaker_emb 行序绑定） ==="
mv /data/datasets/FastSpeech2-QAT/preprocessed_data/AISHELL3/speakers.json \
   /data/datasets/FastSpeech2-QAT/preprocessed_data/AISHELL3/speakers.json.regenerated
cp data_refs/AISHELL3/speakers.json \
   /data/datasets/FastSpeech2-QAT/preprocessed_data/AISHELL3/speakers.json

echo "=== [E] vocoder 蒸馏数据 ==="
python prepare_vocoder_data.py --max-samples 2000

echo "=== 完成 ==="
wc -l /data/datasets/FastSpeech2-QAT/preprocessed_data/AISHELL3/train.txt \
      /data/datasets/FastSpeech2-QAT/preprocessed_data/AISHELL3/val.txt
