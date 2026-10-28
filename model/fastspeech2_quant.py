# coding=utf-8
# fastspeech2_quant.py - FastSpeech2 量化感知训练版本
# 参考 VoxCPM-0.5B 的量化方案，将 nn.Linear / nn.Conv1d / nn.Embedding
# 替换为可量化训练的 QuantizeLinear / QuantizeConv1d / QuantizeEmbedding

import os
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils_quant import (
    QuantizeLinear,
    QuantizeConv1d,
    QuantizeEmbedding,
    weight_quant_fn,
    act_quant_fn,
)
from utils.tools import get_mask_from_lengths, pad
from text.symbols import symbols

import transformer.Constants as Constants


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 量化版 Scaled Dot-Product Attention（无需量化，不含参数）
# ============================================================

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        output = torch.bmm(attn, v)
        return output, attn


# ============================================================
# 量化版 Multi-Head Attention
# ============================================================

class QuantMultiHeadAttention(nn.Module):
    """ Multi-Head Attention module - 使用 QuantizeLinear """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1,
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input"):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = QuantizeLinear(d_model, n_head * d_k,
                                   weight_bits=weight_bits, input_bits=input_bits,
                                   weight_quant_method=weight_quant_method,
                                   input_quant_method=input_quant_method)
        self.w_ks = QuantizeLinear(d_model, n_head * d_k,
                                   weight_bits=weight_bits, input_bits=input_bits,
                                   weight_quant_method=weight_quant_method,
                                   input_quant_method=input_quant_method)
        self.w_vs = QuantizeLinear(d_model, n_head * d_v,
                                   weight_bits=weight_bits, input_bits=input_bits,
                                   weight_quant_method=weight_quant_method,
                                   input_quant_method=input_quant_method)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = QuantizeLinear(n_head * d_v, d_model,
                                 weight_bits=weight_bits, input_bits=input_bits,
                                 weight_quant_method=weight_quant_method,
                                 input_quant_method=input_quant_method)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)

        mask = mask.repeat(n_head, 1, 1)
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


# ============================================================
# 量化版 Positionwise Feed-Forward
# ============================================================

class QuantPositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module - 使用 QuantizeConv1d """

    def __init__(self, d_in, d_hid, kernel_size, dropout=0.1,
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input"):
        super().__init__()

        self.w_1 = QuantizeConv1d(
            d_in, d_hid, kernel_size=kernel_size[0],
            padding=(kernel_size[0] - 1) // 2,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )
        self.w_2 = QuantizeConv1d(
            d_hid, d_in, kernel_size=kernel_size[1],
            padding=(kernel_size[1] - 1) // 2,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


# ============================================================
# 量化版 FFT Block
# ============================================================

class QuantFFTBlock(nn.Module):
    """FFT Block - 使用量化版 Attention 和 FFN"""

    def __init__(self, d_model, n_head, d_k, d_v, d_inner, kernel_size,
                 dropout=0.1, weight_bits=8, input_bits=8,
                 weight_quant_method="vox", input_quant_method="vox_input"):
        super(QuantFFTBlock, self).__init__()
        self.slf_attn = QuantMultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )
        self.pos_ffn = QuantPositionwiseFeedForward(
            d_model, d_inner, kernel_size, dropout=dropout,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )

    def forward(self, enc_input, mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


# ============================================================
# 量化版 ConvNorm（PostNet 中使用）
# ============================================================

class QuantConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain="linear",
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input"):
        super(QuantConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = QuantizeConv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )

    def forward(self, signal):
        return self.conv(signal)


# ============================================================
# 量化版 PostNet
# ============================================================

class QuantPostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    - 使用 QuantizeConv1d
    """

    def __init__(self, n_mel_channels=80, postnet_embedding_dim=512,
                 postnet_kernel_size=5, postnet_n_convolutions=5,
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input"):
        super(QuantPostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                QuantConvNorm(
                    n_mel_channels, postnet_embedding_dim,
                    kernel_size=postnet_kernel_size, stride=1,
                    padding=int((postnet_kernel_size - 1) // 2),
                    dilation=1, w_init_gain="tanh",
                    weight_bits=weight_bits, input_bits=input_bits,
                    weight_quant_method=weight_quant_method,
                    input_quant_method=input_quant_method,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    QuantConvNorm(
                        postnet_embedding_dim, postnet_embedding_dim,
                        kernel_size=postnet_kernel_size, stride=1,
                        padding=int((postnet_kernel_size - 1) // 2),
                        dilation=1, w_init_gain="tanh",
                        weight_bits=weight_bits, input_bits=input_bits,
                        weight_quant_method=weight_quant_method,
                        input_quant_method=input_quant_method,
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            nn.Sequential(
                QuantConvNorm(
                    postnet_embedding_dim, n_mel_channels,
                    kernel_size=postnet_kernel_size, stride=1,
                    padding=int((postnet_kernel_size - 1) // 2),
                    dilation=1, w_init_gain="linear",
                    weight_bits=weight_bits, input_bits=input_bits,
                    weight_quant_method=weight_quant_method,
                    input_quant_method=input_quant_method,
                ),
                nn.BatchNorm1d(n_mel_channels),
            )
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        x = x.contiguous().transpose(1, 2)
        return x


# ============================================================
# 量化版 Encoder
# ============================================================

class QuantEncoder(nn.Module):
    """ Encoder - 使用量化版 FFTBlock """

    def __init__(self, config, weight_bits=8, input_bits=8,
                 weight_quant_method="vox", input_quant_method="vox_input"):
        super(QuantEncoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        n_src_vocab = len(symbols) + 1
        d_word_vec = config["transformer"]["encoder_hidden"]
        n_layers = config["transformer"]["encoder_layer"]
        n_head = config["transformer"]["encoder_head"]
        d_k = d_v = (
            config["transformer"]["encoder_hidden"]
            // config["transformer"]["encoder_head"]
        )
        d_model = config["transformer"]["encoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["encoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                QuantFFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size,
                    dropout=dropout,
                    weight_bits=weight_bits, input_bits=input_bits,
                    weight_quant_method=weight_quant_method,
                    input_quant_method=input_quant_method,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src_seq, mask, return_attns=False):
        enc_slf_attn_list = []
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]

        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        if not self.training and src_seq.shape[1] > self.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(
                src_seq.shape[1], self.d_model
            )[: src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                src_seq.device
            )
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        return enc_output


# ============================================================
# 量化版 Decoder
# ============================================================

class QuantDecoder(nn.Module):
    """ Decoder - 使用量化版 FFTBlock """

    def __init__(self, config, weight_bits=8, input_bits=8,
                 weight_quant_method="vox", input_quant_method="vox_input"):
        super(QuantDecoder, self).__init__()

        n_position = config["max_seq_len"] + 1
        d_word_vec = config["transformer"]["decoder_hidden"]
        n_layers = config["transformer"]["decoder_layer"]
        n_head = config["transformer"]["decoder_head"]
        d_k = d_v = (
            config["transformer"]["decoder_hidden"]
            // config["transformer"]["decoder_head"]
        )
        d_model = config["transformer"]["decoder_hidden"]
        d_inner = config["transformer"]["conv_filter_size"]
        kernel_size = config["transformer"]["conv_kernel_size"]
        dropout = config["transformer"]["decoder_dropout"]

        self.max_seq_len = config["max_seq_len"]
        self.d_model = d_model

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0),
            requires_grad=False,
        )

        self.layer_stack = nn.ModuleList(
            [
                QuantFFTBlock(
                    d_model, n_head, d_k, d_v, d_inner, kernel_size,
                    dropout=dropout,
                    weight_bits=weight_bits, input_bits=input_bits,
                    weight_quant_method=weight_quant_method,
                    input_quant_method=input_quant_method,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, enc_seq, mask, return_attns=False):
        dec_slf_attn_list = []
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]

        if not self.training and enc_seq.shape[1] > self.max_seq_len:
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq + get_sinusoid_encoding_table(
                enc_seq.shape[1], self.d_model
            )[: enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(
                enc_seq.device
            )
        else:
            max_len = min(max_len, self.max_seq_len)
            slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)
            dec_output = enc_seq[:, :max_len, :] + self.position_enc[
                :, :max_len, :
            ].expand(batch_size, -1, -1)
            mask = mask[:, :max_len]
            slf_attn_mask = slf_attn_mask[:, :, :max_len]

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output, mask=mask, slf_attn_mask=slf_attn_mask
            )
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output, mask


# ============================================================
# 量化版 Conv（VariancePredictor 中使用）
# ============================================================

class QuantConv(nn.Module):
    """ Convolution Module - 使用 QuantizeConv1d """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, bias=True, w_init="linear",
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input"):
        super(QuantConv, self).__init__()

        self.conv = QuantizeConv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=bias,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)
        return x


# ============================================================
# 量化版 Variance Predictor
# ============================================================

class QuantVariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor - 使用量化层 """

    def __init__(self, model_config, weight_bits=8, input_bits=8,
                 weight_quant_method="vox", input_quant_method="vox_input"):
        super(QuantVariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        QuantConv(
                            self.input_size, self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                            weight_bits=weight_bits, input_bits=input_bits,
                            weight_quant_method=weight_quant_method,
                            input_quant_method=input_quant_method,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        QuantConv(
                            self.filter_size, self.filter_size,
                            kernel_size=self.kernel, padding=1,
                            weight_bits=weight_bits, input_bits=input_bits,
                            weight_quant_method=weight_quant_method,
                            input_quant_method=input_quant_method,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = QuantizeLinear(
            self.conv_output_size, 1,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


# ============================================================
# 量化版 Variance Adaptor
# ============================================================

class BucketEmbeddingSTE(torch.autograd.Function):
    """bucketize + embedding 查表的直通估计器（STE）。

    forward 与部署完全一致（硬分桶查表）；backward 用相邻 bin 的局部线性化

        d emb / d p ~= (E[i+1] - E[i]) / (edge_hi - edge_lo)

    把输出梯度传回 prediction。这样训练能"看见"翻桶：当选错 embedding 时，
    梯度会推动 prediction 移向能降低输出误差的相邻 bin。尾部 bin 的
    emb_hi 不存在，梯度记 0。

    注意：embedding 权重本身不从这条路径获得梯度（backward 只回传到
    prediction），即 embedding 保持 teacher 锚定——这是有意的，防止
    "把相邻 bin 的 embedding 拉近"这类改变模型语义的退化解。
    """

    @staticmethod
    def forward(ctx, prediction, bins, weight):
        idx = torch.bucketize(prediction, bins)
        ctx.save_for_backward(prediction, bins, weight, idx)
        return F.embedding(idx, weight)

    @staticmethod
    def backward(ctx, grad_output):
        prediction, bins, weight, idx = ctx.saved_tensors
        n_bins = weight.shape[0]
        K = bins.shape[0]
        if K < 2 or n_bins != K + 1:
            return None, None, None
        # 每个 bin 的左右边界（首/尾 bin 用相邻 bin 宽外推）
        pad_lo = (bins[0] - (bins[1] - bins[0])).reshape(1)
        pad_hi = (bins[-1] + (bins[-1] - bins[-2])).reshape(1)
        edges = torch.cat([pad_lo, bins, pad_hi])          # [n_bins + 1]
        lo = edges[idx]
        hi = edges[idx + 1]
        width = (hi - lo).clamp(min=1e-8)
        emb_lo = weight[idx]
        emb_hi = weight[(idx + 1).clamp(max=n_bins - 1)]   # 尾部 bin 梯度为 0
        slope = (emb_hi - emb_lo) / width.unsqueeze(-1)
        grad_pred = (grad_output * slope).sum(dim=-1)
        return grad_pred, None, None


class QuantVarianceAdaptor(nn.Module):
    """ Variance Adaptor - 使用量化层 """

    def __init__(self, preprocess_config, model_config,
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input"):
        super(QuantVarianceAdaptor, self).__init__()
        self.duration_predictor = QuantVariancePredictor(
            model_config, weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = QuantVariancePredictor(
            model_config, weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )
        self.energy_predictor = QuantVariancePredictor(
            model_config, weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
        )

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

        # 训练时对 bucketize 使用直通估计器（见 BucketEmbeddingSTE）。
        # 默认关闭，保持与其他训练脚本的行为不变。
        self.ste_bucket = False

    def _bucket_embedding(self, prediction, bins, embedding):
        """prediction 分桶查表；ste_bucket 开启时走 STE 版本以回传梯度"""
        if self.ste_bucket:
            return BucketEmbeddingSTE.apply(prediction, bins, embedding.weight)
        return embedding(torch.bucketize(prediction, bins))

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self._bucket_embedding(prediction, self.pitch_bins, self.pitch_embedding)
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self._bucket_embedding(prediction, self.energy_bins, self.energy_embedding)
        return prediction, embedding

    def forward(
        self, x, src_mask, mel_mask=None, max_len=None,
        pitch_target=None, energy_target=None, duration_target=None,
        p_control=1.0, e_control=1.0, d_control=1.0,
    ):
        log_duration_prediction = self.duration_predictor(x, src_mask)
        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, e_control
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, e_control
            )
            x = x + energy_embedding

        return (
            x, pitch_prediction, energy_prediction,
            log_duration_prediction, duration_rounded, mel_len, mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator（无参数，不需要量化）"""
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            # Handle both 1D tensor and 0-dim tensor cases
            pred_val = predicted[i]
            if pred_val.dim() == 0:
                # 0-dim tensor (scalar)
                expand_size = pred_val.item()
            else:
                # 1D or higher dim tensor
                expand_size = pred_val.item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


# ============================================================
# 辅助函数：位置编码
# ============================================================

# 缓存已生成的正弦位置编码表（key: n_position/d_hid/padding_idx）。
# 返回的 tensor 是共享对象，调用方只读（现有调用均满足）。
_SINUSOID_TABLE_CACHE = {}


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    """ Sinusoid position encoding table（向量化 + 缓存）

    数学上与原逐元素循环版本完全等价:
        angle[p, h] = p / 10000^(2*(h//2)/d_hid)
        table[:, 0::2] = sin(angle[:, 0::2])
        table[:, 1::2] = cos(angle[:, 1::2])

    原实现是 O(n_position * d_hid) 的纯 Python 嵌套循环，实测 L=1229 时
    每次调用 ~0.6s、L=4000 时 ~1.7s。decoder 的 eval 分支在
    mel_len > max_seq_len 时每次 forward 都会调用它 —— 这是 decoder
    QAT 训练极慢的主因（64% 的标定样本 mel_len>1000）。
    """
    key = (n_position, d_hid, padding_idx)
    cached = _SINUSOID_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    position = np.arange(n_position, dtype=np.float64)[:, None]   # [P, 1]
    hid_idx = np.arange(d_hid)[None, :]                           # [1, H]
    angle = position / np.power(
        10000, (2 * (hid_idx // 2)) / d_hid)                      # [P, H]
    sinusoid_table = np.empty((n_position, d_hid), dtype=np.float64)
    sinusoid_table[:, 0::2] = np.sin(angle[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(angle[:, 1::2])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0.0

    table = torch.FloatTensor(sinusoid_table)
    _SINUSOID_TABLE_CACHE[key] = table
    return table


# ============================================================
# 量化版 FastSpeech2（主模型）
# ============================================================

class QuanFastSpeech2(nn.Module):
    """ FastSpeech2 量化感知训练版本

    所有 nn.Linear → QuantizeLinear
    所有 nn.Conv1d  → QuantizeConv1d
    所有 nn.Embedding → QuantizeEmbedding

    默认：权重 8-bit 量化，激活保持 FP32
    """

    def __init__(self, preprocess_config, model_config,
                 weight_bits=8, input_bits=8, weight_quant_method="vox",
                 input_quant_method="vox_input",
                 quant_modules=("encoder",)):
        """ FastSpeech2 量化感知训练版本

        Args:
            weight_bits: 权重量化位数
            input_bits: 激活量化位数
            weight_quant_method: 权重量化方法
            input_quant_method: 激活量化方法
            quant_modules: tuple[str], 指定哪些模块使用量化版本。
                可选: "encoder", "variance_adaptor", "decoder", "mel_linear", "postnet", "speaker_emb"
                默认: ("encoder",) 仅 encoder 量化，其他模块用 FP32
                示例: ("encoder", "decoder") 量化 encoder 和 decoder
        """
        super(QuanFastSpeech2, self).__init__()
        self.model_config = model_config
        self.quant_modules = set(quant_modules)

        # --- Encoder ---
        if "encoder" in self.quant_modules:
            self.encoder = QuantEncoder(
                model_config,
                weight_bits=weight_bits, input_bits=input_bits,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
            )
        else:
            from transformer import Encoder
            self.encoder = Encoder(model_config)

        # --- Variance Adaptor ---
        if "variance_adaptor" in self.quant_modules:
            self.variance_adaptor = QuantVarianceAdaptor(
                preprocess_config, model_config,
                weight_bits=weight_bits, input_bits=input_bits,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
            )
        else:
            from model.modules import VarianceAdaptor
            self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)

        # --- Decoder ---
        if "decoder" in self.quant_modules:
            self.decoder = QuantDecoder(
                model_config,
                weight_bits=weight_bits, input_bits=input_bits,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
            )
        else:
            from transformer import Decoder
            self.decoder = Decoder(model_config)

        # --- Mel Linear ---
        if "mel_linear" in self.quant_modules:
            self.mel_linear = QuantizeLinear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                weight_bits=weight_bits, input_bits=input_bits,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
            )
        else:
            self.mel_linear = nn.Linear(
                model_config["transformer"]["decoder_hidden"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
            )

        # --- PostNet ---
        if "postnet" in self.quant_modules:
            self.postnet = QuantPostNet(
                n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                weight_bits=weight_bits, input_bits=input_bits,
                weight_quant_method=weight_quant_method,
                input_quant_method=input_quant_method,
            )
        else:
            from transformer import PostNet
            self.postnet = PostNet()

        # --- Speaker Embedding ---
        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            if "speaker_emb" in self.quant_modules:
                self.speaker_emb = QuantizeEmbedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                    weight_bits=weight_bits, input_bits=input_bits,
                    weight_quant_method=weight_quant_method,
                    input_quant_method=input_quant_method,
                )
            else:
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )

    @staticmethod
    def load_from_pretrained(model, preprocess_config, weight_bits=8, input_bits=8,
                             weight_quant_method="vox", input_quant_method="vox_input",
                             quant_modules=("encoder", "variance_adaptor", "decoder",
                                            "mel_linear", "postnet", "speaker_emb")):
        """从预训练的 FastSpeech2 加载权重到量化版本

        Args:
            model: 预训练的 FastSpeech2 实例
            preprocess_config: 预处理配置（FastSpeech2 不保存该属性，需显式传入）
            weight_bits: 权重量化位数
            input_bits: 激活量化位数
            weight_quant_method: 权重量化方法
            input_quant_method: 激活量化方法
            quant_modules: 需要量化的模块，默认全模块
        Returns:
            QuanFastSpeech2 实例
        """
        quant_model = QuanFastSpeech2(
            preprocess_config, model.model_config,
            weight_bits=weight_bits, input_bits=input_bits,
            weight_quant_method=weight_quant_method,
            input_quant_method=input_quant_method,
            quant_modules=quant_modules,
        )

        # 加载权重
        quant_dict = quant_model.state_dict()
        teacher_dict = {k: v for k, v in model.state_dict().items()
                        if k in quant_dict}

        # 量化层的 weight/bias 与原始层完全兼容
        missing = set(quant_dict.keys()) - set(teacher_dict.keys())
        extra = set(teacher_dict.keys()) - set(quant_dict.keys())

        quant_model.load_state_dict(teacher_dict, strict=False)

        if missing:
            print(f"[WARNING] Missing keys: {sorted(missing)[:5]}...")
        if extra:
            print(f"[WARNING] Extra keys: {sorted(extra)[:5]}...")

        print(f"Loaded {len(teacher_dict)}/{len(quant_dict)} weights "
              f"from pretrained model")
        return quant_model
