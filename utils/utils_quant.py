# coding=utf-8
# utils_quant.py — FastSpeech2-QAT 量化器
#
# 端到端 QAT 所需的量化器与量化层：
#   - 权重:  vox（per-tensor 对称, max 标定）/ symmetric（等价 vox）
#   - 激活:  vox_input（可学习 scale_in, nn.Parameter, 随 ckpt 保存）

import math

import torch
import torch.nn as nn


class VoxQuantizer(torch.autograd.Function):
    """per-tensor 对称量化，权重用（2-bit 时 L1 均值定标，其余 max 定标）"""

    @staticmethod
    def forward(ctx, tensor_in, num_bits):
        ctx.save_for_backward(tensor_in)

        tensor_in = tensor_in.float()
        if 2 == num_bits:
            m = tensor_in.norm(p=1).div(tensor_in.nelement())
            abs_max = 1.4 * m
        else:
            abs_max = tensor_in.abs().max()
        eps = torch.finfo(torch.float32).eps
        abs_max = torch.clamp(abs_max, min=eps)

        maxq = 2 ** (num_bits - 1) - 1
        scale = abs_max / maxq

        result = torch.round(tensor_in / scale).clamp(-maxq - 1, maxq)
        result *= scale
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


class VoxQuantizer_input(torch.autograd.Function):
    """激活量化：使用可学习 scale_in（nn.Parameter），STE 反传，scale_in 梯度可学习"""

    @staticmethod
    def forward(ctx, tensor_in, num_bits, scale_in):
        tensor_in = tensor_in.float()

        # 32-bit 激活旁路量化，避免除零
        if num_bits >= 32:
            ctx.save_for_backward(tensor_in)
            return tensor_in

        tensor_in_scaled = tensor_in / scale_in
        maxq = 2 ** (num_bits - 1) - 1
        quantized_val = torch.round(tensor_in_scaled).clamp(-maxq - 1, maxq)

        ctx.save_for_backward(quantized_val)
        return quantized_val * scale_in

    @staticmethod
    def backward(ctx, grad_output):
        quantized_val, = ctx.saved_tensors

        # STE: 强行让 round 的梯度为 1
        grad_input = grad_output.clone()
        # scale_in 的梯度
        grad_scale_in = (grad_output * quantized_val).sum()

        return grad_input, None, grad_scale_in


class SymmetricQuantizer(torch.autograd.Function):
    """对称量化器（无 scale_in 版本）

    scale = max_abs / (2^(num_bits-1) - 1)
    量化范围: [-2^(num_bits-1), 2^(num_bits-1)-1]，STE 反传。
    """

    @staticmethod
    def forward(ctx, tensor_in, num_bits):
        ctx.save_for_backward(tensor_in)

        if num_bits >= 32:
            return tensor_in

        tensor_in = tensor_in.float()

        maxq = 2 ** (num_bits - 1) - 1
        abs_max = tensor_in.abs().max()
        eps = torch.finfo(torch.float32).eps
        abs_max = torch.clamp(abs_max, min=eps)
        scale = abs_max / maxq

        q = torch.round(tensor_in / scale).clamp(-maxq - 1, maxq)
        result = q * scale

        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def act_quant_fn(input, num_bits, symmetric, quant_method, scale_in):
    if num_bits == 32:
        return input

    if quant_method == "vox":
        # 2-bit 有 L1 定标特化
        return VoxQuantizer.apply(input, num_bits)

    if quant_method == "vox_input":
        # 可学习 scale_in
        return VoxQuantizer_input.apply(input, num_bits, scale_in)

    if quant_method == "symmetric":
        return SymmetricQuantizer.apply(input, num_bits)

    raise ValueError("Unknown quant_method: %s" % quant_method)


def weight_quant_fn(weight, num_bits, symmetric, quant_method):
    if num_bits == 32:
        return weight

    if quant_method == "vox":
        quant_fn = VoxQuantizer
    elif quant_method == "symmetric":
        quant_fn = SymmetricQuantizer
    else:
        raise ValueError("Unknown quant_method: %s" % quant_method)

    weight = quant_fn.apply(weight, num_bits)
    return weight


class QuantizeLinear(nn.Linear):
    def __init__(self, *kargs, weight_bits=8, input_bits=8,
                 symmetric=True, weight_quant_method="vox",
                 input_quant_method="vox_input",
                 scale_in=1.0, **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.symmetric = symmetric
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.scale_in = nn.Parameter(torch.tensor(scale_in,
                                     dtype=torch.float32))

    def forward(self, input):
        weight = weight_quant_fn(
            self.weight, num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method)

        input = act_quant_fn(input, num_bits=self.input_bits,
                             symmetric=self.symmetric,
                             quant_method=self.input_quant_method,
                             scale_in=self.scale_in)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(input.dtype)
        weight.data = weight.data.to(input.dtype)
        out = nn.functional.linear(input, weight, self.bias)
        return out


class QuantizeConv1d(nn.Conv1d):
    def __init__(self, *kargs, weight_bits=8, input_bits=8, symmetric=True,
                 weight_quant_method="vox", input_quant_method="vox_input",
                 scale_in=1.0, **kwargs):
        super(QuantizeConv1d, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.symmetric = symmetric
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        self.scale_in = nn.Parameter(torch.tensor(scale_in,
                                     dtype=torch.float32))

    def forward(self, input):
        weight = weight_quant_fn(
            self.weight, num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method)

        input = act_quant_fn(input, num_bits=self.input_bits,
                             symmetric=self.symmetric,
                             quant_method=self.input_quant_method,
                             scale_in=self.scale_in)
        out = nn.functional.conv1d(
            input,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return out


class QuantizeConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *kargs, weight_bits=8, input_bits=8, symmetric=True,
                 weight_quant_method="vox", input_quant_method="vox_input",
                 scale_in=1.0, **kwargs):
        super(QuantizeConvTranspose1d, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.symmetric = symmetric
        self.weight_quant_method = weight_quant_method
        self.input_quant_method = input_quant_method
        # NOTE: scale_in 必须是 nn.Parameter 才能进 state_dict，
        # 否则标定值在保存/加载后丢失（历史坑，与 QuantizeConv1d 一致）。
        self.scale_in = nn.Parameter(torch.tensor(scale_in, dtype=torch.float32))

    def forward(self, input):
        weight = weight_quant_fn(
            self.weight, num_bits=self.weight_bits,
            symmetric=self.symmetric,
            quant_method=self.weight_quant_method)

        input = act_quant_fn(input, num_bits=self.input_bits,
                             symmetric=self.symmetric,
                             quant_method=self.input_quant_method,
                             scale_in=self.scale_in)
        out = nn.functional.conv_transpose1d(
            input,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups
        )
        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self, *kargs, weight_bits=8, input_bits=8, learnable=False,
                 symmetric=True, weight_quant_method="vox",
                 input_quant_method="vox_input", embed_layerwise=False, **kwargs):
        super(QuantizeEmbedding, self).__init__(*kargs, **kwargs)
        self.weight_bits = weight_bits
        self.input_bits = input_bits
        self.learnable = learnable
        self.symmetric = symmetric
        # 兼容原签名：embedding 只量化权重，input_quant_method/embed_layerway 保留形参不生效
        self.input_quant_method = input_quant_method
        self.embed_layerwise = embed_layerwise
        self.weight_quant_method = weight_quant_method
        # vox/symmetric 无需 clip_val
        self.register_buffer('embed_clip_val', None)

    def forward(self, input):
        weight = weight_quant_fn(self.weight, num_bits=self.weight_bits,
                                 symmetric=self.symmetric,
                                 quant_method=self.weight_quant_method)

        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

        return out
