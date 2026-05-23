"""FP16 GEMV 示例：先给出 PyTorch 参考实现，再对照 Triton kernel 实现。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def fp16_gemv_reference(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """参考实现：输入权重矩阵 [out, in] 和向量 [in]，输出 [out]。"""
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    if weight.shape[1] != x.shape[0]:
        raise ValueError(f"shape mismatch: weight={tuple(weight.shape)} x={tuple(x.shape)}")

    # 这里直接交给 PyTorch 的矩阵乘法，作为 Triton 版本的对照基线。
    return torch.matmul(weight, x)


# ── Triton kernel ────────────────────────────────────────────────────────────
# 这个函数直接在 GPU 上跑，每次调用处理一行权重。
# @triton.jit 告诉 Triton："把这个函数编译成 GPU 代码"。
@triton.jit
def _fp16_gemv_kernel(
    w_ptr,            # 权重矩阵在显存中的起始地址
    x_ptr,            # 输入向量在显存中的起始地址
    out_ptr,          # 输出向量在显存中的起始地址
    in_features,      # 输入向量的长度（1536）
    BLOCK: tl.constexpr,  # 每次处理多少列（编译期常量，这里等于 in_features）
):
    # 我是第几号程序？也就是我负责计算第几个输出值（第几行）。
    row = tl.program_id(0)

    # 生成列下标：[0, 1, 2, ..., BLOCK-1]
    # 用来知道"我要读这一行的哪些元素"
    cols = tl.arange(0, BLOCK)

    # 有些 BLOCK 可能比实际长度大，mask 用来忽略越界的位置
    mask = cols < in_features

    # 从显存读取权重矩阵的第 row 行
    # w_ptr + row * in_features 是这一行的起始地址，加上 cols 得到每列的地址
    w_row = tl.load(w_ptr + row * in_features + cols, mask=mask, other=0.0)

    # 从显存读取输入向量 x（所有程序读的都是同一个 x）
    x_vec = tl.load(x_ptr + cols, mask=mask, other=0.0)

    # 点积：先转成 FP32 再做乘加，避免 FP16 累积误差
    # FP16 精度有限，1536 次加法叠下来误差会超标
    result = tl.sum(w_row.to(tl.float32) * x_vec.to(tl.float32), axis=0)

    # 结果转回 FP16 再写出去（输出 tensor 是 FP16）
    tl.store(out_ptr + row, result.to(tl.float16))


# ── Python 入口函数 ──────────────────────────────────────────────────────────
# 这是普通 Python 函数，负责：
#   1. 准备输出 tensor
#   2. 告诉 Triton 启动多少个程序
#   3. 调用上面的 GPU kernel
def fp16_gemv_triton(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out_features, in_features = weight.shape

    # 准备一个空的输出 tensor，形状 [out_features]，放在 GPU 上
    output = torch.empty(out_features, device=weight.device, dtype=weight.dtype)

    # grid = 启动多少个程序
    # 这里启动 out_features（256）个程序，每个程序算一行
    grid = (out_features,)

    _fp16_gemv_kernel[grid](
        weight,       # 权重矩阵
        x,            # 输入向量
        output,       # 输出向量（空的，kernel 会填进去）
        in_features,  # 告诉 kernel 输入长度是多少
        BLOCK=triton.next_power_of_2(in_features),  # BLOCK 取 >= in_features 的最小 2 的幂
    )

    return output
