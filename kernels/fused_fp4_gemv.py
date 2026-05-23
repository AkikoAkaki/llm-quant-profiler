"""Learning scaffold for a fused bitsandbytes fp4 dequant-GEMV kernel."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from phase3_utils import Linear4bitArtifacts


# ── Triton kernel ────────────────────────────────────────────────────────────
# FP16 kernel 的扩展版本：多了 拆包 → 查码本 → 乘 scale 三步
@triton.jit
def _fused_fp4_gemv_kernel(
    packed_ptr,               # packed uint8 权重，已展平 [out * in / 2]
    absmax_ptr,               # block scale，shape [n_blocks]，float32
    code_ptr,                 # FP4 码本，shape [16]，float32
    x_ptr,                   # 输入向量，shape [in_features]，float16
    out_ptr,                  # 输出向量，shape [out_features]，float16
    in_features,              # 输入维度（1536）
    blocksize: tl.constexpr,  # 每个 block 的大小（64）
    BLOCK_IN: tl.constexpr,   # >= in_features // 2，必须是 2 的幂
):
    # 我负责第几行（第几个输出值）？
    row = tl.program_id(0)

    n_packed = in_features // 2       # 每行有多少个 packed byte（768）
    blocks_per_row = in_features // blocksize  # 每行有多少个 scale block（24）

    # col_pairs 是 packed byte 的下标：[0, 1, ..., BLOCK_IN-1]
    # col_pair k 对应原始矩阵的第 2k 和 2k+1 列
    col_pairs = tl.arange(0, BLOCK_IN)
    mask = col_pairs < n_packed

    # ── 步骤 1：读 packed uint8 ──────────────────────────────────────────────
    # row * n_packed 跳到这一行的起始位置，加 col_pairs 取出每个 byte
    packed = tl.load(
        packed_ptr + row * n_packed + col_pairs,
        mask=mask, other=0
    ).to(tl.int32)

    # ── 步骤 2：拆包（一个 byte → 两个 4-bit 值）─────────────────────────────
    # 高 4 位 → 对应偶数列（2k）
    # 低 4 位 → 对应奇数列（2k+1）
    hi = (packed >> 4) & 0xF
    lo = packed & 0xF

    # ── 步骤 3：查码本（4-bit 下标 → 真实浮点值）─────────────────────────────
    # code 是 16 个浮点值，hi/lo 是 0-15 的下标
    hi_fp = tl.load(code_ptr + hi, mask=mask, other=0.0)
    lo_fp = tl.load(code_ptr + lo, mask=mask, other=0.0)

    # ── 步骤 4：乘以 block scale（absmax）────────────────────────────────────
    # col_pair k 对应列 2k，block 下标 = row * blocks_per_row + (2k // blocksize)
    block_idx = row * blocks_per_row + (col_pairs * 2) // blocksize
    scale = tl.load(absmax_ptr + block_idx, mask=mask, other=0.0)
    hi_fp = hi_fp * scale
    lo_fp = lo_fp * scale

    # ── 步骤 5：读输入向量 x，做点积──────────────────────────────────────────
    x_hi = tl.load(x_ptr + col_pairs * 2,     mask=mask, other=0.0)
    x_lo = tl.load(x_ptr + col_pairs * 2 + 1, mask=mask, other=0.0)

    # 用 FP32 累加，避免精度问题（跟 FP16 kernel 一样的处理）
    acc = tl.sum(
        hi_fp.to(tl.float32) * x_hi.to(tl.float32) +
        lo_fp.to(tl.float32) * x_lo.to(tl.float32),
        axis=0
    )

    # ── 步骤 6：写出结果 ──────────────────────────────────────────────────────
    tl.store(out_ptr + row, acc.to(tl.float16))


# ── Python 入口函数 ──────────────────────────────────────────────────────────
def fused_fp4_gemv_triton(artifacts: Linear4bitArtifacts, x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    if x.shape[0] != artifacts.in_features:
        raise ValueError(
            f"input width mismatch: expected {artifacts.in_features}, got {x.shape[0]}"
        )

    out = torch.empty(artifacts.out_features, device=x.device, dtype=torch.float16)

    BLOCK_IN = triton.next_power_of_2(artifacts.in_features // 2)

    _fused_fp4_gemv_kernel[(artifacts.out_features,)](
        artifacts.packed_weight.view(-1),   # 展平 [196608, 1] → [196608]
        artifacts.absmax,
        artifacts.code,
        x,
        out,
        artifacts.in_features,
        artifacts.blocksize,
        BLOCK_IN=BLOCK_IN,
    )

    # bias 在 Python 层加（kernel 外加，不影响 fused 的核心逻辑）
    if artifacts.bias is not None:
        out = out + artifacts.bias

    return out
