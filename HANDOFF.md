# Project Handoff

> 本文件记录项目完整上下文，供新对话快速接管。最后更新：2026-05-22。

---

## 用户背景

- CS + Applied Math 专业，目标申请 MLSys 方向 MS
- 有一定 MLSys 理论基础，实践经验极少（PyTorch、Triton 均零基础）
- 项目代码前期由 AI 生成，用户理解程度处于"大概能解释每一步在做什么"但"不能独立从零写"
- 本项目是 MS 申请的一个重点 artifact，也是进入 MLSys 社区的第一张名片
- 核心叙事：CS 泛全栈/HCI → MLSys，用实测数据证明有动手能力

---

## 项目目的

测量 INT4 量化的"量化税"：bitsandbytes 的 dequantization 会把权重写回 VRAM 再读出来做 matmul，引入额外内存带宽开销，导致 INT4 decode 反而比 FP16 更慢。用 Triton fused kernel 修复这个问题。

**硬件**：RTX 4060 Laptop 8GB（272 GB/s 带宽，ridge point 80 FLOP/Byte）
**模型**：Qwen/Qwen2.5-1.5B-Instruct
**运行环境**：WSL2 Ubuntu（bitsandbytes 和 Triton 不支持 Windows）

---

## 当前代码状态（全部完成）

```
llm-quant-profiler/
├── profiler/
│   ├── hook_profiler.py     # 给所有 Linear 层注册计时 hook
│   ├── metrics.py           # CUDATimer（用 CUDA Events，不是 time.time()）
│   └── phase_detector.py    # 判断 prefill / decode
├── scripts/
│   ├── run_benchmark.py     # 主入口，含 FusedFP4Linear 集成 ← 本次修改
│   ├── run_analysis.py      # Phase 2 分析图生成
│   ├── extract_linear4bit_reference.py  # 查看真实量化层结构
│   ├── verify_fp16_gemv.py  # 验证 FP16 kernel
│   └── verify_fused_fp4.py  # 验证 FP4 fused kernel
├── kernels/
│   ├── fp16_gemv.py         # Triton FP16 GEMV kernel ← 本次实现
│   └── fused_fp4_gemv.py    # Triton fused FP4 dequant+GEMV kernel ← 本次实现
├── analysis/
│   ├── visualize.py         # Phase 2 四张图
│   └── roofline.py          # Roofline 分析
├── CONCEPTS.md              # 核心概念手册（用户学习记录）← 本次新建
├── ROADMAP.md               # 项目进度和技术分析
├── HANDOFF.md               # 本文件
└── phase3_utils.py          # Linear4bitArtifacts 数据类 + 工具函数
```

---

## 三个阶段完成情况

### Phase 1 ✅ Benchmark + Profiler
- `HookProfiler` 给所有 `nn.Linear` / `nn.LayerNorm` 注册 pre/post hook
- 手动 decode 循环（不用 `model.generate()`），每 token 一条记录
- 输出 CSV：`layer_name, layer_type, phase, time_ms, mem_before_mb, ...`

### Phase 2 ✅ Analysis + Visualization
- 四张图已生成（outputs/ 目录）
- **已知 bug**：`visualize.py` 过滤 `layer_type == "Linear"` 漏掉了 `Linear4bit`，分析图的 INT4 layer 数据不完整。修复方式：把所有 `== "Linear"` 改为 `in ["Linear", "Linear4bit"]`。目前未修，不影响 Phase 3。

### Phase 3 ✅ Triton Fused Kernel
见下方详细说明。

---

## Phase 3 实现细节

### Step 1：FP16 GEMV kernel（`kernels/fp16_gemv.py`）

```python
@triton.jit
def _fp16_gemv_kernel(w_ptr, x_ptr, out_ptr, in_features, BLOCK):
    row = tl.program_id(0)          # 我负责第几行
    cols = tl.arange(0, BLOCK)
    mask = cols < in_features
    w_row = tl.load(w_ptr + row * in_features + cols, mask=mask, other=0.0)
    x_vec = tl.load(x_ptr + cols, mask=mask, other=0.0)
    result = tl.sum(w_row.to(tl.float32) * x_vec.to(tl.float32), axis=0)
    tl.store(out_ptr + row, result.to(tl.float16))
```

关键点：用 FP32 累加（FP16 累加 1536 次误差超标），启动 `out_features` 个程序并行。

### Step 2：FP4 Fused GEMV kernel（`kernels/fused_fp4_gemv.py`）

在 FP16 kernel 基础上，读 packed uint8 后加三步：

```
packed uint8（每字节 2 个 FP4 值）
  → 拆包：hi = (packed >> 4) & 0xF，lo = packed & 0xF
  → 查码本：tl.load(code_ptr + hi)   ← gather 读，16 个浮点值的查找表
  → 乘 scale：block_idx = row * (in_features//blocksize) + (col_pairs*2) // blocksize
  → 点积累加（FP32）→ 写出 FP16
```

FP4 格式说明（bitsandbytes）：
- `packed_weight` shape: `[out * in / 2, 1]`，每 uint8 装 2 个 4-bit 值
- `absmax` shape: `[out * in / blocksize]`，每 64 个权重共享 1 个 scale
- `code` shape: `[16]`，FP4 码本（16 个不均匀分布的浮点值）
- blocksize = 64

验证：`python scripts/verify_fused_fp4.py` → `allclose: True`

### Step 3：集成进 benchmark（`scripts/run_benchmark.py`）

新增 `FusedFP4Linear` 类替换 `Linear4bit`：

```python
class FusedFP4Linear(nn.Module):
    def forward(self, x):
        orig_dtype = x.dtype
        if x.dtype != torch.float16:
            x = x.to(torch.float16)   # 模型默认 bfloat16，kernel 要 float16
        # 逐向量调用 fused_fp4_gemv_triton
        ...
        return result.to(orig_dtype)  # 转回原始 dtype
```

`replace_kv_proj_with_fused(model)` 遍历模型，把所有 `k_proj` / `v_proj` 的 `Linear4bit` 替换成 `FusedFP4Linear`。共替换 56 层。

运行方式：
```bash
python scripts/run_benchmark.py --quantization int4-fused-kv --prompt-len 512 --max-new-tokens 128
```

---

## 实测结果

**完整 benchmark**（512 tokens prefill，128 tokens decode，RTX 4060 Laptop）：

| 模式 | Prefill | Decode | 速度 | VRAM |
|------|---------|--------|------|------|
| FP16 | 0.25s | 17.40s | 7.4 tok/s | 3100 MB |
| INT4 bitsandbytes | 0.27s | 22.10s | 5.8 tok/s | 1227 MB |
| INT4 fused k/v | 3.20s | 19.09s | 6.7 tok/s | 1227 MB |

**Decode 改善**：fused k/v 比 bitsandbytes INT4 快 **16%**（22.10s → 19.09s），VRAM 不变。

**Prefill 变慢说明**：fused kernel 是 GEMV（矩阵×向量），专为 decode（seq=1）设计。Prefill 时 seq=512，`FusedFP4Linear.forward()` 里有 Python for 循环跑 512 次，开销极大。这是已知设计限制，不是 bug。Prefill 需要 GEMM，是不同的优化问题（未来工作方向）。

---

## 核心结论（可用于申请材料）

> 对 Qwen2.5-1.5B 在 RTX 4060 Laptop 上做 per-layer profiling，发现 INT4 decode 比 FP16 慢 1.27×（22.10s vs 17.40s），VRAM 省 60%。瓶颈不是计算，而是 bitsandbytes 的 dequant 把权重写回 VRAM 再读出来做 matmul，引入了额外内存带宽开销（算术强度从 1 FLOP/Byte 降到 0.44）。k/v proj 层最惨（8–12x slower per layer），因为 GQA 导致这些矩阵只有 256×1536，计算量少但 dequant 开销不变。自己写了 Triton fused kernel，在 GPU 片上完成 dequant，消除 VRAM round-trip。替换 k/v proj（56 层）后，decode 比 bitsandbytes 快 16%，VRAM 保持相同。Prefill 需要 GEMM 而非 GEMV，是不同的优化问题，已识别为后续工作。

---

## 下一步（按优先级）

### 🔴 必须做（影响申请材料质量）

1. ~~跑完整 benchmark~~ ✅ 已完成，数据见上方

2. **写 README**（技术故事，不是代码文档）：
   - 我在测什么 / 核心发现（数字）/ 为什么慢 / 怎么修 / 结果
   - 目标读者：招生委员会 + MLSys 工程师，30 秒内看懂

3. **推到 GitHub**，公开可访问

### 🟡 有余力做

4. **修 Phase 2 bug**：`visualize.py` 里把 `layer_type == "Linear"` 改为 `in ["Linear", "Linear4bit"]`，重新生成四张分析图

5. **替换所有 Linear4bit 层**（不只 k/v），测试能否超过 FP16

### 🟢 加分项

6. 一篇技术博客（Medium 或个人网站），500 字讲清楚 quantization tax 是什么

---

## 我已掌握的概念

我能基本解释：
- Linear 层 = 矩阵乘向量
- 为什么 INT4 decode 比 FP16 慢（dequant VRAM round-trip）
- 为什么 k/v proj 最惨（GQA 小矩阵 + 固定 dequant 开销）
- CUDA Events vs time.time()（GPU 异步执行）
- Hook 是什么，为什么手动写 decode 循环
- Triton kernel 整体流程（能解释每步目的，不能独立从零写）
- FP4 存储格式（packed uint8、blockwise scale、code 码本）
- 算术强度和 Roofline 的含义

用户尚不熟悉（如需深入需要铺垫）：
- PyTorch API 细节
- Triton 语法细节
- Attention 机制的数学

---

## 重要文件路径

| 文件 | 内容 |
|------|------|
| `CONCEPTS.md` | 核心概念手册（用户学习笔记）|
| `ROADMAP.md` | 详细技术分析和数据解读 |
| `HANDOFF.md` | 本文件 |
| `kernels/fp16_gemv.py` | FP16 GEMV Triton kernel（已实现）|
| `kernels/fused_fp4_gemv.py` | FP4 fused kernel（已实现）|
| `scripts/run_benchmark.py` | 主 benchmark 脚本（含 FusedFP4Linear）|
| `data/` | benchmark CSV 输出（gitignore）|
