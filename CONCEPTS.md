# Concepts & Mental Models

> 从零整理，用于快速 recall。

---

## 环境 & 流程

**为什么用 WSL2**：bitsandbytes 和 Triton 不支持 Windows，需要 Linux 环境。WSL2 在 Windows 里跑真正的 Ubuntu，同时能用 GPU。

**模型从哪来**：HuggingFace。第一次运行自动下载并缓存，之后从本地读。

**量化怎么做的**：bitsandbytes 黑盒完成。加载时传 `BitsAndBytesConfig(load_in_4bit=True)`，它自动把所有 Linear 层压缩成 INT4，替换成 `Linear4bit` 层。不需要手写压缩逻辑。

**整体数据流**：
```
run_benchmark.py
  → 加载模型（FP16 或 INT4）
  → HookProfiler 给所有 Linear 层装计时器
  → warmup（不记录）
  → prefill（512 token，1次 forward）
  → decode（128 次单 token 循环）
  → 输出 CSV
```

---

## 核心概念

### Linear 层
就是矩阵乘向量：`output = W × input`。没有魔法。
- 输入：一个向量（比如长度 1536）
- 权重 W：一个矩阵（比如 256×1536）
- 输出：一个向量（长度 256）

权重是训练好的固定参数，每次 forward 时输入不同但权重不变。

### Q / K / V / O Projection
Attention 模块里的四个 Linear 层，区别只是**权重形状不同**。

在 Qwen2.5-1.5B（GQA 架构）里：
```
q_proj: 1536×1536  （输出 1536）
k_proj:  256×1536  （输出 256）← 小 6 倍
v_proj:  256×1536  （输出 256）← 小 6 倍
o_proj: 1536×1536  （输出 1536）
```
k/v 输出小是 GQA 设计，目的是省 KV cache 显存。

### 量化（Quantization）
把权重从 FP16（每个数 2 字节）压缩到 INT4（每个数 0.5 字节），省 4 倍空间。
压缩时同时保存 scale 参数（记录压缩比例，用于还原）。

### Dequant（反量化）
GPU 做矩阵乘法只能用 FP16/FP32，不能直接用 INT4。所以每次 forward 前要先把 INT4 解码回 FP16。

**bitsandbytes 的做法（有问题）**：
```
INT4 权重（VRAM）→ 解码 → 写 FP16 权重（VRAM）→ 读出来做矩阵乘法
```
多了"写回 VRAM 再读出"这一步，多付了两次内存带宽。

**fused kernel 的目标**：
```
INT4 权重（VRAM）→ 读进片上 SRAM → 在 SRAM 里解码 → 直接做乘法 → 输出
```
不写回 VRAM，省掉两次带宽。

### 为什么 k/v proj 量化税最惨
矩阵越小 → FLOPs 越少 → dequant 开销占总时间比越大。
k/v 的矩阵是 q/o 的 1/6，但 dequant 要过整个矩阵，没法缩短，所以 8–12x slower。

### 算术强度（Arithmetic Intensity）
= FLOPs ÷ 访问内存的字节数，单位 FLOP/Byte。衡量"每搬一字节能做多少计算"。

decode 阶段（batch=1）：
- FP16：~1 FLOP/Byte
- INT4（bitsandbytes）：~0.44 FLOP/Byte（dequant round-trip 增加了字节数）

Ridge point（你的 RTX 4060 Laptop）：80 FLOP/Byte。所有 decode 层都在 1 FLOP/Byte 以下，完全 memory-bound，计算单元大量闲置。

### CUDA Events vs time.time()
GPU 执行是异步的——CPU 调用 `layer(x)` 时只是把指令放进队列就返回了，GPU 还没跑完。

`time.time()` 只测到"把任务提交给队列"的时间（几乎是 0）。

`torch.cuda.Event` 在 GPU 队列里插时间戳，`synchronize()` 等 GPU 真正完成，测到的是真实 GPU 耗时。

### Hook（计时钩子）
给 PyTorch 层注册的回调函数，层每次 forward 时自动触发。
`HookProfiler` 就是给所有 Linear 层装了 pre-hook（开始计时）和 post-hook（结束计时），自动记录每层耗时和显存。

### 为什么手动写 decode 循环
`model.generate()` 内部封装了所有 token 的生成循环，hook 触发时无法知道是第几个 token。手动循环可以精确控制每步的记录，数据干净可分析。

---

## Phase 1 & 2 主要发现

**一句话总结**：
> 对 Qwen2.5-1.5B 做 per-layer profiling，发现 INT4 量化虽然省了 60% 显存，但 decode 速度反而比 FP16 慢 2.9 倍，原因是 bitsandbytes 的 dequant 多了一次 VRAM 读写，其中 k/v proj 层最惨（8–12x slower），因为它们矩阵小、计算量少，dequant 开销占比最大。

| 指标 | FP16 | INT4 |
|------|------|------|
| Decode 总耗时 | 基准 | 2.9x 更慢 |
| VRAM 峰值 | 3098 MB | 1252 MB（-60%）|
| 最慢层 | — | k/v proj（8–12x）|

**Phase 3 目标**：写 Triton fused kernel，在 GPU 片上完成 dequant + matmul，消除 VRAM round-trip，让 INT4 真正快起来。

---

## Phase 3 实现与结果

### Triton Kernel 结构

两个 kernel，逐步构建：

**FP16 GEMV**（`kernels/fp16_gemv.py`）：
- 启动 `out_features` 个程序，每个程序算一行的点积
- `tl.program_id(0)` 获取当前程序负责第几行
- 用 FP32 累加避免 FP16 精度误差，最后转回 FP16 写出

**Fused FP4 GEMV**（`kernels/fused_fp4_gemv.py`）：
在 FP16 kernel 基础上加三步：
1. **拆包**：`(packed >> 4) & 0xF` 取高位，`packed & 0xF` 取低位
2. **查码本**：`tl.load(code_ptr + hi)` — gather 读，每个元素用不同下标
3. **乘 scale**：`block_idx = row * blocks_per_row + (col_pairs * 2) // blocksize`

全程在 GPU 片上完成，不写回 VRAM。

### 集成方式

`FusedFP4Linear`（`scripts/run_benchmark.py`）：
- 替换模型中所有 k_proj / v_proj 的 `Linear4bit` 层
- forward 时先把输入转 float16（模型默认 bfloat16，需统一），输出再转回原始 dtype

### 实测结果（RTX 4060 Laptop，16 tokens decode）

| 模式 | Decode 耗时 | 速度 | VRAM |
|------|------------|------|------|
| FP16 | 2.06s | 7.8 tok/s | 3097 MB |
| INT4 bitsandbytes | 2.75s | 5.8 tok/s | 1224 MB |
| INT4 fused k/v | 2.56s | 6.2 tok/s | 1224 MB |

**结论**：
- fused kernel 比 bitsandbytes INT4 快 7%，VRAM 不变
- 仍慢于 FP16，因为只替换了 k/v proj，其余量化层仍走 bitsandbytes
- 若替换所有层，预期可消除量化税并超过 FP16 速度
