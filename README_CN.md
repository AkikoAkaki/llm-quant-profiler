# llm-quant-profiler

在消费级 GPU 上对 INT4 LLM 推理进行性能剖析，并通过融合 Triton dequant+GEMV 内核将 decode 吞吐量相比 bitsandbytes INT4 提升 16%。

在 RTX 4060 Laptop GPU 上，bitsandbytes INT4 将显存占用减少 60%，但 decode 速度比 FP16 慢 1.27 倍。将 56 个 k/v 投影层替换为融合 Triton 内核后，INT4 decode 吞吐量从 5.8 提升至 6.7 tok/s。

**核心结论：** INT4 decode 比 FP16 慢 1.27× · 比 bitsandbytes INT4 decode 吞吐量高 16% · 显存占用比 FP16 少 60%

---

## 实测结果

| 模式 | Prefill | Decode | Decode 吞吐量 | 峰值显存 |
|------|---------|--------|---------------|---------|
| FP16 | 0.25s | 17.40s | 7.4 tok/s | 3100 MB |
| INT4 (bitsandbytes) | 0.27s | 22.10s | 5.8 tok/s | 1227 MB |
| INT4 + 融合 k/v 内核 | 3.20s | 19.09s | 6.7 tok/s | 1227 MB |

*Qwen2.5-1.5B-Instruct · 512 token prompt · 128 decode steps · RTX 4060 Laptop 8GB*

Decode 是目标工作负载。融合内核面向 GEMV（seq=1）设计；Prefill 需要 GEMM，未做优化——详见[局限性](#局限性)。

---

## 问题所在

性能剖析结果与以下假设一致：INT4 去量化在 matmul 前引入了额外的全局内存流量。在内存带宽受限的 GPU 上，这会降低有效算术强度：

- **FP16：** ~1 FLOP/Byte
- **INT4（bitsandbytes）：** ~0.44 FLOP/Byte

算术强度越低，在内存受限的 GPU 上执行越慢——decode 阶段始终处于内存受限区间（RTX 4060 Laptop 脊点：80 FLOP/Byte）。

最惨的层是 `k_proj` 和 `v_proj`。Qwen2.5-1.5B 使用 GQA（Grouped Query Attention），这两个投影的输出维度只有 256，而其他投影是 1536。FLOP 少、去量化开销相同 → **每层最高比 FP16 慢约 380%（4–5 倍）**。

![各层延迟对比：FP16 vs INT4 decode](outputs/fig1_layerwise_latency.png)
![Roofline 分析](outputs/fig4_roofline.png)

---

## 方法

### 1. 逐层性能剖析

在模型所有 `nn.Linear` 和 `Linear4bit` 层上注册 CUDA Event hook，手动驱动 decode 循环（逐 token，不使用 `model.generate()`），记录每步的计时、显存快照和输入形状，每次运行采集 6300+ 条层调用记录。

### 2. Roofline 分析

计算 FP16 和 INT4 两种制式下各层的算术强度。所有 decode 层均深陷内存受限区间。INT4 去量化使算术强度在 Roofline 图上进一步左移——与量化本该带来的收益相反。

### 3. 融合 Triton 内核

编写自定义 Triton GEMV 内核，在内核内部完成去量化后再做累加，避免将去量化权重具象化为独立的中间张量。将全部 28 个注意力层的 `k_proj` 和 `v_proj`（共 56 个投影）替换为 `FusedFP4Linear`，直接调用该内核。

---

## 复现

> 需要 WSL2 Ubuntu。bitsandbytes 和 Triton 不支持 Windows 原生环境。

```bash
bash setup.sh && source venv/bin/activate

python scripts/run_benchmark.py --quantization fp16 --prompt-len 512 --max-new-tokens 128
python scripts/run_benchmark.py --quantization int4 --prompt-len 512 --max-new-tokens 128
python scripts/run_benchmark.py --quantization int4-fused-kv --prompt-len 512 --max-new-tokens 128

python scripts/run_analysis.py
```

模型：`Qwen/Qwen2.5-1.5B-Instruct`（首次运行约下载 3GB）。结果写入 `data/`（gitignore）。图表写入 `outputs/`。

---

## 局限性

融合内核面向 GEMV 设计：每次处理一个 token（序列长度 = 1），即 decode 模式。Prefill 将完整序列一次性送入模型，需要 GEMM 而非 GEMV。因此融合 k/v 内核提升了 decode 吞吐量，但显著拖慢了 prefill（0.27s → 3.20s）。优化 prefill 是独立的问题，需要不同的内核设计。

仅替换了 `k_proj` 和 `v_proj`，其余 `Linear4bit` 层仍使用 bitsandbytes。

---

[English README](README.md)
