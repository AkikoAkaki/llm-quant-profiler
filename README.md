# llm-quant-profiler

Profiling INT4 LLM inference on a consumer GPU and improving decode throughput by 16%
over bitsandbytes INT4 with a fused Triton dequant+GEMV kernel.

On Qwen2.5-1.5B-Instruct with an RTX 4060 Laptop GPU, bitsandbytes INT4 reduced VRAM by 60% but made decode 1.27× slower than FP16. Replacing 56 k/v projection layers with a fused Triton kernel improved INT4 decode throughput from 5.8 to 6.7 tok/s.

**Key results:** 1.27× INT4 decode slowdown vs FP16 · 16% higher decode throughput than bitsandbytes INT4 · 60% VRAM reduction vs FP16

---

## Results

| Mode | Prefill | Decode | Decode throughput | Peak VRAM |
|------|---------|--------|-------------------|-----------|
| FP16 | 0.25s | 17.40s | 7.4 tok/s | 3100 MB |
| INT4 (bitsandbytes) | 0.27s | 22.10s | 5.8 tok/s | 1227 MB |
| INT4 + fused k/v kernel | 3.20s | 19.09s | 6.7 tok/s | 1227 MB |

*Qwen2.5-1.5B-Instruct · 512-token prompt · 128 decode steps · RTX 4060 Laptop 8GB*

Decode is the target workload. The fused kernel is GEMV-oriented (seq=1); prefill requires GEMM and is not optimized — see [Limitations](#limitations).

---

## The Problem

Profiling is consistent with extra global-memory traffic associated with INT4 dequantization before matmul. On a memory-bound GPU, this lowers effective arithmetic intensity:

- **FP16:** ~1 FLOP/Byte
- **INT4 (bitsandbytes):** ~0.44 FLOP/Byte

Lower arithmetic intensity means slower execution when the GPU is memory-bound — which decode always is (ridge point: 80 FLOP/Byte on RTX 4060 Laptop).

The worst offenders are `k_proj` and `v_proj`. Qwen2.5-1.5B uses GQA (Grouped Query Attention), so these projections output only 256 features vs. 1536 for other projections. Fewer FLOPs, same dequantization overhead → **up to ~380% (4–5×) slower than FP16** per layer.

![Per-layer latency: FP16 vs INT4 decode](outputs/fig1_layerwise_latency.png)
![Roofline analysis](outputs/fig4_roofline.png)

---

## Approach

### 1. Layer-by-layer profiling

Registered CUDA Event hooks on all `nn.Linear` and `Linear4bit` layers across the model. Manually drove the decode loop token-by-token (instead of `model.generate()`) to isolate each step. Captured timing, memory snapshots, and input shapes for 6,300+ layer calls per run.

### 2. Roofline analysis

Computed arithmetic intensity for each layer type under FP16 and INT4 regimes. All decode layers sit deep in the memory-bound region. INT4 dequantization pushes arithmetic intensity further left on the roofline — the opposite of what quantization is supposed to achieve.

### 3. Fused Triton kernel

Wrote a custom Triton GEMV kernel that performs dequantization inside the kernel before accumulation, avoiding materializing dequantized weights as a separate intermediate tensor. Replaced `k_proj` and `v_proj` in all 28 attention layers (56 projections total) with `FusedFP4Linear`, which calls this kernel directly.

---

## Reproduce

> Requires WSL2 Ubuntu. bitsandbytes and Triton do not run on native Windows.

```bash
bash setup.sh && source venv/bin/activate

python scripts/run_benchmark.py --quantization fp16 --prompt-len 512 --max-new-tokens 128
python scripts/run_benchmark.py --quantization int4 --prompt-len 512 --max-new-tokens 128
python scripts/run_benchmark.py --quantization int4-fused-kv --prompt-len 512 --max-new-tokens 128

python scripts/run_analysis.py
```

Model: `Qwen/Qwen2.5-1.5B-Instruct` (~3GB download on first run). Results written to `data/` (gitignored). Charts written to `outputs/`.

---

## Limitations

The fused kernel is GEMV-oriented: it processes one token at a time (sequence length = 1), which is the decode regime. Prefill passes a full sequence through the model at once, requiring GEMM rather than GEMV. This is why the fused k/v kernel improves decode throughput but significantly slows prefill (0.27s → 3.20s). Optimizing prefill is a separate problem requiring a different kernel design.

Only `k_proj` and `v_proj` were replaced. Other `Linear4bit` layers still use bitsandbytes.

---

[中文版 README](README_CN.md)
