# llm-quant-profiler

A single-GPU profiler for the latency and memory tradeoffs of FP16 and bitsandbytes FP4 LLM inference.

On Qwen2.5-1.5B-Instruct and an RTX 4060 Laptop GPU, the fresh no-hook benchmark found that bitsandbytes INT4 used **56.5% less peak allocated VRAM** than FP16 but had **338.9% higher decode latency** (4.39×). A correctness-validated Triton fused k/v prototype was **51.7% slower** than the bitsandbytes INT4 baseline in the same benchmark.

**Canonical results:** [report](PHASE3_REPORT.md) · [machine-readable JSON](results/canonical.json) · [Chinese README](README_CN.md)

## Results

| Mode | Prefill | Decode | Decode throughput | Peak allocated VRAM |
|------|---------|--------|-------------------|---------------------|
| FP16 | 0.068 s (std 0.025) | 5.541 s (std 0.615) | 23.10 tok/s (std 2.56) | 3271.3 MB |
| INT4 (bitsandbytes FP4) | 0.288 s (std 0.037) | 24.319 s (std 0.796) | 5.26 tok/s (std 0.17) | 1423.9 MB |
| INT4 + fused k/v prototype | 5.995 s (std 1.010) | 36.895 s (std 2.634) | 3.47 tok/s (std 0.22) | 1435.0 MB |

*Qwen/Qwen2.5-1.5B-Instruct · 512-token prompt · 128 fixed decode steps · 2 warmups + 5 measured runs per mode · medians with sample standard deviation · PyTorch 2.5.1+cu121 · Transformers 5.3.0 · bitsandbytes 0.49.2 · Triton 3.1.0*

The primary table is measured with **no profiler hooks attached**. CUDA Event layer profiling is run separately and is used only for diagnosis and the figures below.

## What the profile shows

- INT4 reduces allocated memory substantially on this 8GB GPU, but the tested bitsandbytes FP4 path is much slower during batch-1 decode.
- Nine of the ten largest relative layer slowdowns are GQA `k_proj` or `v_proj` layers. The largest single outlier in this run is `model.layers.13.mlp.down_proj`.
- The storage-based Roofline estimate predicts fewer weight bytes for INT4, yet measured latency is much worse. Compression alone therefore does not guarantee a faster kernel.
- In bitsandbytes 0.49.2, eligible batch-1 decode shapes dispatch to a dedicated CUDA `gemv_4bit` path. The remaining bottleneck may involve on-the-fly unpack/scale work, instruction mix, occupancy, launch overhead, or shape-specific kernel efficiency; this repo does not isolate those causes yet.
- The fused k/v implementation is a correctness-first GEMV prototype. Its Python wrapper launches one Triton kernel per token row and does not outperform bitsandbytes end to end.

![Per-layer latency](outputs/fig1_layerwise_latency.png)

![Largest relative INT4 slowdowns](outputs/fig3_top10_slowest.png)

## Figure guide

- **Figure 1 — Decode slowdown by layer:** read the y-axis as INT4 latency change relative to FP16. Orange markers isolate `k_proj`/`v_proj`; the question is whether the tax is isolated or distributed.
- **Figure 2 — Peak allocated VRAM:** a direct memory comparison across modes. It is deliberately not labeled KV-cache growth because this experiment does not isolate cache growth.
- **Figure 3 — Top-10 layer cost:** the optimization priority list. Orange bars are k/v projections; gray bars are other layers.
- **Figure 4 — Storage-traffic Roofline:** faint points are individual layers and X markers are mode medians. All medians are far left of the ridge point; this is a workload-location diagnostic, not a causal explanation.

## Measurement design

`scripts/run_benchmark.py` exposes two explicitly separate paths:

- `--measurement-mode e2e`: attaches no hooks, uses the same requested prompt for prefill and decode, generates a fixed number of decode steps, and reports wall time, throughput, and peak allocated/reserved VRAM.
- `--measurement-mode profile`: attaches CUDA Event hooks to 197 target layers and exports per-layer CSVs. Per-layer synchronization changes wall time, so these runs are never used as the primary performance result.

The canonical runner executes FP16, INT4, and INT4-fused-k/v under both paths, records package/GPU/driver/Git metadata, regenerates the four charts, and writes the public report and JSON summary.

## Reproduce

Run inside WSL2 with an NVIDIA GPU:

```bash
bash setup.sh
source ~/.venvs/llm-quant-profiler/bin/activate
python scripts/run_phase3.py --local-files-only
```

Omit `--local-files-only` on the first run if the Hugging Face model is not cached.

For a quick E2E smoke test:

```bash
python scripts/run_benchmark.py \
  --quantization fp16 \
  --measurement-mode e2e \
  --prompt-len 32 \
  --max-new-tokens 8 \
  --warmup-runs 0 \
  --repeats 1
```

Raw metadata and layer CSVs are written under `data/` and gitignored. Public artifacts are written to `results/canonical.json`, `PHASE3_REPORT.md`, and `outputs/`.

## Interpretation boundaries

- These results apply to one model, one consumer GPU, batch size 1, and the exact software versions above.
- The Roofline figure models packed FP4 storage and block scales. It does not model unpack/codebook/scale instruction cost, occupancy, or cache behavior.
- The measured slowdown is established; its exact CUDA-kernel bottleneck still requires Nsight or equivalent kernel-level profiling.
- The custom kernel replaces only 56 k/v projections. It is not a general INT4 inference engine and should not be compared with optimized production fused kernels.
