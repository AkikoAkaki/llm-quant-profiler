# Phase 3 Quantization Tax Report

## Executive Summary

- Primary performance numbers come from an end-to-end path with no profiler hooks attached.
- Per-layer CUDA Event timings are diagnostic measurements and are reported separately.
- bitsandbytes INT4 decode is **338.9% slower** than FP16.
- The fused k/v prototype is **51.7% slower** than bitsandbytes INT4.
- INT4 peak allocated VRAM is **56.5% lower** than FP16.

## Experiment Setup

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Hardware: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Prompt length: `512` tokens
- Decode length: `128` fixed steps
- E2E protocol: `2` warmups + `5` measured runs per mode
- Python / PyTorch / CUDA: `3.11.15` / `2.5.1+cu121` / `12.1`
- Transformers / bitsandbytes / Triton: `5.3.0` / `0.49.2` / `3.1.0`
- NVIDIA driver: `610.47`

## Canonical End-to-End Results

| Mode | Prefill | Decode | Throughput | Peak allocated VRAM |
|------|---------|--------|------------|---------------------|
| FP16 | 0.068 s (std 0.025) | 5.541 s (std 0.615) | 23.10 tok/s (std 2.56) | 3271.3 MB (std 0.0) |
| INT4 | 0.288 s (std 0.037) | 24.319 s (std 0.796) | 5.26 tok/s (std 0.17) | 1423.9 MB (std 0.0) |
| INT4 + fused k/v | 5.995 s (std 1.010) | 36.895 s (std 2.634) | 3.47 tok/s (std 0.22) | 1435.0 MB (std 0.0) |

The fused path is a correctness-first GEMV prototype for 56 k/v projections, not a full quantized inference engine.

## How to Read the Figures

These figures come from diagnostic profile runs with CUDA Event hooks. They explain where the slowdown is concentrated; they are not the source of the E2E table above.

### Figure 1: Decode slowdown across all layers

![Per-layer latency](outputs/fig1_layerwise_latency.png)

Read the y-axis as INT4 latency change relative to FP16. Positive values are slower; orange markers isolate `k_proj`/`v_proj`. The purpose is to see whether the tax is isolated or distributed across the model.

### Figure 2: Peak allocated VRAM by mode

![Allocated memory during instrumented decode](outputs/fig2_memory_growth.png)

This is a memory comparison, not a KV-cache growth curve. The main readout is the lower INT4 baseline and whether the fused prototype changes it.

### Figure 3: Top-10 layer-level slowdowns

![Largest INT4 slowdowns](outputs/fig3_top10_slowest.png)

This is the optimization priority list. Orange bars are k/v projections; gray bars are other layers. In this run, nine of the ten largest relative slowdowns are k/v, with one MLP outlier.

### Figure 4: Storage-traffic Roofline estimate

![Storage-traffic Roofline estimate](outputs/fig4_roofline.png)

The faint points are individual layers and X markers are mode medians. All medians are far left of the ridge point, so the profile is not compute-saturated; this plot is a location/scale diagnostic, not a causal proof of the slowdown.

## Worst INT4 Layers

| Layer | FP16 hook ms | INT4 hook ms | Fused hook ms | INT4 slowdown |
|-------|--------------|--------------|---------------|---------------|
| `model.layers.13.mlp.down_proj` | 0.310 | 1.501 | 2.291 | +383.7% |
| `model.layers.15.self_attn.k_proj` | 0.271 | 0.656 | 1.148 | +141.7% |
| `model.layers.19.self_attn.v_proj` | 0.219 | 0.510 | 0.773 | +132.7% |
| `model.layers.0.self_attn.v_proj` | 0.249 | 0.572 | 0.876 | +129.3% |
| `model.layers.18.self_attn.v_proj` | 0.221 | 0.504 | 0.766 | +128.3% |
| `model.layers.0.self_attn.k_proj` | 0.298 | 0.672 | 1.157 | +125.6% |
| `model.layers.20.self_attn.v_proj` | 0.219 | 0.492 | 0.748 | +124.4% |
| `model.layers.1.self_attn.v_proj` | 0.227 | 0.502 | 0.840 | +121.4% |
| `model.layers.17.self_attn.k_proj` | 0.264 | 0.578 | 1.135 | +119.0% |
| `model.layers.13.self_attn.v_proj` | 0.218 | 0.475 | 0.767 | +117.7% |

## Interpretation Boundaries

- This experiment directly supports the observed latency and VRAM tradeoff on the stated hardware and software stack.
- For eligible batch-1 decode shapes, bitsandbytes 0.49.2 dispatches to a dedicated CUDA `gemv_4bit` path rather than materializing a full FP16 weight tensor in the Python/PyTorch path.
- The storage-traffic Roofline estimate models packed FP4 weights and block scales, but not unpack/codebook/scale instruction cost, occupancy, launch overhead, or cache behavior. The exact kernel bottleneck still requires Nsight-level profiling.
- The fused k/v result only evaluates the current Python wrapper and Triton prototype. It should not be generalized to optimized fused INT4 kernels.

## Reproducibility

- Machine-readable summary: `results/canonical.json`
- Raw E2E metadata and layer CSVs are generated under `data/phase3/` and intentionally gitignored.
- Recreate all public artifacts from WSL after activating the setup environment: `python scripts/run_phase3.py --local-files-only`.
