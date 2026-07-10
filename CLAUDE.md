# CLAUDE.md

## Project purpose

`llm-quant-profiler` measures the latency and memory tradeoffs of FP16 and bitsandbytes FP4 inference on an RTX 4060 Laptop GPU. It also contains a correctness-first Triton fused FP4 GEMV prototype for Qwen2.5 k/v projections.

For eligible batch-1 decode shapes, bitsandbytes 0.49.2 dispatches to a dedicated CUDA `gemv_4bit` path. Do not claim that the canonical decode baseline materializes a full FP16 weight tensor. The exact low-bit kernel bottleneck remains unverified without Nsight-level profiling.

## Environment

- Run GPU workflows inside WSL2; bitsandbytes and Triton are not supported by this project on native Windows.
- Target model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Create the locked WSL environment with `bash setup.sh`.
- Activate the default Linux-filesystem environment with `source ~/.venvs/llm-quant-profiler/bin/activate`, then use `python` for documented commands.

## Commands

```bash
# Canonical three-mode workflow: no-hook E2E + diagnostic profile
python scripts/run_phase3.py --local-files-only

# Small no-hook smoke test
python scripts/run_benchmark.py \
  --quantization fp16 --measurement-mode e2e \
  --prompt-len 32 --max-new-tokens 8 \
  --warmup-runs 0 --repeats 1 --local-files-only

# Kernel correctness
python scripts/verify_fused_fp4.py --local-files-only

# CPU-only aggregation tests
python -m unittest discover -s tests -v
```

## Measurement contract

- `--measurement-mode e2e` must not attach profiler hooks. It uses one prompt for prefill and decode, generates the requested fixed number of decode steps, and records wall time, throughput, and peak allocated/reserved VRAM.
- `--measurement-mode profile` attaches CUDA Event hooks to `nn.Linear`, `nn.LayerNorm`, bitsandbytes `Linear4bit`, and project-local `FusedFP4Linear`. Per-layer synchronization changes wall time, so profile timing is diagnostic only.
- Canonical E2E defaults: 512-token prompt, 128 decode steps, 8 warmups, 7 measured runs per mode.
- Canonical profile defaults: 1 uninstrumented warmup, 3 instrumented runs per mode.
- Before each E2E mode, require three consecutive GPU-utilization samples at or below 15%; then run all warmups and measured repetitions back-to-back so GPU clock state can reach steady state. Record GPU telemetry before and after each timed run.
- A canonical artifact is emitted only if every mode passes the predeclared stability gate: at most one Tukey outlier in seven runs, retained decode CV at most 15%, and IQR/median at most 30%.
- Always report medians for primary E2E metrics and include sample standard deviation.

## Data flow

```text
scripts/run_phase3.py
  -> scripts/run_benchmark.py --measurement-mode e2e
  -> scripts/run_benchmark.py --measurement-mode profile
  -> scripts/run_analysis.py
       -> results/canonical.json
       -> PHASE3_REPORT.md
       -> outputs/*.png
```

Raw run metadata and layer CSVs stay under gitignored `data/`. The tracked public source of truth is `results/canonical.json`; the report and READMEs must agree with it.

## Development status

| Phase | Status | Main artifacts |
|-------|--------|----------------|
| Layer profiler and benchmark | Done | `profiler/`, `scripts/run_benchmark.py` |
| Analysis and visualization | Done | `analysis/`, `outputs/` |
| Fused k/v prototype | Done as correctness-first prototype | `kernels/fused_fp4_gemv.py`, `scripts/verify_fused_fp4.py` |
| Canonical evidence chain | Done | `scripts/run_phase3.py`, `results/canonical.json`, `PHASE3_REPORT.md` |

Do not hand-edit generated result numbers before updating the canonical JSON and regenerating the report. Do not describe the fused prototype as a production-speed replacement unless a fresh canonical run supports that claim.
