#!/usr/bin/env python3
"""Generate canonical JSON, diagnostic charts, and the public Phase 3 report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.results import MODE_ORDER, build_canonical_result
from analysis.roofline import plot_roofline
from analysis.visualize import (
    MODE_LABELS,
    aggregate_layer_timings,
    compute_quant_tax_layers,
    compute_summary_stats,
    load_benchmark_metadata,
    load_data,
    plot_layerwise_latency,
    plot_memory_growth,
    plot_top10_slowest,
)


def _fmt_stat(stats: dict, precision: int = 2, suffix: str = "") -> str:
    return f"{stats['median']:.{precision}f}{suffix} (std {stats['std']:.{precision}f})"


def _fmt_change(value: float) -> str:
    return f"{abs(value):.1f}% {'slower' if value >= 0 else 'faster'}"


def _relative_path(from_path: Path, target: Path) -> str:
    return os.path.relpath(target, start=from_path.parent).replace("\\", "/")


def _fused_top10(tax_layers: pd.DataFrame, dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    top = tax_layers.head(10).copy()
    if top.empty or "int4-fused-kv_decode" not in dfs:
        return top
    fused = aggregate_layer_timings(dfs["int4-fused-kv_decode"])[
        ["layer_name", "time_ms_mean", "time_ms_std"]
    ].rename(
        columns={
            "time_ms_mean": "time_ms_mean_fused",
            "time_ms_std": "time_ms_std_fused",
        }
    )
    return top.merge(fused, on="layer_name", how="left")


def write_report(
    report_path: Path,
    canonical_path: Path,
    output_dir: Path,
    canonical: dict,
    tax_layers: pd.DataFrame,
    dfs: dict[str, pd.DataFrame],
):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    config = canonical["config"]
    modes = canonical["modes"]
    comparisons = canonical["comparisons"]
    stability = canonical["stability"]
    environment = canonical["environment_by_mode"]["fp16"]

    int4_delta = comparisons["int4_decode_latency_change_vs_fp16_pct"]
    fused_delta = comparisons["fused_decode_latency_change_vs_int4_pct"]
    vram_reduction = comparisons["int4_peak_vram_reduction_vs_fp16_pct"]
    top = _fused_top10(tax_layers, dfs)
    kv_count = (
        int(
            top["layer_name"].str.contains(r"\.self_attn\.[kv]_proj$").sum()
        )
        if not top.empty
        else 0
    )
    if kv_count == len(top):
        top_description = "In this run, all ten entries are k/v projections."
    else:
        top_description = (
            f"In this run, {kv_count} of the ten entries are k/v projections."
        )

    lines = [
        "# Phase 3 Quantization Tax Report",
        "",
        "## Executive Summary",
        "",
        "- Primary performance numbers come from an end-to-end path with no profiler hooks attached.",
        "- Per-layer CUDA Event timings are diagnostic measurements and are reported separately.",
        f"- bitsandbytes INT4 decode is **{_fmt_change(int4_delta)}** than FP16.",
        f"- The fused k/v prototype differs by **{fused_delta:.1f}%** from bitsandbytes INT4; this run does not demonstrate an end-to-end gain.",
        f"- INT4 peak allocated VRAM is **{vram_reduction:.1f}% lower** than FP16.",
        "",
        "## Experiment Setup",
        "",
        f"- Model: `{config['model']}`",
        f"- Hardware: `{environment['gpu_name']}`",
        f"- Prompt length: `{config['prompt_len']}` tokens",
        f"- Decode length: `{config['max_new_tokens']}` fixed steps",
        f"- E2E protocol: `{config['warmup_runs']}` warmups + `{config['repeats']}` measured runs per mode",
        "- Stability gate: Tukey outliers at most "
        f"`{stability['policy']['max_outlier_fraction_pct']:.1f}%`, retained decode CV at most "
        f"`{stability['policy']['max_retained_decode_cv_pct']:.1f}%`, and IQR/median at most "
        f"`{stability['policy']['max_iqr_over_median_pct']:.1f}%`",
        f"- Python / PyTorch / CUDA: `{environment['python_version']}` / `{environment['torch_version']}` / `{environment['torch_cuda_version']}`",
        f"- Transformers / bitsandbytes / Triton: `{environment['transformers_version']}` / `{environment['bitsandbytes_version']}` / `{environment['triton_version']}`",
        f"- NVIDIA driver: `{environment['nvidia_driver_version']}`",
        f"- Windows host power plan: `{environment['host_power_plan']}`",
        f"- Git commit / dirty: `{environment['git_commit']}` / `{environment['git_dirty']}`",
        "- Retained decode CV by mode: "
        + ", ".join(
            f"`{mode}` {stability['by_mode'][mode]['retained_cv_pct']:.1f}%"
            for mode in MODE_ORDER
        ),
        "",
        "## Canonical End-to-End Results",
        "",
        "| Mode | Prefill | Decode | Throughput | Peak allocated VRAM |",
        "|------|---------|--------|------------|---------------------|",
    ]

    for mode in MODE_ORDER:
        summary = modes[mode]["summary"]
        lines.append(
            f"| {MODE_LABELS[mode]} | "
            f"{_fmt_stat(summary['prefill_time_s'], 3, ' s')} | "
            f"{_fmt_stat(summary['decode_time_s'], 3, ' s')} | "
            f"{_fmt_stat(summary['decode_throughput_tps'], 2, ' tok/s')} | "
            f"{_fmt_stat(summary['peak_vram_mb'], 1, ' MB')} |"
        )

    lines += [
        "",
        "The fused path is a correctness-first GEMV prototype for 56 k/v projections, not a full quantized inference engine.",
        "",
        "## How to Read the Figures",
        "",
        "Figures 1, 3, and 4 come from diagnostic profile runs with CUDA Event hooks. Figure 2 uses canonical E2E peak-memory data. The diagnostic figures explain where the slowdown is concentrated; they are not the source of the E2E table above.",
        "",
        "### Figure 1: Decode slowdown across all layers",
        "",
        f"![Per-layer latency]({_relative_path(report_path, output_dir / 'fig1_layerwise_latency.png')})",
        "",
        "Read the y-axis as INT4 latency change relative to FP16. Positive values are slower; orange markers isolate `k_proj`/`v_proj`. The purpose is to see whether the tax is isolated or distributed across the model.",
        "",
        "### Figure 2: Peak allocated VRAM by mode",
        "",
        f"![Canonical E2E peak allocated memory]({_relative_path(report_path, output_dir / 'fig2_memory_growth.png')})",
        "",
        "This is a memory comparison, not a KV-cache growth curve. The main readout is the lower INT4 baseline and whether the fused prototype changes it.",
        "",
        "### Figure 3: Top-10 layer-level slowdowns",
        "",
        f"![Largest INT4 slowdowns]({_relative_path(report_path, output_dir / 'fig3_top10_slowest.png')})",
        "",
        "This is the optimization priority list. Orange bars are k/v projections; gray bars are other layers. "
        + top_description,
        "",
        "### Figure 4: Storage-traffic Roofline estimate",
        "",
        f"![Storage-traffic Roofline estimate]({_relative_path(report_path, output_dir / 'fig4_roofline.png')})",
        "",
        "The faint points are individual layers and X markers are mode medians. All medians are far left of the ridge point, so the profile is not compute-saturated; this plot is a location/scale diagnostic, not a causal proof of the slowdown.",
        "",
        "## Worst INT4 Layers",
        "",
        "| Layer | FP16 hook ms | INT4 hook ms | Fused hook ms | INT4 slowdown |",
        "|-------|--------------|--------------|---------------|---------------|",
    ]

    for _, row in top.iterrows():
        fused = row.get("time_ms_mean_fused")
        fused_text = "n/a" if pd.isna(fused) else f"{fused:.3f}"
        lines.append(
            f"| `{row['layer_name']}` | {row['time_ms_mean_fp16']:.3f} | "
            f"{row['time_ms_mean_int4']:.3f} | {fused_text} | +{row['slowdown_pct']:.1f}% |"
        )

    lines += [
        "",
        "## Interpretation Boundaries",
        "",
        "- This experiment directly supports the observed latency and VRAM tradeoff on the stated hardware and software stack.",
        "- For eligible batch-1 decode shapes, bitsandbytes 0.49.2 dispatches to a dedicated CUDA `gemv_4bit` path rather than materializing a full FP16 weight tensor in the Python/PyTorch path.",
        "- The storage-traffic Roofline estimate models packed FP4 weights and block scales, but not unpack/codebook/scale instruction cost, occupancy, launch overhead, or cache behavior. The exact kernel bottleneck still requires Nsight-level profiling.",
        "- The fused k/v result only evaluates the current Python wrapper and Triton prototype. It should not be generalized to optimized fused INT4 kernels.",
        "",
        "## Reproducibility",
        "",
        f"- Machine-readable summary: `{_relative_path(report_path, canonical_path)}`",
        "- Raw E2E metadata and layer CSVs are generated under `data/phase3/` and intentionally gitignored.",
        "- Recreate all public artifacts from WSL after activating the setup environment: `python scripts/run_phase3.py --local-files-only`.",
        "- On this Windows laptop, use `powershell -ExecutionPolicy Bypass -File scripts/run_phase3_windows.ps1 -LocalFilesOnly` to temporarily select and then restore the High performance host plan.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate canonical Phase 3 artifacts")
    parser.add_argument("--e2e-data-dir", required=True)
    parser.add_argument("--profile-data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--canonical-json", default="results/canonical.json")
    parser.add_argument("--report-path", default="PHASE3_REPORT.md")
    args = parser.parse_args()

    e2e_dir = Path(args.e2e_data_dir)
    profile_dir = Path(args.profile_data_dir)
    output_dir = Path(args.output_dir)
    canonical_path = Path(args.canonical_json)
    report_path = Path(args.report_path)

    e2e_metadata = [
        entry
        for entry in load_benchmark_metadata(str(e2e_dir))
        if entry.get("measurement_mode") == "e2e"
    ]
    dfs = load_data(str(profile_dir))
    if len(e2e_metadata) != len(MODE_ORDER):
        raise SystemExit(f"expected {len(MODE_ORDER)} E2E metadata files, found {len(e2e_metadata)}")
    if not dfs:
        raise SystemExit(f"no profile CSV files found under {profile_dir}")

    profile_summary = compute_summary_stats(dfs)
    tax_layers = compute_quant_tax_layers(dfs)
    canonical = build_canonical_result(
        e2e_metadata,
        profile_summary.to_dict(orient="records"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_layerwise_latency(dfs, str(output_dir))
    plot_memory_growth(canonical, str(output_dir))
    plot_top10_slowest(dfs, str(output_dir))
    plot_roofline(dfs, str(output_dir))
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.write_text(
        json.dumps(canonical, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(
        report_path,
        canonical_path,
        output_dir,
        canonical,
        tax_layers,
        dfs,
    )
    print(f"Saved canonical JSON: {canonical_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
