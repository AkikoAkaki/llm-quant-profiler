"""Phase 2 analysis helpers and chart generation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

MODE_ORDER = ("fp16", "int4", "int4-fused-kv")
MODE_LABELS = {
    "fp16": "FP16",
    "int4": "INT4",
    "int4-fused-kv": "INT4 + fused k/v",
}
MODE_COLORS = {
    "fp16": "#4C72B0",
    "int4": "#DD8452",
    "int4-fused-kv": "#55A868",
}

CSV_NAME_RE = re.compile(r"^(fp16|int4|int4-fused-kv)_(prefill|decode)\.csv$")
LAYER_ORDER = {
    "input_layernorm": 0,
    "self_attn.q_proj": 1,
    "self_attn.k_proj": 2,
    "self_attn.v_proj": 3,
    "self_attn.o_proj": 4,
    "post_attention_layernorm": 5,
    "mlp.gate_proj": 6,
    "mlp.up_proj": 7,
    "mlp.down_proj": 8,
    "norm": 9,
    "lm_head": 10,
}


def is_linear_layer_type(layer_type: str) -> bool:
    """Return True for both FP16 and bitsandbytes linear modules."""
    return str(layer_type).startswith("Linear") or str(layer_type) == "FusedFP4Linear"


def _sort_key(layer_name: str) -> tuple:
    match = re.search(r"model\.layers\.(\d+)\.(.+)", layer_name)
    if match:
        layer_idx = int(match.group(1))
        suffix = match.group(2)
        return (0, layer_idx, LAYER_ORDER.get(suffix, 999), suffix)
    return (1, 0, LAYER_ORDER.get(layer_name, 999), layer_name)


def shorten_layer_name(layer_name: str) -> str:
    """Compact layer labels for plots and tables."""
    return layer_name.replace("model.layers.", "L")


def _infer_decode_steps(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill decode_step for older CSVs that predate the explicit column."""
    df = df.copy()
    df["decode_step"] = df.groupby("layer_name").cumcount()
    return df


def _normalize_df(df: pd.DataFrame, phase: str, fallback_run_id: str) -> pd.DataFrame:
    df = df.copy()

    if "run_id" not in df.columns:
        df["run_id"] = fallback_run_id
    df["run_id"] = df["run_id"].fillna(fallback_run_id).astype(str)

    if "decode_step" not in df.columns:
        if phase == "decode":
            df = _infer_decode_steps(df)
        else:
            df["decode_step"] = pd.NA

    df["decode_step"] = pd.to_numeric(df["decode_step"], errors="coerce").astype("Int64")
    return df


def load_data(data_dir: str) -> dict[str, pd.DataFrame]:
    """Recursively load all matching benchmark CSVs under a directory."""
    root = Path(data_dir)
    dfs: dict[str, list[pd.DataFrame]] = {}

    for path in sorted(root.rglob("*.csv")):
        match = CSV_NAME_RE.match(path.name)
        if not match:
            continue

        quant, phase = match.groups()
        key = f"{quant}_{phase}"
        fallback_run_id = path.parent.name if path.parent != root else path.stem
        df = pd.read_csv(path)
        df = _normalize_df(df, phase=phase, fallback_run_id=fallback_run_id)
        df["source_csv"] = str(path)
        dfs.setdefault(key, []).append(df)

    return {
        key: pd.concat(parts, ignore_index=True)
        for key, parts in dfs.items()
        if parts
    }


def load_benchmark_metadata(data_dir: str) -> list[dict]:
    """Load per-run metadata JSON files when available."""
    root = Path(data_dir)
    results = []
    for path in sorted(root.rglob("benchmark_metadata.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def aggregate_layer_timings(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-layer timings across decode steps and repeated runs."""
    if df.empty:
        return pd.DataFrame(columns=[
            "layer_name",
            "layer_type",
            "time_ms_mean",
            "time_ms_std",
            "input_shape",
            "output_shape",
        ])

    per_run = (
        df.groupby(["run_id", "layer_name"], as_index=False)
        .agg(
            time_ms=("time_ms", "mean"),
            layer_type=("layer_type", "first"),
            input_shape=("input_shape", "first"),
            output_shape=("output_shape", "first"),
        )
    )

    agg = (
        per_run.groupby("layer_name", as_index=False)
        .agg(
            layer_type=("layer_type", "first"),
            time_ms_mean=("time_ms", "mean"),
            time_ms_std=("time_ms", "std"),
            input_shape=("input_shape", "first"),
            output_shape=("output_shape", "first"),
        )
    )
    agg["time_ms_std"] = agg["time_ms_std"].fillna(0.0)
    agg["sort_key"] = agg["layer_name"].map(_sort_key)
    agg = agg.sort_values("sort_key").drop(columns="sort_key").reset_index(drop=True)
    return agg


def compute_summary_stats(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute per-phase summary stats across repeated runs."""
    rows = []
    for quant in MODE_ORDER:
        for phase in ("prefill", "decode"):
            key = f"{quant}_{phase}"
            if key not in dfs:
                continue

            df = dfs[key]
            total_per_run = df.groupby("run_id")["time_ms"].sum()
            peak_per_run = df.groupby("run_id")["mem_peak_mb"].max()
            per_run_layer = (
                df.groupby(["run_id", "layer_name"], as_index=False)["time_ms"]
                .mean()
            )
            avg_layer_per_run = per_run_layer.groupby("run_id")["time_ms"].mean()
            layer_agg = aggregate_layer_timings(df)
            slowest_layer = layer_agg.iloc[layer_agg["time_ms_mean"].idxmax()]

            rows.append({
                "quantization": quant,
                "phase": phase,
                "num_runs": int(df["run_id"].nunique()),
                "records": int(len(df)),
                "num_layers": int(df["layer_name"].nunique()),
                "total_time_ms_mean": float(total_per_run.mean()),
                "total_time_ms_std": float(total_per_run.std(ddof=1) if len(total_per_run) > 1 else 0.0),
                "peak_vram_mb_mean": float(peak_per_run.mean()),
                "peak_vram_mb_std": float(peak_per_run.std(ddof=1) if len(peak_per_run) > 1 else 0.0),
                "avg_layer_time_ms_mean": float(avg_layer_per_run.mean()),
                "avg_layer_time_ms_std": float(avg_layer_per_run.std(ddof=1) if len(avg_layer_per_run) > 1 else 0.0),
                "slowest_layer": str(slowest_layer["layer_name"]),
                "slowest_layer_time_ms_mean": float(slowest_layer["time_ms_mean"]),
                "slowest_layer_time_ms_std": float(slowest_layer["time_ms_std"]),
            })

    return pd.DataFrame(rows)


def compute_quant_tax_layers(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Find decode layers where INT4 is slower than FP16."""
    if "fp16_decode" not in dfs or "int4_decode" not in dfs:
        return pd.DataFrame()

    fp16 = aggregate_layer_timings(dfs["fp16_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    int4 = aggregate_layer_timings(dfs["int4_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    merged = fp16.merge(int4, on="layer_name", suffixes=("_fp16", "_int4"))
    merged["slowdown_pct"] = (merged["time_ms_mean_int4"] / merged["time_ms_mean_fp16"] - 1.0) * 100.0
    merged = merged[merged["slowdown_pct"] > 0].sort_values("slowdown_pct", ascending=False)
    return merged.reset_index(drop=True)


def plot_layerwise_latency(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Figure 1: decode slowdown distribution across all layers."""
    output_dir = Path(output_dir)
    if "fp16_decode" not in dfs or "int4_decode" not in dfs:
        print("Skipping layerwise slowdown chart: decode data missing.")
        return

    fp16 = aggregate_layer_timings(dfs["fp16_decode"]).set_index("layer_name")
    int4 = aggregate_layer_timings(dfs["int4_decode"]).set_index("layer_name")
    layers = fp16.index.intersection(int4.index)
    fp16_mean = fp16.loc[layers, "time_ms_mean"]
    int4_mean = int4.loc[layers, "time_ms_mean"]
    slowdown = (int4_mean / fp16_mean - 1.0) * 100.0
    ordered_layers = sorted(layers, key=_sort_key)
    slowdown = slowdown.reindex(ordered_layers)
    x = np.arange(len(ordered_layers))
    is_kv = np.array([
        name.endswith(".k_proj") or name.endswith(".v_proj")
        for name in ordered_layers
    ])

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.scatter(
        x[~is_kv], slowdown.to_numpy()[~is_kv],
        s=18, color="#999999", alpha=0.65, label="Other layers",
    )
    ax.scatter(
        x[is_kv], slowdown.to_numpy()[is_kv],
        s=26, color=MODE_COLORS["int4"], alpha=0.9, label="k_proj / v_proj",
    )
    median_slowdown = float(slowdown.median())
    ax.axhline(
        median_slowdown, color=MODE_COLORS["int4"], linestyle="--", linewidth=1.2,
        label=f"All-layer median: {median_slowdown:.0f}%",
    )
    top_layer = slowdown.idxmax()
    top_index = ordered_layers.index(top_layer)
    ax.annotate(
        f"{shorten_layer_name(top_layer)}\n{slowdown.loc[top_layer]:+.0f}%",
        xy=(top_index, slowdown.loc[top_layer]),
        xytext=(12, -28), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
        fontsize=9,
    )
    ax.set_title("Decode slowdown by layer: INT4 relative to FP16", fontsize=13, fontweight="bold")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("INT4 latency change (%)")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.legend(loc="upper right", fontsize=9)
    ax.text(
        0.01, 0.02,
        "Positive values are slower; colored markers isolate GQA k/v projections.",
        transform=ax.transAxes, fontsize=9, color="#555555",
    )
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig1_layerwise_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_memory_growth(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Figure 2: peak allocated VRAM by mode."""
    output_dir = Path(output_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Peak allocated VRAM by mode",
                 fontsize=14, fontweight="bold")
    modes, values, errors = [], [], []
    for quant in MODE_ORDER:
        key = f"{quant}_decode"
        if key not in dfs:
            continue
        per_run = dfs[key].groupby("run_id")["mem_peak_mb"].max()
        modes.append(quant)
        values.append(float(per_run.mean()))
        errors.append(float(per_run.std(ddof=1)) if len(per_run) > 1 else 0.0)

    x = np.arange(len(modes))
    bars = ax.bar(
        x, values, yerr=errors, capsize=4,
        color=[MODE_COLORS[mode] for mode in modes], alpha=0.88,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.025,
            f"{value:.0f} MB",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODE_LABELS[mode] for mode in modes])
    ax.set_ylabel("Peak allocated VRAM (MB)")
    ax.text(
        0.01, 0.96,
        "This is a memory comparison, not a KV-cache growth measurement.",
        transform=ax.transAxes, va="top", fontsize=9, color="#555555",
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f} MB"))
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig2_memory_growth.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_top10_slowest(dfs: dict[str, pd.DataFrame], output_dir: str):
    """Figure 3: top-10 decode layers by relative slowdown."""
    output_dir = Path(output_dir)
    if "fp16_decode" not in dfs or "int4_decode" not in dfs:
        print("Skipping Top-10 chart: decode data missing.")
        return

    fp16 = aggregate_layer_timings(dfs["fp16_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    int4 = aggregate_layer_timings(dfs["int4_decode"])[["layer_name", "time_ms_mean", "time_ms_std"]]
    combined = fp16.merge(int4, on="layer_name", suffixes=("_fp16", "_int4"))
    combined["slowdown_pct"] = (combined["time_ms_mean_int4"] / combined["time_ms_mean_fp16"] - 1.0) * 100.0
    top10 = combined[combined["slowdown_pct"] > 0].nlargest(10, "slowdown_pct").sort_values("slowdown_pct")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Where INT4 pays the largest layer-level cost", fontsize=12, fontweight="bold")
    y = np.arange(len(top10))
    colors = [
        MODE_COLORS["int4"] if (name.endswith(".k_proj") or name.endswith(".v_proj"))
        else "#777777"
        for name in top10["layer_name"]
    ]
    bars = ax.barh(y, top10["slowdown_pct"], color=colors, alpha=0.88)
    for bar, pct in zip(bars, top10["slowdown_pct"]):
        ax.text(bar.get_width() + 6, bar.get_y() + bar.get_height() / 2,
                f"+{pct:.0f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([shorten_layer_name(name) for name in top10["layer_name"]], fontsize=8)
    ax.set_xlabel("INT4 latency increase vs FP16 (%)")
    ax.text(
        0.99, 0.02,
        "Orange = k_proj/v_proj; gray = other layer.",
        transform=ax.transAxes, ha="right", fontsize=9, color="#555555",
    )
    sns.despine(ax=ax)

    plt.tight_layout()
    out = output_dir / "fig3_top10_slowest.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
