"""Pure helpers for canonical benchmark aggregation."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone


MODE_ORDER = ("fp16", "int4", "int4-fused-kv")
MAX_CANONICAL_DECODE_CV_PCT = 15.0
MAX_CANONICAL_OUTLIER_FRACTION_PCT = 15.0
MAX_CANONICAL_IQR_OVER_MEDIAN_PCT = 30.0


def summarize_values(values: list[float]) -> dict[str, float | int]:
    """Return stable summary statistics for a non-empty list."""
    if not values:
        raise ValueError("cannot summarize an empty list")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "median": statistics.median(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "min": min(numeric),
        "max": max(numeric),
    }


def percent_change(new: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("baseline must be non-zero")
    return (new / baseline - 1.0) * 100.0


def percent_reduction(new: float, baseline: float) -> float:
    return -percent_change(new, baseline)


def coefficient_of_variation_pct(summary: dict) -> float:
    mean = float(summary["mean"])
    if mean == 0:
        return 0.0
    return abs(float(summary["std"]) / mean) * 100.0


def summarize_stability(values: list[float]) -> dict[str, float | int]:
    """Summarize repeat stability with a predeclared Tukey outlier policy."""
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("cannot assess stability without samples")
    raw = summarize_values(numeric)
    if len(numeric) < 4:
        q1 = raw["min"]
        q3 = raw["max"]
        retained = numeric
    else:
        q1, _, q3 = statistics.quantiles(numeric, n=4, method="inclusive")
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        retained = [value for value in numeric if lower <= value <= upper]
    retained_summary = summarize_values(retained)
    outlier_count = len(numeric) - len(retained)
    median = float(raw["median"])
    iqr_over_median_pct = (
        abs((q3 - q1) / median) * 100.0 if median != 0 else 0.0
    )
    return {
        "count": len(numeric),
        "retained_count": len(retained),
        "outlier_count": outlier_count,
        "outlier_fraction_pct": outlier_count / len(numeric) * 100.0,
        "raw_cv_pct": coefficient_of_variation_pct(raw),
        "retained_cv_pct": coefficient_of_variation_pct(retained_summary),
        "iqr_over_median_pct": iqr_over_median_pct,
    }


def build_canonical_result(
    e2e_entries: list[dict],
    profile_summary: list[dict],
    max_decode_cv_pct: float = MAX_CANONICAL_DECODE_CV_PCT,
) -> dict:
    """Combine three E2E mode results with diagnostic profile summaries."""
    by_mode = {
        entry.get("quantization"): entry
        for entry in e2e_entries
        if entry.get("measurement_mode") == "e2e"
    }
    missing = [mode for mode in MODE_ORDER if mode not in by_mode]
    if missing:
        raise ValueError(f"missing E2E metadata for: {', '.join(missing)}")

    config_keys = ("model", "prompt_len", "max_new_tokens", "warmup_runs", "repeats")
    config = {key: by_mode["fp16"]["config"][key] for key in config_keys}
    for mode in MODE_ORDER[1:]:
        candidate = {key: by_mode[mode]["config"][key] for key in config_keys}
        if candidate != config:
            raise ValueError(f"inconsistent canonical config for {mode}")

    modes = {}
    environments = {}
    for mode in MODE_ORDER:
        entry = by_mode[mode]
        modes[mode] = {
            "runs": entry["runs"],
            "summary": entry["summary"],
        }
        environments[mode] = entry["environment"]

    fp16 = modes["fp16"]["summary"]
    int4 = modes["int4"]["summary"]
    fused = modes["int4-fused-kv"]["summary"]

    fp16_decode = fp16["decode_time_s"]["median"]
    int4_decode = int4["decode_time_s"]["median"]
    fused_decode = fused["decode_time_s"]["median"]
    fp16_vram = fp16["peak_vram_mb"]["median"]
    int4_vram = int4["peak_vram_mb"]["median"]
    stability_by_mode = {
        mode: summarize_stability(
            [float(run["decode_time_s"]) for run in modes[mode]["runs"]]
        )
        for mode in MODE_ORDER
    }
    unstable = {}
    for mode, metrics in stability_by_mode.items():
        reasons = []
        if metrics["retained_cv_pct"] > max_decode_cv_pct:
            reasons.append(f"retained CV {metrics['retained_cv_pct']:.1f}%")
        if metrics["outlier_fraction_pct"] > MAX_CANONICAL_OUTLIER_FRACTION_PCT:
            reasons.append(f"outliers {metrics['outlier_fraction_pct']:.1f}%")
        if metrics["iqr_over_median_pct"] > MAX_CANONICAL_IQR_OVER_MEDIAN_PCT:
            reasons.append(f"IQR/median {metrics['iqr_over_median_pct']:.1f}%")
        if reasons:
            unstable[mode] = reasons
    if unstable:
        details = "; ".join(
            f"{mode}: {', '.join(reasons)}" for mode, reasons in unstable.items()
        )
        raise ValueError(
            "unstable canonical E2E decode timings: "
            f"{details}"
        )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "environment_by_mode": environments,
        "modes": modes,
        "comparisons": {
            "int4_decode_latency_change_vs_fp16_pct": percent_change(
                int4_decode, fp16_decode
            ),
            "fused_decode_latency_change_vs_int4_pct": percent_change(
                fused_decode, int4_decode
            ),
            "int4_peak_vram_reduction_vs_fp16_pct": percent_reduction(
                int4_vram, fp16_vram
            ),
        },
        "stability": {
            "policy": {
                "tukey_fence_multiplier": 1.5,
                "max_retained_decode_cv_pct": max_decode_cv_pct,
                "max_outlier_fraction_pct": MAX_CANONICAL_OUTLIER_FRACTION_PCT,
                "max_iqr_over_median_pct": MAX_CANONICAL_IQR_OVER_MEDIAN_PCT,
            },
            "by_mode": stability_by_mode,
            "passed": True,
        },
        "diagnostic_profile": profile_summary,
    }
