"""Pure helpers for canonical benchmark aggregation."""

from __future__ import annotations

import statistics
from datetime import datetime, timezone


MODE_ORDER = ("fp16", "int4", "int4-fused-kv")


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


def build_canonical_result(
    e2e_entries: list[dict],
    profile_summary: list[dict],
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
        "diagnostic_profile": profile_summary,
    }
