#!/usr/bin/env python3
"""Run the canonical three-mode E2E benchmark and diagnostic profile in WSL."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


MODES = ("fp16", "int4", "int4-fused-kv")


def require_wsl():
    proc_version = Path("/proc/version")
    text = proc_version.read_text(encoding="utf-8", errors="ignore") if proc_version.exists() else ""
    if os.name == "nt" or "microsoft" not in text.lower():
        raise SystemExit("run_phase3.py must be executed inside WSL2")


def run_checked(args: list[str], cwd: Path):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=cwd)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def benchmark_command(
    repo_root: Path,
    mode: str,
    measurement_mode: str,
    output_dir: Path,
    model: str,
    prompt_len: int,
    max_new_tokens: int,
    warmups: int,
    repeats: int,
    local_files_only: bool,
    max_idle_gpu_util: float,
    idle_samples: int,
    idle_timeout_s: float,
    cooldown_s: float,
) -> list[str]:
    command = [
        sys.executable,
        str(repo_root / "scripts" / "run_benchmark.py"),
        "--model", model,
        "--quantization", mode,
        "--measurement-mode", measurement_mode,
        "--prompt-len", str(prompt_len),
        "--max-new-tokens", str(max_new_tokens),
        "--warmup-runs", str(warmups),
        "--repeats", str(repeats),
        "--run-id", f"{mode}_{measurement_mode}",
        "--output-dir", str(output_dir),
        "--metadata-path", str(output_dir / "benchmark_metadata.json"),
        "--max-idle-gpu-util", str(max_idle_gpu_util),
        "--idle-samples", str(idle_samples),
        "--idle-timeout-s", str(idle_timeout_s),
        "--cooldown-s", str(cooldown_s),
    ]
    if local_files_only:
        command.append("--local-files-only")
    return command


def main():
    require_wsl()
    parser = argparse.ArgumentParser(description="Canonical Phase 3 workflow")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--e2e-warmups", type=int, default=8)
    parser.add_argument("--e2e-repeats", type=int, default=7)
    parser.add_argument("--profile-warmups", type=int, default=1)
    parser.add_argument("--profile-repeats", type=int, default=3)
    parser.add_argument("--max-idle-gpu-util", type=float, default=15.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-timeout-s", type=float, default=120.0)
    parser.add_argument("--cooldown-s", type=float, default=0.0)
    parser.add_argument("--mode-order-seed", type=int, default=20260711)
    parser.add_argument("--data-root", default="data/phase3")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--canonical-json", default="results/canonical.json")
    parser.add_argument("--report-path", default="PHASE3_REPORT.md")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("phase3_%Y%m%d_%H%M%S")
    run_root = (repo_root / args.data_root / timestamp).resolve()
    e2e_root = run_root / "e2e"
    profile_root = run_root / "profile"
    output_dir = (repo_root / args.output_dir).resolve()
    canonical_json = (repo_root / args.canonical_json).resolve()
    report_path = (repo_root / args.report_path).resolve()

    e2e_modes = list(MODES)
    random.Random(args.mode_order_seed).shuffle(e2e_modes)

    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "e2e_warmups": args.e2e_warmups,
        "e2e_repeats": args.e2e_repeats,
        "profile_warmups": args.profile_warmups,
        "profile_repeats": args.profile_repeats,
        "modes": list(MODES),
        "e2e_mode_order": e2e_modes,
        "mode_order_seed": args.mode_order_seed,
        "max_idle_gpu_util_pct": args.max_idle_gpu_util,
        "idle_samples": args.idle_samples,
        "idle_timeout_s": args.idle_timeout_s,
        "cooldown_s": args.cooldown_s,
        "local_files_only": bool(args.local_files_only),
        "run_root": str(run_root),
    }

    for mode in e2e_modes:
        mode_dir = e2e_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        run_checked(
            benchmark_command(
                repo_root,
                mode,
                "e2e",
                mode_dir,
                args.model,
                args.prompt_len,
                args.max_new_tokens,
                args.e2e_warmups,
                args.e2e_repeats,
                args.local_files_only,
                args.max_idle_gpu_util,
                args.idle_samples,
                args.idle_timeout_s,
                args.cooldown_s,
            ),
            repo_root,
        )

    for mode in MODES:
        mode_dir = profile_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        run_checked(
            benchmark_command(
                repo_root,
                mode,
                "profile",
                mode_dir,
                args.model,
                args.prompt_len,
                args.max_new_tokens,
                args.profile_warmups,
                args.profile_repeats,
                args.local_files_only,
                args.max_idle_gpu_util,
                args.idle_samples,
                args.idle_timeout_s,
                args.cooldown_s,
            ),
            repo_root,
        )

    run_checked(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_analysis.py"),
            "--e2e-data-dir", str(e2e_root),
            "--profile-data-dir", str(profile_root),
            "--output-dir", str(output_dir),
            "--canonical-json", str(canonical_json),
            "--report-path", str(report_path),
        ],
        repo_root,
    )

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(run_root / "phase3_manifest.json", manifest)
    print("\n=== Canonical workflow complete ===")
    print(f"Raw data:  {run_root}")
    print(f"Canonical: {canonical_json}")
    print(f"Report:    {report_path}")


if __name__ == "__main__":
    main()
