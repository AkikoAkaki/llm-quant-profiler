#!/usr/bin/env python3
"""Legacy two-mode diagnostic profile workflow (FP16 vs INT4)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.roofline import plot_roofline
from analysis.visualize import (
    load_data,
    plot_layerwise_latency,
    plot_memory_growth,
    plot_top10_slowest,
)


def require_wsl():
    text = Path("/proc/version").read_text(encoding="utf-8", errors="ignore")
    if os.name == "nt" or "microsoft" not in text.lower():
        raise SystemExit("run_phase2.py must be executed inside WSL2")


def main():
    require_wsl()
    parser = argparse.ArgumentParser(description="FP16 vs INT4 diagnostic layer profile")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-root", default="data/phase2")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("phase2_%Y%m%d_%H%M%S")
    run_root = (repo_root / args.data_root / timestamp).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    for mode in ("fp16", "int4"):
        mode_dir = run_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(repo_root / "scripts" / "run_benchmark.py"),
            "--model", args.model,
            "--quantization", mode,
            "--measurement-mode", "profile",
            "--prompt-len", str(args.prompt_len),
            "--max-new-tokens", str(args.max_new_tokens),
            "--warmup-runs", str(args.warmup_runs),
            "--repeats", str(args.repeats),
            "--run-id", f"{mode}_profile",
            "--output-dir", str(mode_dir),
        ]
        if args.local_files_only:
            command.append("--local-files-only")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=repo_root)

    dfs = load_data(str(run_root))
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_layerwise_latency(dfs, str(output_dir))
    plot_memory_growth(dfs, str(output_dir))
    plot_top10_slowest(dfs, str(output_dir))
    plot_roofline(dfs, str(output_dir))
    print(f"Raw profiles: {run_root}")
    print(f"Charts:       {output_dir}")


if __name__ == "__main__":
    main()
