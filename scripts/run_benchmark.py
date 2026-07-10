#!/usr/bin/env python3
"""Run either uninstrumented E2E inference or diagnostic layer profiling."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.results import summarize_values
from kernels.fused_fp4_gemv import fused_fp4_gemv_triton
from phase3_utils import Linear4bitArtifacts, extract_linear4bit_artifacts
from profiler.hook_profiler import HookProfiler


class FusedFP4Linear(nn.Module):
    """Correctness-first wrapper around the project-local fused FP4 GEMV."""

    def __init__(self, artifacts: Linear4bitArtifacts):
        super().__init__()
        self.in_features = artifacts.in_features
        self.out_features = artifacts.out_features
        self.blocksize = artifacts.blocksize
        self.register_buffer("packed_weight", artifacts.packed_weight.clone())
        self.register_buffer("absmax", artifacts.absmax.clone())
        self.register_buffer("code", artifacts.code.clone())
        if artifacts.bias is not None:
            self.register_buffer("bias_vec", artifacts.bias.clone())
        else:
            self.bias_vec = None

    def _make_artifacts(self) -> Linear4bitArtifacts:
        return Linear4bitArtifacts(
            layer_path="",
            in_features=self.in_features,
            out_features=self.out_features,
            packed_weight=self.packed_weight,
            bias=self.bias_vec,
            absmax=self.absmax,
            code=self.code,
            blocksize=self.blocksize,
            quant_type="fp4",
            quant_dtype="float16",
            weight_device=str(self.packed_weight.device),
            packed_weight_shape=tuple(self.packed_weight.shape),
            logical_weight_shape=(self.out_features, self.in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        original_shape = x.shape
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        x_flat = x.reshape(-1, self.in_features)
        artifacts = self._make_artifacts()
        outputs = torch.empty(
            x_flat.shape[0],
            self.out_features,
            device=x.device,
            dtype=torch.float16,
        )
        for index in range(x_flat.shape[0]):
            outputs[index] = fused_fp4_gemv_triton(artifacts, x_flat[index])
        result = outputs.reshape(*original_shape[:-1], self.out_features)
        return result.to(original_dtype)


def replace_kv_proj_with_fused(model: nn.Module) -> int:
    """Replace every quantized k/v projection and return the replacement count."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if not (name.endswith(".k_proj") or name.endswith(".v_proj")):
            continue
        if module.__class__.__name__ != "Linear4bit":
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        artifacts = extract_linear4bit_artifacts(module, name)
        setattr(parent, parts[-1], FusedFP4Linear(artifacts))
        replaced += 1
    print(f"Replaced {replaced} k_proj/v_proj layers with FusedFP4Linear")
    return replaced


def load_model(model_id: str, quantization: str, local_files_only: bool = False):
    print(f"Loading {model_id} in {quantization}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    if quantization == "fp16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="cuda",
            local_files_only=local_files_only,
        )
    else:
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=config,
            device_map="cuda",
            local_files_only=local_files_only,
        )
        if quantization == "int4-fused-kv":
            replace_kv_proj_with_fused(model)
    model.eval()
    print(f"Model loaded. VRAM allocated: {torch.cuda.memory_allocated() / 1e6:.0f} MB")
    return model, tokenizer


def make_prompt_ids(tokenizer, target_len: int, device: str = "cuda"):
    if target_len < 1:
        raise ValueError("prompt length must be positive")
    text = "The quick brown fox jumps over the lazy dog. " * (target_len + 1)
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)
    if ids.shape[1] < target_len:
        raise RuntimeError(f"failed to construct a {target_len}-token prompt")
    return ids[:, :target_len].to(device)


def _decode_steps(model, prefill_outputs, max_new_tokens: int):
    past_key_values = prefill_outputs.past_key_values
    next_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for _ in range(max_new_tokens):
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return outputs if max_new_tokens else prefill_outputs


@torch.inference_mode()
def run_e2e_iteration(model, prompt_ids, max_new_tokens: int) -> dict:
    """Measure one fixed-length inference iteration without profiler hooks."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    prefill_outputs = model(prompt_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_time_s = time.perf_counter() - started

    started = time.perf_counter()
    final_outputs = _decode_steps(model, prefill_outputs, max_new_tokens)
    torch.cuda.synchronize()
    decode_time_s = time.perf_counter() - started

    result = {
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "decode_steps": max_new_tokens,
        "decode_throughput_tps": (
            max_new_tokens / decode_time_s if decode_time_s > 0 else 0.0
        ),
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1e6,
        "peak_reserved_vram_mb": torch.cuda.max_memory_reserved() / 1e6,
    }
    del final_outputs, prefill_outputs
    return result


@torch.inference_mode()
def run_profile_iteration(
    model,
    profiler: HookProfiler,
    prompt_ids,
    max_new_tokens: int,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Collect instrumented per-layer timings for one fixed-length iteration."""
    profiler.run_id = run_id
    profiler.clear_records()
    profiler.current_phase = "prefill"
    profiler.current_decode_step = None
    profiler.recording = True
    prefill_outputs = model(prompt_ids, use_cache=True)
    torch.cuda.synchronize()
    prefill_df = profiler.to_dataframe().copy()

    profiler.clear_records()
    profiler.current_phase = "decode"
    for step in range(max_new_tokens):
        profiler.current_decode_step = step
        if step == 0:
            past_key_values = prefill_outputs.past_key_values
            next_token = prefill_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    profiler.current_decode_step = None
    decode_df = profiler.to_dataframe().copy()

    def phase_summary(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {"records": 0, "layers": 0, "hook_total_time_ms": 0.0}
        return {
            "records": int(len(frame)),
            "layers": int(frame["layer_name"].nunique()),
            "hook_total_time_ms": float(frame["time_ms"].sum()),
            "peak_vram_mb": float(frame["mem_peak_mb"].max()),
        }

    return prefill_df, decode_df, {
        "run_id": run_id,
        "prefill": phase_summary(prefill_df),
        "decode": phase_summary(decode_df),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_gpu_number(value: str) -> float | None:
    cleaned = value.strip().replace("[N/A]", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def query_gpu_state() -> dict:
    """Return a lightweight GPU state snapshot outside the timed region."""
    fields = (
        "pstate",
        "temperature.gpu",
        "clocks.current.graphics",
        "clocks.current.memory",
        "power.draw",
        "utilization.gpu",
        "memory.used",
    )
    output = _command_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        raise RuntimeError("nvidia-smi GPU telemetry is unavailable")
    values = [value.strip() for value in output.splitlines()[0].split(",")]
    if len(values) != len(fields):
        raise RuntimeError(f"unexpected nvidia-smi output: {output}")
    return {
        "pstate": values[0],
        "temperature_c": _parse_gpu_number(values[1]),
        "graphics_clock_mhz": _parse_gpu_number(values[2]),
        "memory_clock_mhz": _parse_gpu_number(values[3]),
        "power_w": _parse_gpu_number(values[4]),
        "utilization_pct": _parse_gpu_number(values[5]),
        "memory_used_mb": _parse_gpu_number(values[6]),
    }


def wait_for_gpu_idle(
    max_utilization_pct: float,
    consecutive_samples: int,
    timeout_s: float,
) -> dict:
    """Require a quiet GPU before entering a timed benchmark region."""
    deadline = time.monotonic() + timeout_s
    quiet = 0
    last_state = None
    while time.monotonic() < deadline:
        last_state = query_gpu_state()
        utilization = last_state["utilization_pct"]
        if utilization is not None and utilization <= max_utilization_pct:
            quiet += 1
            if quiet >= consecutive_samples:
                return last_state
        else:
            quiet = 0
        time.sleep(1.0)
    raise RuntimeError(
        "GPU did not become idle before benchmark: "
        f"threshold={max_utilization_pct:.1f}% last_state={last_state}"
    )


def _git_dirty(repo_root: Path) -> bool | None:
    """Ignore platform EOL presentation while detecting real repository changes."""
    try:
        tracked = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "diff",
                "--quiet",
                "--ignore-space-at-eol",
                "HEAD",
                "--",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if tracked.returncode not in (0, 1):
            return None
        untracked = _command_output(
            ["git", "-C", str(repo_root), "ls-files", "--others", "--exclude-standard"]
        )
        return tracked.returncode == 1 or bool(untracked)
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    git_commit = _command_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    return {
        "host_platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "transformers_version": _package_version("transformers"),
        "accelerate_version": _package_version("accelerate"),
        "bitsandbytes_version": _package_version("bitsandbytes"),
        "triton_version": _package_version("triton"),
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_mb": torch.cuda.get_device_properties(0).total_memory / 1e6,
        "nvidia_driver_version": _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "git_commit": git_commit,
        "git_dirty": _git_dirty(repo_root),
    }


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata: {path}")


def run_e2e(
    model,
    prompt_ids,
    max_new_tokens: int,
    warmups: int,
    repeats: int,
    max_idle_gpu_util: float,
    idle_samples: int,
    idle_timeout_s: float,
    cooldown_s: float,
):
    for index in range(warmups):
        print(f"Warmup {index + 1}/{warmups}")
        wait_for_gpu_idle(max_idle_gpu_util, idle_samples, idle_timeout_s)
        run_e2e_iteration(model, prompt_ids, max_new_tokens)
        if cooldown_s:
            time.sleep(cooldown_s)
    runs = []
    for index in range(repeats):
        gpu_before = wait_for_gpu_idle(max_idle_gpu_util, idle_samples, idle_timeout_s)
        result = run_e2e_iteration(model, prompt_ids, max_new_tokens)
        result["gpu_before"] = gpu_before
        result["gpu_after"] = query_gpu_state()
        result["run_index"] = index + 1
        runs.append(result)
        print(
            f"Run {index + 1}/{repeats}: prefill={result['prefill_time_s']:.3f}s "
            f"decode={result['decode_time_s']:.3f}s "
            f"throughput={result['decode_throughput_tps']:.2f} tok/s"
        )
        if cooldown_s and index + 1 < repeats:
            time.sleep(cooldown_s)
    summary = {
        key: summarize_values([float(run[key]) for run in runs])
        for key in (
            "prefill_time_s",
            "decode_time_s",
            "decode_throughput_tps",
            "peak_vram_mb",
            "peak_reserved_vram_mb",
        )
    }
    return runs, summary


def run_profile(model, prompt_ids, max_new_tokens: int, warmups: int, repeats: int, run_id: str):
    for index in range(warmups):
        print(f"Uninstrumented warmup {index + 1}/{warmups}")
        run_e2e_iteration(model, prompt_ids, max_new_tokens)

    profiler = HookProfiler(model)
    profiler.attach()
    prefill_frames = []
    decode_frames = []
    runs = []
    try:
        for index in range(repeats):
            repeat_id = f"{run_id}_rep{index + 1:02d}"
            prefill_df, decode_df, summary = run_profile_iteration(
                model,
                profiler,
                prompt_ids,
                max_new_tokens,
                repeat_id,
            )
            prefill_frames.append(prefill_df)
            decode_frames.append(decode_df)
            runs.append(summary)
            print(
                f"Profile {index + 1}/{repeats}: "
                f"prefill_records={len(prefill_df)} decode_records={len(decode_df)}"
            )
    finally:
        profiler.detach()
    return (
        pd.concat(prefill_frames, ignore_index=True),
        pd.concat(decode_frames, ignore_index=True),
        runs,
    )


def main():
    parser = argparse.ArgumentParser(description="LLM quantization benchmark and profiler")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--quantization",
        choices=["fp16", "int4", "int4-fused-kv"],
        default="fp16",
    )
    parser.add_argument(
        "--measurement-mode",
        choices=["e2e", "profile"],
        default="e2e",
        help="e2e disables hooks; profile emits diagnostic per-layer CSVs",
    )
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-idle-gpu-util", type=float, default=15.0)
    parser.add_argument("--idle-samples", type=int, default=3)
    parser.add_argument("--idle-timeout-s", type=float, default=120.0)
    parser.add_argument("--cooldown-s", type=float, default=3.0)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--run-id", default="single-run")
    parser.add_argument("--metadata-path")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    warmups = args.warmup_runs
    if warmups is None:
        warmups = 3 if args.measurement_mode == "e2e" else 1
    repeats = args.repeats
    if repeats is None:
        repeats = 7 if args.measurement_mode == "e2e" else 3
    if warmups < 0 or repeats < 1 or args.idle_samples < 1:
        parser.error("warmups must be non-negative; repeats and idle samples must be positive")
    if args.max_idle_gpu_util < 0 or args.idle_timeout_s <= 0 or args.cooldown_s < 0:
        parser.error("GPU idle settings must be non-negative and timeout must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(args.metadata_path) if args.metadata_path else output_dir / "benchmark_metadata.json"

    started_at = datetime.now(timezone.utc).isoformat()
    model, tokenizer = load_model(args.model, args.quantization, args.local_files_only)
    prompt_ids = make_prompt_ids(tokenizer, args.prompt_len)

    config = {
        "model": args.model,
        "prompt_len": args.prompt_len,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": warmups,
        "repeats": repeats,
        "local_files_only": bool(args.local_files_only),
        "max_idle_gpu_util_pct": args.max_idle_gpu_util,
        "idle_samples": args.idle_samples,
        "idle_timeout_s": args.idle_timeout_s,
        "cooldown_s": args.cooldown_s,
    }

    if args.measurement_mode == "e2e":
        runs, summary = run_e2e(
            model,
            prompt_ids,
            args.max_new_tokens,
            warmups,
            repeats,
            args.max_idle_gpu_util,
            args.idle_samples,
            args.idle_timeout_s,
            args.cooldown_s,
        )
        payload = {
            "schema_version": 2,
            "measurement_mode": "e2e",
            "quantization": args.quantization,
            "run_id": args.run_id,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "environment": collect_environment(),
            "runs": runs,
            "summary": summary,
        }
    else:
        prefill_df, decode_df, runs = run_profile(
            model,
            prompt_ids,
            args.max_new_tokens,
            warmups,
            repeats,
            args.run_id,
        )
        prefill_path = output_dir / f"{args.quantization}_prefill.csv"
        decode_path = output_dir / f"{args.quantization}_decode.csv"
        prefill_df.to_csv(prefill_path, index=False)
        decode_df.to_csv(decode_path, index=False)
        print(f"Saved profile CSVs: {prefill_path}, {decode_path}")
        payload = {
            "schema_version": 2,
            "measurement_mode": "profile",
            "quantization": args.quantization,
            "run_id": args.run_id,
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "environment": collect_environment(),
            "runs": runs,
        }

    write_json(metadata_path, payload)


if __name__ == "__main__":
    main()
