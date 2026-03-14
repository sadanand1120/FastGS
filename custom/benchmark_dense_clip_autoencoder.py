#!/usr/bin/env python3
"""Fixed benchmark harness for train_dense_clip_langsplat_autoencoder.py.

This file is the immutable benchmark contract for performance work on the
dense CLIP autoencoder trainer. It intentionally runs the target script as a
subprocess, captures its structured summary, and keeps raw stdout/stderr logs
separate from the benchmark summary shown to the user.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


CUSTOM_DIR = Path(__file__).resolve().parent
REPO_ROOT = CUSTOM_DIR.parent
TARGET_SCRIPT = CUSTOM_DIR / "train_dense_clip_langsplat_autoencoder.py"
DEFAULT_IMAGES_DIR = CUSTOM_DIR / "trial_images"
DEFAULT_LOG_DIR = CUSTOM_DIR / "dense_clip_optimization_logs"
PROFILES = {
    "smoke": {"num_epochs": 2},
    "quality": {"num_epochs": 20},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark harness for dense CLIP autoencoder optimization")
    parser.add_argument("--profile", choices=sorted(PROFILES.keys()), required=True)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--scale-images", type=int, default=1000, help="Target image count for linear runtime projection")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--keep-output", action="store_true", help="Keep the target script output root on successful runs")
    return parser.parse_args()


def git_short_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def project_summary(summary: dict, scale_images: int) -> dict[str, float | int]:
    num_images = int(summary.get("num_images", 0))
    if num_images <= 0:
        return {
            "projected_num_images": scale_images,
            "projected_num_feature_vectors": 0,
            "projected_extract_seconds": 0.0,
            "projected_cache_seconds": 0.0,
            "projected_train_seconds": 0.0,
            "projected_export_seconds": 0.0,
            "projected_total_seconds": 0.0,
        }

    scale = float(scale_images) / float(num_images)
    return {
        "projected_num_images": scale_images,
        "projected_num_feature_vectors": int(round(float(summary["num_feature_vectors"]) * scale)),
        "projected_extract_seconds": float(summary["extract_seconds"]) * scale,
        "projected_cache_seconds": float(summary["cache_seconds"]) * scale,
        "projected_train_seconds": float(summary["train_seconds"]) * scale,
        "projected_export_seconds": float(summary["export_seconds"]) * scale,
        "projected_total_seconds": float(summary["total_seconds"]) * scale,
    }


def run_profile(profile: str, images_dir: Path, log_dir: Path, keep_output: bool, scale_images: int) -> tuple[int, dict]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Benchmark images directory not found: {images_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(tempfile.mkdtemp(prefix=f"dense_clip_{profile}_", dir="/tmp"))
    log_path = log_dir / f"{stamp}_{profile}.log"
    summary_path = output_root / "clip_autoencoder_ckpt" / "run_summary.json"
    profile_cfg = PROFILES[profile]
    cmd = [
        sys.executable,
        str(TARGET_SCRIPT),
        "--images-dir",
        str(images_dir),
        "--output-root",
        str(output_root),
        "--reextract",
        "--viz-num-images",
        "0",
        "--num-epochs",
        str(profile_cfg["num_epochs"]),
    ]

    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    result: dict[str, object] = {
        "commit": git_short_commit(),
        "profile": profile,
        "log_path": str(log_path),
    }

    if proc.returncode != 0:
        result.update(
            {
                "status": "crash",
                "returncode": proc.returncode,
                "output_root": str(output_root),
            }
        )
        return 1, result

    if not summary_path.exists():
        result.update(
            {
                "status": "crash",
                "returncode": 2,
                "output_root": str(output_root),
            }
        )
        return 1, result

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    projected = project_summary(summary, scale_images=scale_images)
    result.update(summary)
    result.update(projected)
    result["status"] = "ok"

    if keep_output:
        result["output_root"] = str(output_root)
    else:
        shutil.rmtree(output_root)
        result["output_root"] = "<deleted>"

    return 0, result


def main() -> None:
    args = parse_args()
    code, result = run_profile(
        profile=args.profile,
        images_dir=args.images_dir.resolve(),
        log_dir=args.log_dir.resolve(),
        keep_output=args.keep_output,
        scale_images=args.scale_images,
    )

    print("---")
    print(f"commit:                     {result.get('commit', 'unknown')}")
    print(f"profile:                    {result['profile']}")
    print(f"status:                     {result['status']}")
    if result["status"] == "ok":
        print(f"best_eval_mse:              {float(result['best_eval_mse']):.6f}")
        print(f"extract_seconds:            {float(result['extract_seconds']):.3f}")
        print(f"cache_seconds:              {float(result['cache_seconds']):.3f}")
        print(f"train_seconds:              {float(result['train_seconds']):.3f}")
        print(f"export_seconds:             {float(result['export_seconds']):.3f}")
        print(f"total_seconds:              {float(result['total_seconds']):.3f}")
        print(f"projected_1000_total_seconds: {float(result['projected_total_seconds']):.3f}")
        print(f"peak_vram_mb:               {float(result['peak_vram_mb']):.1f}")
        print(f"num_feature_vectors:        {int(result['num_feature_vectors'])}")
    else:
        print(f"returncode:                 {result.get('returncode', 1)}")
    print(f"log_path:                   {result['log_path']}")
    print(f"output_root:                {result.get('output_root', '<unknown>')}")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
