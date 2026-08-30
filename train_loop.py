#!/usr/bin/env python3
"""
LoRA Training Loop with Auto Brain Toggle

Automatically kills the Qwen 3 6B ("brain") process before training,
and restarts it after training completes.

Usage:
    # Run with all defaults
    python train_loop.py

    # Custom epochs
    python train_loop.py --epochs 20

    # Custom data dir
    python train_loop.py --data_dir data/cropped --output_dir output/models/my_lora

    # With Temporal LoRA mode
    python train_loop.py --mode temporal --video_dir data/temporal

    # Dry run (show what would happen without training)
    python train_loop.py --dry-run
"""

import argparse
import os
import sys
import time
import subprocess
import signal
import shutil
import json
import re
from pathlib import Path

# ─────────────────────────────────────────────────────
# Configuration (editable)
# ─────────────────────────────────────────────────────
BRAIN_PROCESS_NAME = "llama-server"    # Actual process name (llama.cpp server)
BRAIN_CMD = None                      # Command to restart brain (None = don't restart)
# Example: "nohup python -m qwen_server --model qwen3.6b > /dev/null 2>&1 &"

TRAIN_SCRIPT = Path(__file__).parent / "train_lora.py"
TEMPORAL_TRAIN_SCRIPT = Path(__file__).parent / "train_video_lora.py"

# ─────────────────────────────────────────────────────

def log(msg, level="INFO"):
    """Print timestamped log message."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}")


def find_process(pid_name):
    """Find PIDs matching a process name pattern."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pid_name],
            capture_output=True, text=True, timeout=10
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        return pids
    except Exception:
        return []


def kill_process(pid_name, signal=signal.SIGTERM):
    """Kill all processes matching pid_name."""
    pids = find_process(pid_name)
    if not pids:
        log(f"No running processes found matching '{pid_name}'", "WARN")
        return False

    log(f"Found {len(pids)} process(es) matching '{pid_name}': {pids}")
    for pid in pids:
        try:
            os.kill(pid, signal)
            log(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            log(f"PID {pid} already dead", "WARN")
        except PermissionError:
            # Try SIGKILL
            try:
                os.kill(pid, signal.SIGKILL)
                log(f"Sent SIGKILL to PID {pid} (was refused SIGTERM)", "WARN")
            except Exception as e:
                log(f"Failed to kill PID {pid}: {e}", "ERROR")

    # Wait for processes to die
    log("Waiting for processes to terminate...")
    for pid in pids:
        for _ in range(30):  # Wait up to 30 seconds
            try:
                os.kill(pid, 0)  # Check if process exists
                time.sleep(1)
            except ProcessLookupError:
                log(f"PID {pid} terminated successfully", "OK")
                break
        else:
            # Force kill
            try:
                os.kill(pid, signal.SIGKILL)
                log(f"PID {pid} force-killed", "WARN")
            except Exception:
                pass

    return True


def start_brain(cmd):
    """Start the brain process."""
    if not cmd:
        log("No BRAIN_CMD set, skipping brain start", "WARN")
        return None

    log(f"Starting brain: {cmd}")
    try:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)  # Give it time to start
        log(f"Brain started (PID: {proc.pid})", "OK")
        return proc
    except Exception as e:
        log(f"Failed to start brain: {e}", "ERROR")
        return None


def check_gpu_usage():
    """Check if GPU is busy."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            mem = result.stdout.strip().split("\n")
            total_mem = sum(int(m.replace(" MiB", "").strip()) for m in mem)
            return total_mem
    except Exception:
        pass
    return 0


def run_training(args):
    """Run the actual training."""
    if args.mode == "temporal":
        script = TEMPORAL_TRAIN_SCRIPT
        cmd = [
            sys.executable, str(script),
            "--mode", "temporal",
            "--video_dir", args.video_dir,
            "--output_dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--resolution", args.resolution,
            "--seq_length", str(args.seq_length),
            "--fps", str(args.fps),
            "--save_every", str(args.save_every),
        ]
        mode_name = "Temporal LoRA"
    else:
        script = TRAIN_SCRIPT
        cmd = [
            sys.executable, str(script),
            "--base_model", args.base_model,
            "--data_dir", args.data_dir,
            "--output_dir", args.output_dir,
            "--lora_rank", str(args.lora_rank),
            "--lora_alpha", str(args.lora_alpha),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--resolution", args.resolution,
            "--seed", str(args.seed),
        ]
        mode_name = "Standard LoRA"

    log(f"Command: {' '.join(cmd)}")

    # Set CUDA device if available
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    start_time = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - start_time

    return proc.returncode == 0, elapsed


def save_checkpoint(args, status, elapsed, epoch=None):
    """Save training checkpoint metadata."""
    checkpoint_file = Path(args.output_dir) / "train_checkpoint.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "status": status,
        "mode": args.mode,
        "epochs_completed": epoch,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f, indent=2)

    log(f"Checkpoint saved: {checkpoint_file}")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA Training Loop with Auto Brain Toggle",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Data paths
    parser.add_argument("--data_dir", type=str, default="data/cropped",
                        help="Directory with training images (standard mode)")
    parser.add_argument("--video_dir", type=str, default="data/temporal",
                        help="Directory with temporal video sequences")
    parser.add_argument("--output_dir", type=str, default="output/models/lora",
                        help="Output directory for trained LoRA weights")

    # Training parameters
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "temporal"],
                        help="Training mode")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--base_model", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Base diffusion model")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--resolution", type=str, default="512,512",
                        help="Training resolution (W,H)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seq_length", type=int, default=8,
                        help="Frame sequence length (temporal mode)")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Target frame rate for sequence extraction")
    parser.add_argument("--save_every", type=int, default=2,
                        help="Save checkpoint every N epochs")

    # Brain toggle
    parser.add_argument("--brain_name", type=str, default=BRAIN_PROCESS_NAME,
                        help="Process name to kill during training")
    parser.add_argument("--brain_cmd", type=str, default=BRAIN_CMD,
                        help="Command to restart brain after training")

    # Misc
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without training")
    parser.add_argument("--no-brain-toggle", action="store_true",
                        help="Skip killing brain process")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")

    args = parser.parse_args()

    # ── Setup output directory ──
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("LoRA Training Loop with Auto Brain Toggle")
    log("=" * 70)
    log(f"Mode: {args.mode}")
    log(f"Epochs: {args.epochs}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.learning_rate}")
    log(f"Output: {args.output_dir}")

    # ── Dry Run ──
    if args.dry_run:
        log("\n=== DRY RUN ===")
        log(f"Brain process to kill: '{args.brain_name}'")
        brain_pids = find_process(args.brain_name)
        log(f"Currently running: {brain_pids if brain_pids else 'None'}")
        log(f"Data dir exists: {os.path.isdir(args.data_dir)}")
        log(f"Script exists: {TRAIN_SCRIPT.exists()}")
        log(f"Output dir: {args.output_dir}")
        if args.mode == "temporal":
            log(f"Video dir exists: {os.path.isdir(args.video_dir)}")
            log(f"Temporal script exists: {TEMPORAL_TRAIN_SCRIPT.exists()}")
        log("=== END DRY RUN ===")
        return

    # ── Step 1: Kill brain ──
    if not args.no_brain_toggle:
        log("\n" + "=" * 70)
        log("Step 1: Killing brain process...")
        log("=" * 70)
        kill_process(args.brain_name)
        log("Brain process terminated", "OK")
    else:
        log("\n⏭️  Skipping brain toggle (--no-brain-toggle)")

    # ── Step 2: Check GPU ──
    log("\n" + "=" * 70)
    log("Step 2: Checking GPU status...")
    log("=" * 70)
    gpu_mem = check_gpu_usage()
    log(f"GPU memory used: {gpu_mem} MiB")
    if gpu_mem > 0:
        log("GPU is in use — brain process was successfully killed", "OK")
    else:
        log("GPU appears idle", "WARN")

    # ── Step 3: Run training ──
    log("\n" + "=" * 70)
    log("Step 3: Starting training...")
    log("=" * 70)

    success, elapsed = run_training(args)
    elapsed_min = elapsed / 60

    if success:
        log(f"Training completed successfully! ({elapsed_min:.1f} minutes)", "OK")
        save_checkpoint(args, "completed", elapsed)
    else:
        log(f"Training failed! ({elapsed_min:.1f} minutes)", "ERROR")
        save_checkpoint(args, "failed", elapsed)
        # Don't restart brain if training failed
        sys.exit(1)

    # ── Step 4: Restart brain ──
    if not args.no_brain_toggle and args.brain_cmd:
        log("\n" + "=" * 70)
        log("Step 4: Restarting brain process...")
        log("=" * 70)
        start_brain(args.brain_cmd)
        log("Brain process restarted", "OK")
    elif not args.no_brain_toggle and not args.brain_cmd:
        log("\n⏭️  Brain killed but BRAIN_CMD not set — brain stays off", "WARN")

    # ── Done ──
    log("\n" + "=" * 70)
    log("Training loop finished!", "OK")
    log(f"Output: {args.output_dir}")
    log(f"Total time: {elapsed_min:.1f} minutes")
    log("=" * 70)

    # List generated files
    out_dir = Path(args.output_dir)
    files = list(out_dir.glob("*"))
    if files:
        log("Generated files:")
        for f in files:
            size = f.stat().st_size
            if size > 1024 * 1024:
                log(f"  {f.name} ({size / 1024 / 1024:.1f} MB)")
            else:
                log(f"  {f.name} ({size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
