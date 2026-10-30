#!/usr/bin/env python3
"""
GPU Inspection Script
Inspects NVIDIA GPUs, free/used VRAM, and displays which models/processes are active on each GPU.
"""

import os
import subprocess
import sys
from pathlib import Path

# Ensure execution in .venv if available
_script_dir = Path(__file__).resolve().parent
_venv_python = _script_dir / ".venv" / "bin" / "python"
if _venv_python.exists() and sys.executable != str(_venv_python):
    os.execv(str(_venv_python), [str(_venv_python)] + sys.argv)


def get_cmdline(pid: str) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().decode("utf-8", errors="replace").replace("\0", " ").strip()
            return cmd if cmd else "[unknown]"
    except Exception:
        return "[process exited or permission denied]"


def main():
    print("=" * 75)
    print(" NVIDIA GPU & MODEL STATUS CHECK")
    print("=" * 75)

    # 1. Check with PyTorch
    try:
        import torch

        if not torch.cuda.is_available():
            print("CUDA is NOT available in PyTorch.")
        else:
            print(f"CUDA Available: True (PyTorch {torch.__version__})")
            print(f"Device Count  : {torch.cuda.device_count()}\n")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                free_b, tot_b = torch.cuda.mem_get_info(i)
                alloc_b = torch.cuda.memory_allocated(i)
                print(
                    f"GPU {i} [{name}]:\n"
                    f"  Total Memory    : {tot_b / 1e9:6.2f} GB\n"
                    f"  Free Memory     : {free_b / 1e9:6.2f} GB\n"
                    f"  In-Use Memory   : {(tot_b - free_b) / 1e9:6.2f} GB\n"
                    f"  PyTorch Allocated: {alloc_b / 1e9:6.2f} GB\n"
                )
    except ImportError:
        print("PyTorch is not installed in the current environment.\n")

    # 2. Query running compute processes via nvidia-smi
    print("-" * 75)
    print(" ACTIVE PROCESSES & MODELS PER GPU")
    print("-" * 75)

    try:
        # Get map of bus_id -> (gpu_index, gpu_name)
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_bus_id,name", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()

        bus_map = {}
        for line in gpu_info:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                bus_map[parts[1]] = (parts[0], parts[2])

        apps_info = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()

        if not apps_info or all(not l.strip() for l in apps_info):
            print("No active compute processes found on any GPU.")
            return

        for line in apps_info:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue

            bus_id, pid, proc_name, used_mem = parts[0], parts[1], parts[2], parts[3]
            gpu_idx, gpu_name = bus_map.get(bus_id, ("?", "Unknown GPU"))
            cmdline = get_cmdline(pid)

            print(f"▶ GPU {gpu_idx} ({gpu_name}) | PID {pid} | Memory: {used_mem}")
            print(f"  Process: {proc_name}")
            print(f"  Command: {cmdline}\n")

    except Exception as e:
        print(f"Failed to query nvidia-smi: {e}")

    print("=" * 75)


if __name__ == "__main__":
    main()
