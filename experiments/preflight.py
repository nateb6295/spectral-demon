#!/usr/bin/env python3
"""Pre-flight check for spectral experiments on AGX.

Verifies GPU availability, estimates run time, checks Gemma status,
and confirms experiment files are syntactically valid.

Usage: python3 preflight.py [experiment_script]
"""

import subprocess, sys, os, importlib.util, json
from pathlib import Path

def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
            return True, f"{name} ({mem:.1f}GB total, ~{free:.1f}GB free)"
        return False, "CUDA not available"
    except ImportError:
        return False, "torch not installed"

def check_gemma():
    try:
        r = subprocess.run(["systemctl", "--user", "is-active", "chronicle-gemma"],
                          capture_output=True, text=True, timeout=5)
        active = r.stdout.strip() == "active"
        return active, "RUNNING — must stop before experiment (GPU conflict)" if active else "stopped — GPU available"
    except Exception as e:
        return None, f"check failed: {e}"

def check_ollama():
    try:
        r = subprocess.run(["pgrep", "-f", "ollama"], capture_output=True, text=True, timeout=5)
        running = r.returncode == 0
        return running, "RUNNING — may compete for memory" if running else "not running"
    except Exception:
        return None, "check failed"

def check_experiment(script_path):
    p = Path(script_path)
    if not p.exists():
        return False, f"not found: {p}"
    if p.suffix == '.py':
        spec = importlib.util.spec_from_file_location("exp", p)
        try:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return True, "syntax OK"
        except Exception as e:
            return False, f"import error: {e}"
    elif p.suffix == '.sh':
        r = subprocess.run(["bash", "-n", str(p)], capture_output=True, text=True)
        return r.returncode == 0, "syntax OK" if r.returncode == 0 else f"syntax error: {r.stderr.strip()}"
    return True, "unchecked (not .py/.sh)"

def estimate_time(script_path):
    p = Path(script_path)
    if not p.exists():
        return "unknown"
    content = p.read_text()
    n_modes = content.count("--phase2-mode") or 1
    n_models = content.count("--model") or 1
    if "for MODE in" in content:
        modes = content.split("for MODE in")[1].split(";")[0].split()
        n_modes = len(modes)
    turns = 25
    per_run_min = turns * 0.5
    total = n_modes * per_run_min
    return f"~{total:.0f} min ({n_modes} runs × ~{per_run_min:.0f} min each)"

def check_results_dir():
    d = Path(__file__).parent.parent / "results"
    if not d.exists():
        return False, f"missing: {d}"
    count = len(list(d.glob("exp_convergence_v2_*.json")))
    return True, f"{count} existing result files"

if __name__ == "__main__":
    script = sys.argv[1] if len(sys.argv) > 1 else None

    print("=== Spectral Experiment Pre-flight ===\n")

    ok, msg = check_gpu()
    print(f"  GPU:     {'✓' if ok else '✗'} {msg}")

    gemma_on, msg = check_gemma()
    print(f"  Gemma:   {'⚠' if gemma_on else '✓'} {msg}")

    ollama_on, msg = check_ollama()
    print(f"  Ollama:  {'⚠' if ollama_on else '✓'} {msg}")

    ok, msg = check_results_dir()
    print(f"  Results: {'✓' if ok else '✗'} {msg}")

    if script:
        print(f"\n  Experiment: {script}")
        ok, msg = check_experiment(script)
        print(f"  Syntax:  {'✓' if ok else '✗'} {msg}")
        est = estimate_time(script)
        print(f"  Est time: {est}")

    print()
    if gemma_on:
        print("  ⚠ BLOCKED: Stop Gemma first → systemctl --user stop chronicle-gemma")
    elif not check_gpu()[0]:
        print("  ✗ BLOCKED: No GPU available")
    else:
        print("  ✓ READY to run")
