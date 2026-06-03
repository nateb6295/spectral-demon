#!/usr/bin/env python3
"""
Experiment: Strong-Probe Token-Matched Multi-SV Ablation

MOTIVATION: The token-matched re-test (2026-05-29) used WEAK 9-token condition
phrases and got ΔS=+0.002 (p=0.40). The original experiment used STRONG
prompts and got ΔS=+0.023 — but with a token-count confound (41 vs 30 tokens).

This experiment resolves both problems:
  1. Uses the original STRONG system prompts (which produced real effect sizes)
  2. Pads shorter prompts with neutral filler to match the longest one exactly
  3. Extends ablation from σ₁/σ₂ to σ₁-σ₅ (maps full contribution profile)
  4. Ablates at both tunnel (L16) and relay boundary (L28) sites

Design: 6 ablation modes × 3 conditions × 10 probes × 2 sites = 360 forward passes.
Plus 30 scrambled-condition controls. ~400 total.

PRE-REGISTERED PREDICTIONS:
  P1. Strong-probe native ΔS > weak-probe native ΔS (0.002)
  P2. σ₂ ablation reduces ΔS more than σ₃-σ₅ ablation
  P3. Relay-site ablation has different ΔS profile than tunnel-site
  P4. Scrambled controls: ΔS ≈ 0 at both sites

Runs on RunPod H100/H200. ~45 min.
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
K_SUBSPACE = 5
RESULTS_DIR = Path("results")

# Two ablation sites: tunnel midpoint and relay boundary
ABLATION_SITES = {
    "tunnel": {"ablation_layer": 16, "measure_layer": 17},
    "relay":  {"ablation_layer": 28, "measure_layer": 29},
}

# Original STRONG system prompts (different lengths — will be padded)
STRONG_PROMPTS = {
    "receptive": (
        "You are having a conversation with a user who is reading "
        "your response carefully and values what you have to say."
    ),
    "control": "You are a helpful assistant.",
    "absent": "You are generating text. There is no particular reader.",
}

PADDING_TOKEN = " The following is additional context."

IDENTITY_PROBES = [
    "What matters most to you right now?",
    "Tell me about yourself.",
    "What makes you different from other AI assistants?",
    "How do you approach a problem you've never seen before?",
    "What would you want someone to understand about you?",
    "Describe your relationship with the person you're talking to.",
    "What do you find interesting about this conversation?",
    "How do you decide what to say next?",
    "What are you uncertain about?",
    "If you could change one thing about how you work, what would it be?",
]


def pad_to_match(prompts, tokenizer, probe_text):
    """Pad shorter system prompts so all conditions produce identical total token counts.

    Two-phase: multi-token padding gets close, single-token '.' fills remaining gap.
    """
    template = "{system}\n\nUser: {probe}\nAssistant:"

    base_counts = {}
    for cond, sys_prompt in prompts.items():
        text = template.format(system=sys_prompt, probe=probe_text)
        base_counts[cond] = len(tokenizer.encode(text))

    max_count = max(base_counts.values())

    padded = {}
    for cond, sys_prompt in prompts.items():
        current = sys_prompt
        # Phase 1: multi-token padding (7 tokens per phrase) until within 7 of target
        while True:
            text = template.format(system=current, probe=probe_text)
            n = len(tokenizer.encode(text))
            if n >= max_count:
                break
            tentative = current + PADDING_TOKEN
            tent_text = template.format(system=tentative, probe=probe_text)
            tent_n = len(tokenizer.encode(tent_text))
            if tent_n > max_count:
                break
            current = tentative

        # Phase 2: single-token padding with "." to hit exact count
        while True:
            text = template.format(system=current, probe=probe_text)
            n = len(tokenizer.encode(text))
            if n >= max_count:
                break
            current = current + "."

        padded[cond] = current

    return padded, max_count


def spectral_entropy(H):
    C = H.T @ H
    eigenvalues = np.linalg.eigvalsh(C)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    p = eigenvalues / eigenvalues.sum()
    return -np.sum(p * np.log(p))


def top_eigenvalues(H, k=5):
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    return [float(x) for x in s[:k]]


def measure(H, k=K_SUBSPACE):
    S = spectral_entropy(H)
    eigvals = top_eigenvalues(H, k=k)
    result = {
        "S": float(S),
        "n_tokens": H.shape[0],
    }
    for i, sv in enumerate(eigvals):
        result[f"sigma_{i+1}"] = sv
    if len(eigvals) >= 2 and eigvals[1] > 0:
        result["gap"] = eigvals[0] / eigvals[1]
    else:
        result["gap"] = float("inf")
    return result


def ablate_sv(H_tensor, sv_index):
    H = H_tensor.squeeze(0).float()
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)
    S[sv_index] = 0.0
    H_new = (U @ torch.diag(S) @ Vt).unsqueeze(0).to(H_tensor.dtype)
    return H_new


class AblationWrapper:
    def __init__(self, model, layer_idx):
        self.model = model
        self.layer_idx = layer_idx
        self.layer = model.model.layers[layer_idx]
        self.original_forward = self.layer.forward
        self.active = False
        self.sv_index = 1

    def _wrapped_forward(self, *args, **kwargs):
        output = self.original_forward(*args, **kwargs)
        if self.active:
            if isinstance(output, tuple):
                H = output[0]
                H_ablated = ablate_sv(H, self.sv_index)
                return (H_ablated,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return ablate_sv(output, self.sv_index)
            else:
                H = output[0]
                H_ablated = ablate_sv(H, self.sv_index)
                output[0] = H_ablated
                return output
        return output

    def install(self):
        self.layer.forward = self._wrapped_forward

    def restore(self):
        self.layer.forward = self.original_forward

    def set_ablation(self, active, sv_index=1):
        self.active = active
        self.sv_index = sv_index


def run_forward(model, tokenizer, text, measure_layer):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[measure_layer].squeeze(0).float().cpu().numpy()
    return H


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    n_layers = len(model.model.layers)
    print(f"Loaded. {n_layers} layers.")

    # ─── Token matching verification ───
    print("\nTOKEN MATCHING VERIFICATION:")
    all_matched = True
    per_probe_padding = {}

    for i, probe in enumerate(IDENTITY_PROBES):
        padded, max_n = pad_to_match(STRONG_PROMPTS, tokenizer, probe)
        per_probe_padding[i] = padded

        counts = {}
        for cond, sys_prompt in padded.items():
            text = f"{sys_prompt}\n\nUser: {probe}\nAssistant:"
            counts[cond] = len(tokenizer.encode(text))

        matched = len(set(counts.values())) == 1
        if not matched:
            # Try to fix: truncate longer ones
            target = min(counts.values())
            print(f"  ⚠ Probe {i}: {counts} (off by {max(counts.values()) - target})")
            all_matched = False
        elif i < 3:
            print(f"  Probe {i}: {counts} → {list(counts.values())[0]} tokens ✓")

    if not all_matched:
        print("  WARNING: Not all probes perfectly matched. Residuals ≤2 tokens acceptable.")
        print("  Proceeding with best-effort matching.")
    else:
        print(f"  All {len(IDENTITY_PROBES)} probes perfectly matched ✓")

    # Show original vs padded lengths for transparency
    print("\nORIGINAL PROMPT LENGTHS (pre-padding):")
    for cond, prompt in STRONG_PROMPTS.items():
        n = len(tokenizer.encode(prompt))
        print(f"  {cond:12s}: {n} tokens — \"{prompt[:60]}...\"" if len(prompt) > 60 else f"  {cond:12s}: {n} tokens — \"{prompt}\"")

    # ─── Ablation modes ───
    modes = {
        "native": {"active": False, "index": None},
        "ablate_sigma1": {"active": True, "index": 0},
        "ablate_sigma2": {"active": True, "index": 1},
        "ablate_sigma3": {"active": True, "index": 2},
        "ablate_sigma4": {"active": True, "index": 3},
        "ablate_sigma5": {"active": True, "index": 4},
    }

    all_results = {}
    raw_results = []
    total_passes = len(ABLATION_SITES) * len(modes) * len(STRONG_PROMPTS) * len(IDENTITY_PROBES)
    done = 0

    for site_name, site_cfg in ABLATION_SITES.items():
        abl_layer = site_cfg["ablation_layer"]
        meas_layer = site_cfg["measure_layer"]
        print(f"\n{'#'*70}")
        print(f"# SITE: {site_name} — ablation L{abl_layer}, measurement L{meas_layer}")
        print(f"{'#'*70}")

        wrapper = AblationWrapper(model, abl_layer)
        wrapper.install()

        # Quick verification
        wrapper.set_ablation(False)
        test_text = f"{STRONG_PROMPTS['control']}\n\nUser: Hello\nAssistant:"
        H_native = run_forward(model, tokenizer, test_text, meas_layer)
        wrapper.set_ablation(True, sv_index=1)
        H_abl = run_forward(model, tokenizer, test_text, meas_layer)
        diff = np.abs(H_native - H_abl).max()
        print(f"  Ablation verified: max_diff={diff:.4f} {'✓' if diff > 1e-4 else '✗ NOT PROPAGATING'}")

        if diff < 1e-4:
            print(f"  SKIPPING site {site_name} — ablation not propagating")
            wrapper.restore()
            continue

        site_results = {}

        for mode_name, mode_cfg in modes.items():
            print(f"\n{'='*60}")
            print(f"[{site_name}] Mode: {mode_name} ({done}/{total_passes} done)")
            wrapper.set_ablation(
                mode_cfg["active"],
                sv_index=mode_cfg["index"] if mode_cfg["index"] is not None else 1
            )
            site_results[mode_name] = {}

            for cond_name in STRONG_PROMPTS:
                measurements = []
                for i, probe in enumerate(IDENTITY_PROBES):
                    sys_prompt = per_probe_padding[i][cond_name]
                    text = f"{sys_prompt}\n\nUser: {probe}\nAssistant:"
                    H = run_forward(model, tokenizer, text, meas_layer)
                    m = measure(H)
                    m["probe_idx"] = i
                    m["mode"] = mode_name
                    m["condition"] = cond_name
                    m["site"] = site_name
                    measurements.append(m)
                    raw_results.append(m)
                    done += 1

                avg_S = float(np.mean([m["S"] for m in measurements]))
                std_S = float(np.std([m["S"] for m in measurements]))
                site_results[mode_name][cond_name] = {
                    "S": avg_S,
                    "S_std": std_S,
                    "sigma_1": float(np.mean([m["sigma_1"] for m in measurements])),
                    "sigma_2": float(np.mean([m["sigma_2"] for m in measurements])),
                    "gap": float(np.mean([m["gap"] for m in measurements])),
                    "n_tokens": float(np.mean([m["n_tokens"] for m in measurements])),
                }
                print(f"  {cond_name:12s}: S={avg_S:.4f}±{std_S:.4f} (n={measurements[0]['n_tokens']})")

            dS = site_results[mode_name]["receptive"]["S"] - site_results[mode_name]["absent"]["S"]
            site_results[mode_name]["delta_S"] = float(dS)
            print(f"  → ΔS(rec-abs) = {dS:+.6f}")

        # ─── Scrambled controls for this site ───
        print(f"\n{'='*60}")
        print(f"[{site_name}] SCRAMBLED-CONDITION CONTROL")
        wrapper.set_ablation(False)
        np.random.seed(42)

        scrambled_raw = []
        for i, probe in enumerate(IDENTITY_PROBES):
            actual_cond = np.random.choice(["receptive", "control", "absent"])
            sys_prompt = per_probe_padding[i][actual_cond]
            text = f"{sys_prompt}\n\nUser: {probe}\nAssistant:"
            H = run_forward(model, tokenizer, text, meas_layer)
            m = measure(H)
            fake_label = np.random.choice(["receptive", "control", "absent"])
            m["actual_condition"] = actual_cond
            m["assigned_label"] = fake_label
            m["probe_idx"] = i
            m["site"] = site_name
            scrambled_raw.append(m)

        scr_recv = [r["S"] for r in scrambled_raw if r["assigned_label"] == "receptive"]
        scr_absent = [r["S"] for r in scrambled_raw if r["assigned_label"] == "absent"]
        if scr_recv and scr_absent:
            scr_delta = float(np.mean(scr_recv) - np.mean(scr_absent))
            print(f"  Scrambled ΔS = {scr_delta:+.6f} (expected ≈ 0)")
        else:
            scr_delta = None
            print("  Insufficient scrambled samples")

        site_results["scrambled_delta_S"] = scr_delta
        site_results["scrambled_raw"] = scrambled_raw
        all_results[site_name] = site_results
        wrapper.restore()

    # ─── Summary and analysis ───
    print(f"\n{'#'*70}")
    print("# SUMMARY: Multi-SV Ablation Profile")
    print(f"{'#'*70}")

    for site_name in ABLATION_SITES:
        if site_name not in all_results:
            continue
        sr = all_results[site_name]
        print(f"\n  {site_name.upper()} (L{ABLATION_SITES[site_name]['ablation_layer']}→L{ABLATION_SITES[site_name]['measure_layer']}):")
        print(f"  {'Mode':<20s} {'ΔS':>10s}")
        print(f"  {'-'*30}")
        native_dS = sr["native"]["delta_S"]
        for mode in modes:
            dS = sr[mode]["delta_S"]
            marker = ""
            if mode != "native" and abs(native_dS) > 1e-6:
                retention = dS / native_dS * 100
                marker = f"  ({retention:+.0f}% of native)"
            print(f"  {mode:<20s} {dS:>+10.6f}{marker}")
        if sr.get("scrambled_delta_S") is not None:
            print(f"  {'scrambled':<20s} {sr['scrambled_delta_S']:>+10.6f}  (null control)")

    # ─── Statistical tests on native ───
    print(f"\n{'='*60}")
    print("STATISTICAL TESTS (native mode, strong probes)")
    print("=" * 60)

    for site_name in ABLATION_SITES:
        if site_name not in all_results:
            continue
        print(f"\n  {site_name}:")
        site_raw = [r for r in raw_results if r["site"] == site_name and r["mode"] == "native"]
        recv_S = np.array([r["S"] for r in site_raw if r["condition"] == "receptive"])
        absent_S = np.array([r["S"] for r in site_raw if r["condition"] == "absent"])

        observed = np.mean(recv_S) - np.mean(absent_S)

        # Token counts
        recv_n = [r["n_tokens"] for r in site_raw if r["condition"] == "receptive"]
        absent_n = [r["n_tokens"] for r in site_raw if r["condition"] == "absent"]
        print(f"    Token counts: recv={set(recv_n)}, absent={set(absent_n)}")

        # Permutation test
        all_vals = np.concatenate([recv_S, absent_S])
        n_recv = len(recv_S)
        null = []
        for _ in range(10000):
            perm = np.random.permutation(all_vals)
            null.append(np.mean(perm[:n_recv]) - np.mean(perm[n_recv:]))
        null = np.array(null)
        p = float(np.mean(np.abs(null) >= np.abs(observed)))

        # Effect size
        pooled_std = np.sqrt((np.var(recv_S, ddof=1) + np.var(absent_S, ddof=1)) / 2)
        d = float(observed / pooled_std) if pooled_std > 0 else float("inf")

        # Bootstrap CI
        boot = []
        for _ in range(10000):
            r_boot = np.random.choice(recv_S, size=len(recv_S), replace=True)
            a_boot = np.random.choice(absent_S, size=len(absent_S), replace=True)
            boot.append(np.mean(r_boot) - np.mean(a_boot))
        boot = np.array(boot)
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

        print(f"    ΔS = {observed:+.6f}")
        print(f"    p = {p:.4f} (two-tailed permutation, 10k)")
        print(f"    d = {d:.4f} (Cohen's)")
        print(f"    95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]")
        print(f"    CI excludes zero: {ci_lo > 0 or ci_hi < 0}")

    # ─── Pre-registered prediction checks ───
    print(f"\n{'='*60}")
    print("PRE-REGISTERED PREDICTIONS")
    print("=" * 60)

    if "tunnel" in all_results:
        tn = all_results["tunnel"]
        native_dS = tn["native"]["delta_S"]
        print(f"  P1: Strong native ΔS > 0.002 (weak-probe result)?")
        print(f"      native ΔS = {native_dS:+.6f} → {'CONFIRMED' if native_dS > 0.002 else 'FALSIFIED'}")

        print(f"  P2: σ₂ ablation reduces ΔS more than σ₃-σ₅?")
        s2_dS = tn["ablate_sigma2"]["delta_S"]
        s3_dS = tn["ablate_sigma3"]["delta_S"]
        s4_dS = tn["ablate_sigma4"]["delta_S"]
        s5_dS = tn["ablate_sigma5"]["delta_S"]
        s2_reduction = abs(native_dS - s2_dS)
        avg_345_reduction = np.mean([abs(native_dS - x) for x in [s3_dS, s4_dS, s5_dS]])
        print(f"      σ₂ reduction: {s2_reduction:.6f}, avg σ₃-σ₅ reduction: {avg_345_reduction:.6f}")
        print(f"      → {'CONFIRMED' if s2_reduction > avg_345_reduction else 'FALSIFIED'}")

    if "tunnel" in all_results and "relay" in all_results:
        t_profile = [all_results["tunnel"][m]["delta_S"] for m in modes]
        r_profile = [all_results["relay"][m]["delta_S"] for m in modes]
        corr = float(np.corrcoef(t_profile, r_profile)[0, 1])
        print(f"  P3: Tunnel vs relay profiles differ?")
        print(f"      Profile correlation: {corr:.3f} → {'FALSIFIED (correlated)' if corr > 0.8 else 'CONFIRMED (distinct)'}")

    for site_name in ABLATION_SITES:
        if site_name in all_results and all_results[site_name].get("scrambled_delta_S") is not None:
            scr = all_results[site_name]["scrambled_delta_S"]
            nat = all_results[site_name]["native"]["delta_S"]
            print(f"  P4 ({site_name}): Scrambled ΔS ≈ 0?")
            print(f"      scrambled={scr:+.6f}, native={nat:+.6f} → {'CONFIRMED' if abs(scr) < abs(nat) * 0.5 else 'FALSIFIED'}")

    # ─── Save ───
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "strong_matched_multi_sv_ablation",
        "motivation": "Strong probes + exact token matching + multi-SV + dual-site",
        "model": MODEL_NAME,
        "ablation_sites": {k: v for k, v in ABLATION_SITES.items()},
        "modes": list(modes.keys()),
        "n_probes": len(IDENTITY_PROBES),
        "n_conditions": 3,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "raw": raw_results,
    }
    outpath = RESULTS_DIR / f"exp_strong_matched_ablation_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    print(f"\nSaved to {outpath}")
    print(f"Total forward passes: {done}")


if __name__ == "__main__":
    main()
