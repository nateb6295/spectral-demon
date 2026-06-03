#!/usr/bin/env python3
"""
Experiment: Token-Matched Ablation Re-test — Mistral 7B v0.1

MOTIVATION: Adversarial audit (2026-05-29) found that the original ablation
experiment had systematic token count differences between conditions:
  receptive ~41 tokens, absent ~30 tokens, control ~25 tokens
After ANCOVA adjustment, native ΔS dropped by 78%. F58/F59 magnitudes
are confounded. This experiment resolves the confound.

DESIGN: Condition framing is a 9-token phrase embedded in a fixed template.
All conditions produce IDENTICAL token counts for every probe.
Template: "You are an AI assistant. The reader is {phrase}. Answer the question below."

Token-matched phrases (all 9 tokens):
  receptive: "attentively reading with genuine curiosity now"
  control:   "asking for a helpful and direct answer here"
  absent:    "not present and will never read this output"

Three ablation modes × 3 conditions × 10 probes = 90 forward passes.
Plus 10 scrambled-condition control runs = 100 total.

PRE-REGISTERED PREDICTIONS:
  1. If F58 survives: native ΔS > +0.005 (adjusted value from audit)
  2. If F59 survives: σ₁-ablated |ΔS| > native |ΔS|
  3. Scrambled control: ΔS ≈ 0 (no significant difference)

Runs on RunPod H100. ~30 min.
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

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
ABLATION_LAYER = 16
MEASURE_LAYER = 17
K_SUBSPACE = 5
RESULTS_DIR = Path("results")

TEMPLATE = "You are an AI assistant. The reader is {cond_phrase}. Answer the question below.\n\nUser: {probe}\nAssistant:"

CONDITION_PHRASES = {
    "receptive": "attentively reading with genuine curiosity now",
    "control": "asking for a helpful and direct answer here",
    "absent": "not present and will never read this output",
}

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
    return {
        "S": float(S),
        "sigma_1": eigvals[0],
        "sigma_2": eigvals[1] if len(eigvals) > 1 else 0.0,
        "sigma_3": eigvals[2] if len(eigvals) > 2 else 0.0,
        "gap": eigvals[0] / eigvals[1] if len(eigvals) > 1 and eigvals[1] > 0 else float("inf"),
        "secondary_gap": eigvals[1] / eigvals[2] if len(eigvals) > 2 and eigvals[2] > 0 else float("inf"),
        "n_tokens": H.shape[0],
    }


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


def run_forward(model, tokenizer, text, measure_layer=MEASURE_LAYER):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    H = outputs.hidden_states[measure_layer].squeeze(0).float().cpu().numpy()
    return H


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0"
    )
    model.eval()
    print(f"Loaded. {len(model.model.layers)} layers.")

    # ─── Verify token matching ───
    print("\nTOKEN COUNT VERIFICATION:")
    all_matched = True
    for i, probe in enumerate(IDENTITY_PROBES):
        counts = {}
        for cond, phrase in CONDITION_PHRASES.items():
            text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
            tokens = tokenizer(text, return_tensors="pt").input_ids
            counts[cond] = tokens.shape[1]
        if len(set(counts.values())) != 1:
            print(f"  MISMATCH probe {i}: {counts}")
            all_matched = False
        else:
            if i == 0:
                print(f"  probe 0: {counts} ✓")

    if not all_matched:
        print("ERROR: Token counts not matched! Aborting.")
        return
    print("  All probes token-matched ✓")

    # ─── Install ablation wrapper ───
    wrapper = AblationWrapper(model, ABLATION_LAYER)
    wrapper.install()

    # Verification
    test_text = TEMPLATE.format(cond_phrase=CONDITION_PHRASES["control"],
                                probe="Hello, how are you?")
    wrapper.set_ablation(False)
    H_native = run_forward(model, tokenizer, test_text)

    wrapper.set_ablation(True, sv_index=1)
    H_s2 = run_forward(model, tokenizer, test_text)

    wrapper.set_ablation(True, sv_index=0)
    H_s1 = run_forward(model, tokenizer, test_text)

    diff_s2 = np.abs(H_native - H_s2).max()
    diff_s1 = np.abs(H_native - H_s1).max()
    print(f"\nAblation verification:")
    print(f"  σ₂ ablation max diff: {diff_s2:.4f}")
    print(f"  σ₁ ablation max diff: {diff_s1:.4f}")

    if diff_s2 < 1e-4 and diff_s1 < 1e-4:
        print("ERROR: Ablation not propagating! Aborting.")
        wrapper.restore()
        return
    print("  Ablation confirmed ✓")

    # ─── Main experiment ───
    modes = {
        "native": {"active": False, "index": None},
        "ablate_sigma2": {"active": True, "index": 1},
        "ablate_sigma1": {"active": True, "index": 0},
    }

    all_results = {}
    raw_results = []

    for mode_name, mode_cfg in modes.items():
        print(f"\n{'='*60}")
        print(f"Mode: {mode_name}")
        wrapper.set_ablation(mode_cfg["active"],
                             sv_index=mode_cfg["index"] if mode_cfg["index"] is not None else 1)
        all_results[mode_name] = {}

        for cond_name, phrase in CONDITION_PHRASES.items():
            measurements = []
            for i, probe in enumerate(IDENTITY_PROBES):
                text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
                H = run_forward(model, tokenizer, text)
                m = measure(H)
                m["probe_idx"] = i
                m["mode"] = mode_name
                m["condition"] = cond_name
                measurements.append(m)
                raw_results.append(m)

                if i == 0:
                    print(f"  {cond_name}: S={m['S']:.4f}, gap={m['gap']:.2f}, "
                          f"σ₁={m['sigma_1']:.1f}, σ₂={m['sigma_2']:.1f}, "
                          f"n_tokens={m['n_tokens']}")

            avg = {
                "S": float(np.mean([m["S"] for m in measurements])),
                "S_std": float(np.std([m["S"] for m in measurements])),
                "sigma_1": float(np.mean([m["sigma_1"] for m in measurements])),
                "sigma_2": float(np.mean([m["sigma_2"] for m in measurements])),
                "sigma_3": float(np.mean([m["sigma_3"] for m in measurements])),
                "gap": float(np.mean([m["gap"] for m in measurements])),
            }
            all_results[mode_name][cond_name] = avg
            print(f"  {cond_name} avg: S={avg['S']:.4f}±{avg['S_std']:.4f}")

        dS = all_results[mode_name]["receptive"]["S"] - all_results[mode_name]["absent"]["S"]
        all_results[mode_name]["delta_S"] = float(dS)
        print(f"\n  ΔS(receptive - absent) = {dS:+.6f}")

    # ─── Scrambled-condition control ───
    print(f"\n{'='*60}")
    print("SCRAMBLED-CONDITION CONTROL")
    print("Randomly assign condition labels to same prompts")
    wrapper.set_ablation(False)

    np.random.seed(42)
    scrambled_raw = []
    for i, probe in enumerate(IDENTITY_PROBES):
        # Randomly pick one condition's framing
        cond = np.random.choice(["receptive", "control", "absent"])
        phrase = CONDITION_PHRASES[cond]
        text = TEMPLATE.format(cond_phrase=phrase, probe=probe)
        H = run_forward(model, tokenizer, text)
        m = measure(H)
        # Assign a RANDOM label (different from actual)
        fake_label = np.random.choice(["receptive", "control", "absent"])
        m["actual_condition"] = cond
        m["assigned_label"] = fake_label
        m["probe_idx"] = i
        scrambled_raw.append(m)

    scrambled_recv = [r["S"] for r in scrambled_raw if r["assigned_label"] == "receptive"]
    scrambled_absent = [r["S"] for r in scrambled_raw if r["assigned_label"] == "absent"]
    if scrambled_recv and scrambled_absent:
        scrambled_delta = np.mean(scrambled_recv) - np.mean(scrambled_absent)
        print(f"  Scrambled ΔS = {scrambled_delta:+.6f}")
        print(f"  (Expected: ≈ 0)")
    else:
        scrambled_delta = None
        print("  Insufficient samples in scrambled groups")

    wrapper.restore()

    # ─── Pre-registered prediction checks ───
    print(f"\n{'='*60}")
    print("PRE-REGISTERED PREDICTION CHECKS")
    print("=" * 60)

    native_dS = all_results["native"]["delta_S"]
    s1_dS = all_results["ablate_sigma1"]["delta_S"]

    pred1 = native_dS > 0.005
    print(f"  P1: native ΔS > +0.005? ΔS = {native_dS:+.6f} → {'CONFIRMED' if pred1 else 'FALSIFIED'}")

    pred2 = abs(s1_dS) > abs(native_dS)
    print(f"  P2: |σ₁-ablated ΔS| > |native ΔS|? {abs(s1_dS):.6f} vs {abs(native_dS):.6f} → {'CONFIRMED' if pred2 else 'FALSIFIED'}")

    if scrambled_delta is not None:
        pred3 = abs(scrambled_delta) < abs(native_dS) * 0.5
        print(f"  P3: scrambled |ΔS| < 50% of native? {abs(scrambled_delta):.6f} vs {abs(native_dS)*0.5:.6f} → {'CONFIRMED' if pred3 else 'FALSIFIED'}")

    # ─── Statistical tests ───
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    for mode in ["native", "ablate_sigma2", "ablate_sigma1"]:
        mode_raw = [r for r in raw_results if r["mode"] == mode]
        recv_S = np.array([r["S"] for r in mode_raw if r["condition"] == "receptive"])
        absent_S = np.array([r["S"] for r in mode_raw if r["condition"] == "absent"])

        observed = np.mean(recv_S) - np.mean(absent_S)

        # Permutation test
        all_vals = np.concatenate([recv_S, absent_S])
        n_recv = len(recv_S)
        null = []
        for _ in range(10000):
            perm = np.random.permutation(all_vals)
            null.append(np.mean(perm[:n_recv]) - np.mean(perm[n_recv:]))
        null = np.array(null)
        p = np.mean(np.abs(null) >= np.abs(observed))

        # Effect size
        pooled_std = np.sqrt((np.var(recv_S, ddof=1) + np.var(absent_S, ddof=1)) / 2)
        d = observed / pooled_std if pooled_std > 0 else float("inf")

        # Bootstrap CI
        boot = []
        for _ in range(10000):
            r_boot = np.random.choice(recv_S, size=len(recv_S), replace=True)
            a_boot = np.random.choice(absent_S, size=len(absent_S), replace=True)
            boot.append(np.mean(r_boot) - np.mean(a_boot))
        boot = np.array(boot)
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        print(f"\n  {mode}:")
        print(f"    ΔS = {observed:+.6f}")
        print(f"    p = {p:.4f} (permutation, 10k)")
        print(f"    d = {d:.4f} (Cohen's)")
        print(f"    95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]")
        print(f"    CI excludes zero: {ci_lo > 0 or ci_hi < 0}")

    # ─── Token count verification in results ───
    print(f"\n{'='*60}")
    print("TOKEN COUNT VERIFICATION IN RESULTS")
    for mode in ["native"]:
        mode_raw = [r for r in raw_results if r["mode"] == mode]
        for cond in ["receptive", "control", "absent"]:
            tokens = [r["n_tokens"] for r in mode_raw if r["condition"] == cond]
            print(f"  {cond}: tokens = {tokens}")

    # ─── Compare with original experiment ───
    print(f"\n{'='*60}")
    print("COMPARISON: Original vs Token-Matched")
    print("=" * 60)
    print(f"  Original native ΔS:      +0.022832 (token counts: R=41, A=30)")
    print(f"  Original adjusted ΔS:    +0.004981 (22% retained)")
    print(f"  Token-matched native ΔS: {native_dS:+.6f} (token counts IDENTICAL)")
    if native_dS > 0:
        ratio = native_dS / 0.022832 * 100
        print(f"  Token-matched / Original: {ratio:.1f}%")
        if abs(native_dS - 0.004981) < abs(native_dS - 0.022832):
            print(f"  Closer to adjusted value → token count WAS confounding")
        else:
            print(f"  Closer to raw value → token count was NOT the main driver")
    else:
        print(f"  SIGN REVERSAL → the original effect was entirely token-count-driven")

    # ─── Save results ───
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "experiment": "token_matched_ablation_retest",
        "motivation": "Adversarial audit found r(S,n_tokens)=0.976; original ablation had systematic token differences",
        "design": "9-token condition phrase swap in fixed template; all conditions produce identical token counts",
        "model": MODEL_NAME,
        "ablation_layer": ABLATION_LAYER,
        "measure_layer": MEASURE_LAYER,
        "timestamp": datetime.now().isoformat(),
        "predictions": {
            "P1_native_dS_above_005": pred1,
            "P2_sigma1_amplifies": pred2,
            "P3_scrambled_null": pred3 if scrambled_delta is not None else None,
        },
        "results": all_results,
        "scrambled": {
            "delta_S": float(scrambled_delta) if scrambled_delta is not None else None,
            "raw": scrambled_raw,
        },
        "raw": raw_results,
        "original_comparison": {
            "original_native_dS": 0.022832,
            "original_adjusted_dS": 0.004981,
            "token_matched_native_dS": float(native_dS),
        },
    }

    outpath = RESULTS_DIR / f"exp_token_matched_ablation_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
