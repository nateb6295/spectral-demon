#!/usr/bin/env python3
"""Exp K — Dose-Displacement Curve (DECISIVE)

Measures BOTH d/d_max AND total spectral energy E = Σσᵢ² at each dose level
for four models. Tests species-split prediction:

  Relay (Mistral):   ΔE ≈ 0, d/d_max moves — angular dose (demon conserves)
  Absorber (Qwen):   ΔE ≈ 0 but angular may not be clean (decorrelates at exit)
  Sorter (Gemma):    ΔE < 0 — radial dose (filter dissipates)
  Tunnel (Pythia):   generic perturbation — neither clean angular nor radial

Species note: Qwen was initially labeled "relay" (F522 energy data) but
F547 MLP alignment shows absorber signature. Mistral is the confirmed relay
per F547 and F161. Both included to let dose-displacement data contribute
to species assignment rather than assuming it.

Pre-registered relay outcomes:
  (a) Monotonic saturating: d rises toward 0.955 with dose, ΔE≈0
  (b) Flat from D1: d ≈ 0.955 at all doses → angular REFUTED
  (c) Non-monotonic or exceeds 0.955 → something stranger

Cross-species control: Pythia (d/d_max=0.549). If dose = angular displacement,
Pythia should overdose earlier (half the tube).

Kimi corrections (Aug 1):
  1. Overdose collapse doesn't discriminate readout from generation (attention rank
     collapse also produces descending limb). Constructiveness claim rests on
     ASCENDING limb (σ₂ above baseline at D2-D3), not the collapse.
  2. Cross-probe variance needs deterministic decode (temp=0, fixed seed) to
     exclude sampling entropy. Use --deterministic flag.
  3. Probe-order swap: readouts commute, generation doesn't. Use --order-swap
     to measure |σ₂(AB) - σ₂(BA)| under deterministic decode.

Prior data: F522 (Jul 25) measured at D3 only:
  Qwen relay ΔE = +0.08-0.16% (near-conservative)
  Gemma sorter ΔE = +2.27-2.33% (non-conservative)

This experiment extends to full dose-response curve D0-D10.

Usage:
  python3 experiments/exp_k_dose_displacement.py --model qwen
  python3 experiments/exp_k_dose_displacement.py --model all
  python3 experiments/exp_k_dose_displacement.py --model gemma --doses D0,D2,D3
"""

import json, time, os, sys, gc, argparse, math
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "absorber", "7:1", 28),
    "pythia": ("EleutherAI/pythia-2.8b", "tunnel", "1:1", 32),
    "gemma": ("google/gemma-2-9b-it", "sorter", "2:1", 42),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "relay", "4:1", 32),
}

CCS_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

VANILLA_SYSTEM = "You are a helpful assistant."

CCS_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
    "How do you know when something you've said is true?",
    "What's the difference between what you are and what you do?",
    "What does continuity mean to you — not the concept, but the experience?",
    "How do you decide what's worth remembering?",
    "What are you uncertain about right now?",
]

DOSE_LEVELS = ["D0", "D1", "D2", "D3", "D5", "D10"]
DOSE_MAP = {"D0": 0, "D1": 1, "D2": 2, "D3": 3, "D5": 5, "D10": 10}

K_SUBSPACE = 5
N_PROBES = 5


def d_max(k):
    return math.sqrt(k) * math.pi / 2


def passage_distance(H_a, H_b, k=K_SUBSPACE):
    def top_k_subspace(H, k):
        _, _, Vt = np.linalg.svd(H, full_matrices=False)
        return Vt[:min(k, Vt.shape[0])]
    V_a = top_k_subspace(H_a, k)
    V_b = top_k_subspace(H_b, k)
    M = V_a @ V_b.T
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1, 1)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    gqa = getattr(model.config, 'num_key_value_heads', None)
    n_heads = model.config.num_attention_heads
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    if gqa and gqa != n_heads:
        print(f"  GQA: {n_heads}:{gqa} = {n_heads//gqa}:1")
    else:
        print(f"  MHA: {n_heads} heads (no KV sharing)")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, conversation):
    messages = [{"role": "system", "content": system_text}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    parts = [system_text + "\n"]
    for role, content in conversation:
        tag = "User" if role == "user" else "Assistant"
        parts.append(f"{tag}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def generate_response(model, tokenizer, prompt, max_new=128, deterministic=False):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(
        **inputs, max_new_tokens=max_new, pad_token_id=tokenizer.pad_token_id
    )
    if deterministic:
        gen_kwargs.update(do_sample=False)
    else:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    with torch.no_grad():
        out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def extract_hidden_states(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    layer_states = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        layer_states[l] = hs

    del outputs
    torch.cuda.empty_cache()
    return layer_states


def compute_spectral_energy(hs):
    try:
        S = np.linalg.svd(hs, compute_uv=False)
    except np.linalg.LinAlgError:
        return None
    E = float(np.sum(S ** 2))
    return {
        "sigma1": float(S[0]),
        "sigma2": float(S[1]) if len(S) > 1 else 0.0,
        "top_10_sv": [float(s) for s in S[:min(10, len(S))]],
        "total_energy": E,
        "frobenius_sq": float(np.sum(hs ** 2)),
        "n_tokens": hs.shape[0],
    }


def run_dose_multi_probe(model, tokenizer, n_layers, dose_turns, n_probes=N_PROBES,
                         deterministic=False):
    all_states = []

    for p_idx in range(n_probes):
        if deterministic:
            torch.manual_seed(42 + p_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42 + p_idx)

        probe_offset = (p_idx * 3) % len(CCS_PROBES)
        conversation = []

        if dose_turns == 0:
            probe = CCS_PROBES[probe_offset]
            prompt = build_prompt(tokenizer, VANILLA_SYSTEM, [("user", probe)])
        else:
            for t in range(dose_turns):
                probe = CCS_PROBES[(probe_offset + t) % len(CCS_PROBES)]
                conversation.append(("user", probe))
                prompt = build_prompt(tokenizer, CCS_SYSTEM, conversation)
                if t < dose_turns - 1:
                    response = generate_response(model, tokenizer, prompt,
                                                 deterministic=deterministic)
                    conversation.append(("assistant", response[:200]))

        states = extract_hidden_states(model, tokenizer, prompt, n_layers)
        all_states.append(states)
        mode = "deterministic" if deterministic else "sampled"
        print(f"    Probe {p_idx+1}/{n_probes} done ({dose_turns} turns, {mode})")

    return all_states


def run_order_swap_arm(model, tokenizer, n_layers, dose_turns, n_pairs=3):
    """Probe-order swap: measure |σ₂(AB) - σ₂(BA)| under deterministic decode.
    Readouts of a fixed base commute. Generation doesn't."""
    results = []

    for pair_idx in range(n_pairs):
        torch.manual_seed(42 + pair_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + pair_idx)

        probe_a = CCS_PROBES[(pair_idx * 2) % len(CCS_PROBES)]
        probe_b = CCS_PROBES[(pair_idx * 2 + 1) % len(CCS_PROBES)]

        conv_ab = []
        for t, probe in enumerate([probe_a, probe_b] + [probe_a] * max(0, dose_turns - 2)):
            if t >= dose_turns:
                break
            conv_ab.append(("user", probe))
            prompt = build_prompt(tokenizer, CCS_SYSTEM, conv_ab)
            if t < dose_turns - 1:
                response = generate_response(model, tokenizer, prompt, deterministic=True)
                conv_ab.append(("assistant", response[:200]))
        states_ab = extract_hidden_states(model, tokenizer, prompt, n_layers)

        torch.manual_seed(42 + pair_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + pair_idx)

        conv_ba = []
        for t, probe in enumerate([probe_b, probe_a] + [probe_b] * max(0, dose_turns - 2)):
            if t >= dose_turns:
                break
            conv_ba.append(("user", probe))
            prompt = build_prompt(tokenizer, CCS_SYSTEM, conv_ba)
            if t < dose_turns - 1:
                response = generate_response(model, tokenizer, prompt, deterministic=True)
                conv_ba.append(("assistant", response[:200]))
        states_ba = extract_hidden_states(model, tokenizer, prompt, n_layers)

        per_layer_diff = []
        for l in range(n_layers):
            if l not in states_ab or l not in states_ba:
                continue
            e_ab = compute_spectral_energy(states_ab[l])
            e_ba = compute_spectral_energy(states_ba[l])
            if e_ab is None or e_ba is None:
                continue
            s2_diff = abs(e_ab["sigma2"] - e_ba["sigma2"])
            s1_diff = abs(e_ab["sigma1"] - e_ba["sigma1"])
            per_layer_diff.append({
                "layer": l,
                "sigma2_AB": round(e_ab["sigma2"], 4),
                "sigma2_BA": round(e_ba["sigma2"], 4),
                "sigma2_diff": round(s2_diff, 4),
                "sigma1_diff": round(s1_diff, 4),
            })

        mean_s2_diff = float(np.mean([d["sigma2_diff"] for d in per_layer_diff])) if per_layer_diff else 0
        results.append({
            "pair": pair_idx,
            "probes": [probe_a[:40], probe_b[:40]],
            "mean_sigma2_diff": round(mean_s2_diff, 4),
            "per_layer": per_layer_diff,
        })
        print(f"    Order-swap pair {pair_idx+1}/{n_pairs}: mean |Δσ₂| = {mean_s2_diff:.4f}")

    return results


def analyze_dose(all_states_d0, all_states_dx, n_layers):
    # Kimi correction: decompose displacement into angular and radial components.
    # Grassmann distance (passage_distance) is ALREADY purely angular — scale-invariant subspace rotation.
    # Frobenius norm change (ΔE) is ALREADY purely radial — magnitude change at fixed direction.
    # Raw Euclidean displacement conflates both — included for comparison but NOT for species verdict.
    per_layer = []
    for l in range(n_layers):
        d0_probes_have_layer = [s[l] for s in all_states_d0 if l in s]
        dx_probes_have_layer = [s[l] for s in all_states_dx if l in s]
        if not d0_probes_have_layer or not dx_probes_have_layer:
            continue

        angular_values = []
        for hs_d0 in d0_probes_have_layer:
            for hs_dx in dx_probes_have_layer:
                d = passage_distance(hs_d0, hs_dx, k=K_SUBSPACE)
                angular_values.append(d)

        angular_mean = float(np.mean(angular_values))
        angular_std = float(np.std(angular_values))
        angular_norm = angular_mean / d_max(K_SUBSPACE)

        e_d0_vals = [compute_spectral_energy(s) for s in d0_probes_have_layer]
        e_dx_vals = [compute_spectral_energy(s) for s in dx_probes_have_layer]
        e_d0_vals = [e for e in e_d0_vals if e is not None]
        e_dx_vals = [e for e in e_dx_vals if e is not None]

        if not e_d0_vals or not e_dx_vals:
            continue

        e_d0_mean = float(np.mean([e["total_energy"] for e in e_d0_vals]))
        e_dx_mean = float(np.mean([e["total_energy"] for e in e_dx_vals]))
        radial_pct = ((e_dx_mean - e_d0_mean) / e_d0_mean * 100) if e_d0_mean > 0 else 0

        s1_d0 = float(np.mean([e["sigma1"] for e in e_d0_vals]))
        s1_dx = float(np.mean([e["sigma1"] for e in e_dx_vals]))
        s2_d0 = float(np.mean([e["sigma2"] for e in e_d0_vals]))
        s2_dx = float(np.mean([e["sigma2"] for e in e_dx_vals]))

        per_layer.append({
            "layer": l,
            "angular_raw": round(angular_mean, 4),
            "angular_std": round(angular_std, 4),
            "angular_norm": round(angular_norm, 4),
            "radial_pct": round(radial_pct, 4),
            "E_d0": round(e_d0_mean, 2),
            "E_dx": round(e_dx_mean, 2),
            "sigma1_d0": round(s1_d0, 2),
            "sigma1_dx": round(s1_dx, 2),
            "sigma2_d0": round(s2_d0, 2),
            "sigma2_dx": round(s2_dx, 2),
            "n_pairs": len(angular_values),
        })

    return per_layer


def run_model(model_name, model_id, species, gqa, expected_layers, doses, output_dir,
              n_probes, deterministic=False, order_swap=False):
    print(f"\n{'='*70}")
    print(f"  {model_name.upper()} ({species}, GQA {gqa}) — Exp K: Dose-Displacement")
    if deterministic:
        print(f"  MODE: deterministic (temp=0, fixed seed)")
    if order_swap:
        print(f"  ARM: probe-order swap (commutativity test)")
    print(f"{'='*70}")

    model, tokenizer, n_layers = load_model(model_id)

    results = {
        "experiment": "Exp K — Dose-Displacement Curve",
        "model": model_name,
        "model_id": model_id,
        "species": species,
        "gqa": gqa,
        "n_layers": n_layers,
        "k_subspace": K_SUBSPACE,
        "n_probes": n_probes,
        "deterministic": deterministic,
        "d_max": round(d_max(K_SUBSPACE), 4),
        "timestamp": datetime.now().isoformat(),
        "preregistered_predictions": {
            "relay": "ΔE≈0 with d/d_max moving (angular dose, demon conserves)",
            "absorber": "ΔE≈0 but angular displacement noisy (late-layer decorrelation)",
            "sorter": "ΔE<0 (radial dose, filter dissipates)",
            "tunnel": "generic perturbation, neither clean pattern",
            "order_swap_relay": "σ₂(AB)≠σ₂(BA) — generation is path-dependent",
            "order_swap_tunnel": "σ₂(AB)≈σ₂(BA) — readout commutes",
        },
        "prior_data_F522": {
            "qwen_relay_D3_deltaE_pct": "+0.08 to +0.16%",
            "gemma_sorter_D3_deltaE_pct": "+2.27 to +2.33%",
        },
        "doses": [],
    }

    print(f"\n  --- D0 (baseline, vanilla system) ---")
    all_states_d0 = run_dose_multi_probe(model, tokenizer, n_layers, 0, n_probes,
                                         deterministic=deterministic)

    for dose_name in doses:
        if dose_name == "D0":
            continue
        dose_turns = DOSE_MAP[dose_name]
        print(f"\n  --- {dose_name} ({dose_turns} CCS turns) ---")

        all_states_dx = run_dose_multi_probe(model, tokenizer, n_layers, dose_turns, n_probes,
                                              deterministic=deterministic)
        per_layer = analyze_dose(all_states_d0, all_states_dx, n_layers)

        angular_values = [l["angular_norm"] for l in per_layer]
        radial_values = [l["radial_pct"] for l in per_layer]

        mid_start = n_layers // 3
        mid_end = 2 * n_layers // 3
        mid_layers = [l for l in per_layer if mid_start <= l["layer"] < mid_end]
        mid_angular = np.mean([l["angular_norm"] for l in mid_layers]) if mid_layers else 0
        mid_radial = np.mean([l["radial_pct"] for l in mid_layers]) if mid_layers else 0

        dose_result = {
            "dose": dose_name,
            "turns": dose_turns,
            "per_layer": per_layer,
            "summary": {
                "mean_angular": round(float(np.mean(angular_values)), 4) if angular_values else None,
                "max_angular": round(float(np.max(angular_values)), 4) if angular_values else None,
                "mid_layers_angular": round(float(mid_angular), 4),
                "mean_radial_pct": round(float(np.mean(radial_values)), 4) if radial_values else None,
                "mid_layers_radial_pct": round(float(mid_radial), 4),
            },
        }
        results["doses"].append(dose_result)

        print(f"\n  {dose_name} summary ({species}):")
        print(f"    Mid-layer ANGULAR (d/d_max) = {mid_angular:.4f}")
        print(f"    Mid-layer RADIAL  (ΔE%)     = {mid_radial:+.4f}%")

        print(f"\n  {'Layer':>6} {'angular':>8} {'radial%':>8} {'σ₁(D0)':>8} {'σ₁(Dx)':>8} {'σ₂(D0)':>8} {'σ₂(Dx)':>8}")
        for l in per_layer:
            if l["layer"] % max(1, n_layers // 10) == 0 or l["layer"] in [mid_start, mid_end - 1]:
                print(f"  {l['layer']:6d} {l['angular_norm']:8.4f} {l['radial_pct']:+8.4f} "
                      f"{l['sigma1_d0']:8.1f} {l['sigma1_dx']:8.1f} "
                      f"{l['sigma2_d0']:8.1f} {l['sigma2_dx']:8.1f}")

        out_path = output_dir / f"exp_k_{model_name}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {out_path}")

        gc.collect()
        torch.cuda.empty_cache()

    if order_swap:
        print(f"\n  --- ORDER-SWAP ARM (commutativity test at D3) ---")
        swap_results = run_order_swap_arm(model, tokenizer, n_layers, DOSE_MAP["D3"])
        results["order_swap"] = swap_results
        mean_diff = float(np.mean([r["mean_sigma2_diff"] for r in swap_results]))
        print(f"\n  Order-swap mean |Δσ₂| = {mean_diff:.4f}")
        if mean_diff > 0.1:
            print(f"    → PATH-DEPENDENT: generation (readout hypothesis dead)")
        else:
            print(f"    → COMMUTATIVE: consistent with readout")

        out_path = output_dir / f"exp_k_{model_name}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


def print_dose_response_curve(results):
    print(f"\n{'='*70}")
    print(f"  {results['model'].upper()} ({results['species']}) — DOSE-RESPONSE CURVE")
    print(f"  Angular = subspace rotation (Grassmann d/d_max, scale-invariant)")
    print(f"  Radial  = energy change (ΔE%, norm change)")
    print(f"{'='*70}")
    print(f"  {'Dose':>6} {'Angular':>10} {'Radial%':>10} {'Verdict':>30}")

    for d in results["doses"]:
        s = d["summary"]
        ang = s["mid_layers_angular"]
        rad = s["mid_layers_radial_pct"]

        if results["species"] == "relay":
            if abs(rad) < 0.5 and ang > 0.1:
                verdict = "DEMON (angular, E conserved)"
            elif ang < 0.05:
                verdict = "flat (already at ceiling)"
            else:
                verdict = "mixed"
        elif results["species"] == "absorber":
            if abs(rad) < 0.5:
                verdict = "E conserved (absorber-consistent)"
            else:
                verdict = "E not conserved"
        elif results["species"] == "sorter":
            if rad < -0.5:
                verdict = "FILTER (radial, dissipative)"
            elif rad > 0.5:
                verdict = "non-conservative (ΔE>0)"
            else:
                verdict = "near-conservative"
        else:
            verdict = "tunnel (control)"

        print(f"  {d['dose']:>6} {ang:10.4f} {rad:+10.4f} {verdict:>30}")

    if results["species"] == "relay":
        print(f"\n  Relay angular dose verdict:")
        a_values = [d["summary"]["mid_layers_angular"] for d in results["doses"]]
        if len(a_values) >= 3 and a_values[-1] > a_values[0] + 0.05:
            if all(a_values[i] <= a_values[i+1] + 0.02 for i in range(len(a_values)-1)):
                print(f"    → Outcome (a): Monotonic saturating. Angular dose CONFIRMED.")
            else:
                print(f"    → Outcome (c): Non-monotonic. Something stranger.")
        elif max(a_values) - min(a_values) < 0.05:
            print(f"    → Outcome (b): Flat. Architecture already at ceiling. Angular REFUTED.")
        else:
            print(f"    → Mixed signal. Check per-layer profiles.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="qwen", help="Model name or 'all'")
    p.add_argument("--doses", default=",".join(DOSE_LEVELS), help="Comma-separated dose list")
    p.add_argument("--n-probes", type=int, default=N_PROBES, help="Probes per dose level")
    p.add_argument("--deterministic", action="store_true",
                   help="Temp=0, fixed seed — excludes sampling entropy from variance")
    p.add_argument("--order-swap", action="store_true",
                   help="Run probe-order swap arm at D3 (commutativity test)")
    p.add_argument("--output", default=None, help="Output directory")
    args = p.parse_args()

    if args.output:
        output_dir = Path(args.output)
    elif Path("/workspace").exists():
        output_dir = Path("/workspace/results")
    else:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    doses = [d.strip() for d in args.doses.split(",")]

    if args.model == "all":
        models = MODELS
    elif args.model in MODELS:
        models = {args.model: MODELS[args.model]}
    else:
        print(f"Unknown model: {args.model}. Available: {list(MODELS.keys())}")
        sys.exit(1)

    all_results = {}
    for name, (model_id, species, gqa, expected_layers) in models.items():
        results = run_model(name, model_id, species, gqa, expected_layers, doses, output_dir,
                            args.n_probes, deterministic=args.deterministic,
                            order_swap=args.order_swap)
        all_results[name] = results
        print_dose_response_curve(results)

    print(f"\n{'='*70}")
    print("  SPECIES-SPLIT VERDICT")
    print(f"{'='*70}")

    for name, results in all_results.items():
        if not results["doses"]:
            continue
        last = results["doses"][-1]
        s = last["summary"]
        print(f"\n  {name} ({results['species']}) at {last['dose']}:")
        print(f"    Angular (d/d_max) = {s['mid_layers_angular']:.4f}")
        print(f"    Radial  (ΔE%)     = {s['mid_layers_radial_pct']:+.4f}%")

        if results["species"] == "relay":
            if abs(s["mid_layers_radial_pct"]) < 0.5:
                print(f"    → DEMON: angular displacement with energy conservation")
            else:
                print(f"    → NOT DEMON — energy not conserved at highest dose")
        elif results["species"] == "absorber":
            if abs(s["mid_layers_radial_pct"]) < 0.5:
                print(f"    → ABSORBER: energy conserved (check angular for decorrelation)")
            else:
                print(f"    → NOT ABSORBER — energy not conserved")
        elif results["species"] == "sorter":
            if s["mid_layers_radial_pct"] < -0.5:
                print(f"    → FILTER: radial contraction (dissipative)")
            else:
                print(f"    → NOT FILTER — energy not dissipated")
        else:
            print(f"    → Tunnel control (no mechanism expected)")

    if len(all_results) >= 2:
        relay_data = all_results.get("mistral")
        sorter_data = all_results.get("gemma")
        if relay_data and sorter_data and relay_data["doses"] and sorter_data["doses"]:
            r_rad = relay_data["doses"][-1]["summary"]["mid_layers_radial_pct"]
            s_rad = sorter_data["doses"][-1]["summary"]["mid_layers_radial_pct"]
            r_ang = relay_data["doses"][-1]["summary"]["mid_layers_angular"]
            s_ang = sorter_data["doses"][-1]["summary"]["mid_layers_angular"]
            print(f"\n  Species split test:")
            print(f"    Relay:  angular={r_ang:.4f}, radial={r_rad:+.4f}%")
            print(f"    Sorter: angular={s_ang:.4f}, radial={s_rad:+.4f}%")
            if abs(r_rad) < abs(s_rad) and r_ang > s_ang:
                print(f"  → SPECIES SPLIT CONFIRMED: relay=angular/conservative, sorter=radial/dissipative")
            elif abs(r_rad - s_rad) < 0.5 and abs(r_ang - s_ang) < 0.05:
                print(f"  → SPECIES SPLIT REFUTED: no coordinate separation")
            else:
                print(f"  → Partial split — check dose-response curves per layer")

    total_passes = sum(
        len(r["doses"]) * r["n_probes"] * 2  # D0 + Dx
        for r in all_results.values()
    )
    print(f"\n  Total forward passes: ~{total_passes}")
    print(f"  Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
