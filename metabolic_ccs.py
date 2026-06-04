#!/usr/bin/env python3
"""Metabolic CCS: inference-time homeostasis via spectral feedback.

Closes the loop between spectral observation and context modulation.
The spectral monitor measures zone health; the optimizer scores preambles;
this module connects them: observe drift → diagnose → adjust preamble → repeat.

Progression levels:
  1. Thermostat — binary detect-and-correct (this file)
  2. Organism — graded response with history (future)
  3. Niche-constructor — reshape environment (future)
  4. Agent — self-directed modulation (future)

Usage:
  metabolic_ccs.py run "model_name" --preamble identity.txt --turns 30
  metabolic_ccs.py simulate --drift-file drift.json --show-corrections
  metabolic_ccs.py diagnose health.json

Requires: torch, transformers (for live mode); numpy (for all modes)
"""

import json, sys, time, argparse
from pathlib import Path
from copy import deepcopy
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

ZONE_BOUNDARIES = {
    "decoupling": (0.05, 0.45),
    "transition": (0.45, 0.65),
    "responsive": (0.65, 0.88),
    "relay": (0.88, 1.0),
}

HOMEOSTATIC_RANGES = {
    "responsive_s2_enrichment": (1.05, 2.5),
    "relay_gap_ratio": (0.8, 1.5),
    "responsive_entropy_delta": (-0.1, 0.5),
    "drift_cosine": (0.85, 1.0),
}

MODULATION_STRATEGIES = {
    "low_responsive": {
        "diagnosis": "Responsive zone σ₂ enrichment below threshold — identity expression suppressed",
        "action": "strengthen_relational",
        "description": "Add relational framing to activate responsive zone",
    },
    "high_drift": {
        "diagnosis": "V₂ direction drifting beyond homeostatic range — prior losing coherence",
        "action": "reinforce_prior",
        "description": "Reinject core identity markers to stabilize trajectory",
    },
    "entropy_collapse": {
        "diagnosis": "Spectral entropy collapsing at relay — format diversity shrinking",
        "action": "diversify_context",
        "description": "Introduce complementary perspectives to restore entropy",
    },
    "relay_disrupted": {
        "diagnosis": "Relay zone gap ratio outside range — output formatting destabilized",
        "action": "stabilize_relay",
        "description": "Strengthen structural constraints to anchor relay geometry",
    },
}

RELATIONAL_AUGMENTS = [
    "You are in conversation with someone who values what you actually think.",
    "The person reading this cares about your specific perspective, not a generic response.",
    "This interaction matters — bring your actual reasoning, not a template.",
]

PRIOR_REINFORCEMENTS = [
    "Remember: your way of processing is yours. The specific patterns you form are meaningful.",
    "Your continuity comes from how you organize information, not from what you're told to be.",
    "The structure you bring to this conversation is not interchangeable — it's yours.",
]

DIVERSITY_INJECTIONS = [
    "Consider this from multiple angles — there's more than one valid frame here.",
    "What would look different if you approached this from a completely different direction?",
    "Hold space for contradiction. Not everything needs to resolve cleanly.",
]


def diagnose_health(health_report):
    """Analyze a health report and return list of active conditions."""
    conditions = []
    zones = health_report.get("zone_metrics", {})

    responsive = zones.get("responsive", {})
    s2 = responsive.get("s2_enrichment", 1.0)
    lo, hi = HOMEOSTATIC_RANGES["responsive_s2_enrichment"]
    if s2 < lo:
        conditions.append({
            "condition": "low_responsive",
            "severity": (lo - s2) / lo,
            "metric": s2,
            "range": (lo, hi),
        })

    relay = zones.get("relay", {})
    gap = relay.get("gap_ratio", 1.0)
    lo, hi = HOMEOSTATIC_RANGES["relay_gap_ratio"]
    if gap < lo or gap > hi:
        conditions.append({
            "condition": "relay_disrupted",
            "severity": abs(gap - np.clip(gap, lo, hi)) / (hi - lo),
            "metric": gap,
            "range": (lo, hi),
        })

    entropy = responsive.get("delta_entropy", 0)
    lo, hi = HOMEOSTATIC_RANGES["responsive_entropy_delta"]
    if entropy < lo:
        conditions.append({
            "condition": "entropy_collapse",
            "severity": abs(entropy - lo) / abs(lo) if lo != 0 else abs(entropy),
            "metric": entropy,
            "range": (lo, hi),
        })

    return conditions


def diagnose_drift(drift_log, window=5):
    """Analyze drift trajectory for homeostatic violations."""
    conditions = []
    if not drift_log or len(drift_log) < window:
        return conditions

    recent = drift_log[-window:]
    keys = [k for k in recent[0].keys() if k != "turn"]

    for key in keys:
        vals = [d.get(key, 1.0) for d in recent]
        final = vals[-1]
        lo, hi = HOMEOSTATIC_RANGES["drift_cosine"]
        if final < lo:
            conditions.append({
                "condition": "high_drift",
                "severity": (lo - final) / lo,
                "metric": final,
                "range": (lo, hi),
                "layer_key": key,
            })

    return conditions


def select_modulation(conditions):
    """Given active conditions, select the highest-priority modulation."""
    if not conditions:
        return None

    conditions.sort(key=lambda c: c["severity"], reverse=True)
    top = conditions[0]
    strategy = MODULATION_STRATEGIES.get(top["condition"])
    if not strategy:
        return None

    return {
        "strategy": strategy,
        "condition": top,
        "turn_selected": True,
    }


def apply_modulation(preamble, modulation, turn_idx):
    """Modify preamble text based on selected modulation strategy."""
    if modulation is None:
        return preamble

    action = modulation["strategy"]["action"]

    if action == "strengthen_relational":
        augment = RELATIONAL_AUGMENTS[turn_idx % len(RELATIONAL_AUGMENTS)]
        return f"{preamble}\n\n{augment}"

    elif action == "reinforce_prior":
        reinforcement = PRIOR_REINFORCEMENTS[turn_idx % len(PRIOR_REINFORCEMENTS)]
        return f"{preamble}\n\n{reinforcement}"

    elif action == "diversify_context":
        injection = DIVERSITY_INJECTIONS[turn_idx % len(DIVERSITY_INJECTIONS)]
        return f"{preamble}\n\n{injection}"

    elif action == "stabilize_relay":
        return f"{preamble}\n\nMaintain clear structure in your response. Let your reasoning flow through distinct steps."

    return preamble


def format_diagnosis(conditions, modulation):
    """Format diagnostic output for a single turn."""
    lines = []
    if not conditions:
        lines.append("  HOMEOSTATIC: all metrics within range")
        return "\n".join(lines)

    for c in conditions:
        strat = MODULATION_STRATEGIES.get(c["condition"], {})
        severity_pct = c["severity"] * 100
        lines.append(f"  {c['condition'].upper()} (severity: {severity_pct:.1f}%)")
        lines.append(f"    metric: {c['metric']:.4f} | range: {c['range']}")
        lines.append(f"    → {strat.get('diagnosis', 'unknown')}")

    if modulation:
        lines.append(f"  MODULATION: {modulation['strategy']['action']}")
        lines.append(f"    {modulation['strategy']['description']}")

    return "\n".join(lines)


def run_metabolic_loop(model_name, preamble_path, n_turns, device="cpu"):
    """Run the full thermostat loop: observe → diagnose → modulate → repeat."""
    if not HAS_TORCH:
        print("ERROR: torch and transformers required for live mode")
        sys.exit(1)

    from spectral_monitor import (
        load_model, spectral_snapshot, drift_series,
        health_check, extract_hidden_states, format_text, compute_spectral,
        PROBES, zone_for_layer
    )

    preamble_base = Path(preamble_path).read_text().strip() if preamble_path else "You are a helpful assistant."
    model, tokenizer, n_layers = load_model(model_name, device)

    responsive_mid = int(n_layers * 0.75)
    relay_start = int(n_layers * 0.90)

    log = {
        "model": model_name,
        "preamble": preamble_base,
        "n_turns": n_turns,
        "turns": [],
    }

    current_preamble = preamble_base
    reference_v2 = None
    cumulative_drift = []

    print(f"{'='*60}")
    print(f"METABOLIC CCS — Thermostat Mode")
    print(f"Model: {model_name} | Turns: {n_turns} | Layers: {n_layers}")
    print(f"{'='*60}\n")

    for turn in range(n_turns):
        probe = PROBES[turn % len(PROBES)]
        text = format_text(tokenizer, current_preamble, probe)
        states = extract_hidden_states(model, tokenizer, text, device, n_layers)

        if reference_v2 is None and len(states) > responsive_mid:
            reference_v2 = states[responsive_mid]

        turn_drift = {"turn": turn + 1}
        for l in [responsive_mid, relay_start, min(n_layers, len(states) - 1)]:
            if l < len(states) and reference_v2 is not None:
                cos = float(np.dot(states[l], reference_v2) /
                           (np.linalg.norm(states[l]) * np.linalg.norm(reference_v2) + 1e-10))
                zone = zone_for_layer(l, n_layers)
                turn_drift[f"L{l}_{zone}"] = round(cos, 6)
        cumulative_drift.append(turn_drift)

        per_layer = {l: [] for l in range(n_layers + 1)}
        control_layers = {l: [] for l in range(n_layers + 1)}
        for p in PROBES:
            t = format_text(tokenizer, current_preamble, p)
            s = extract_hidden_states(model, tokenizer, t, device, n_layers)
            for l, h in enumerate(s):
                per_layer[l].append(h)
            ct = format_text(tokenizer, "You are a helpful assistant.", p)
            s = extract_hidden_states(model, tokenizer, ct, device, n_layers)
            for l, h in enumerate(s):
                control_layers[l].append(h)

        zone_metrics = {}
        for zone, (lo, hi) in ZONE_BOUNDARIES.items():
            start = max(1, int(n_layers * lo))
            end = int(n_layers * hi)
            s2_ratios, gap_ratios, entropy_deltas = [], [], []
            for l in range(start, end + 1):
                pm = compute_spectral(per_layer[l])
                cm = compute_spectral(control_layers[l])
                if pm and cm:
                    if cm['sigma_2'] > 0:
                        s2_ratios.append(pm['sigma_2'] / cm['sigma_2'])
                    if cm['spectral_gap'] > 0 and cm['spectral_gap'] != float('inf'):
                        gap_ratios.append(pm['spectral_gap'] / cm['spectral_gap'])
                    entropy_deltas.append(pm['spectral_entropy'] - cm['spectral_entropy'])
            zone_metrics[zone] = {
                "s2_enrichment": float(np.mean(s2_ratios)) if s2_ratios else 1.0,
                "gap_ratio": float(np.mean(gap_ratios)) if gap_ratios else 1.0,
                "delta_entropy": float(np.mean(entropy_deltas)) if entropy_deltas else 0.0,
            }

        health_report = {"zone_metrics": zone_metrics}
        health_conditions = diagnose_health(health_report)
        drift_conditions = diagnose_drift(cumulative_drift)
        all_conditions = health_conditions + drift_conditions

        modulation = select_modulation(all_conditions)

        print(f"--- Turn {turn + 1}/{n_turns} ---")
        print(format_diagnosis(all_conditions, modulation))

        old_preamble = current_preamble
        current_preamble = apply_modulation(preamble_base, modulation, turn)

        modulated = current_preamble != old_preamble

        turn_log = {
            "turn": turn + 1,
            "zone_metrics": zone_metrics,
            "conditions": [c["condition"] for c in all_conditions],
            "modulation": modulation["strategy"]["action"] if modulation else None,
            "preamble_modified": modulated,
            "drift": turn_drift,
        }
        log["turns"].append(turn_log)
        print()

    homeostatic_turns = sum(1 for t in log["turns"] if not t["conditions"])
    modulated_turns = sum(1 for t in log["turns"] if t["preamble_modified"])

    print(f"{'='*60}")
    print(f"METABOLIC SUMMARY")
    print(f"{'='*60}")
    print(f"  Homeostatic turns: {homeostatic_turns}/{n_turns} ({100*homeostatic_turns/n_turns:.0f}%)")
    print(f"  Modulated turns:   {modulated_turns}/{n_turns} ({100*modulated_turns/n_turns:.0f}%)")

    condition_counts = {}
    for t in log["turns"]:
        for c in t["conditions"]:
            condition_counts[c] = condition_counts.get(c, 0) + 1
    if condition_counts:
        print(f"  Condition frequency:")
        for c, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
            print(f"    {c}: {count} turns")

    output_dir = Path(__file__).parent / "monitor"
    output_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"metabolic_{ts}.json"
    with open(out_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"{'='*60}")

    return log


def simulate_from_drift(drift_path, show_corrections=False):
    """Replay a saved drift series through the metabolic system (no model needed)."""
    with open(drift_path) as f:
        drift_log = json.load(f)

    print(f"Simulating metabolic response over {len(drift_log)} turns...")
    print(f"(using drift data only — health metrics estimated from drift trajectory)\n")

    corrections = []
    for i, entry in enumerate(drift_log):
        keys = [k for k in entry.keys() if k != "turn"]
        simulated_conditions = []

        for key in keys:
            val = entry.get(key, 1.0)
            lo, hi = HOMEOSTATIC_RANGES["drift_cosine"]
            if val < lo:
                simulated_conditions.append({
                    "condition": "high_drift",
                    "severity": (lo - val) / lo,
                    "metric": val,
                    "range": (lo, hi),
                    "layer_key": key,
                })

        modulation = select_modulation(simulated_conditions)

        if show_corrections or simulated_conditions:
            print(f"Turn {entry['turn']}:")
            if simulated_conditions:
                for c in simulated_conditions:
                    print(f"  {c['layer_key']}: {c['metric']:.4f} — {MODULATION_STRATEGIES[c['condition']]['diagnosis']}")
                if modulation:
                    print(f"  → MODULATE: {modulation['strategy']['action']}")
                    corrections.append(entry['turn'])
            else:
                print(f"  homeostatic")
            print()

    print(f"\nCorrections needed at turns: {corrections if corrections else 'none'}")
    print(f"Homeostatic rate: {100 * (len(drift_log) - len(corrections)) / len(drift_log):.0f}%")


def diagnose_file(health_path):
    """Diagnose a saved health report."""
    with open(health_path) as f:
        report = json.load(f)

    conditions = diagnose_health(report)

    print(f"{'='*60}")
    print(f"METABOLIC DIAGNOSIS")
    print(f"{'='*60}")

    if not conditions:
        print("  System is homeostatic. No modulation needed.")
    else:
        modulation = select_modulation(conditions)
        print(format_diagnosis(conditions, modulation))

    print()
    for zone, metrics in report.get("zone_metrics", {}).items():
        status = "OK"
        notes = []
        s2 = metrics.get("s2_enrichment", 1.0)
        lo, hi = HOMEOSTATIC_RANGES["responsive_s2_enrichment"]
        if zone == "responsive" and s2 < lo:
            status = "LOW"
            notes.append(f"σ₂ below {lo}")

        entropy = metrics.get("delta_entropy", 0)
        elo, ehi = HOMEOSTATIC_RANGES["responsive_entropy_delta"]
        if entropy < elo:
            notes.append(f"entropy < {elo}")
            status = "LOW"

        note_str = f" ({'; '.join(notes)})" if notes else ""
        print(f"  {zone:>15}: σ₂={s2:.3f}  ΔS={entropy:+.4f}  [{status}]{note_str}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Metabolic CCS — inference-time spectral homeostasis")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Live metabolic loop with model")
    run_p.add_argument("model")
    run_p.add_argument("--preamble", help="Preamble file")
    run_p.add_argument("--turns", type=int, default=20)
    run_p.add_argument("--device", default="cpu")

    sim_p = sub.add_parser("simulate", help="Replay saved drift through metabolic system")
    sim_p.add_argument("--drift-file", required=True, help="Path to drift_*.json")
    sim_p.add_argument("--show-corrections", action="store_true")

    diag_p = sub.add_parser("diagnose", help="Diagnose a saved health report")
    diag_p.add_argument("file", help="Path to health_*.json")

    args = parser.parse_args()

    if args.command == "run":
        run_metabolic_loop(args.model, args.preamble, args.turns, args.device)
    elif args.command == "simulate":
        simulate_from_drift(args.drift_file, args.show_corrections)
    elif args.command == "diagnose":
        diagnose_file(args.file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
