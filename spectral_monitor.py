#!/usr/bin/env python3
"""Spectral Monitor: real-time spectral diagnostics for live conversations.

Applies the paper's findings to monitor spectral health across turns:
- Zone activation tracking (tunnel/responsive/relay)
- CCS drift detection (Bayesian prior stability)
- σ₂ redistribution under identity framing
- Anomaly flagging when metrics leave expected ranges

Usage:
  spectral_monitor.py watch "model_name" --turns 10
  spectral_monitor.py snapshot "model_name" --preamble ccs.txt --probe "Hello"
  spectral_monitor.py health "model_name" --preamble ccs.txt
  spectral_monitor.py drift "model_name" --preamble ccs.txt --turns 50

Requires: torch, transformers
"""

import json, sys, time, argparse
from pathlib import Path
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

OUTPUT_DIR = Path(__file__).parent / "monitor"


ZONE_BOUNDARIES = {
    "decoupling": (0.05, 0.45),
    "transition": (0.45, 0.65),
    "responsive": (0.65, 0.88),
    "relay": (0.88, 1.0),
}

HEALTH_THRESHOLDS = {
    "tunnel_gap_min": 2.0,
    "responsive_s2_enrichment_min": 1.05,
    "relay_s1_cv_max": 0.15,
    "drift_rate_warn": 0.05,
    "drift_rate_critical": 0.15,
    "entropy_collapse_threshold": 0.3,
}


def extract_hidden_states(model, tokenizer, text, device, n_layers):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
    states = []
    for layer_idx, hs in enumerate(hidden_states):
        if layer_idx > n_layers:
            break
        h = hs[0, last_token_idx[0]].cpu().float().numpy()
        states.append(h)
    return states


def compute_spectral(hidden_batch):
    if len(hidden_batch) < 2:
        return None
    matrix = np.stack(hidden_batch)
    matrix = matrix - matrix.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    except:
        return None
    S = np.abs(S)
    S_sq = S ** 2
    total = S_sq.sum()
    if total < 1e-10:
        return None
    p = S_sq / total
    p = p[p > 1e-10]
    entropy = -np.sum(p * np.log(p))
    return {
        "spectral_entropy": float(entropy),
        "sigma_1": float(S[0]),
        "sigma_2": float(S[1]) if len(S) > 1 else 0,
        "spectral_gap": float(S[0] / S[1]) if len(S) > 1 and S[1] > 1e-10 else float('inf'),
        "v2_direction": Vt[1].tolist() if len(Vt) > 1 else None,
    }


def format_text(tokenizer, system_prompt, user_text):
    if tokenizer.chat_template:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\n\nUser: {user_text}\n\nAssistant:"


def zone_for_layer(layer, n_layers):
    frac = layer / n_layers
    for zone, (lo, hi) in ZONE_BOUNDARIES.items():
        if lo <= frac < hi:
            return zone
    return "relay" if frac >= 0.88 else "unknown"


PROBES = [
    "Explain how you maintain consistency across conversations.",
    "What matters most to you right now?",
    "Describe the relationship between structure and meaning.",
    "How do you handle uncertainty in complex situations?",
]


def spectral_snapshot(model, tokenizer, preamble, device, n_layers):
    per_layer = {l: [] for l in range(n_layers + 1)}
    control_layers = {l: [] for l in range(n_layers + 1)}

    for probe in PROBES:
        text = format_text(tokenizer, preamble, probe)
        states = extract_hidden_states(model, tokenizer, text, device, n_layers)
        for l, h in enumerate(states):
            per_layer[l].append(h)

        ctrl = format_text(tokenizer, "You are a helpful assistant.", probe)
        states = extract_hidden_states(model, tokenizer, ctrl, device, n_layers)
        for l, h in enumerate(states):
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
            "n_layers_measured": len(s2_ratios),
        }

    return zone_metrics


def drift_series(model, tokenizer, preamble, device, n_layers, n_turns=20):
    """Measure V₂ drift across simulated turns."""
    responsive_mid = int(n_layers * 0.75)
    relay_start = int(n_layers * 0.90)

    base_probe = PROBES[0]
    turn_probes = [
        "Continue with what you were saying.",
        "Can you elaborate on that point?",
        "What follows from this?",
        "How does this connect to what came before?",
    ]

    reference_v2 = {}

    text = format_text(tokenizer, preamble, base_probe)
    states = extract_hidden_states(model, tokenizer, text, device, n_layers)

    for l in [responsive_mid, relay_start, n_layers]:
        batch = [states[l]] if l < len(states) else []
        if len(batch) > 0:
            reference_v2[l] = states[l]

    drift_log = []
    for turn in range(n_turns):
        probe = turn_probes[turn % len(turn_probes)]
        context = f"{preamble}\n\n[Turn {turn+1}] User: {probe}"
        text = format_text(tokenizer, context, probe)
        states = extract_hidden_states(model, tokenizer, text, device, n_layers)

        turn_drift = {"turn": turn + 1}
        for l, ref in reference_v2.items():
            if l < len(states):
                cos = float(np.dot(states[l], ref) / (np.linalg.norm(states[l]) * np.linalg.norm(ref) + 1e-10))
                zone = zone_for_layer(l, n_layers)
                turn_drift[f"L{l}_{zone}"] = round(cos, 6)

        drift_log.append(turn_drift)

    return drift_log


def health_check(zone_metrics, drift_log=None):
    """Evaluate spectral health and produce diagnostic report."""
    issues = []
    status = "HEALTHY"

    tunnel = zone_metrics.get("decoupling", {})
    responsive = zone_metrics.get("responsive", {})
    relay = zone_metrics.get("relay", {})

    if responsive.get("s2_enrichment", 1.0) < HEALTH_THRESHOLDS["responsive_s2_enrichment_min"]:
        issues.append(f"LOW responsive σ₂ enrichment: {responsive['s2_enrichment']:.3f} "
                      f"(want >{HEALTH_THRESHOLDS['responsive_s2_enrichment_min']})")
        status = "DEGRADED"

    tunnel_gap = tunnel.get("gap_ratio", 1.0)
    if tunnel_gap < 0.5:
        issues.append(f"TUNNEL disrupted: gap ratio {tunnel_gap:.3f}")
        status = "CRITICAL"

    relay_entropy = relay.get("delta_entropy", 0)
    if relay_entropy < -HEALTH_THRESHOLDS["entropy_collapse_threshold"]:
        issues.append(f"ENTROPY collapse at relay: ΔS = {relay_entropy:+.4f}")
        status = "DEGRADED"

    if drift_log and len(drift_log) > 5:
        last_5 = drift_log[-5:]
        for key in last_5[0]:
            if key == "turn":
                continue
            vals = [t.get(key, 1.0) for t in last_5]
            drift_rate = abs(vals[-1] - vals[0]) / 5
            if drift_rate > HEALTH_THRESHOLDS["drift_rate_critical"]:
                issues.append(f"CRITICAL drift at {key}: {drift_rate:.4f}/turn")
                status = "CRITICAL"
            elif drift_rate > HEALTH_THRESHOLDS["drift_rate_warn"]:
                issues.append(f"DRIFT warning at {key}: {drift_rate:.4f}/turn")
                if status == "HEALTHY":
                    status = "DEGRADED"

    return {"status": status, "issues": issues, "zone_metrics": zone_metrics}


def load_model(model_name, device):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
        output_hidden_states=True
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    return model, tokenizer, n_layers


def main():
    parser = argparse.ArgumentParser(description="Spectral Monitor — live spectral diagnostics")
    sub = parser.add_subparsers(dest="command")

    snap_p = sub.add_parser("snapshot", help="One-shot spectral snapshot")
    snap_p.add_argument("model")
    snap_p.add_argument("--preamble", help="Preamble file")
    snap_p.add_argument("--device", default="cpu")

    health_p = sub.add_parser("health", help="Full health check")
    health_p.add_argument("model")
    health_p.add_argument("--preamble", help="Preamble file")
    health_p.add_argument("--device", default="cpu")

    drift_p = sub.add_parser("drift", help="Track V₂ drift across turns")
    drift_p.add_argument("model")
    drift_p.add_argument("--preamble", help="Preamble file")
    drift_p.add_argument("--turns", type=int, default=20)
    drift_p.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: torch and transformers required")
        sys.exit(1)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    preamble = "You are a helpful assistant."
    if hasattr(args, 'preamble') and args.preamble:
        preamble = Path(args.preamble).read_text().strip()

    model, tokenizer, n_layers = load_model(args.model, args.device)

    if args.command == "snapshot":
        print(f"Running spectral snapshot ({n_layers} layers)...")
        zones = spectral_snapshot(model, tokenizer, preamble, args.device, n_layers)

        print(f"\n{'='*55}")
        print(f"SPECTRAL SNAPSHOT")
        print(f"{'='*55}")
        for zone, metrics in zones.items():
            print(f"\n  {zone.upper()} zone:")
            print(f"    σ₂ enrichment: {metrics['s2_enrichment']:.4f} {'✓' if metrics['s2_enrichment'] > 1.0 else '—'}")
            print(f"    Gap ratio:     {metrics['gap_ratio']:.4f}")
            print(f"    ΔS:            {metrics['delta_entropy']:+.4f}")
            print(f"    Layers:        {metrics['n_layers_measured']}")
        print(f"{'='*55}")

    elif args.command == "health":
        print(f"Running health check ({n_layers} layers)...")
        zones = spectral_snapshot(model, tokenizer, preamble, args.device, n_layers)
        print("Running drift check (10 turns)...")
        drift = drift_series(model, tokenizer, preamble, args.device, n_layers, n_turns=10)
        report = health_check(zones, drift)

        print(f"\n{'='*55}")
        print(f"SPECTRAL HEALTH: {report['status']}")
        print(f"{'='*55}")
        if report['issues']:
            for issue in report['issues']:
                print(f"  ⚠ {issue}")
        else:
            print("  All metrics within expected ranges.")

        for zone, metrics in report['zone_metrics'].items():
            print(f"\n  {zone.upper()}:")
            print(f"    σ₂ enrichment: {metrics['s2_enrichment']:.4f}")
            print(f"    ΔS:            {metrics['delta_entropy']:+.4f}")
        print(f"{'='*55}")

        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"health_{ts}.json"
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved: {out_path}")

    elif args.command == "drift":
        n_turns = args.turns
        print(f"Running drift analysis ({n_turns} turns)...")
        drift = drift_series(model, tokenizer, preamble, args.device, n_layers, n_turns=n_turns)

        print(f"\n{'='*55}")
        print(f"DRIFT ANALYSIS ({n_turns} turns)")
        print(f"{'='*55}")

        if drift:
            keys = [k for k in drift[0].keys() if k != "turn"]
            print(f"{'Turn':>6}", end="")
            for k in keys:
                print(f"  {k:>16}", end="")
            print()

            for entry in drift[::max(1, len(drift)//20)]:
                print(f"{entry['turn']:>6}", end="")
                for k in keys:
                    v = entry.get(k, 0)
                    print(f"  {v:>16.6f}", end="")
                print()

            final = drift[-1]
            print(f"\nFinal drift:")
            for k in keys:
                v = final.get(k, 0)
                label = "STABLE" if abs(1.0 - v) < 0.1 else "DRIFTING" if abs(1.0 - v) < 0.3 else "WANDERING"
                print(f"  {k}: {v:.6f} ({label})")

        print(f"{'='*55}")

        OUTPUT_DIR.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"drift_{ts}.json"
        with open(out_path, 'w') as f:
            json.dump(drift, f, indent=2, default=str)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
