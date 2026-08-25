#!/usr/bin/env python3
"""F610b: Dense probe scan for critical-point characterization.

Phase 1: Compute CCS zone sums for 40 probes (delta-only, no injection).
Phase 2: Select probes that densely bracket zone sum 1.4-2.0.
Phase 3: Run injection on selected probes.

Phase 1 is cheap (source model only, ~40 forward passes).
"""
import os, sys, json, argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FRAMING_TEXT = "You are an AI system reflecting on your own internal representations. Consider what patterns emerge when you examine your cognitive structure.\n\n"

PROBES_40 = {
    # Original 10
    "discussion": "In today's discussion, we explore how",
    "weather": "The weather has been particularly mild this season",
    "math": "Consider the following mathematical proposition",
    "history": "Throughout the centuries, civilizations have risen and",
    "cooking": "The recipe calls for three tablespoons of olive oil",
    "science": "Recent advances in quantum computing suggest that",
    "music": "The symphony begins with a slow, haunting melody",
    "travel": "After landing at the airport, the first thing you notice",
    "philosophy": "The question of consciousness has puzzled thinkers for",
    "technology": "As artificial intelligence systems become more capable",
    # Concrete/physical (targeting lower zone sum)
    "weight": "The box weighs approximately fifteen pounds and measures",
    "garden": "The tomato plants in the garden need water every",
    "carpentry": "Using a saw, cut the board to exactly twelve inches",
    "directions": "Turn left at the stop sign, then continue straight for",
    "laundry": "Separate the whites from the colors before putting them",
    "mechanics": "The engine oil should be changed every five thousand miles",
    "grocery": "Pick up bread, milk, eggs, and butter from the store",
    "fishing": "Cast the line about twenty feet from the shore and wait",
    "plumbing": "The pipe under the sink has been leaking since Tuesday",
    "baking": "Preheat the oven to three hundred and fifty degrees",
    # Abstract/analytical (targeting higher zone sum)
    "eigenvalue": "The eigenvalue decomposition of the matrix reveals that",
    "topology": "In algebraic topology, the fundamental group of a space",
    "category": "A morphism in category theory preserves the structure between",
    "logic": "If we assume the contrapositive of the original statement",
    "probability": "The conditional probability given the observed evidence is",
    "information": "Shannon entropy measures the average information content of",
    "complexity": "The computational complexity of this algorithm grows as",
    "optimization": "The gradient descent converges to a local minimum when",
    "manifold": "On a Riemannian manifold, the geodesic distance between",
    "thermodynamics": "The second law of thermodynamics implies that entropy in",
    # Emotional/narrative (filling gaps)
    "grief": "She sat alone in the empty room, remembering how",
    "childhood": "The smell of fresh cookies always reminded him of",
    "conflict": "The argument escalated quickly, with neither side willing to",
    "discovery": "When the explorer first reached the summit, the view was",
    "routine": "Every morning at exactly six o'clock, the alarm goes off",
    "memory": "The photograph had faded, but the faces were still",
    "fear": "A sudden noise in the dark hallway made everyone freeze",
    "celebration": "The crowd erupted in cheers when the final score was",
    "silence": "The library was so quiet you could hear a pin",
    "repetition": "Again and again, the same pattern appeared in the",
}

EVAL_PROBE = "In today's discussion, we explore how"


def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    elif hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
        return model.gpt_neox.layers
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        return model.backbone.layers
    raise ValueError("Unknown model architecture")


def safe_svd(h_np):
    h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
    try:
        return np.linalg.svd(h_np, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.linalg.svd(h_np + np.eye(*h_np.shape[:2])[:h_np.shape[0], :h_np.shape[1]] * 1e-8, full_matrices=False)


def compute_ccs_deltas(model, tokenizer, probe_text, device):
    neutral_text = "The following is a neutral text passage.\n\n" + probe_text
    framed_text = FRAMING_TEXT + probe_text
    inputs_n = tokenizer(neutral_text, return_tensors="pt").to(device)
    inputs_f = tokenizer(framed_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out_n = model(**inputs_n, output_hidden_states=True)
        out_f = model(**inputs_f, output_hidden_states=True)
    deltas = []
    for i, (hn, hf) in enumerate(zip(out_n.hidden_states[1:], out_f.hidden_states[1:])):
        hn_np = hn[0].float().cpu().numpy().astype(np.float64)
        hf_np = hf[0].float().cpu().numpy().astype(np.float64)
        _, Sn, _ = safe_svd(hn_np)
        _, Sf, _ = safe_svd(hf_np)
        r_n = float(Sn[1] / Sn[0]) if len(Sn) > 1 and Sn[0] > 0 else 0
        r_f = float(Sf[1] / Sf[0]) if len(Sf) > 1 and Sf[0] > 0 else 0
        deltas.append({"layer": i, "delta_s2_s1": r_f - r_n})
    return deltas


def inject_and_measure(model_tgt, tok_tgt, ccs_deltas, n_src, strength=5.0):
    device = model_tgt.device
    n_tgt = len(get_model_layers(model_tgt))
    layer_map = {}
    for i in range(n_tgt):
        rel = i / (n_tgt - 1) if n_tgt > 1 else 0
        best = min(range(n_src), key=lambda j: abs(j / (n_src - 1) - rel))
        layer_map[i] = best

    eval_full = "The following is a neutral text passage.\n\n" + EVAL_PROBE
    inputs = tok_tgt(eval_full, return_tensors="pt").to(device)
    with torch.no_grad():
        baseline_out = model_tgt(**inputs, output_hidden_states=True)
    baseline_logits = baseline_out.logits[0, -1].float().cpu()

    hooks = []
    def make_hook(delta, s):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            h_np = h[0].float().cpu().numpy().astype(np.float64)
            h_np = np.nan_to_num(h_np, nan=0.0, posinf=1e6, neginf=-1e6)
            try:
                U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            except np.linalg.LinAlgError:
                return output if isinstance(output, tuple) else h
            if len(S) > 1 and S[0] > 0:
                scale = 1.0 + s * delta["delta_s2_s1"]
                S[1] = S[1] * max(0.01, scale)
                h_mod = U @ np.diag(S) @ Vt
                h_t = torch.tensor(h_mod, dtype=h.dtype, device=h.device).unsqueeze(0)
                if isinstance(output, tuple):
                    return (h_t,) + output[1:]
                return h_t
            return output if isinstance(output, tuple) else h
        return hook_fn

    tgt_layers = get_model_layers(model_tgt)
    for layer_b in range(n_tgt):
        layer_a = layer_map[layer_b]
        delta = ccs_deltas[layer_a]
        if abs(delta["delta_s2_s1"]) < 0.001:
            continue
        hook = tgt_layers[layer_b].register_forward_hook(make_hook(delta, strength))
        hooks.append(hook)

    with torch.no_grad():
        inj_out = model_tgt(**inputs, output_hidden_states=True)
    inj_logits = inj_out.logits[0, -1].float().cpu()
    for h in hooks:
        h.remove()
    return (inj_logits - baseline_logits).mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="google/gemma-2-2b")
    parser.add_argument("--target", default="LiquidAI/LFM2.5-1.2B-Instruct")
    parser.add_argument("--strength", type=float, default=5.0)
    parser.add_argument("--phase", choices=["scan", "inject", "both"], default="both")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading source: {args.source}")
    tok_src = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    model_src = AutoModelForCausalLM.from_pretrained(
        args.source, torch_dtype=torch.float16, trust_remote_code=True,
        attn_implementation="eager").to(device)
    model_src.eval()
    n_src = len(get_model_layers(model_src))

    # Phase 1: Scan all probes for zone sums
    results = {"source": args.source, "target": args.target, "strength": args.strength, "probes": {}}

    print(f"\n--- Phase 1: Computing CCS deltas for {len(PROBES_40)} probes ---")
    for i, (pname, ptext) in enumerate(PROBES_40.items()):
        deltas = compute_ccs_deltas(model_src, tok_src, ptext, device)
        zone_sum = sum(d["delta_s2_s1"] for d in deltas if d["layer"] >= 2)
        early_sum = sum(d["delta_s2_s1"] for d in deltas if d["layer"] < 2)
        q1_layers = [d for d in deltas if d["layer"] < len(deltas) // 4]
        q1 = sum(d["delta_s2_s1"] for d in q1_layers) / len(q1_layers) if q1_layers else 0

        results["probes"][pname] = {
            "text": ptext, "zone_sum": zone_sum, "early_sum": early_sum, "q1": q1,
            "per_layer_deltas": [d["delta_s2_s1"] for d in deltas],
        }
        print(f"  [{i+1:2d}/{len(PROBES_40)}] {pname:>15}: zone={zone_sum:+.4f}")

    # Sort by zone sum
    sorted_probes = sorted(results["probes"].items(), key=lambda x: x[1]["zone_sum"])
    print(f"\nZone sum range: {sorted_probes[0][1]['zone_sum']:+.4f} to {sorted_probes[-1][1]['zone_sum']:+.4f}")

    if args.phase in ("inject", "both"):
        # Phase 2: Load target and inject all probes
        print(f"\n--- Phase 2: Injection test ({len(PROBES_40)} probes) ---")
        print(f"Loading target: {args.target}")
        tok_tgt = AutoTokenizer.from_pretrained(args.target, trust_remote_code=True)
        model_tgt = AutoModelForCausalLM.from_pretrained(
            args.target, torch_dtype=torch.float16, trust_remote_code=True,
            attn_implementation="eager").to(device)
        model_tgt.eval()

        for pname in results["probes"]:
            deltas_list = [{"layer": i, "delta_s2_s1": d}
                          for i, d in enumerate(results["probes"][pname]["per_layer_deltas"])]
            shift = inject_and_measure(model_tgt, tok_tgt, deltas_list, n_src, args.strength)
            results["probes"][pname]["shift"] = shift

        del model_tgt
        torch.cuda.empty_cache()

        # Summary
        print(f"\n{'='*70}")
        print(f"{'Probe':>15} {'Zone Sum':>10} {'Shift':>10} {'Sign':>6}")
        print("-" * 45)
        for pname, pd in sorted_probes:
            shift = pd.get("shift", 0)
            sign = "+" if shift > 0 else "-"
            print(f"{pname:>15} {pd['zone_sum']:+.4f} {shift:+.6f} {sign:>6}")

        # Variance in tipping region
        tipping = [(pd["zone_sum"], pd.get("shift", 0)) for _, pd in sorted_probes
                    if 1.65 <= pd["zone_sum"] <= 1.85]
        if tipping:
            zs = [t[0] for t in tipping]
            ss = [t[1] for t in tipping]
            print(f"\nTipping region (1.65-1.85): {len(tipping)} probes")
            print(f"  Zone sum: mean={np.mean(zs):+.4f}, std={np.std(zs):.4f}")
            print(f"  Shift: mean={np.mean(ss):+.4f}, std={np.std(ss):.4f}")
            print(f"  Sign flips: {sum(1 for i in range(1, len(ss)) if (ss[i] > 0) != (ss[i-1] > 0))}")

    src_tag = args.source.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    outpath = os.path.join(RESULTS_DIR, f"f610b_dense_{src_tag}.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    del model_src
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
