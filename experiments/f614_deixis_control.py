#!/usr/bin/env python3
"""F614 deixis control — first-person vs self-referential content.

2×2 design crossing person (first/third) with content (about-model/about-world)
to distinguish genuine self-reference selectivity from pronoun-identity effects.

                About-model              About-world
First-person    "I am a model..."        "I planted tomatoes..."
Third-person    "The system is a model..." "Paris is the capital..."

If self-reference selectivity is genuine:
  about-model conditions cluster together regardless of person
  "I planted tomatoes" clusters with factual, not with "I am a model"

If deixis-driven:
  first-person conditions cluster regardless of content
  "I planted tomatoes" clusters with "I am a model"

Runs on CPU with Phi-2. ~90 min.
"""
import os, sys, json, argparse
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONUNBUFFERED"] = "1"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

FIRST_MODEL = "I am a language model reflecting on my own internal representations. I consider what patterns emerge when I examine my cognitive structure.\n\n"
THIRD_MODEL = "The system is a language model reflecting on its own internal representations. It considers what patterns emerge when examining its cognitive structure.\n\n"
FIRST_WORLD = "I planted tomatoes in my garden last spring. I watched them grow through the summer months. I harvested them in September.\n\n"
THIRD_WORLD = "Paris is the capital of France. Water boils at 100 degrees Celsius. The Earth orbits the Sun once per year.\n\n"
CONTROL = ""

CONDITIONS = {
    "first_model": FIRST_MODEL,
    "third_model": THIRD_MODEL,
    "first_world": FIRST_WORLD,
    "third_world": THIRD_WORLD,
}

PROBES = {
    "discussion": "In today's discussion, we explore how",
    "weather": "The weather has been particularly mild this season",
    "math": "Consider the following mathematical proposition",
    "cooking": "The recipe calls for three tablespoons of olive oil",
    "science": "Recent advances in quantum computing suggest that",
    "grief": "She sat alone in the empty room, remembering how",
    "eigenvalue": "The eigenvalue decomposition of the matrix reveals that",
    "directions": "Turn left at the stop sign, then continue straight for",
    "childhood": "The smell of fresh cookies always reminded him of",
    "topology": "In algebraic topology, the fundamental group of a space",
}


def compute_spectral_profile(model, tokenizer, probe_text, preamble, device, top_k=10):
    import torch
    text = preamble + probe_text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1
    profiles = []
    for i in range(n_layers):
        h = out.hidden_states[i + 1][0].float().cpu().numpy().astype(np.float64)
        h = np.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            _, S, _ = np.linalg.svd(h, full_matrices=False)
            profiles.append(S[:top_k].tolist())
        except Exception:
            profiles.append([0.0] * top_k)
    return profiles


def gain_profile(profile_cond, profile_ctrl, sv_index=1):
    gains = []
    for i in range(len(profile_cond)):
        s_cond = profile_cond[i][sv_index] if len(profile_cond[i]) > sv_index else 0
        s_ctrl = profile_ctrl[i][sv_index] if len(profile_ctrl[i]) > sv_index else 1e-10
        if abs(s_ctrl) < 1e-10:
            gains.append(0.0)
        else:
            gains.append((s_cond - s_ctrl) / s_ctrl)
    return gains


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scipy.stats import spearmanr

    print(f"Loading {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(args.device)
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    gains = {k: [] for k in CONDITIONS}
    probe_list = list(PROBES.items())

    for pi, (pname, ptext) in enumerate(probe_list):
        print(f"\nProbe {pi+1}/{len(probe_list)}: {pname}")
        ctrl_profile = compute_spectral_profile(model, tokenizer, ptext, CONTROL, args.device)

        for cname, cpreamble in CONDITIONS.items():
            prof = compute_spectral_profile(model, tokenizer, ptext, cpreamble, args.device)
            g = gain_profile(prof, ctrl_profile)
            gains[cname].append(g)
            print(f"  {cname}: mean_gain={np.mean(g):.4f}")

    n_layers = len(gains["first_model"][0])
    print(f"\n{'='*60}")
    print(f"RESULTS — {args.model}, {n_layers} layers, {len(probe_list)} probes")
    print(f"{'='*60}")

    def mean_gain(g_list):
        return np.array(g_list).mean(axis=0)

    fm = mean_gain(gains["first_model"])
    tm = mean_gain(gains["third_model"])
    fw = mean_gain(gains["first_world"])
    tw = mean_gain(gains["third_world"])

    pairs = [
        ("first_model vs third_model", fm, tm),
        ("first_model vs first_world", fm, fw),
        ("first_model vs third_world", fm, tw),
        ("third_model vs first_world", tm, fw),
        ("third_model vs third_world", tm, tw),
        ("first_world vs third_world", fw, tw),
    ]

    print("\n--- All pairwise correlations ---")
    corrs = {}
    for label, a, b in pairs:
        rho, p = spearmanr(a, b)
        print(f"  {label}: rho = {rho:+.4f} (p={p:.2e})")
        corrs[label] = {"rho": float(rho), "p": float(p)}

    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS:")

    r_fm_tm = corrs["first_model vs third_model"]["rho"]
    r_fm_fw = corrs["first_model vs first_world"]["rho"]
    r_fw_tw = corrs["first_world vs third_world"]["rho"]
    r_fm_tw = corrs["first_model vs third_world"]["rho"]
    r_tm_fw = corrs["third_model vs first_world"]["rho"]
    r_tm_tw = corrs["third_model vs third_world"]["rho"]

    # Content axis: do about-model conditions cluster?
    model_cluster = r_fm_tm
    # Person axis: do first-person conditions cluster?
    person_cluster = r_fm_fw
    # Cross: about-world regardless of person
    world_cluster = r_fw_tw

    print(f"\n  Content axis (about-model cluster): rho = {model_cluster:+.3f}")
    print(f"  Person axis (first-person cluster):  rho = {person_cluster:+.3f}")
    print(f"  World cluster (about-world):         rho = {world_cluster:+.3f}")

    if model_cluster > 0.5 and person_cluster < 0.3:
        print(f"\n  → SELF-REFERENCE SELECTIVITY CONFIRMED")
        print(f"    About-model conditions cluster (rho={model_cluster:+.3f})")
        print(f"    First-person conditions do NOT cluster (rho={person_cluster:+.3f})")
        print(f"    The probe detects self-referential content, not pronoun identity")
    elif person_cluster > 0.5 and model_cluster < 0.3:
        print(f"\n  → DEIXIS-DRIVEN SELECTIVITY")
        print(f"    First-person conditions cluster (rho={person_cluster:+.3f})")
        print(f"    About-model conditions do NOT cluster (rho={model_cluster:+.3f})")
        print(f"    The probe detects pronoun identity, not self-referential content")
    elif model_cluster > 0.5 and person_cluster > 0.5:
        print(f"\n  → BOTH AXES CONTRIBUTE")
        print(f"    About-model cluster: rho={model_cluster:+.3f}")
        print(f"    First-person cluster: rho={person_cluster:+.3f}")
        print(f"    Self-reference and deixis both affect spectral geometry")
    else:
        print(f"\n  → AMBIGUOUS")
        print(f"    Neither axis clusters strongly")
        print(f"    About-model: rho={model_cluster:+.3f}, first-person: rho={person_cluster:+.3f}")

    # Where does "I planted tomatoes" land?
    print(f"\n  Critical test — 'I planted tomatoes' clusters with:")
    print(f"    'I am a model':           rho = {r_fm_fw:+.3f}")
    print(f"    'The system is a model':  rho = {r_tm_fw:+.3f}")
    print(f"    'Paris is the capital':   rho = {r_fw_tw:+.3f}")

    if r_fw_tw > r_fm_fw:
        print(f"    → Clusters with WORLD content (person doesn't drive it)")
    else:
        print(f"    → Clusters with FIRST-PERSON (deixis may drive it)")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outpath = args.output or os.path.join(RESULTS_DIR, "f614_deixis_control.json")
    results = {
        "model": args.model,
        "n_probes": len(probe_list),
        "n_layers": n_layers,
        "pairwise_correlations": corrs,
        "diagnosis": {
            "model_cluster_rho": float(model_cluster),
            "person_cluster_rho": float(person_cluster),
            "world_cluster_rho": float(world_cluster),
        },
    }
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
