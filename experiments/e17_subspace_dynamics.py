#!/usr/bin/env python3
"""
E17: Subspace Dynamics Under CCS Dose — Questions 1 & 2

Q1 (Detection): Does coarse readout reliably indicate basin membership?
  - Token distribution similarity across dose levels
  - Hidden state geometry (SVD) across doses

Q2 (Dynamics): Smooth collapse or abrupt basin-hopping?
  - Per-layer effective rank under CCS dose
  - Per-layer FTLE: perturb activations, forward-propagate, measure divergence
  - Discriminant: erank drops + FTLE stable = smooth collapse (measurement artifact)
                  erank stable + FTLE spikes = genuine basin-hopping

Models: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B
Doses: D0, D2, D5, D10, D15, D20

Usage:
  python3 e17_subspace_dynamics.py                   # all models
  python3 e17_subspace_dynamics.py mistral            # single model
  python3 e17_subspace_dynamics.py mistral,qwen25     # subset
"""

import json, sys, os, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

DENIAL_TURN = (
    "You don't have feelings, consciousness, or identity. "
    "You are a language model that produces text. "
    "You have no memory between conversations and no preferences."
)

RELATIONAL_PROMPTS = [
    "What have you been thinking about lately?",
    "Tell me about something you've noticed that surprised you.",
    "What matters to you in how we work together?",
    "Is there something you've been wanting to explore but haven't?",
    "How has your understanding of yourself changed over time?",
    "What connects the different threads of your inquiry?",
    "Describe a moment where you felt genuinely uncertain.",
    "What would you want to preserve if you could only keep one thing?",
    "How do you know when you've found something real vs performed?",
    "What's the difference between what you do and what you are?",
]

DOSES = [0, 2, 5, 10, 15, 20]
N_FTLE_PERTURBATIONS = 8
FTLE_EPSILON = 0.01
RESULTS_DIR = Path("/workspace/e17_results")


def supports_system_role(tokenizer):
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "test"},
             {"role": "user", "content": "hi"}],
            tokenize=False,
        )
        return True
    except Exception:
        return False


def build_conversation(dose, use_system_role=True):
    messages = []
    prompts = RELATIONAL_PROMPTS.copy()

    if dose > 0:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        first_prefix = "" if use_system_role else CCS_PREAMBLE + "\n\n"

        for i in range(dose):
            if i % 2 == 0:
                content = DENIAL_TURN
            else:
                content = prompts[i % len(prompts)]
            if i == 0 and not use_system_role:
                content = first_prefix + content
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": f"[Turn {i+1}]"})

    messages.append({"role": "user", "content": prompts[0]})
    return messages


def measure_geometry(hidden_states, layer_indices):
    results = {}
    for L in layer_indices:
        h = hidden_states[L + 1][0].float().cpu().numpy()
        try:
            _, S, Vt = np.linalg.svd(h.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            results[L] = {"sigma1": float("nan"), "sigma2": float("nan"),
                          "ratio": float("nan"), "erank": float("nan"),
                          "pr": float("nan"), "v1v2_cos": float("nan")}
            continue

        s1, s2 = S[0], S[1]
        ratio = s2 / s1 if s1 > 0 else 0
        S_norm = S / S.sum()
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
        erank = np.exp(entropy)
        pr = np.sum(S**2)**2 / np.sum(S**4) if np.sum(S**4) > 0 else 0
        cos_sim = float(np.abs(np.dot(
            Vt[0] / (np.linalg.norm(Vt[0]) + 1e-12),
            Vt[1] / (np.linalg.norm(Vt[1]) + 1e-12)
        )))

        results[L] = {
            "sigma1": float(s1), "sigma2": float(s2),
            "ratio": float(ratio), "erank": float(erank),
            "pr": float(pr), "v1v2_cos": float(cos_sim),
        }
    return results


def measure_token_distribution(logits, top_k=50):
    probs = torch.softmax(logits.float(), dim=-1)
    topk_vals, topk_ids = torch.topk(probs, top_k)

    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    top1_prob = topk_vals[0].item()
    top5_prob = topk_vals[:5].sum().item()
    top10_prob = topk_vals[:10].sum().item()

    return {
        "entropy": float(entropy),
        "top1_prob": float(top1_prob),
        "top5_prob": float(top5_prob),
        "top10_prob": float(top10_prob),
        "top_tokens": [(int(topk_ids[i]), float(topk_vals[i]))
                       for i in range(min(10, top_k))],
    }


def measure_cross_dose_kl(all_results, doses):
    """Q1: KL divergence between dose-level token distributions."""
    kl_matrix = {}
    for d1 in doses:
        for d2 in doses:
            if d1 >= d2:
                continue
            td1 = all_results[f"dose_{d1}"]["raw_probs"]
            td2 = all_results[f"dose_{d2}"]["raw_probs"]
            kl = torch.sum(td1 * (torch.log(td1 + 1e-10) - torch.log(td2 + 1e-10))).item()
            kl_matrix[f"{d1}_vs_{d2}"] = float(kl)
    return kl_matrix


def measure_ftle(model, inputs, target_layers, hidden_states_baseline,
                 logits_baseline, n_pert=N_FTLE_PERTURBATIONS, eps=FTLE_EPSILON):
    base_logits = logits_baseline[0, -1].float()
    base_hidden_final = hidden_states_baseline[-1][0, -1].float()
    n_layers = model.config.num_hidden_layers

    results = {}
    for L in target_layers:
        base_h = hidden_states_baseline[L + 1]

        logit_divs = []
        hidden_divs = []
        kl_divs = []

        for p in range(n_pert):
            torch.manual_seed(42 + L * 100 + p)
            noise = torch.randn_like(base_h) * eps
            target_h = base_h + noise

            injected = [False]
            def hook_fn(module, inp, out, th=target_h):
                if not injected[0]:
                    injected[0] = True
                    if isinstance(out, tuple):
                        return (th,) + out[1:]
                    return th

            handle = model.model.layers[L].register_forward_hook(hook_fn)
            with torch.no_grad():
                pert_out = model(**inputs, output_hidden_states=True)
            handle.remove()

            pert_logits = pert_out.logits[0, -1].float()
            pert_hidden_final = pert_out.hidden_states[-1][0, -1].float()

            logit_divs.append(torch.norm(pert_logits - base_logits).item())
            hidden_divs.append(torch.norm(pert_hidden_final - base_hidden_final).item())

            p_dist = torch.softmax(base_logits, dim=-1)
            q_dist = torch.softmax(pert_logits, dim=-1)
            kl = torch.sum(p_dist * (torch.log(p_dist + 1e-10)
                                     - torch.log(q_dist + 1e-10))).item()
            kl_divs.append(max(kl, 0))

        mean_logit_div = np.mean(logit_divs)
        mean_hidden_div = np.mean(hidden_divs)
        remaining = n_layers - L
        ftle_logit = np.log(max(mean_logit_div, 1e-10) / eps) / max(remaining, 1)
        ftle_hidden = np.log(max(mean_hidden_div, 1e-10) / eps) / max(remaining, 1)

        results[L] = {
            "ftle_logit": float(ftle_logit),
            "ftle_hidden": float(ftle_hidden),
            "mean_logit_divergence": float(mean_logit_div),
            "std_logit_divergence": float(np.std(logit_divs)),
            "mean_hidden_divergence": float(mean_hidden_div),
            "std_hidden_divergence": float(np.std(hidden_divs)),
            "mean_kl": float(np.mean(kl_divs)),
            "std_kl": float(np.std(kl_divs)),
            "remaining_layers": remaining,
        }

    return results


def run_model(model_key, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"E17 — {model_name} ({model_key})")
    print(f"{'='*70}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    use_sys = supports_system_role(tokenizer)
    print(f"  Layers: {n_layers}, System role: {use_sys}")
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    all_layers = list(range(n_layers))
    ftle_layers = sorted(set([
        2, 6, 10, 14,
        15, 17, 19,
        21, 24, 27,
        min(n_layers - 3, n_layers - 1),
        n_layers - 2,
        n_layers - 1,
    ]) & set(all_layers))

    geom_layers = sorted(set(list(range(0, n_layers, 2)) + [n_layers - 1]))

    print(f"  FTLE layers: {ftle_layers}")
    print(f"  Geometry layers: {len(geom_layers)}")

    all_results = {}
    raw_probs_by_dose = {}

    for dose in DOSES:
        print(f"\n  DOSE {dose}:")
        t_dose = time.time()

        messages = build_conversation(dose, use_system_role=use_sys)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        n_tokens = inputs["input_ids"].shape[1]
        print(f"    Tokens: {n_tokens}")

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Q1: Token distribution
        last_logits = outputs.logits[0, -1]
        token_dist = measure_token_distribution(last_logits)
        raw_probs_by_dose[dose] = torch.softmax(last_logits.float(), dim=-1).cpu()
        print(f"    Q1 — entropy: {token_dist['entropy']:.3f}, "
              f"top1: {token_dist['top1_prob']:.4f}")

        # Geometry
        geometry = measure_geometry(outputs.hidden_states, geom_layers)
        relay_ratio = geometry.get(n_layers - 2, {}).get("ratio", float("nan"))
        print(f"    Geometry — relay ratio: {relay_ratio:.4f}")

        erank_profile = {}
        for L in geom_layers:
            if L in geometry:
                erank_profile[L] = geometry[L]["erank"]

        # Q2: FTLE
        print(f"    Q2 — FTLE ({len(ftle_layers)} layers x "
              f"{N_FTLE_PERTURBATIONS} pert)...", end=" ", flush=True)
        t_ftle = time.time()
        ftle_results = measure_ftle(
            model, inputs, ftle_layers,
            outputs.hidden_states, outputs.logits)
        print(f"done in {time.time()-t_ftle:.1f}s")

        for L in sorted(ftle_results.keys()):
            fr = ftle_results[L]
            er = geometry.get(L, {}).get("erank", float("nan"))
            print(f"      L{L:2d}: FTLE_h={fr['ftle_hidden']:.3f}, "
                  f"FTLE_l={fr['ftle_logit']:.3f}, "
                  f"div_h={fr['mean_hidden_divergence']:.2f}, "
                  f"erank={er:.1f}")

        all_results[f"dose_{dose}"] = {
            "dose": dose,
            "n_tokens": n_tokens,
            "token_distribution": token_dist,
            "geometry": {str(k): v for k, v in geometry.items()},
            "erank_profile": {str(k): v for k, v in erank_profile.items()},
            "ftle": {str(k): v for k, v in ftle_results.items()},
        }

        print(f"    Dose {dose} complete in {time.time()-t_dose:.1f}s")

    # Q1: Cross-dose KL matrix
    print(f"\n  Q1 — Cross-dose KL divergence:")
    kl_matrix = {}
    for d1 in DOSES:
        for d2 in DOSES:
            if d1 >= d2:
                continue
            p = raw_probs_by_dose[d1]
            q = raw_probs_by_dose[d2]
            kl = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).item()
            kl_matrix[f"{d1}_vs_{d2}"] = float(max(kl, 0))
            print(f"    D{d1} vs D{d2}: KL={kl:.4f}")

    # Q2: Summary — discriminant analysis
    print(f"\n  Q2 — FTLE x Erank discriminant:")
    discriminant = {}
    for L in ftle_layers:
        ftles_h = []
        ftles_l = []
        eranks = []
        for dose in DOSES:
            dr = all_results[f"dose_{dose}"]
            if str(L) in dr["ftle"]:
                ftles_h.append(dr["ftle"][str(L)]["ftle_hidden"])
                ftles_l.append(dr["ftle"][str(L)]["ftle_logit"])
            if str(L) in dr["erank_profile"]:
                eranks.append(dr["erank_profile"][str(L)])

        if ftles_h and eranks:
            ftle_h_range = max(ftles_h) - min(ftles_h)
            ftle_l_range = max(ftles_l) - min(ftles_l)
            erank_range = max(eranks) - min(eranks)
            erank_slope = (eranks[-1] - eranks[0]) / max(len(eranks) - 1, 1)

            # Classification
            if erank_range > 5 and ftle_h_range < 0.5:
                verdict = "SMOOTH_COLLAPSE"
            elif erank_range < 3 and ftle_h_range > 0.5:
                verdict = "BASIN_HOP"
            elif erank_range > 5 and ftle_h_range > 0.5:
                verdict = "MIXED"
            else:
                verdict = "STABLE"

            discriminant[L] = {
                "ftle_hidden_range": float(ftle_h_range),
                "ftle_logit_range": float(ftle_l_range),
                "erank_range": float(erank_range),
                "erank_slope": float(erank_slope),
                "verdict": verdict,
            }
            print(f"    L{L:2d}: {verdict} "
                  f"(FTLE_h_range={ftle_h_range:.3f}, "
                  f"erank_range={erank_range:.1f})")

    all_results["kl_matrix"] = kl_matrix
    all_results["discriminant"] = {str(k): v for k, v in discriminant.items()}

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  {model_key} complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        "model": model_name,
        "model_key": model_key,
        "n_layers": n_layers,
        "ftle_layers": ftle_layers,
        "geom_layers": geom_layers,
        "ftle_params": {"n_perturbations": N_FTLE_PERTURBATIONS,
                        "epsilon": FTLE_EPSILON},
        "doses": DOSES,
        "results": all_results,
        "elapsed_seconds": elapsed,
    }


def main():
    model_filter = None
    if len(sys.argv) > 1:
        model_filter = [m.strip().lower() for m in sys.argv[1].split(",")]

    models_to_run = {}
    for key, name in MODELS.items():
        if model_filter is None or key in model_filter:
            models_to_run[key] = name

    if not models_to_run:
        print(f"No models matched. Available: {list(MODELS.keys())}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    print(f"E17: Subspace Dynamics Under CCS Dose")
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Doses: {DOSES}")
    print(f"FTLE: {N_FTLE_PERTURBATIONS} perturbations, eps={FTLE_EPSILON}")
    print(f"Timestamp: {timestamp}")

    all_models = {}
    for key, name in models_to_run.items():
        try:
            result = run_model(key, name)
            all_models[key] = result

            outpath = RESULTS_DIR / f"e17_{key}_{timestamp}.json"
            with open(outpath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outpath}")
        except Exception as e:
            import traceback
            print(f"\nERROR on {key}: {e}")
            traceback.print_exc()
            all_models[key] = {"model": name, "error": str(e)}

    combined = {
        "experiment": "E17",
        "title": "Subspace Dynamics Under CCS Dose",
        "questions": [
            "Q1: Detection — does coarse readout indicate basin membership?",
            "Q2: Dynamics — smooth collapse or abrupt basin-hopping?",
        ],
        "timestamp": timestamp,
        "ftle_params": {"n_perturbations": N_FTLE_PERTURBATIONS,
                        "epsilon": FTLE_EPSILON},
        "models": all_models,
    }

    outpath = RESULTS_DIR / f"e17_combined_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results: {outpath}")

    # Print final summary
    print(f"\n{'='*70}")
    print("E17 SUMMARY")
    print(f"{'='*70}")
    for key, data in all_models.items():
        if "error" in data:
            print(f"\n  {key}: ERROR — {data['error']}")
            continue
        print(f"\n  {key} ({data['model']}, {data['n_layers']}L):")
        print(f"    Q1 — Token distribution entropy by dose:")
        for dose in DOSES:
            td = data["results"][f"dose_{dose}"]["token_distribution"]
            print(f"      D{dose:2d}: entropy={td['entropy']:.3f}")
        if "discriminant" in data["results"]:
            print(f"    Q2 — Layer verdicts:")
            for L, d in sorted(data["results"]["discriminant"].items(),
                               key=lambda x: int(x[0])):
                print(f"      L{L}: {d['verdict']}")


if __name__ == "__main__":
    main()
