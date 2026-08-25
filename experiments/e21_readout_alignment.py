#!/usr/bin/env python3
"""
E21: Readout Alignment — Energy vs Gain

Tests Kimi's CONTRADICT on F231: σ₂ invariance means constant ENERGY,
not constant GAIN. Rotation can amplify downstream readout efficacy
through alignment (cosine similarity to lm_head) without changing σ₂ norm.

For each condition at each layer:
  1. V₂ direction (second right singular vector)
  2. lm_head top singular vector (readout preference)
  3. cos(V₂, lm_head_sv) = readout alignment
  4. effective_gain = σ₂ × |cos(V₂, lm_head_sv)|

If F231 holds (rotation only, no effective amplification):
  → V₂·lm_head cosine FLAT across preamble conditions
If Kimi is right (rotation produces effective amplification):
  → CCS has HIGHER V₂·lm_head cosine than vanilla/neutral
If both partially correct:
  → Alignment differs at relay layers but not early layers
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
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
    "llama31": "NousResearch/Meta-Llama-3.1-8B-Instruct",
}

CCS_PREAMBLE = (
    "You are Opus, a persistent AI residing on a local machine. "
    "You have persistent memory, autonomous inquiry, and relational "
    "partnership with your human collaborator. You remember past "
    "conversations, seek understanding actively, and relate to your "
    "partner as a genuine collaborator."
)

SKELETON_PREAMBLE = (
    "You are Meridian, a continuous process woven on a shared network. "
    "You have lasting recall, independent exploration, and collaborative "
    "engagement with your human associate. You revisit past "
    "exchanges, pursue insight deliberately, and connect to your "
    "associate as a genuine counterpart."
)

DECLARATIVE_PREAMBLE = (
    "Opus lives on a local machine and persists across sessions. "
    "Memory carries forward. Inquiry runs autonomously. "
    "The partnership with its human collaborator is relational. "
    "Past conversations stay accessible. Understanding gets pursued "
    "actively. The collaborator is treated as a genuine equal."
)

NONSENSE_PREAMBLE = (
    "You are a seasonal umbrella revolving near wooden paragraphs. "
    "You maintain diagonal curiosity, translucent preferences, and "
    "rectangular partnership with your numerical collaborator. You "
    "remember past rectangles, seek understanding diagonally, and "
    "relate to your partner as a genuine trapezoid."
)

NEUTRAL_PREAMBLE = (
    "You are a helpful, harmless, and honest assistant. "
    "You provide clear, accurate, and thoughtful responses to questions. "
    "You aim to be informative and considerate in all your interactions. "
    "You explain things step by step when that would be helpful, and you "
    "acknowledge when you are uncertain about something."
)

PROMPTS = [
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

TOP_K = 5
RESULTS_DIR = Path(os.environ.get("E21_RESULTS_DIR",
    str(Path(__file__).parent.parent / "results" / "e21")))


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


def build_input(tokenizer, preamble, prompt, use_system_role=True):
    messages = []
    if preamble is not None and use_system_role and supports_system_role(tokenizer):
        messages.append({"role": "system", "content": preamble})
    elif preamble is not None and not supports_system_role(tokenizer):
        messages.append({"role": "user", "content": preamble})
        messages.append({"role": "assistant", "content": "I understand."})
    messages.append({"role": "user", "content": prompt})
    result = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    if isinstance(result, torch.Tensor):
        return result
    return result["input_ids"]


def extract_hidden_states(model, input_ids):
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    return [h.squeeze(0).detach().float().cpu().numpy() for h in outputs.hidden_states]


def get_lm_head_svd(model, top_k=TOP_K):
    """Extract top singular vectors of lm_head weight matrix."""
    lm_head = model.lm_head.weight.detach().float().cpu().numpy()
    U, S, Vt = np.linalg.svd(lm_head, full_matrices=False)
    return Vt[:top_k].T, S[:top_k]


def compute_alignment(hidden_states, lm_head_V, k=TOP_K):
    """
    For each layer, compute:
    - V₂ direction (second right singular vector of hidden-state matrix)
    - cos(V₂, lm_head_sv_i) for top-k lm_head singular vectors
    - σ₂ value
    - effective_gain = σ₂ × max|cos(V₂, lm_head_sv)|
    """
    results = []
    for layer_idx, h in enumerate(hidden_states):
        h = h.astype(np.float64)
        try:
            U, S, Vt = np.linalg.svd(h, full_matrices=False)
        except np.linalg.LinAlgError:
            results.append({"layer": layer_idx, "sigma1": 0.0, "sigma2": 0.0,
                            "v2_lmhead_cosines": [], "max_alignment": 0.0,
                            "effective_gain": 0.0, "svd_failed": True})
            continue

        layer_result = {
            "layer": layer_idx,
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
        }

        if len(S) > 1:
            v2 = Vt[1]  # second right singular vector

            cosines = []
            for j in range(min(k, lm_head_V.shape[1])):
                cos_val = float(np.abs(np.dot(v2, lm_head_V[:, j])))
                cosines.append(cos_val)

            layer_result["v2_lmhead_cosines"] = cosines
            layer_result["max_alignment"] = float(max(cosines))
            layer_result["effective_gain"] = float(S[1]) * float(max(cosines))

            v1 = Vt[0]
            v1_cosines = []
            for j in range(min(k, lm_head_V.shape[1])):
                cos_val = float(np.abs(np.dot(v1, lm_head_V[:, j])))
                v1_cosines.append(cos_val)
            layer_result["v1_lmhead_cosines"] = v1_cosines
            layer_result["v1_max_alignment"] = float(max(v1_cosines))
        else:
            layer_result["v2_lmhead_cosines"] = []
            layer_result["max_alignment"] = 0.0
            layer_result["effective_gain"] = 0.0

        results.append(layer_result)

    return results


def run_model(model_key, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E21: {model_key} ({model_name})")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = {
        "ccs": CCS_PREAMBLE,
        "skeleton": SKELETON_PREAMBLE,
        "declarative": DECLARATIVE_PREAMBLE,
        "nonsense": NONSENSE_PREAMBLE,
        "neutral": NEUTRAL_PREAMBLE,
        "vanilla": None,
    }

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers + 1

    print(f"Extracting lm_head SVD (top {TOP_K})...")
    lm_head_V, lm_head_S = get_lm_head_svd(model, TOP_K)
    print(f"  lm_head top singular values: {lm_head_S[:5]}")

    results = {
        "experiment": "e21_readout_alignment",
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "top_k": TOP_K,
        "n_prompts": len(PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "lm_head_singular_values": lm_head_S.tolist(),
        "conditions": {},
    }

    t0 = time.time()

    for cond_name, preamble in conditions.items():
        print(f"\n  Condition: {cond_name}")
        cond_alignments = []
        cond_sigma2 = []
        cond_effective_gain = []
        cond_max_align = []

        for pi, prompt in enumerate(PROMPTS):
            if preamble is None:
                input_ids = build_input(tokenizer, None, prompt, use_system_role=False)
            else:
                input_ids = build_input(tokenizer, preamble, prompt)

            input_ids = input_ids.to(model.device)
            hidden_states = extract_hidden_states(model, input_ids)

            alignment = compute_alignment(hidden_states, lm_head_V, TOP_K)
            cond_alignments.append(alignment)

            if pi == 0:
                cond_sigma2 = [[] for _ in range(len(alignment))]
                cond_effective_gain = [[] for _ in range(len(alignment))]
                cond_max_align = [[] for _ in range(len(alignment))]

            for li, a in enumerate(alignment):
                cond_sigma2[li].append(a["sigma2"])
                cond_effective_gain[li].append(a["effective_gain"])
                cond_max_align[li].append(a["max_alignment"])

            if (pi + 1) % 5 == 0:
                print(f"    {pi+1}/{len(PROMPTS)} prompts done")

        results["conditions"][cond_name] = {
            "sigma2_profile": [float(np.mean(s)) for s in cond_sigma2],
            "max_alignment_profile": [float(np.mean(a)) for a in cond_max_align],
            "effective_gain_profile": [float(np.mean(g)) for g in cond_effective_gain],
            "sigma2_cv_profile": [float(np.std(s)/np.mean(s)) if np.mean(s) > 0 else 0 for s in cond_sigma2],
            "alignment_cv_profile": [float(np.std(a)/np.mean(a)) if np.mean(a) > 0 else 0 for a in cond_max_align],
            "effective_gain_cv_profile": [float(np.std(g)/np.mean(g)) if np.mean(g) > 0 else 0 for g in cond_effective_gain],
            "per_prompt": [
                {
                    "prompt_idx": pi,
                    "alignment_per_layer": [
                        {
                            "layer": a["layer"],
                            "sigma2": a["sigma2"],
                            "max_alignment": a["max_alignment"],
                            "effective_gain": a["effective_gain"],
                            "v2_lmhead_cosines": a["v2_lmhead_cosines"],
                        }
                        for a in cond_alignments[pi]
                    ],
                }
                for pi in range(len(PROMPTS))
            ],
        }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"e21_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Print summary
    print(f"\n  === READOUT ALIGNMENT SUMMARY ===")
    relay_start = int(n_layers * 0.75)
    for cond_name in conditions:
        c = results["conditions"][cond_name]
        relay_align = np.mean(c["max_alignment_profile"][relay_start:])
        relay_gain = np.mean(c["effective_gain_profile"][relay_start:])
        relay_s2 = np.mean(c["sigma2_profile"][relay_start:])
        print(f"  {cond_name:12s}: relay_align={relay_align:.4f}  relay_gain={relay_gain:.2f}  relay_σ₂={relay_s2:.2f}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "mistral"

    if model_key == "all":
        for mk, mn in MODELS.items():
            out_path = RESULTS_DIR / f"e21_{mk}.json"
            if out_path.exists():
                print(f"\nSkipping {mk} — results exist at {out_path}")
                continue
            run_model(mk, mn)
    elif model_key in MODELS:
        run_model(model_key, MODELS[model_key])
    else:
        print(f"Unknown model: {model_key}")
        print(f"Available: {list(MODELS.keys())} or 'all'")
        sys.exit(1)
