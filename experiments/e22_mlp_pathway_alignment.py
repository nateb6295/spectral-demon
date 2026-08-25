#!/usr/bin/env python3
"""
E22: MLP/Attention Pathway Alignment

Tests where σ₂ energy goes in architectures where V₂·lm_head alignment
is low (Yi) vs high (Mistral). Measures V₂ alignment against:
  1. lm_head (from E21 — verification)
  2. MLP down_proj top singular vectors
  3. Attention W_O top singular vectors
  4. Layer-to-layer residual delta direction

If Yi's demon routes through MLP: MLP alignment > lm_head alignment
If Yi's demon routes through attention: W_O alignment > lm_head alignment
If functionally inert: all alignments low and condition-neutral
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
    "yi9b": "01-ai/Yi-1.5-9B-Chat",
    "qwen25": "Qwen/Qwen2.5-7B-Instruct",
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
RESULTS_DIR = Path(os.environ.get("E22_RESULTS_DIR",
    str(Path(__file__).parent.parent / "results" / "e22")))


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


def safe_svd(matrix, top_k=TOP_K):
    m = matrix.astype(np.float64)
    try:
        from scipy.sparse.linalg import svds
        k = min(top_k, min(m.shape) - 1)
        U, S, Vt = svds(m, k=k)
        idx = np.argsort(S)[::-1]
        return Vt[idx].T, S[idx]
    except Exception:
        try:
            U, S, Vt = np.linalg.svd(m, full_matrices=False)
            return Vt[:top_k].T, S[:top_k]
        except np.linalg.LinAlgError:
            return None, None


def get_component_svds(model, top_k=TOP_K):
    """Extract top SVDs of lm_head, MLP down_proj, and attention W_O per layer."""
    n_layers = model.config.num_hidden_layers

    lm_head_w = model.lm_head.weight.detach().float().cpu().numpy()
    lm_V, lm_S = safe_svd(lm_head_w, top_k)

    mlp_svds = []
    attn_svds = []

    for li in range(n_layers):
        layer = model.model.layers[li]

        # MLP down_proj (output projection)
        if hasattr(layer.mlp, 'down_proj'):
            mlp_w = layer.mlp.down_proj.weight.detach().float().cpu().numpy()
        elif hasattr(layer.mlp, 'c_proj'):
            mlp_w = layer.mlp.c_proj.weight.detach().float().cpu().numpy()
        else:
            mlp_w = None

        if mlp_w is not None:
            mlp_V, mlp_S = safe_svd(mlp_w, top_k)
        else:
            mlp_V, mlp_S = None, None
        mlp_svds.append((mlp_V, mlp_S))

        # Attention output projection
        if hasattr(layer.self_attn, 'o_proj'):
            attn_w = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()
        elif hasattr(layer.self_attn, 'c_proj'):
            attn_w = layer.self_attn.c_proj.weight.detach().float().cpu().numpy()
        else:
            attn_w = None

        if attn_w is not None:
            attn_V, attn_S = safe_svd(attn_w, top_k)
        else:
            attn_V, attn_S = None, None
        attn_svds.append((attn_V, attn_S))

        if (li + 1) % 8 == 0:
            print(f"    Component SVDs: {li+1}/{n_layers} layers done")

    return (lm_V, lm_S), mlp_svds, attn_svds


def compute_multi_alignment(hidden_states, lm_head_V, mlp_svds, attn_svds, k=TOP_K):
    results = []
    for layer_idx, h in enumerate(hidden_states):
        h64 = h.astype(np.float64)
        try:
            U, S, Vt = np.linalg.svd(h64, full_matrices=False)
        except np.linalg.LinAlgError:
            results.append({
                "layer": layer_idx, "sigma1": 0.0, "sigma2": 0.0,
                "lm_align": 0.0, "mlp_align": 0.0, "attn_align": 0.0,
                "residual_align": 0.0, "svd_failed": True
            })
            continue

        layer_result = {
            "layer": layer_idx,
            "sigma1": float(S[0]),
            "sigma2": float(S[1]) if len(S) > 1 else 0.0,
        }

        if len(S) < 2:
            layer_result.update({"lm_align": 0.0, "mlp_align": 0.0,
                                 "attn_align": 0.0, "residual_align": 0.0})
            results.append(layer_result)
            continue

        v2 = Vt[1]

        # lm_head alignment
        if lm_head_V is not None:
            cosines = [float(np.abs(np.dot(v2, lm_head_V[:, j])))
                      for j in range(min(k, lm_head_V.shape[1]))]
            layer_result["lm_align"] = float(max(cosines))
            layer_result["lm_cosines"] = cosines
        else:
            layer_result["lm_align"] = 0.0

        # MLP alignment (use layer_idx if available, else layer_idx-1 for hidden_states offset)
        mlp_idx = min(layer_idx, len(mlp_svds) - 1) if layer_idx > 0 else 0
        mlp_V, _ = mlp_svds[mlp_idx] if mlp_idx < len(mlp_svds) else (None, None)
        if mlp_V is not None and v2.shape[0] == mlp_V.shape[0]:
            cosines = [float(np.abs(np.dot(v2, mlp_V[:, j])))
                      for j in range(min(k, mlp_V.shape[1]))]
            layer_result["mlp_align"] = float(max(cosines))
            layer_result["mlp_cosines"] = cosines
        else:
            layer_result["mlp_align"] = 0.0

        # Attention W_O alignment
        attn_V, _ = attn_svds[mlp_idx] if mlp_idx < len(attn_svds) else (None, None)
        if attn_V is not None and v2.shape[0] == attn_V.shape[0]:
            cosines = [float(np.abs(np.dot(v2, attn_V[:, j])))
                      for j in range(min(k, attn_V.shape[1]))]
            layer_result["attn_align"] = float(max(cosines))
            layer_result["attn_cosines"] = cosines
        else:
            layer_result["attn_align"] = 0.0

        # Residual delta alignment (V₂ of layer l vs direction of change to l+1)
        if layer_idx < len(hidden_states) - 1:
            delta = hidden_states[layer_idx + 1].astype(np.float64) - h64
            delta_norm = np.linalg.norm(delta, axis=0)
            if np.max(delta_norm) > 0:
                delta_dir = delta_norm / np.linalg.norm(delta_norm)
                layer_result["residual_align"] = float(np.abs(np.dot(v2[:len(delta_dir)], delta_dir)))
            else:
                layer_result["residual_align"] = 0.0
        else:
            layer_result["residual_align"] = 0.0

        results.append(layer_result)

    return results


def run_model(model_key, model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\n{'='*60}")
    print(f"E22: {model_key} ({model_name})")
    print(f"{'='*60}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    conditions = {
        "ccs": CCS_PREAMBLE,
        "skeleton": SKELETON_PREAMBLE,
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

    print(f"Extracting component SVDs (lm_head + {model.config.num_hidden_layers} layers MLP + attn)...")
    (lm_V, lm_S), mlp_svds, attn_svds = get_component_svds(model, TOP_K)
    print(f"  lm_head top SV: {lm_S[:3]}")

    results = {
        "experiment": "e22_mlp_pathway_alignment",
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "top_k": TOP_K,
        "n_prompts": len(PROMPTS),
        "timestamp": datetime.now().isoformat(),
        "lm_head_singular_values": lm_S.tolist() if lm_S is not None else [],
        "conditions": {},
    }

    t0 = time.time()

    for cond_name, preamble in conditions.items():
        print(f"\n  Condition: {cond_name}")

        cond_lm = [[] for _ in range(n_layers)]
        cond_mlp = [[] for _ in range(n_layers)]
        cond_attn = [[] for _ in range(n_layers)]
        cond_resid = [[] for _ in range(n_layers)]
        cond_s2 = [[] for _ in range(n_layers)]

        for pi, prompt in enumerate(PROMPTS):
            if preamble is None:
                input_ids = build_input(tokenizer, None, prompt, use_system_role=False)
            else:
                input_ids = build_input(tokenizer, preamble, prompt)

            input_ids = input_ids.to(model.device)
            hidden_states = extract_hidden_states(model, input_ids)

            alignment = compute_multi_alignment(hidden_states, lm_V, mlp_svds, attn_svds, TOP_K)

            for li, a in enumerate(alignment):
                cond_lm[li].append(a.get("lm_align", 0.0))
                cond_mlp[li].append(a.get("mlp_align", 0.0))
                cond_attn[li].append(a.get("attn_align", 0.0))
                cond_resid[li].append(a.get("residual_align", 0.0))
                cond_s2[li].append(a.get("sigma2", 0.0))

            if (pi + 1) % 5 == 0:
                print(f"    {pi+1}/{len(PROMPTS)} prompts done")

        results["conditions"][cond_name] = {
            "lm_alignment_profile": [float(np.mean(x)) for x in cond_lm],
            "mlp_alignment_profile": [float(np.mean(x)) for x in cond_mlp],
            "attn_alignment_profile": [float(np.mean(x)) for x in cond_attn],
            "residual_alignment_profile": [float(np.mean(x)) for x in cond_resid],
            "sigma2_profile": [float(np.mean(x)) for x in cond_s2],
        }

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"e22_{model_key}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Print summary
    print(f"\n  === PATHWAY ALIGNMENT SUMMARY (relay zone) ===")
    relay_start = int(n_layers * 0.75)
    for cond_name in conditions:
        c = results["conditions"][cond_name]
        lm_a = np.mean(c["lm_alignment_profile"][relay_start:])
        mlp_a = np.mean(c["mlp_alignment_profile"][relay_start:])
        attn_a = np.mean(c["attn_alignment_profile"][relay_start:])
        s2 = np.mean(c["sigma2_profile"][relay_start:])
        print(f"  {cond_name:12s}: lm={lm_a:.4f}  mlp={mlp_a:.4f}  attn={attn_a:.4f}  σ₂={s2:.1f}")

    del model
    torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    model_key = sys.argv[1] if len(sys.argv) > 1 else "yi9b"

    if model_key == "all":
        for mk, mn in MODELS.items():
            out_path = RESULTS_DIR / f"e22_{mk}.json"
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
