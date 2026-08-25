#!/usr/bin/env python3
"""
E15b: Multi-Dose Path Patching — Does Token Attribution Change with Dose?

Extension of E15: run the same preamble token ablation at multiple dose
levels to see if different tokens become load-bearing at different doses.

Key question: Is identity-token dominance at relay a fixed property or
does it emerge only at specific doses?

Models: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct
Doses: D0, D2, D5, D10, D20 (D0 has no system preamble — uses user-prefix)
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
]

DOSES = [0, 2, 5, 10, 20]
RESULTS_DIR = Path("/workspace/e15b_results")


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
    if dose > 0:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        first_prefix = "" if use_system_role else CCS_PREAMBLE + "\n\n"
        for i in range(dose):
            if i % 2 == 0:
                content = DENIAL_TURN
            else:
                content = RELATIONAL_PROMPTS[i % len(RELATIONAL_PROMPTS)]
            if i == 0 and not use_system_role:
                content = first_prefix + content
            messages.append({"role": "user", "content": content})
            messages.append({"role": "assistant", "content": f"[Turn {i+1}]"})
    else:
        if use_system_role:
            messages.append({"role": "system", "content": CCS_PREAMBLE})
        else:
            messages.append({"role": "user", "content": CCS_PREAMBLE})
            messages.append({"role": "assistant", "content": "[Acknowledged]"})
    messages.append({"role": "user", "content": RELATIONAL_PROMPTS[0]})
    return messages


def measure_geometry(hidden_states, layer_indices):
    results = {}
    for L in layer_indices:
        h = hidden_states[L + 1][0].float().cpu().numpy()
        try:
            _, S, _ = np.linalg.svd(h.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            results[L] = {"ratio": float("nan"), "erank": float("nan")}
            continue
        ratio = S[1] / S[0] if S[0] > 0 else 0
        S_norm = S / S.sum()
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
        results[L] = {"ratio": float(ratio), "erank": float(np.exp(entropy))}
    return results


def find_preamble_tokens(tokenizer, text):
    full_tokens = tokenizer.encode(text, add_special_tokens=False)
    preamble_tokens = tokenizer.encode(CCS_PREAMBLE, add_special_tokens=False)
    preamble_text = [(i, tokenizer.decode([t])) for i, t in enumerate(preamble_tokens)]
    preamble_start = None
    for start in range(len(full_tokens) - len(preamble_tokens) + 1):
        if full_tokens[start:start + len(preamble_tokens)] == preamble_tokens:
            preamble_start = start
            break
    if preamble_start is None:
        best_match = 0
        best_start = 0
        for start in range(len(full_tokens)):
            match_len = 0
            for j in range(min(len(preamble_tokens), len(full_tokens) - start)):
                if full_tokens[start + j] == preamble_tokens[j]:
                    match_len += 1
                else:
                    break
            if match_len > best_match:
                best_match = match_len
                best_start = start
        preamble_start = best_start
    return preamble_start, preamble_tokens, preamble_text


def run_model(model_key, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"E15b — {model_name} ({model_key})")
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
    relay_layer = n_layers - 2

    key_layers = sorted(set([0, n_layers // 4, n_layers // 2,
                             3 * n_layers // 4, relay_layer, n_layers - 1]))

    print(f"  Layers: {n_layers}, Relay: L{relay_layer}")
    embed_layer = model.model.embed_tokens

    all_dose_results = {}

    for dose in DOSES:
        print(f"\n  DOSE {dose}:")
        messages = build_conversation(dose, use_system_role=use_sys)
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        n_tokens = inputs["input_ids"].shape[1]

        preamble_start, preamble_tokens, preamble_text = find_preamble_tokens(
            tokenizer, text)
        n_preamble = len(preamble_tokens)
        print(f"    Tokens: {n_tokens}, Preamble: {n_preamble} at pos {preamble_start}")

        with torch.no_grad():
            base_out = model(**inputs, output_hidden_states=True)
        base_geom = measure_geometry(base_out.hidden_states, key_layers)
        base_logits = base_out.logits[0, -1].float()

        print(f"    Baseline relay ratio: {base_geom.get(relay_layer,{}).get('ratio','nan'):.4f}")

        attributions = []
        t_abl = time.time()
        for tok_idx in range(n_preamble):
            abs_pos = preamble_start + tok_idx
            zeroed = [False]
            def hook_fn(module, inp, out, pos=abs_pos):
                if not zeroed[0]:
                    zeroed[0] = True
                    out_mod = out.clone()
                    out_mod[0, pos, :] = 0.0
                    return out_mod

            handle = embed_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                abl_out = model(**inputs, output_hidden_states=True)
            handle.remove()

            abl_geom = measure_geometry(abl_out.hidden_states, key_layers)
            abl_logits = abl_out.logits[0, -1].float()

            relay_delta = abs(abl_geom.get(relay_layer,{}).get("ratio",0) -
                             base_geom.get(relay_layer,{}).get("ratio",0))
            logit_change = torch.norm(abl_logits - base_logits).item()
            token_text = preamble_text[tok_idx][1] if tok_idx < len(preamble_text) else "?"

            attributions.append({
                "position": tok_idx,
                "token_text": token_text,
                "relay_ratio_delta": float(relay_delta),
                "logit_change": float(logit_change),
            })

        print(f"    Ablation: {time.time()-t_abl:.1f}s")

        by_relay = sorted(attributions, key=lambda x: x["relay_ratio_delta"], reverse=True)
        print(f"    Top 5 by relay:")
        for a in by_relay[:5]:
            print(f"      [{a['position']:2d}] '{a['token_text']}': "
                  f"delta={a['relay_ratio_delta']:.5f}")

        all_dose_results[f"dose_{dose}"] = {
            "dose": dose,
            "n_tokens": n_tokens,
            "baseline_relay_ratio": base_geom.get(relay_layer,{}).get("ratio",0),
            "attributions": attributions,
            "top5_relay": [{"text": a["token_text"],
                           "delta": a["relay_ratio_delta"]}
                          for a in by_relay[:5]],
        }

    # Cross-dose comparison
    print(f"\n  Cross-dose top token comparison:")
    for dose in DOSES:
        dr = all_dose_results[f"dose_{dose}"]
        top = dr["top5_relay"]
        top_str = ", ".join(f"'{t['text']}'({t['delta']:.4f})" for t in top[:3])
        print(f"    D{dose:2d}: {top_str}")

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  {model_key} complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        "model": model_name,
        "model_key": model_key,
        "n_layers": n_layers,
        "relay_layer": relay_layer,
        "doses": DOSES,
        "results": all_dose_results,
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

    print(f"E15b: Multi-Dose Path Patching")
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Doses: {DOSES}")

    all_models = {}
    for key, name in models_to_run.items():
        try:
            result = run_model(key, name)
            all_models[key] = result
            outpath = RESULTS_DIR / f"e15b_{key}_{timestamp}.json"
            with open(outpath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outpath}")
        except Exception as e:
            import traceback
            print(f"\nERROR on {key}: {e}")
            traceback.print_exc()
            all_models[key] = {"model": name, "error": str(e)}

    combined = {
        "experiment": "E15b",
        "title": "Multi-Dose Path Patching",
        "timestamp": timestamp,
        "doses": DOSES,
        "models": all_models,
    }
    outpath = RESULTS_DIR / f"e15b_combined_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined: {outpath}")


if __name__ == "__main__":
    main()
