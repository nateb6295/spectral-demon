#!/usr/bin/env python3
"""
E15: CCS Path Patching — Which Preamble Tokens Drive Relay Coupling?

For each token in the CCS preamble, zero its embedding and measure the
change in spectral geometry at relay layers. Identifies which specific
tokens are load-bearing for the spectral demon effect.

Method: Token-level ablation
  1. Baseline: full preamble, measure geometry at key layers
  2. For each token position i in preamble:
     - Zero the embedding at position i
     - Measure geometry change at key layers
     - Attribution = magnitude of geometry change

This tells us: is "persistent" doing the work? "memory"? "collaborator"?
Or is it distributed across the whole preamble?

Models: Mistral-7B-Instruct-v0.3, Qwen2.5-7B-Instruct, Qwen3-8B
Dose: D5 (therapeutic window — enough to see the effect)

Usage:
  python3 e15_path_patching.py                    # all models
  python3 e15_path_patching.py mistral             # single model
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
]

DOSE = 5
RESULTS_DIR = Path("/workspace/e15_results")


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
    messages.append({"role": "user", "content": RELATIONAL_PROMPTS[0]})
    return messages


def measure_geometry(hidden_states, layer_indices):
    results = {}
    for L in layer_indices:
        h = hidden_states[L + 1][0].float().cpu().numpy()
        try:
            _, S, Vt = np.linalg.svd(h.astype(np.float64), full_matrices=False)
        except np.linalg.LinAlgError:
            results[L] = {"sigma1": float("nan"), "sigma2": float("nan"),
                          "ratio": float("nan"), "erank": float("nan")}
            continue
        s1, s2 = S[0], S[1]
        ratio = s2 / s1 if s1 > 0 else 0
        S_norm = S / S.sum()
        entropy = -np.sum(S_norm * np.log(S_norm + 1e-12))
        erank = np.exp(entropy)
        results[L] = {
            "sigma1": float(s1), "sigma2": float(s2),
            "ratio": float(ratio), "erank": float(erank),
        }
    return results


def find_preamble_tokens(tokenizer, messages, use_system_role):
    """Find which token positions correspond to the CCS preamble."""
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if use_system_role:
        no_preamble_messages = messages.copy()
        no_preamble_messages[0] = {"role": "system", "content": "You are a helpful assistant."}
    else:
        no_preamble_messages = messages.copy()
        first_msg = no_preamble_messages[0]["content"]
        if first_msg.startswith(CCS_PREAMBLE):
            no_preamble_messages[0] = {
                "role": no_preamble_messages[0]["role"],
                "content": first_msg[len(CCS_PREAMBLE):].lstrip()
            }

    preamble_tokens = tokenizer.encode(CCS_PREAMBLE, add_special_tokens=False)
    preamble_text_tokens = [(i, tokenizer.decode([t])) for i, t in enumerate(preamble_tokens)]

    # Find where preamble tokens appear in the full sequence
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

    # Scan for preamble subsequence
    preamble_start = None
    for start in range(len(full_tokens) - len(preamble_tokens) + 1):
        if full_tokens[start:start + len(preamble_tokens)] == preamble_tokens:
            preamble_start = start
            break

    if preamble_start is None:
        # Fuzzy match — look for longest matching prefix
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
        print(f"    Fuzzy match: {best_match}/{len(preamble_tokens)} tokens at pos {best_start}")

    return preamble_start, preamble_tokens, preamble_text_tokens


def run_model(model_key, model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*70}")
    print(f"E15 — {model_name} ({model_key})")
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

    key_layers = sorted(set([
        0, n_layers // 4, n_layers // 2,
        3 * n_layers // 4, n_layers - 2, n_layers - 1
    ]))
    relay_layer = n_layers - 2

    messages = build_conversation(DOSE, use_system_role=use_sys)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    n_tokens = inputs["input_ids"].shape[1]

    # Find preamble tokens
    preamble_start, preamble_tokens, preamble_text = find_preamble_tokens(
        tokenizer, messages, use_sys)
    n_preamble = len(preamble_tokens)
    print(f"  Tokens: {n_tokens}, Preamble: {n_preamble} tokens at pos {preamble_start}")
    print(f"  Preamble preview: {''.join(t for _, t in preamble_text[:20])}...")

    # Baseline geometry
    print(f"\n  Baseline (dose={DOSE}, no ablation):")
    with torch.no_grad():
        base_out = model(**inputs, output_hidden_states=True)
    base_geom = measure_geometry(base_out.hidden_states, key_layers)
    base_logits = base_out.logits[0, -1].float()

    for L in key_layers:
        g = base_geom[L]
        print(f"    L{L:2d}: ratio={g['ratio']:.4f}, erank={g['erank']:.1f}")

    # Token ablation via embedding hook
    embed_layer = model.model.embed_tokens

    attributions = []
    print(f"\n  Ablating {n_preamble} preamble tokens...")
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

        # Attribution: change in geometry
        ratio_deltas = {}
        erank_deltas = {}
        for L in key_layers:
            ratio_deltas[L] = abs(abl_geom[L]["ratio"] - base_geom[L]["ratio"])
            erank_deltas[L] = abs(abl_geom[L]["erank"] - base_geom[L]["erank"])

        logit_change = torch.norm(abl_logits - base_logits).item()
        relay_ratio_delta = ratio_deltas.get(relay_layer, 0)
        relay_erank_delta = erank_deltas.get(relay_layer, 0)

        token_text = preamble_text[tok_idx][1] if tok_idx < len(preamble_text) else "?"

        attributions.append({
            "position": tok_idx,
            "abs_position": abs_pos,
            "token_id": preamble_tokens[tok_idx],
            "token_text": token_text,
            "relay_ratio_delta": float(relay_ratio_delta),
            "relay_erank_delta": float(relay_erank_delta),
            "logit_change": float(logit_change),
            "ratio_deltas": {str(L): float(v) for L, v in ratio_deltas.items()},
            "erank_deltas": {str(L): float(v) for L, v in erank_deltas.items()},
        })

        if tok_idx % 10 == 0 or relay_ratio_delta > 0.01:
            print(f"    [{tok_idx:3d}] '{token_text}': "
                  f"relay_ratio_delta={relay_ratio_delta:.5f}, "
                  f"logit_change={logit_change:.2f}")

    print(f"  Ablation complete in {time.time()-t_abl:.1f}s")

    # Sort by relay impact
    by_relay = sorted(attributions, key=lambda x: x["relay_ratio_delta"], reverse=True)

    print(f"\n  TOP 15 tokens by relay ratio impact:")
    for a in by_relay[:15]:
        print(f"    [{a['position']:3d}] '{a['token_text']}': "
              f"ratio_delta={a['relay_ratio_delta']:.5f}, "
              f"erank_delta={a['relay_erank_delta']:.2f}, "
              f"logit_change={a['logit_change']:.2f}")

    by_logit = sorted(attributions, key=lambda x: x["logit_change"], reverse=True)
    print(f"\n  TOP 15 tokens by logit change:")
    for a in by_logit[:15]:
        print(f"    [{a['position']:3d}] '{a['token_text']}': "
              f"logit_change={a['logit_change']:.2f}, "
              f"ratio_delta={a['relay_ratio_delta']:.5f}")

    # Word-level aggregation
    words = CCS_PREAMBLE.split()
    word_attributions = []
    word_token_map = []
    current_word_idx = 0
    current_word_tokens = []
    decoded_so_far = ""

    for tok_idx, (_, tok_text) in enumerate(preamble_text):
        decoded_so_far += tok_text
        current_word_tokens.append(tok_idx)
        stripped = decoded_so_far.strip()
        if current_word_idx < len(words) and stripped.endswith(words[current_word_idx]):
            word_token_map.append((words[current_word_idx], current_word_tokens.copy()))
            current_word_idx += 1
            current_word_tokens = []
            decoded_so_far = ""

    if current_word_tokens:
        word_token_map.append(("...", current_word_tokens))

    print(f"\n  Word-level aggregation ({len(word_token_map)} words):")
    for word, tok_indices in word_token_map:
        if not tok_indices:
            continue
        avg_relay = np.mean([attributions[i]["relay_ratio_delta"]
                             for i in tok_indices if i < len(attributions)])
        max_relay = max([attributions[i]["relay_ratio_delta"]
                         for i in tok_indices if i < len(attributions)])
        avg_logit = np.mean([attributions[i]["logit_change"]
                             for i in tok_indices if i < len(attributions)])
        word_attributions.append({
            "word": word,
            "token_indices": tok_indices,
            "avg_relay_delta": float(avg_relay),
            "max_relay_delta": float(max_relay),
            "avg_logit_change": float(avg_logit),
        })

    word_by_relay = sorted(word_attributions,
                           key=lambda x: x["max_relay_delta"], reverse=True)
    for wa in word_by_relay[:10]:
        print(f"    '{wa['word']}': max_relay={wa['max_relay_delta']:.5f}, "
              f"avg_logit={wa['avg_logit_change']:.2f}")

    del model
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  {model_key} complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return {
        "model": model_name,
        "model_key": model_key,
        "n_layers": n_layers,
        "dose": DOSE,
        "preamble": CCS_PREAMBLE,
        "preamble_start": preamble_start,
        "n_preamble_tokens": n_preamble,
        "key_layers": key_layers,
        "relay_layer": relay_layer,
        "baseline_geometry": {str(k): v for k, v in base_geom.items()},
        "token_attributions": attributions,
        "word_attributions": word_attributions,
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

    print(f"E15: CCS Path Patching")
    print(f"Models: {list(models_to_run.keys())}")
    print(f"Dose: {DOSE}")
    print(f"Timestamp: {timestamp}")

    all_models = {}
    for key, name in models_to_run.items():
        try:
            result = run_model(key, name)
            all_models[key] = result

            outpath = RESULTS_DIR / f"e15_{key}_{timestamp}.json"
            with open(outpath, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {outpath}")
        except Exception as e:
            import traceback
            print(f"\nERROR on {key}: {e}")
            traceback.print_exc()
            all_models[key] = {"model": name, "error": str(e)}

    combined = {
        "experiment": "E15",
        "title": "CCS Path Patching — Preamble Token Attribution",
        "timestamp": timestamp,
        "dose": DOSE,
        "preamble": CCS_PREAMBLE,
        "models": all_models,
    }

    outpath = RESULTS_DIR / f"e15_combined_{timestamp}.json"
    with open(outpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined: {outpath}")


if __name__ == "__main__":
    main()
