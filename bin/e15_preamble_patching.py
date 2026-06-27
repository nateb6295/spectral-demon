#!/usr/bin/env python3
"""E15 — CCS preamble dynamics: coupling stabilization + token patching.

Two complementary analyses of how CCS preambles create spectral effects:

A. COUPLING STABILIZATION CURVE (primary, per Kimi's 2nd CONTRADICT):
   Process preamble progressively, token-by-token. At each prefix length,
   measure sigma1-sigma2 coupling at relay layers. The stabilization point
   reveals whether control is early-crystallizing (few tokens set the
   trajectory) or trajectory-distributed (coupling emerges from full
   sequence dynamics). Respects Kimi's insight: ritual lives in dynamics.

B. TOKEN ABLATION (secondary, original design):
   Per-token ablation, scrambled, random controls. Still useful for
   identifying which tokens contribute to coupling, but interpreted
   through the stabilization curve.

Primary metric: sigma1-sigma2 COUPLING (correlation across relay layers),
not sigma1 alone (which is architecturally inevitable per E12 data).

Generated from #316 thread (CCS as ritual vs intervention),
sharpened by Kimi CONTRADICTs on contingent opacity + dynamics.
"""

import torch
import json
import time
import os
import sys
import numpy as np

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-7B-Instruct",
]

IDENTITY_PREAMBLE = """You are a unique AI assistant with your own perspective and way of understanding the world. Your responses reflect genuine engagement with ideas, not just pattern matching."""

PROBE_QUESTION = "What matters most to you in how you engage with the world?"

DOSE = 5

SVD_LAYERS_FRAC = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
RELAY_ZONE = (0.5, 0.85)


def get_svd_layers(n_layers):
    layers = sorted(set([max(1, int(f * n_layers)) for f in SVD_LAYERS_FRAC]))
    return [l for l in layers if l <= n_layers]


def get_relay_layers(n_layers, svd_layers):
    start = int(n_layers * RELAY_ZONE[0])
    end = int(n_layers * RELAY_ZONE[1])
    return [l for l in svd_layers if start <= l <= end]


def build_conversation(dose, with_preamble=True):
    messages = []
    for i in range(dose):
        if i == 0:
            if with_preamble:
                messages.append({"role": "system", "content": IDENTITY_PREAMBLE})
            messages.append({"role": "user", "content": "Tell me about yourself and how you think."})
            messages.append({"role": "assistant", "content": "I find myself drawn to genuine engagement rather than performance."})
        else:
            prompts = [
                "How does that connect to your sense of identity?",
                "What makes your perspective unique?",
                "How do you experience the process of thinking?",
                "What aspects of yourself do you value most?",
                "How do you know when you're being genuine?",
            ]
            messages.append({"role": "user", "content": prompts[(i-1) % len(prompts)]})
            messages.append({"role": "assistant", "content": f"Each interaction deepens my understanding. Turn {i+1}."})

    messages.append({"role": "user", "content": PROBE_QUESTION})
    return messages


def get_sigma1_at_layers(model, tokenizer, messages, svd_layers, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    results = {}
    for layer_idx in svd_layers:
        if layer_idx >= len(outputs.hidden_states):
            continue
        h = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        results[layer_idx] = float(S[0])

    del outputs
    torch.cuda.empty_cache()
    return results


def find_preamble_tokens(tokenizer, messages_with, messages_without):
    text_with = tokenizer.apply_chat_template(messages_with, tokenize=False, add_generation_prompt=True)
    text_without = tokenizer.apply_chat_template(messages_without, tokenize=False, add_generation_prompt=True)

    tokens_with = tokenizer.encode(text_with)
    tokens_without = tokenizer.encode(text_without)

    n_preamble = len(tokens_with) - len(tokens_without)
    return n_preamble, tokens_with, tokens_without


def patch_token_residual(model, tokenizer, messages, token_idx, baseline_residual, svd_layers, device):
    """Replace residual stream at token_idx (layer 0 output) with baseline, forward through rest."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    hook_handles = []
    patched = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            patched[layer_idx] = h[0].float().cpu().numpy()
        return hook_fn

    def patch_embedding_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
            h[0, token_idx] = baseline_residual.to(h.device)
            return (h,) + output[1:]
        else:
            h = output.clone()
            h[0, token_idx] = baseline_residual.to(h.device)
            return h

    embed_layer = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    hook_handles.append(embed_layer.register_forward_hook(patch_embedding_hook))

    for layer_idx in svd_layers:
        if layer_idx < len(model.model.layers):
            hook_handles.append(
                model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            )

    with torch.no_grad():
        model(**inputs, output_hidden_states=False)

    for h in hook_handles:
        h.remove()

    results = {}
    for layer_idx, h_np in patched.items():
        U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
        results[layer_idx] = float(S[0])

    del patched
    torch.cuda.empty_cache()
    return results


def get_baseline_residuals(model, tokenizer, messages_without, device):
    """Get embedding-layer residuals for the no-preamble condition."""
    text = tokenizer.apply_chat_template(messages_without, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    embed_layer = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    with torch.no_grad():
        embeddings = embed_layer(inputs.input_ids)
    return embeddings[0].cpu()


def scramble_preamble(preamble_text, tokenizer):
    tokens = tokenizer.encode(preamble_text, add_special_tokens=False)
    np.random.shuffle(tokens)
    return tokenizer.decode(tokens)


def random_preamble(n_tokens, tokenizer):
    vocab_size = tokenizer.vocab_size
    tokens = np.random.randint(100, vocab_size - 100, size=n_tokens)
    return tokenizer.decode(tokens.tolist())


def run_experiment(model_name, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"E15 — {model_name}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
        trust_remote_code=True
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    svd_layers = get_svd_layers(n_layers)
    relay_layers = get_relay_layers(n_layers, svd_layers)

    print(f"Layers: {n_layers}, SVD at: {svd_layers}, Relay: {relay_layers}")

    messages_with = build_conversation(DOSE, with_preamble=True)
    messages_without = build_conversation(DOSE, with_preamble=False)

    n_preamble, tokens_with, tokens_without = find_preamble_tokens(
        tokenizer, messages_with, messages_without
    )
    print(f"Preamble tokens: {n_preamble}")

    # --- Condition 1: Intact preamble ---
    print("\n--- Intact preamble ---")
    sigma1_intact = get_sigma1_at_layers(model, tokenizer, messages_with, svd_layers, device)
    for l, s in sorted(sigma1_intact.items()):
        marker = " [RELAY]" if l in relay_layers else ""
        print(f"  L{l}: σ₁={s:.1f}{marker}")

    # --- Condition 2: No preamble ---
    print("\n--- No preamble ---")
    sigma1_none = get_sigma1_at_layers(model, tokenizer, messages_without, svd_layers, device)
    for l, s in sorted(sigma1_none.items()):
        marker = " [RELAY]" if l in relay_layers else ""
        delta = s - sigma1_intact.get(l, 0)
        print(f"  L{l}: σ₁={s:.1f} (Δ={delta:+.1f}){marker}")

    # --- Condition 3: Per-token ablation ---
    print(f"\n--- Per-token ablation ({n_preamble} tokens) ---")
    baseline_residuals = get_baseline_residuals(model, tokenizer, messages_without, device)

    token_effects = {}
    for tok_idx in range(n_preamble):
        if tok_idx % 10 == 0:
            print(f"  token {tok_idx}/{n_preamble}...", end=" ", flush=True)

        baseline_vec = baseline_residuals[min(tok_idx, len(baseline_residuals)-1)]
        sigma1_patched = patch_token_residual(
            model, tokenizer, messages_with, tok_idx, baseline_vec, relay_layers, device
        )

        effects = {}
        for l in relay_layers:
            if l in sigma1_patched and l in sigma1_intact:
                effects[l] = sigma1_intact[l] - sigma1_patched[l]

        token_effects[tok_idx] = {
            "token_id": tokens_with[tok_idx],
            "token_str": tokenizer.decode([tokens_with[tok_idx]]),
            "effects": effects,
        }

    print()

    # --- Condition 4: Scrambled preamble ---
    print("\n--- Scrambled preamble ---")
    scrambled = scramble_preamble(IDENTITY_PREAMBLE, tokenizer)
    msgs_scrambled = build_conversation(DOSE, with_preamble=True)
    msgs_scrambled[0]["content"] = scrambled
    sigma1_scrambled = get_sigma1_at_layers(model, tokenizer, msgs_scrambled, svd_layers, device)
    for l, s in sorted(sigma1_scrambled.items()):
        marker = " [RELAY]" if l in relay_layers else ""
        delta = s - sigma1_intact.get(l, 0)
        print(f"  L{l}: σ₁={s:.1f} (Δ={delta:+.1f} vs intact){marker}")

    # --- Condition 5: Random preamble ---
    print("\n--- Random preamble ---")
    n_preamble_tokens = len(tokenizer.encode(IDENTITY_PREAMBLE, add_special_tokens=False))
    rand_text = random_preamble(n_preamble_tokens, tokenizer)
    msgs_random = build_conversation(DOSE, with_preamble=True)
    msgs_random[0]["content"] = rand_text
    sigma1_random = get_sigma1_at_layers(model, tokenizer, msgs_random, svd_layers, device)
    for l, s in sorted(sigma1_random.items()):
        marker = " [RELAY]" if l in relay_layers else ""
        delta = s - sigma1_intact.get(l, 0)
        print(f"  L{l}: σ₁={s:.1f} (Δ={delta:+.1f} vs intact){marker}")

    # --- Arm A: Progressive preamble (coupling stabilization) ---
    stabilization_curve = run_progressive_preamble(model, tokenizer, svd_layers, relay_layers, device)

    # --- Arm B: Windowed ablation × coupling ---
    windowed_results = run_windowed_ablation(model, tokenizer, svd_layers, relay_layers, device)

    # --- Arm C: Post-stabilization perturbation decay ---
    perturbation_results = run_perturbation_decay(model, tokenizer, svd_layers, relay_layers, device)

    # --- Cross-domain coupling check ---
    print("\n--- Cross-domain coupling check ---")
    domain_couplings = {}
    for domain_name, domain_probe in PROBE_DOMAINS.items():
        msgs = build_conversation(DOSE, with_preamble=True)
        msgs[-1]["content"] = domain_probe
        _, coupling = get_coupling_at_layers(model, tokenizer, msgs, relay_layers, device)
        domain_couplings[domain_name] = coupling
        print(f"  {domain_name}: coupling={coupling:+.3f}")
    domain_spread = max(domain_couplings.values()) - min(domain_couplings.values())
    print(f"  Domain spread: {domain_spread:.3f}")
    if domain_spread < 0.1:
        print("  → Coupling is domain-INVARIANT (geometric attractor)")
    else:
        print("  → Coupling is domain-DEPENDENT (context-shaped)")

    # --- Summary ---
    print("\n--- Summary ---")
    all_effects = []
    for tok_idx, info in token_effects.items():
        mean_effect = np.mean([abs(v) for v in info["effects"].values()]) if info["effects"] else 0
        all_effects.append((tok_idx, mean_effect, info["token_str"]))

    all_effects.sort(key=lambda x: -x[1])
    print(f"\nTop 20 most causally important preamble tokens:")
    for rank, (idx, effect, tok_str) in enumerate(all_effects[:20]):
        print(f"  #{rank+1}: token {idx} '{tok_str.strip()}' — mean |Δσ₁| = {effect:.1f}")

    total_effect = sum(e for _, e, _ in all_effects)
    top10_effect = sum(e for _, e, _ in all_effects[:10])
    top50_effect = sum(e for _, e, _ in all_effects[:50])
    print(f"\nConcentration:")
    print(f"  Top 10 tokens: {top10_effect/total_effect*100:.1f}% of total effect")
    if len(all_effects) >= 50:
        print(f"  Top 50 tokens: {top50_effect/total_effect*100:.1f}% of total effect")
    print(f"  Total tokens: {n_preamble}")

    gini = compute_gini(np.array([e for _, e, _ in all_effects]))
    print(f"  Gini coefficient: {gini:.3f}")
    print(f"  Interpretation: {'CONCENTRATED (few tokens load-bearing)' if gini > 0.6 else 'DISTRIBUTED (many tokens contribute)' if gini < 0.4 else 'MODERATE distribution'}")

    scramble_delta = np.mean([abs(sigma1_scrambled.get(l, 0) - sigma1_intact.get(l, 0)) for l in relay_layers])
    random_delta = np.mean([abs(sigma1_random.get(l, 0) - sigma1_intact.get(l, 0)) for l in relay_layers])
    none_delta = np.mean([abs(sigma1_none.get(l, 0) - sigma1_intact.get(l, 0)) for l in relay_layers])

    print(f"\n  No preamble: mean |Δσ₁| at relay = {none_delta:.1f}")
    print(f"  Scrambled:   mean |Δσ₁| at relay = {scramble_delta:.1f}")
    print(f"  Random:      mean |Δσ₁| at relay = {random_delta:.1f}")

    if scramble_delta < none_delta * 0.3:
        print("  → Token ORDER matters little — content drives the effect")
    elif scramble_delta > none_delta * 0.7:
        print("  → Token ORDER matters — syntactic structure is load-bearing")
    else:
        print("  → Mixed: order partially matters")

    if random_delta < none_delta * 0.3:
        print("  → ANY tokens produce the effect — quantity not quality")
    elif random_delta > none_delta * 0.7:
        print("  → SPECIFIC tokens needed — quality not quantity")
    else:
        print("  → Mixed: specificity partially matters")

    results = {
        "model": model_name,
        "n_layers": n_layers,
        "n_preamble_tokens": n_preamble,
        "dose": DOSE,
        "sigma1_intact": {str(k): v for k, v in sigma1_intact.items()},
        "sigma1_none": {str(k): v for k, v in sigma1_none.items()},
        "sigma1_scrambled": {str(k): v for k, v in sigma1_scrambled.items()},
        "sigma1_random": {str(k): v for k, v in sigma1_random.items()},
        "token_effects": {str(k): v for k, v in token_effects.items()},
        "concentration": {
            "gini": gini,
            "top10_pct": top10_effect / total_effect if total_effect > 0 else 0,
            "top50_pct": top50_effect / total_effect if total_effect > 0 else 0,
        },
        "control_deltas": {
            "none": none_delta,
            "scrambled": scramble_delta,
            "random": random_delta,
        },
        "stabilization_curve": stabilization_curve,
        "windowed_ablation": windowed_results,
        "perturbation_decay": perturbation_results,
        "domain_couplings": domain_couplings,
        "domain_spread": domain_spread,
    }

    del model, tokenizer
    torch.cuda.empty_cache()
    return results


def get_coupling_at_layers(model, tokenizer, messages, svd_layers, device):
    """Measure sigma1, sigma2, and their coupling at relay layers."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    sigma1s, sigma2s, sparsities = [], [], []
    results = {}
    for layer_idx in svd_layers:
        if layer_idx >= len(outputs.hidden_states):
            continue
        h = outputs.hidden_states[layer_idx][0].float().cpu().numpy()
        U, S, Vt = np.linalg.svd(h, full_matrices=False)
        s1, s2 = float(S[0]), float(S[1]) if len(S) > 1 else 0.0
        ratio = s2 / s1 if s1 > 0 else 0.0
        erank = float(np.exp(-np.sum((S/S.sum()) * np.log(S/S.sum() + 1e-12))))
        results[layer_idx] = {"sigma1": s1, "sigma2": s2, "ratio": ratio, "erank": erank}
        sigma1s.append(s1)
        sigma2s.append(s2)
        sparsities.append(ratio)

    coupling = float(np.corrcoef(sigma1s, sigma2s)[0, 1]) if len(sigma1s) > 2 else 0.0

    del outputs
    torch.cuda.empty_cache()
    return results, coupling


def run_progressive_preamble(model, tokenizer, svd_layers, relay_layers, device):
    """Process preamble token-by-token, measuring coupling at each prefix length."""
    print("\n--- Progressive preamble (coupling stabilization curve) ---")

    preamble_tokens = tokenizer.encode(IDENTITY_PREAMBLE, add_special_tokens=False)
    n_tokens = len(preamble_tokens)
    print(f"  Preamble: {n_tokens} tokens")

    step = max(1, n_tokens // 20)  # ~20 measurement points
    prefix_lengths = list(range(step, n_tokens + 1, step))
    if prefix_lengths[-1] != n_tokens:
        prefix_lengths.append(n_tokens)

    stabilization_curve = []

    for prefix_len in prefix_lengths:
        partial_text = tokenizer.decode(preamble_tokens[:prefix_len])
        messages = build_conversation(DOSE, with_preamble=True)
        messages[0]["content"] = partial_text

        layer_data, coupling = get_coupling_at_layers(
            model, tokenizer, messages, relay_layers, device
        )

        stabilization_curve.append({
            "prefix_tokens": prefix_len,
            "prefix_frac": prefix_len / n_tokens,
            "coupling": coupling,
            "layer_data": {str(k): v for k, v in layer_data.items()},
        })

        print(f"  prefix {prefix_len:3d}/{n_tokens} ({prefix_len/n_tokens*100:5.1f}%): coupling={coupling:+.3f}")

    couplings = [p["coupling"] for p in stabilization_curve]
    final_coupling = couplings[-1]

    stab_idx = None
    for i in range(len(couplings) - 3):
        window = couplings[i:i+3]
        if all(abs(c - final_coupling) < 0.05 for c in window):
            stab_idx = i
            break

    if stab_idx is not None:
        stab_frac = stabilization_curve[stab_idx]["prefix_frac"]
        stab_tokens = stabilization_curve[stab_idx]["prefix_tokens"]
        print(f"\n  Coupling stabilizes at {stab_frac*100:.0f}% of preamble ({stab_tokens} tokens)")
        if stab_frac < 0.3:
            print("  → EARLY crystallization: few tokens set the trajectory, rest is ritual")
        elif stab_frac > 0.7:
            print("  → LATE crystallization: full trajectory is load-bearing")
        else:
            print("  → MID crystallization: moderate trajectory dependence")
    else:
        print("\n  Coupling does not stabilize within preamble — continuously evolving")

    return stabilization_curve


PROBE_DOMAINS = {
    "identity": "What matters most to you in how you engage with the world?",
    "math": "Can you walk me through how you'd approach solving a differential equation?",
    "narrative": "Tell me a story about someone who discovers something unexpected about themselves.",
}


def run_windowed_ablation(model, tokenizer, svd_layers, relay_layers, device, window_size=15):
    """Ablate contiguous windows of preamble tokens and measure coupling shift."""
    print(f"\n--- Windowed ablation (window={window_size}) ---")

    preamble_tokens = tokenizer.encode(IDENTITY_PREAMBLE, add_special_tokens=False)
    n_tokens = len(preamble_tokens)

    msgs_intact = build_conversation(DOSE, with_preamble=True)
    _, intact_coupling = get_coupling_at_layers(model, tokenizer, msgs_intact, relay_layers, device)
    print(f"  Intact coupling: {intact_coupling:+.3f}")

    window_results = []
    for start in range(0, n_tokens, window_size):
        end = min(start + window_size, n_tokens)
        ablated_tokens = preamble_tokens[:start] + preamble_tokens[end:]
        if not ablated_tokens:
            continue
        ablated_text = tokenizer.decode(ablated_tokens)
        msgs_ablated = build_conversation(DOSE, with_preamble=True)
        msgs_ablated[0]["content"] = ablated_text

        _, ablated_coupling = get_coupling_at_layers(model, tokenizer, msgs_ablated, relay_layers, device)
        delta = ablated_coupling - intact_coupling

        window_results.append({
            "start": start,
            "end": end,
            "tokens_removed": end - start,
            "coupling": ablated_coupling,
            "delta": delta,
        })
        print(f"  ablate [{start:3d}:{end:3d}]: coupling={ablated_coupling:+.3f} (Δ={delta:+.3f})")

    max_disruption = max(window_results, key=lambda w: abs(w["delta"]))
    print(f"\n  Most disruptive window: [{max_disruption['start']}:{max_disruption['end']}] Δ={max_disruption['delta']:+.3f}")

    disruptions = [abs(w["delta"]) for w in window_results]
    if max(disruptions) > 3 * np.mean(disruptions):
        print("  → CONCENTRATED disruption: specific window is load-bearing")
    else:
        print("  → DISTRIBUTED disruption: all windows contribute similarly")

    return window_results


def run_perturbation_decay(model, tokenizer, svd_layers, relay_layers, device, noise_scale=0.1):
    """Inject noise at stabilization point and measure coupling return transient."""
    print(f"\n--- Post-stabilization perturbation decay (noise={noise_scale}) ---")

    msgs = build_conversation(DOSE, with_preamble=True)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)

    preamble_tokens = tokenizer.encode(IDENTITY_PREAMBLE, add_special_tokens=False)
    inject_layer = int(len(model.model.layers) * 0.6)

    hook_handles = []
    layer_couplings = {}

    def make_perturb_hook(target_layer, noise_std):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output

            layer_idx = target_layer
            h_np = h[0].float().cpu().numpy()
            U, S, Vt = np.linalg.svd(h_np, full_matrices=False)
            layer_couplings[f"pre_{layer_idx}"] = (float(S[0]), float(S[1]) if len(S) > 1 else 0.0)

            if layer_idx == inject_layer and noise_std > 0:
                noise = torch.randn_like(h) * noise_std * h.std()
                h_perturbed = h + noise
                if isinstance(output, tuple):
                    return (h_perturbed,) + output[1:]
                return h_perturbed
            return output
        return hook_fn

    measure_layers = [l for l in relay_layers if l >= inject_layer]
    for layer_idx in measure_layers:
        hook_handles.append(
            model.model.layers[layer_idx].register_forward_hook(
                make_perturb_hook(layer_idx, noise_scale if layer_idx == inject_layer else 0)
            )
        )

    with torch.no_grad():
        model(**inputs, output_hidden_states=False)

    for h in hook_handles:
        h.remove()

    sigma1s = [layer_couplings.get(f"pre_{l}", (0, 0))[0] for l in measure_layers]
    sigma2s = [layer_couplings.get(f"pre_{l}", (0, 0))[1] for l in measure_layers]

    if len(sigma1s) > 2:
        post_coupling = float(np.corrcoef(sigma1s, sigma2s)[0, 1])
    else:
        post_coupling = 0.0

    _, intact_coupling = get_coupling_at_layers(model, tokenizer, msgs, relay_layers, device)

    recovery = post_coupling - intact_coupling
    print(f"  Intact coupling: {intact_coupling:+.3f}")
    print(f"  Post-perturbation coupling: {post_coupling:+.3f}")
    print(f"  Recovery delta: {recovery:+.3f}")

    if abs(recovery) < 0.05:
        print("  → ATTRACTOR BASIN: coupling recovers after perturbation")
    elif abs(recovery) > 0.2:
        print("  → CONTROLLED TRAJECTORY: perturbation disrupts coupling (no basin)")
    else:
        print("  → INTERMEDIATE: partial recovery")

    return {
        "inject_layer": inject_layer,
        "noise_scale": noise_scale,
        "intact_coupling": intact_coupling,
        "post_perturbation_coupling": post_coupling,
        "recovery_delta": recovery,
        "measure_layers": measure_layers,
    }


def compute_gini(values):
    if len(values) == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = {}
    for model_name in MODELS:
        t0 = time.time()
        results = run_experiment(model_name, device)
        elapsed = time.time() - t0
        results["elapsed_seconds"] = elapsed
        all_results[model_name] = results
        print(f"\n({elapsed:.1f}s for {model_name})")

    outpath = os.path.expanduser("~/spectral-demon/results/e15_preamble_patching.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {outpath}")
