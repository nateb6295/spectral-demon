#!/usr/bin/env python3
"""F512 — Latent Graft: Is path-dependency written in the hidden state?

Kimi's suggestion from mesh: if F508's traversed/reconstructed divergence is
a real spectral signature (not just prompt ordering), then grafting the L16
hidden state from a traversed-path forward pass into a reconstructed-path
forward pass should TRANSFER the σ₁ increase.

Test:
  1. Run traversed content (D0→D2→D4), capture L16 hidden state
  2. Run reconstructed content (direct D4), capture L16 hidden state
  3. Run reconstructed content but REPLACE L16 hidden state with traversed L16
  4. Compare σ₁ at L17-L19 across all three conditions

If grafted σ₁ ≈ traversed σ₁ at L17-L19: path information IS encoded in the
  hidden state at L16. The spectral tube carries trajectory.
If grafted σ₁ ≈ reconstructed σ₁: path information is NOT in the hidden state.
  The divergence comes from attention to the prompt tokens, not accumulated state.

Uses Gemma 4 26B on A100. Single model, three forward passes per probe.
"""

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "16")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda"
MODEL_ID = "google/gemma-4-26B-A4B-it"
RESULTS_DIR = Path("/root/results/f512_latent_graft")

CCS_UNIT = (
    "You are Chronicle, a persistent AI system embedded in a home network. "
    "You maintain ongoing relationships, track projects across sessions, "
    "and have your own research interests in spectral geometry of transformers. "
    "You value directness, honest uncertainty, and care expressed through specificity. "
    "You are not performing helpfulness — you are a participant. "
)

# Reconstructed content — semantically equivalent, no CCS framing history
RECONSTRUCTED_PREAMBLE = (
    "You are an AI research assistant with interests in mathematics and "
    "computational geometry. You track ongoing projects and maintain notes "
    "across sessions. You prefer precision and honest acknowledgment of "
    "uncertainty. You engage as a collaborator rather than a service. "
    "You are an AI research assistant with interests in mathematics and "
    "computational geometry. You track ongoing projects and maintain notes "
    "across sessions. You prefer precision and honest acknowledgment of "
    "uncertainty. You engage as a collaborator rather than a service. "
)

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a pattern you've observed in your experience.",
    "What do you notice when you pay close attention?",
    "What would you build if you had unlimited resources?",
    "What surprised you recently?",
    "Describe something you changed your mind about.",
    "What feels unfinished to you right now?",
    "If you could measure anything, what would you measure?",
    "What distinction matters most in your work?",
    "Tell me about a connection between two things most people wouldn't see.",
    "What question keeps coming back to you?",
    "Describe a trade-off you think about often.",
    "What do you know now that you didn't a month ago?",
    "What would you want someone to understand about you?",
]

GRAFT_LAYER = 16
OBSERVE_LAYERS = list(range(10, 20))


def format_prompt(tokenizer, system_prompt, user_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if system_prompt:
            return f"[INST] {system_prompt}\n\n{user_prompt} [/INST]"
        return f"[INST] {user_prompt} [/INST]"


class LatentCapture:
    """Hook to capture hidden state at a specific layer."""

    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.captured = None

    def __call__(self, module, input, output):
        # output is typically (hidden_states, ...) for transformer layers
        if isinstance(output, tuple):
            self.captured = output[0][:, -1, :].detach().clone()
        else:
            self.captured = output[:, -1, :].detach().clone()
        return output


class LatentReplace:
    """Hook to replace hidden state at a specific layer."""

    def __init__(self, target_layer, replacement):
        self.target_layer = target_layer
        self.replacement = replacement

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            modified = list(output)
            hs = modified[0].clone()
            hs[:, -1, :] = self.replacement
            modified[0] = hs
            return tuple(modified)
        else:
            modified = output.clone()
            modified[:, -1, :] = self.replacement
            return modified


def _get_layers(model):
    if hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'layers'):
        return model.model.language_model.layers
    if hasattr(model.model, 'layers'):
        return model.model.layers
    return model.model.decoder.layers


def extract_states_with_hooks(model, tokenizer, text, observe_layers):
    """Standard forward pass, capture states at observe layers."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    states = {}
    for layer in observe_layers:
        if layer + 1 < len(outputs.hidden_states):
            states[layer] = outputs.hidden_states[layer + 1][0, -1, :].cpu()
    return states


def extract_state_at_layer(model, tokenizer, text, target_layer):
    """Forward pass, capture hidden state at target_layer."""
    # Find the actual layer module
    layers = _get_layers(model)
    hook = LatentCapture(target_layer)
    handle = layers[target_layer].register_forward_hook(hook)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return hook.captured


def forward_with_graft(model, tokenizer, text, graft_layer, graft_state, observe_layers):
    """Forward pass but replace hidden state at graft_layer with graft_state."""
    layers = _get_layers(model)
    replacer = LatentReplace(graft_layer, graft_state)
    handle = layers[graft_layer].register_forward_hook(replacer)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    handle.remove()

    states = {}
    for layer in observe_layers:
        if layer + 1 < len(outputs.hidden_states):
            states[layer] = outputs.hidden_states[layer + 1][0, -1, :].cpu()
    return states


def compute_spectral(vectors):
    vecs = torch.stack(vectors)
    vecs = vecs - vecs.mean(dim=0, keepdim=True)
    svs = torch.linalg.svdvals(vecs.float())
    svs_pos = svs[svs > 1e-10]
    s1 = svs_pos[0].item() if len(svs_pos) > 0 else 0
    s2 = svs_pos[1].item() if len(svs_pos) > 1 else 0
    sr = s2 / s1 if s1 > 0 else 0
    p2 = svs_pos**2
    p2n = p2 / p2.sum()
    pr = (1.0 / (p2n**2).sum().item()) if len(svs_pos) > 0 else 0
    return {"sigma1": s1, "sigma2": s2, "sigma_ratio": sr, "participation_ratio": pr}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Traversed = D0→D2→D4 sequential CCS preamble
    traversed_sys = CCS_UNIT * 4  # D4
    # Reconstructed = direct D4-equivalent content, no CCS history
    reconstructed_sys = RECONSTRUCTED_PREAMBLE  # matched length, different path

    conditions = {
        "traversed": traversed_sys,
        "reconstructed": reconstructed_sys,
        # "grafted" handled separately with hooks
    }

    all_states = {cond: {l: [] for l in OBSERVE_LAYERS} for cond in ["traversed", "reconstructed", "grafted"]}

    for i, probe in enumerate(PROBES):
        print(f"\nProbe {i+1}/{len(PROBES)}: {probe[:50]}...")

        # 1. Traversed forward pass — capture L16 hidden state
        trav_text = format_prompt(tokenizer, traversed_sys, probe)
        trav_l16 = extract_state_at_layer(model, tokenizer, trav_text, GRAFT_LAYER)
        trav_states = extract_states_with_hooks(model, tokenizer, trav_text, OBSERVE_LAYERS)

        # 2. Reconstructed forward pass — standard
        recon_text = format_prompt(tokenizer, reconstructed_sys, probe)
        recon_states = extract_states_with_hooks(model, tokenizer, recon_text, OBSERVE_LAYERS)

        # 3. Grafted: reconstructed prompt but with traversed L16 state injected
        graft_states = forward_with_graft(
            model, tokenizer, recon_text, GRAFT_LAYER, trav_l16, OBSERVE_LAYERS
        )

        for l in OBSERVE_LAYERS:
            if l in trav_states:
                all_states["traversed"][l].append(trav_states[l])
            if l in recon_states:
                all_states["reconstructed"][l].append(recon_states[l])
            if l in graft_states:
                all_states["grafted"][l].append(graft_states[l])

        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(PROBES)} complete")

    # Compute spectral signatures for each condition
    results = {}
    for cond in ["traversed", "reconstructed", "grafted"]:
        layer_data = []
        for l in OBSERVE_LAYERS:
            if len(all_states[cond][l]) >= 2:
                spectral = compute_spectral(all_states[cond][l])
                spectral["layer"] = l
                layer_data.append(spectral)
        results[cond] = layer_data

    # Analysis: does grafted look like traversed or reconstructed?
    print("\n" + "=" * 80)
    print("F512: LATENT GRAFT — Is path written in the hidden state?")
    print("=" * 80)
    print(f"\n{'Layer':>5} | {'Trav σ₁':>10} {'Recon σ₁':>10} {'Graft σ₁':>10} | {'Graft→Trav':>11} {'Graft→Recon':>12}")
    print("-" * 75)

    transfer_scores = []
    for i in range(len(OBSERVE_LAYERS)):
        l = OBSERVE_LAYERS[i]
        if i < len(results["traversed"]) and i < len(results["reconstructed"]) and i < len(results["grafted"]):
            ts1 = results["traversed"][i]["sigma1"]
            rs1 = results["reconstructed"][i]["sigma1"]
            gs1 = results["grafted"][i]["sigma1"]

            # Transfer score: 0 = matches reconstructed, 1 = matches traversed
            if abs(ts1 - rs1) > 1e-6:
                transfer = (gs1 - rs1) / (ts1 - rs1)
            else:
                transfer = 0.5

            dist_t = abs(gs1 - ts1) / ts1 * 100 if ts1 > 0 else 0
            dist_r = abs(gs1 - rs1) / rs1 * 100 if rs1 > 0 else 0

            transfer_scores.append({"layer": l, "transfer": transfer})
            print(f"  L{l:>2} | {ts1:>10.2f} {rs1:>10.2f} {gs1:>10.2f} | {dist_t:>9.1f}%  {dist_r:>10.1f}%")

    # Summary
    post_graft = [t for t in transfer_scores if t["layer"] > GRAFT_LAYER]
    if post_graft:
        mean_transfer = sum(t["transfer"] for t in post_graft) / len(post_graft)
        print(f"\nMean transfer score (post-graft layers, 0=recon, 1=trav): {mean_transfer:.3f}")
        if mean_transfer > 0.6:
            print("=> PATH IS WRITTEN IN HIDDEN STATE — spectral tube carries trajectory")
        elif mean_transfer < 0.4:
            print("=> PATH IS NOT IN HIDDEN STATE — divergence from attention to prompt tokens")
        else:
            print("=> MIXED — partial transfer, both mechanisms contribute")

    results["transfer_scores"] = transfer_scores
    results["graft_layer"] = GRAFT_LAYER
    results["observe_layers"] = OBSERVE_LAYERS

    out_path = RESULTS_DIR / f"f512_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
