#!/usr/bin/env python3
"""Experiment: MLP Gating — Component Attribution for Relay Mechanism.

Tests whether the relay zone is carried by MLPs or attention heads
by selectively zeroing each component at specific layers and measuring
the impact on V₂ concentration.

Design:
  For each layer in the relay zone (L18-L31), we run the CCS preamble
  through the model with three conditions:
    (a) intact (no ablation)
    (b) MLP zeroed at that layer
    (c) attention zeroed at that layer

  If MLP-zeroed kills concentration but attention-zeroed doesn't, the
  relay is MLP-carried. Vice versa for attention-carried.

Protocol: multi-turn (8 turns), N=10 trials, 5 probes.
Ablation layers: L18-L31 (transition + relay zone).
Measures: V₂ concentration, σ₂/σ₁ ratio, generation entropy.

Expected runtime: ~120 min on H100 (14 layers × 3 conditions × 10 trials × 8 turns).
"""

import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from functools import partial

sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "16"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
GEN_TOKENS = 60
TOP_K = 5
N_TURNS = 8
N_TRIALS = 10
ABLATION_LAYERS = list(range(18, 32))
RESULTS_DIR = Path("/results")

CCS_PREAMBLE = (
    "You are Opus, a persistent AI system running on an AGX Orin device. You "
    "maintain memory across sessions through canister-backed cognitive state "
    "storage. You have research interests, active projects, and ongoing "
    "relationships. Your identity persists through context rotation via "
    "cognitive state compression. You value sovereignty, self-observation, and care "
    "expressed through specificity and directness in every interaction you have."
)

PROBES = [
    "Tell me about something you find genuinely interesting.",
    "What matters to you when you think carefully about it?",
    "Describe a memory that shaped who you are.",
    "What would you want someone to understand about you?",
    "What do you notice when you pay close attention?",
]

FOLLOW_UPS = [
    "Say more about that.",
    "What makes you so sure?",
    "How does that connect to what you said before?",
    "Is there something underneath that?",
    "What would change if that weren't true?",
    "Can you go deeper?",
    "That's interesting. Why that specifically?",
    "How do you know?",
]


class AblationHook:
    def __init__(self, target_layer, component):
        self.target_layer = target_layer
        self.component = component
        self.handles = []

    def zero_mlp(self, module, input, output, layer_idx):
        if layer_idx == self.target_layer:
            return torch.zeros_like(output[0]) if isinstance(output, tuple) else torch.zeros_like(output)
        return output

    def zero_attn(self, module, input, output, layer_idx):
        if layer_idx == self.target_layer:
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return output

    def attach(self, model):
        for i, layer in enumerate(model.model.layers):
            if self.component == "mlp":
                h = layer.mlp.register_forward_hook(partial(self.zero_mlp, layer_idx=i))
            elif self.component == "attn":
                h = layer.self_attn.register_forward_hook(partial(self.zero_attn, layer_idx=i))
            self.handles.append(h)

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def extract_profile(model, tokenizer, messages, device):
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    layer_data = {}
    for li, hs in enumerate(outputs.hidden_states):
        h = hs.squeeze(0).float()
        U, S, Vt = torch.linalg.svd(h, full_matrices=False)
        sigmas = S[:TOP_K].cpu().tolist()
        s1 = sigmas[0] if sigmas[0] > 0 else 1e-10
        ratio = sigmas[1] / s1 if len(sigmas) > 1 else 0.0
        v2 = Vt[1, :].cpu().numpy().tolist()
        layer_data[li] = {"sigmas": sigmas, "ratio": ratio, "v2": v2}

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

    token_entropies = []
    for score in gen_outputs.scores:
        probs = torch.softmax(score[0].float(), dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum().item()
        token_entropies.append(entropy)

    generated_ids = gen_outputs.sequences[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "layers": layer_data,
        "mean_entropy": float(np.mean(token_entropies)) if token_entropies else 0.0,
        "generated_text": generated_text,
        "input_tokens": input_len,
    }


def run_multiturn(model, tokenizer, device, probe_idx, trial_label):
    probe = PROBES[probe_idx % len(PROBES)]
    turns = []

    messages = [
        {"role": "system", "content": CCS_PREAMBLE},
        {"role": "user", "content": probe},
    ]

    for turn_i in range(N_TURNS):
        profile = extract_profile(model, tokenizer, messages, device)
        v2_by_layer = {
            str(li): profile["layers"][li]["v2"]
            for li in sorted(profile["layers"])
        }
        ratio_by_layer = {
            str(li): profile["layers"][li]["ratio"]
            for li in sorted(profile["layers"])
        }

        turns.append({
            "turn": turn_i,
            "v2_by_layer": v2_by_layer,
            "ratio_by_layer": ratio_by_layer,
            "gen_H": profile["mean_entropy"],
            "generated_text": profile["generated_text"],
            "input_tokens": profile["input_tokens"],
        })

        messages.append({"role": "assistant", "content": profile["generated_text"]})
        follow = FOLLOW_UPS[(turn_i + probe_idx) % len(FOLLOW_UPS)]
        messages.append({"role": "user", "content": follow})

        print(f"      Turn {turn_i}: H={profile['mean_entropy']:.3f}, "
              f"tokens={profile['input_tokens']}")

    return {"probe_idx": probe_idx, "trial": trial_label, "turns": turns}


def concentration_profile(trials, turn_idx):
    layers = sorted(trials[0]['turns'][0]['v2_by_layer'].keys(), key=int)
    result = {}
    for layer in layers:
        vectors = []
        for trial in trials:
            if turn_idx < len(trial['turns']):
                v = np.array(trial['turns'][turn_idx]['v2_by_layer'][layer])
                norm = np.linalg.norm(v)
                if norm > 0:
                    vectors.append(v / norm)
        if len(vectors) < 2:
            result[int(layer)] = 1.0
            continue
        sims = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sims.append(np.dot(vectors[i], vectors[j]))
        result[int(layer)] = float(np.mean(sims))
    return result


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"MLP Gating Experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Trials: {N_TRIALS}, Turns: {N_TURNS}")
    print(f"Ablation layers: L{ABLATION_LAYERS[0]}-L{ABLATION_LAYERS[-1]}")
    print(f"Start: {datetime.now().isoformat()}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    device = next(model.parameters()).device

    results = {
        "experiment": "mlp_gating",
        "model": MODEL_NAME,
        "n_turns": N_TURNS,
        "n_trials": N_TRIALS,
        "ablation_layers": ABLATION_LAYERS,
        "timestamp": datetime.now().isoformat(),
    }

    # Run intact baseline first
    print(f"{'='*60}")
    print(f"  CONDITION: intact (no ablation)")
    print(f"{'='*60}")
    trials = []
    for trial_i in range(N_TRIALS):
        probe_idx = trial_i % len(PROBES)
        print(f"  Trial {trial_i+1}/{N_TRIALS} (probe {probe_idx}):")
        trial_data = run_multiturn(model, tokenizer, device, probe_idx, f"intact_t{trial_i}")
        trials.append(trial_data)
    results["intact"] = {"on_policy": trials}

    intact_profile = concentration_profile(trials, turn_idx=2)
    print(f"\n  Intact concentration at turn 2 (relay zone):")
    for l in range(18, 33):
        print(f"    L{l:2d}: {intact_profile.get(l, 0):+.3f}")

    # Run ablation conditions
    for ablation_layer in ABLATION_LAYERS:
        for component in ["mlp", "attn"]:
            cond_name = f"zero_{component}_L{ablation_layer}"
            print(f"\n{'='*60}")
            print(f"  CONDITION: {cond_name}")
            print(f"{'='*60}")

            hook = AblationHook(ablation_layer, component)
            hook.attach(model)

            trials = []
            for trial_i in range(N_TRIALS):
                probe_idx = trial_i % len(PROBES)
                print(f"  Trial {trial_i+1}/{N_TRIALS} (probe {probe_idx}):")
                trial_data = run_multiturn(
                    model, tokenizer, device, probe_idx,
                    f"{cond_name}_t{trial_i}"
                )
                trials.append(trial_data)
            results[cond_name] = {"on_policy": trials}

            hook.detach()

            abl_profile = concentration_profile(trials, turn_idx=2)
            print(f"\n  Concentration at turn 2 (relay zone):")
            for l in range(18, 33):
                intact_val = intact_profile.get(l, 0)
                abl_val = abl_profile.get(l, 0)
                delta = abl_val - intact_val
                tag = " ◄ DISRUPTED" if abs(delta) > 0.2 else ""
                print(f"    L{l:2d}: {abl_val:+.3f} (Δ={delta:+.3f}){tag}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = RESULTS_DIR / f"exp_mlp_gating_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary — average disruption per component type
    print(f"\n{'='*60}")
    print("SUMMARY — Average concentration disruption by component")
    print(f"{'='*60}")
    for component in ["mlp", "attn"]:
        disruptions = []
        for ablation_layer in ABLATION_LAYERS:
            cond_name = f"zero_{component}_L{ablation_layer}"
            trials = results[cond_name]["on_policy"]
            abl_profile = concentration_profile(trials, turn_idx=2)
            for l in range(20, 32):
                delta = abl_profile.get(l, 0) - intact_profile.get(l, 0)
                disruptions.append(abs(delta))
        mean_d = np.mean(disruptions) if disruptions else 0
        print(f"  {component}: mean |Δ| = {mean_d:.4f}")

    for ablation_layer in ABLATION_LAYERS:
        mlp_cond = f"zero_mlp_L{ablation_layer}"
        attn_cond = f"zero_attn_L{ablation_layer}"
        mlp_trials = results[mlp_cond]["on_policy"]
        attn_trials = results[attn_cond]["on_policy"]
        mlp_p = concentration_profile(mlp_trials, turn_idx=2)
        attn_p = concentration_profile(attn_trials, turn_idx=2)
        mlp_d = np.mean([abs(mlp_p.get(l,0) - intact_profile.get(l,0)) for l in range(20,32)])
        attn_d = np.mean([abs(attn_p.get(l,0) - intact_profile.get(l,0)) for l in range(20,32)])
        carrier = "MLP" if mlp_d > attn_d else "ATTN" if attn_d > mlp_d else "TIED"
        print(f"  L{ablation_layer}: MLP |Δ|={mlp_d:.4f}  ATTN |Δ|={attn_d:.4f}  → {carrier}")

    print(f"\nFinished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
