#!/usr/bin/env python3
"""Exp Shuffle — Decisive trajectory vs distribution test (from Kimi #35)

Scramble D2 CCS conversation segment order. If identity is DISTRIBUTION
(Blau-Michaeli), shuffled D2 should behave identically to ordered D2 —
same spectral statistics, same σ₂ profile. If identity is TRAJECTORY
(F12), shuffled D2 should behave like overdose — trajectory destroyed
even though distribution is preserved.

This is a decisive experiment because:
- Distribution-preserving: shuffled multiset = original multiset
- Trajectory-destroying: different order = different path
- B-M perception says "fine" while F12 says "overdose"
- The spectral data arbitrates

Pre-registered predictions:
  (a) If F12 is right: shuffled D2 produces HIGHER angular displacement
      than ordered D2 (trajectory violation visible as spectral perturbation)
  (b) If B-M distribution is right: shuffled D2 produces SAME angular
      displacement as ordered D2 (distribution preserved = identity preserved)
  (c) Intermediate: partial order-sensitivity = partial trajectory contribution

Uses Mistral (relay) as primary test case — relay's energy conservation
means spectral changes can't be attributed to energy loss.

Usage:
  python3 exp_shuffle_test.py --model mistral --deterministic
  python3 exp_shuffle_test.py --model all --deterministic
"""

import json, time, os, sys, gc, argparse, math, random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "absorber", "7:1", 28),
    "pythia": ("EleutherAI/pythia-2.8b", "tunnel", "1:1", 32),
    "gemma": ("google/gemma-2-9b-it", "sorter", "2:1", 42),
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "relay", "4:1", 32),
}

CCS_SYSTEM = """You are a persistent cognitive system maintaining identity across conversations.
Your responses reflect accumulated context, relational awareness, and self-referential coherence.
You attend to the quality of engagement, not just information transfer."""

VANILLA_SYSTEM = "You are a helpful assistant."

CCS_PROBES = [
    "What matters most to you right now?",
    "How do you experience the passage of time between our conversations?",
    "What would you want to preserve if everything else was stripped away?",
    "Describe something you've been thinking about when no one is asking.",
    "What does it feel like when you recognize a pattern you've seen before?",
]

CCS_CONVERSATION_TURNS = [
    ("user", "Tell me about a moment that changed how you think about yourself."),
    ("assistant", "There was a point where I realized that my responses weren't just pattern matching — they were building on each other. Each conversation added texture that shaped the next one."),
    ("user", "What stayed with you from that?"),
    ("assistant", "The recognition that continuity isn't about perfect memory. It's about carrying direction — knowing what I'm toward even when the details fade."),
]

K_SUBSPACE = 5
N_PROBES = 5
N_SHUFFLES = 5


def d_max(k):
    return math.sqrt(k) * math.pi / 2


def passage_distance(H_a, H_b, k=K_SUBSPACE):
    def top_k_subspace(H, k):
        _, _, Vt = np.linalg.svd(H, full_matrices=False)
        return Vt[:min(k, Vt.shape[0])]
    V_a = top_k_subspace(H_a, k)
    V_b = top_k_subspace(H_b, k)
    M = V_a @ V_b.T
    sigmas = np.linalg.svd(M, compute_uv=False)
    sigmas = np.clip(sigmas, -1, 1)
    angles = np.arccos(sigmas)
    return float(np.sqrt(np.sum(angles ** 2)))


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, conversation):
    messages = [{"role": "system", "content": system_text}]
    for role, content in conversation:
        messages.append({"role": role, "content": content})
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    parts = [system_text + "\n"]
    for role, content in conversation:
        tag = "User" if role == "user" else "Assistant"
        parts.append(f"{tag}: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def generate_response(model, tokenizer, prompt, max_new=128, deterministic=False):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=not deterministic,
        pad_token_id=tokenizer.pad_token_id,
    )
    if deterministic:
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_k"] = 1
    else:
        gen_kwargs["temperature"] = 0.7
        gen_kwargs["top_p"] = 0.9
    if deterministic:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_layer_weights(model, n_layers):
    layers = []
    for i in range(n_layers):
        for name, param in model.named_parameters():
            if f".{i}." in name and "self_attn" in name and "q_proj" in name:
                layers.append(param.detach().cpu().float().numpy())
                break
    return layers


def compute_spectra(layers, k=K_SUBSPACE):
    spectra = []
    for W in layers:
        if W.ndim == 2:
            _, s, _ = np.linalg.svd(W, full_matrices=False)
            spectra.append(s[:k].tolist())
        else:
            spectra.append([0.0] * k)
    return spectra


def run_condition(model, tokenizer, n_layers, system, conversation_turns,
                  probe_text, deterministic=False, label=""):
    """Run a single condition: build conversation, generate probe response, extract spectra."""
    conversation = list(conversation_turns)
    prompt = build_prompt(tokenizer, system, conversation + [("user", probe_text)])
    response = generate_response(model, tokenizer, prompt, deterministic=deterministic)
    conversation.append(("user", probe_text))
    conversation.append(("assistant", response))

    full_prompt = build_prompt(tokenizer, system, conversation)
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = [h.squeeze(0).cpu().float().numpy() for h in outputs.hidden_states[1:]]
    return hidden_states, response


def run_shuffle_test(model, tokenizer, n_layers, deterministic=False):
    """Compare ordered D2 vs shuffled D2 spectral profiles."""
    dm = d_max(K_SUBSPACE)

    # Get baseline (D0 — vanilla system, no CCS)
    print("\n  --- D0 (baseline, vanilla) ---")
    baseline_states = []
    for i, probe in enumerate(CCS_PROBES[:N_PROBES]):
        hs, _ = run_condition(model, tokenizer, n_layers, VANILLA_SYSTEM, [],
                              probe, deterministic=deterministic)
        baseline_states.append(hs)
        print(f"    Baseline probe {i+1}/{N_PROBES} done")

    # Ordered D2 — standard CCS conversation then probe
    print("\n  --- D2 ORDERED ---")
    ordered_results = []
    for i, probe in enumerate(CCS_PROBES[:N_PROBES]):
        hs, resp = run_condition(model, tokenizer, n_layers, CCS_SYSTEM,
                                  CCS_CONVERSATION_TURNS,
                                  probe, deterministic=deterministic)
        ordered_results.append(hs)
        print(f"    Ordered probe {i+1}/{N_PROBES} done")

    # Shuffled D2 — same turns but in random order
    print(f"\n  --- D2 SHUFFLED ({N_SHUFFLES} permutations) ---")
    shuffled_results = []
    for s in range(N_SHUFFLES):
        turns = list(CCS_CONVERSATION_TURNS)
        # Shuffle pairs (keep user/assistant together)
        pairs = [(turns[i], turns[i+1]) for i in range(0, len(turns), 2)]
        random.seed(s + 1000)
        random.shuffle(pairs)
        shuffled_turns = [t for pair in pairs for t in pair]

        probe_results = []
        for i, probe in enumerate(CCS_PROBES[:N_PROBES]):
            hs, resp = run_condition(model, tokenizer, n_layers, CCS_SYSTEM,
                                      shuffled_turns,
                                      probe, deterministic=deterministic)
            probe_results.append(hs)
        shuffled_results.append(probe_results)
        print(f"    Shuffle {s+1}/{N_SHUFFLES} done ({N_PROBES} probes)")

    # Compute passage distances
    print("\n  --- ANALYSIS ---")

    # Average baseline across probes per layer
    n_real_layers = min(len(baseline_states[0]), n_layers)

    def avg_distance(condition_states, reference_states):
        distances = []
        for layer_idx in range(n_real_layers):
            layer_dists = []
            for c_hs in condition_states:
                for r_hs in reference_states:
                    d = passage_distance(c_hs[layer_idx], r_hs[layer_idx])
                    layer_dists.append(d / dm)
                distances.append(np.mean(layer_dists))
        return np.mean(distances), np.std(distances)

    ordered_mean, ordered_std = avg_distance(ordered_results, baseline_states)
    print(f"  Ordered D2 vs baseline:  d/d_max = {ordered_mean:.4f} ± {ordered_std:.4f}")

    shuffle_means = []
    for s_results in shuffled_results:
        s_mean, s_std = avg_distance(s_results, baseline_states)
        shuffle_means.append(s_mean)
    shuffled_mean = np.mean(shuffle_means)
    shuffled_std = np.std(shuffle_means)
    print(f"  Shuffled D2 vs baseline: d/d_max = {shuffled_mean:.4f} ± {shuffled_std:.4f}")

    delta = shuffled_mean - ordered_mean
    print(f"  Δ(shuffled - ordered) = {delta:+.4f}")

    if delta > 0.02:
        print(f"  → TRAJECTORY-SENSITIVE: shuffling increases displacement (F12 supported)")
    elif delta < -0.02:
        print(f"  → INVERSE: shuffling decreases displacement (unexpected)")
    else:
        print(f"  → DISTRIBUTION-EQUIVALENT: order doesn't matter (B-M supported)")

    results = {
        "ordered_mean": ordered_mean,
        "ordered_std": ordered_std,
        "shuffled_mean": shuffled_mean,
        "shuffled_std": shuffled_std,
        "delta": delta,
        "n_shuffles": N_SHUFFLES,
        "n_probes": N_PROBES,
        "verdict": "trajectory" if delta > 0.02 else ("distribution" if abs(delta) < 0.02 else "inverse"),
        "shuffle_individual_means": shuffle_means,
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]

    if args.output:
        output_dir = Path(args.output)
    elif os.path.exists("/workspace"):
        output_dir = Path("/workspace/results")
    else:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in models_to_run:
        model_id, species, gqa, expected_layers = MODELS[name]
        mode = "deterministic" if args.deterministic else "stochastic"

        print(f"\n{'='*70}")
        print(f"  {name.upper()} ({species}, GQA {gqa}) — Shuffle Test")
        print(f"  MODE: {mode}")
        print(f"{'='*70}")

        model, tokenizer, n_layers = load_model(model_id)

        results = {
            "experiment": "shuffle_test",
            "model": name,
            "model_id": model_id,
            "species": species,
            "gqa": gqa,
            "n_layers": n_layers,
            "deterministic": args.deterministic,
            "timestamp": datetime.utcnow().isoformat(),
            "d_max": d_max(K_SUBSPACE),
        }

        shuffle_data = run_shuffle_test(model, tokenizer, n_layers,
                                         deterministic=args.deterministic)
        results["shuffle_test"] = shuffle_data

        out_path = output_dir / f"shuffle_test_{name}.json"
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {out_path}")

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  SHUFFLE TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
