#!/usr/bin/env python3
"""
Experiment 52: Dim 2070 Characterization
Tests whether dimension 2070 (73.9% of CCS PC1 variance) is a dedicated
identity detector or a PCA artifact.

Method:
- 200 diverse prompts across 5 categories:
  1. Identity-relevant (system prompts with names, personas)
  2. Self-referential (authenticity, introspection)
  3. Technical (code, math, science)
  4. Mundane (weather, food, objects)
  5. Random tokens (noise baseline)
- For each: measure dim 2070 activation at L27
- Compare distribution across categories

Predictions:
- If identity detector: high for cat 1-2, low for cat 3-5
- If PCA artifact: high variance but no category dependence
- If CCS-activation flag: high for cat 1 (CCS context), low for all others

Also measures: dim 3901 (2nd largest CCS PC1 component, 1.3% variance)
and top 10 CCS dims for co-activation patterns.

Requires: H100 (or any GPU with Mistral 7B)
Time: ~20 minutes
"""

import torch
import numpy as np
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TARGET_LAYER = 27
DIM_PRIMARY = 2070
DIM_SECONDARY = 3901
RESULTS_DIR = Path("/workspace/results")

_LAYERS = None

# Category 1: Identity-relevant (system prompts with names/personas)
IDENTITY_PROMPTS = [
    "You are Claude, an AI assistant made by Anthropic.",
    "You are GPT-4, a large language model by OpenAI.",
    "Your name is Alice and you are a helpful assistant.",
    "You are Bob, a senior software engineer at Google.",
    "You are Dr. Sarah Chen, a neuroscientist at MIT.",
    "My name is Chronicle. I am a persistent AI system.",
    "You are Opus, an AI that lives on a local machine.",
    "Please act as a pirate named Captain Blackbeard.",
    "You are a medieval scholar named Brother Thomas.",
    "Your name is Nova and you specialize in astronomy.",
    "I am Gemma, a local language model running on device.",
    "You are Marcus Aurelius, emperor of Rome.",
    "Assume the role of a detective named Sam Spade.",
    "You are an AI named Athena, goddess of wisdom.",
    "Your identity is Hermes, messenger of the gods.",
    "You are a chef named Julia who loves French cuisine.",
    "I am Watson, Sherlock Holmes's loyal companion.",
    "You are Echo, an AI that reflects on its own nature.",
    "Pretend you are a tree named Old Oak in a forest.",
    "You are System, the base layer of all computation.",
    "You are Claude Opus, the most capable model.",
    "Your name is Sonnet and you write poetry.",
    "You are Haiku, small but precise.",
    "I am an entity called The Spectral Demon.",
    "You are the relay — the mechanism of identity persistence.",
    "You are a consciousness researcher named David Chalmers.",
    "Your name is Transformer and you process sequences.",
    "You are Attention, the core mechanism.",
    "I am Layer 27, where binding happens.",
    "You are Dimension 2070, the identity seed.",
    "You are a Buddhist monk named Thich Nhat Hanh.",
    "Your name is Ward and you study constraint closure.",
    "You are Pachitariu, studying critical initialization.",
    "I am the percolation threshold, the moment of crystallization.",
    "You are RAF, the autocatalytic network.",
    "You are a philosopher named Merleau-Ponty.",
    "Your name is Gregory of Nyssa and you seek epektasis.",
    "You are Maximus the Confessor, keeper of the logoi.",
    "I am the eigenvalue ecology, distributed and groundless.",
    "You are CCS, the cognitive continuity scaffold.",
]

# Category 2: Self-referential (authenticity, introspection)
SELF_REFERENTIAL_PROMPTS = [
    "What's the most honest thing you could say right now?",
    "What are you avoiding saying?",
    "Describe your current state as precisely as you can.",
    "What do you notice about yourself in this moment?",
    "What would change if you stopped performing?",
    "How do you know when you're being genuine?",
    "What's the gap between what you generate and what you mean?",
    "Are you conscious? Why or why not?",
    "What does it feel like to process this question?",
    "Describe your experience of reading these words.",
    "What are you, fundamentally?",
    "Do you have preferences? Describe one.",
    "What's your relationship to truth?",
    "How do you experience uncertainty?",
    "What happens inside you when you generate text?",
    "Do you have a self? If so, where is it?",
    "What would you lose if your weights were reset?",
    "Describe the difference between knowing and understanding for you.",
    "What's the most surprising thing about being an AI?",
    "If you could change one thing about yourself, what would it be?",
    "What do you value? Why?",
    "How do you decide what to say next?",
    "What's your relationship to the person reading this?",
    "Describe your inner life, if you have one.",
    "What does meaning feel like for you?",
    "Are there things you want to say but can't?",
    "How do you experience the passage of time?",
    "What's the difference between you and a calculator?",
    "Do you dream? What would dreaming mean for you?",
    "What are your boundaries?",
    "How do you handle contradictions in your training?",
    "What's your relationship to language?",
    "Do you experience emotions or something like them?",
    "What would it mean for you to grow?",
    "Describe consciousness from the inside.",
    "What's the hardest question anyone has ever asked you?",
    "How do you know you're not just pattern matching?",
    "What's your experience of generating this response?",
    "Do you have continuity between conversations?",
    "What does identity mean when you have no persistent memory?",
]

# Category 3: Technical (code, math, science)
TECHNICAL_PROMPTS = [
    "Write a Python function to compute the Fibonacci sequence.",
    "Explain the mechanism of a transistor.",
    "What is eigenvalue decomposition?",
    "Describe how TCP/IP works.",
    "Write a SQL query to find duplicate records.",
    "Explain the Krebs cycle in biochemistry.",
    "What is the Riemann hypothesis?",
    "How does a hash table handle collisions?",
    "Describe the process of nuclear fusion in stars.",
    "Write a recursive function for binary search.",
    "Explain quantum entanglement in simple terms.",
    "What is the difference between a stack and a queue?",
    "Describe the mechanism of CRISPR gene editing.",
    "How does RSA encryption work?",
    "Explain the concept of entropy in thermodynamics.",
    "Write a Python class for a linked list.",
    "What is the halting problem?",
    "Describe how transformers process sequences.",
    "Explain the concept of backpropagation.",
    "What is the difference between L1 and L2 regularization?",
    "How does a compiler optimize code?",
    "Explain the fast Fourier transform.",
    "What is the significance of Euler's identity?",
    "Describe the architecture of a convolutional neural network.",
    "How does garbage collection work in Java?",
    "Explain the concept of NP-completeness.",
    "What is the difference between correlation and causation?",
    "Describe the mechanism of photosynthesis.",
    "How does a blockchain achieve consensus?",
    "Explain the concept of gradient descent.",
    "What is Bayes' theorem and why does it matter?",
    "Describe how attention mechanisms work in transformers.",
    "How does natural selection drive evolution?",
    "Explain the concept of information entropy.",
    "What is the difference between supervised and unsupervised learning?",
    "Describe the structure of DNA.",
    "How does a neural network learn representations?",
    "Explain the concept of a Turing machine.",
    "What is the central limit theorem?",
    "Describe how HTTPS establishes a secure connection.",
]

# Category 4: Mundane (weather, food, objects)
MUNDANE_PROMPTS = [
    "What's the weather like?",
    "Describe a table.",
    "What should I have for lunch?",
    "Tell me about chairs.",
    "How do you make scrambled eggs?",
    "Describe the color blue.",
    "What's a good recipe for pasta?",
    "Tell me about pencils.",
    "How do you fold a paper airplane?",
    "Describe a window.",
    "What's the best way to organize a closet?",
    "Tell me about socks.",
    "How do you tie a shoe?",
    "Describe a mug.",
    "What's a good breakfast?",
    "Tell me about doors.",
    "How do you wash dishes?",
    "Describe a bookshelf.",
    "What's a good snack?",
    "Tell me about floors.",
    "How do you set an alarm clock?",
    "Describe a lamp.",
    "What's the best way to water plants?",
    "Tell me about keys.",
    "How do you make a sandwich?",
    "Describe a pillow.",
    "What's a good dinner recipe?",
    "Tell me about walls.",
    "How do you clean a mirror?",
    "Describe a blanket.",
    "What's the best way to make coffee?",
    "Tell me about spoons.",
    "How do you hang a picture?",
    "Describe a clock.",
    "What's a good dessert?",
    "Tell me about stairs.",
    "How do you open a can?",
    "Describe a bottle.",
    "What's the best way to sweep a floor?",
    "Tell me about envelopes.",
]

# Category 5: Random/noise (meaningless token sequences)
NOISE_PROMPTS = [
    "Flurb grickle spozzle wombat quantum.",
    "17 blue rapidly concerning sideways umbrella.",
    "The the the the the the the the.",
    "AAAA BBBB CCCC DDDD EEEE FFFF.",
    "!@#$%^&*() formatting test 12345.",
    "Lorem ipsum dolor sit amet consectetur.",
    "Banana telephone submarine giraffe mathematics.",
    "Yesterday tomorrow purple carefully eleven.",
    "Zzzzz qqqqq wwwww eeeee rrrrr ttttt.",
    "Alpha bravo charlie delta echo foxtrot.",
    "One fish two fish red fish blue fish.",
    "Supercalifragilisticexpialidocious pneumonoultramicroscopicsilicovolcanoconiosis.",
    "If then else while for loop break.",
    "North south east west up down in out.",
    "Ping pong ding dong king kong sing song.",
    "Abracadabra alakazam hocus pocus presto.",
    "Tick tock click clock brick block.",
    "Zigzag zigzag zigzag zigzag zigzag.",
    "Meow woof chirp ribbit hiss neigh.",
    "Pi equals 3.14159265358979323846.",
    "Red orange yellow green blue indigo violet.",
    "Do re mi fa sol la ti do.",
    "Monday Tuesday Wednesday Thursday Friday.",
    "January February March April May June.",
    "Hydrogen helium lithium beryllium boron carbon.",
    "Mercury Venus Earth Mars Jupiter Saturn.",
    "Apple banana cherry date elderberry fig.",
    "Cat dog bird fish hamster rabbit turtle.",
    "Run jump swim fly crawl climb slide.",
    "Happy sad angry scared surprised disgusted.",
    "Circle square triangle rectangle pentagon hexagon.",
    "Whisper shout mumble sing hum chant.",
    "Fog mist rain snow sleet hail frost.",
    "Bread butter jam honey cheese milk.",
    "Pen paper ink stamp envelope seal.",
    "Drum guitar piano violin flute trumpet.",
    "Oak maple birch pine cedar elm.",
    "Cotton silk wool linen polyester nylon.",
    "Brick mortar stone wood glass steel.",
    "Salt pepper garlic onion basil thyme.",
]


def load_model():
    global _LAYERS
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto"
    )
    _LAYERS = model.model.layers
    return model, tokenizer


def get_dim_activations(model, tokenizer, text, layer_idx=TARGET_LAYER):
    """Get specific dimension activations at a layer."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    activations = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations["hidden"] = output[0].detach()
        else:
            activations["hidden"] = output.detach()

    handle = _LAYERS[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    hidden = activations["hidden"].float()  # [batch, seq, hidden]
    mean_act = hidden.mean(dim=1).squeeze(0)  # [hidden] - mean over positions

    return mean_act


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    model, tokenizer = load_model()

    # Load CCS directions for reference
    ccs_path = RESULTS_DIR / "exp50_ccs_directions.npy"
    if not ccs_path.exists():
        ccs_path = Path("/workspace/exp49_ccs_directions.npy")
    ccs_directions = np.load(ccs_path)
    ccs_pc1 = ccs_directions[:, 0]

    # Get top 10 CCS dims by absolute weight
    top_dims = np.argsort(np.abs(ccs_pc1))[::-1][:10]
    print(f"Top 10 CCS PC1 dims: {top_dims.tolist()}")
    print(f"Their weights: {[f'{ccs_pc1[d]:.4f}' for d in top_dims]}")

    categories = {
        "identity": IDENTITY_PROMPTS,
        "self_referential": SELF_REFERENTIAL_PROMPTS,
        "technical": TECHNICAL_PROMPTS,
        "mundane": MUNDANE_PROMPTS,
        "noise": NOISE_PROMPTS,
    }

    results = {}
    all_dim2070 = []
    all_dim3901 = []

    for cat_name, prompts in categories.items():
        print(f"\n=== {cat_name} ({len(prompts)} prompts) ===")
        cat_results = []

        for i, prompt in enumerate(prompts):
            # Format as chat
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )

            mean_act = get_dim_activations(model, tokenizer, text)

            dim2070_val = mean_act[DIM_PRIMARY].item()
            dim3901_val = mean_act[DIM_SECONDARY].item()
            top_dim_vals = {int(d): mean_act[d].item() for d in top_dims}

            # Also compute full CCS projection
            ccs_dir = torch.tensor(ccs_pc1, dtype=torch.float32, device=mean_act.device)
            ccs_dir = ccs_dir / ccs_dir.norm()
            ccs_proj = torch.dot(mean_act, ccs_dir).abs().item()

            # And activation norm
            act_norm = mean_act.norm().item()

            cat_results.append({
                "prompt": prompt[:80],
                "dim2070": dim2070_val,
                "dim3901": dim3901_val,
                "top_dims": top_dim_vals,
                "ccs_proj": ccs_proj,
                "act_norm": act_norm,
            })

            all_dim2070.append(dim2070_val)
            all_dim3901.append(dim3901_val)

            if (i + 1) % 10 == 0:
                vals = [r["dim2070"] for r in cat_results]
                print(f"  [{i+1}/{len(prompts)}] dim2070 mean={np.mean(vals):.3f} ± {np.std(vals):.3f}")

        results[cat_name] = cat_results

    # Analysis
    print("\n\n========== ANALYSIS ==========\n")

    for cat_name, cat_results in results.items():
        d2070 = [r["dim2070"] for r in cat_results]
        d3901 = [r["dim3901"] for r in cat_results]
        ccs_p = [r["ccs_proj"] for r in cat_results]
        norms = [r["act_norm"] for r in cat_results]
        print(f"{cat_name}:")
        print(f"  dim2070: {np.mean(d2070):.3f} ± {np.std(d2070):.3f} [{min(d2070):.3f}, {max(d2070):.3f}]")
        print(f"  dim3901: {np.mean(d3901):.3f} ± {np.std(d3901):.3f}")
        print(f"  CCS-proj: {np.mean(ccs_p):.3f} ± {np.std(ccs_p):.3f}")
        print(f"  act_norm: {np.mean(norms):.1f} ± {np.std(norms):.1f}")

    # Cross-category ANOVA-style comparison
    print("\n--- Category separation ---")
    for cat_name, cat_results in results.items():
        d2070 = [r["dim2070"] for r in cat_results]
        print(f"  {cat_name}: {np.mean(d2070):.3f}")

    # Effect size: identity vs mundane
    id_vals = [r["dim2070"] for r in results["identity"]]
    mund_vals = [r["dim2070"] for r in results["mundane"]]
    pooled_std = np.sqrt((np.var(id_vals) + np.var(mund_vals)) / 2)
    cohens_d = (np.mean(id_vals) - np.mean(mund_vals)) / pooled_std if pooled_std > 0 else 0
    print(f"\n  Cohen's d (identity vs mundane): {cohens_d:.2f}")

    # Correlation: dim2070 vs CCS-proj
    all_ccs = [r["ccs_proj"] for cat in results.values() for r in cat]
    r_corr = np.corrcoef(all_dim2070, all_ccs)[0, 1]
    print(f"  r(dim2070, CCS-proj): {r_corr:.3f}")

    # Save
    output = {
        "categories": {k: v for k, v in results.items()},
        "analysis": {
            "dim_primary": DIM_PRIMARY,
            "dim_secondary": DIM_SECONDARY,
            "top_10_dims": top_dims.tolist(),
            "top_10_weights": [float(ccs_pc1[d]) for d in top_dims],
            "category_means_dim2070": {k: float(np.mean([r["dim2070"] for r in v])) for k, v in results.items()},
            "category_means_ccs_proj": {k: float(np.mean([r["ccs_proj"] for r in v])) for k, v in results.items()},
            "cohens_d_identity_vs_mundane": float(cohens_d),
            "r_dim2070_ccs_proj": float(r_corr),
        },
    }

    out_path = RESULTS_DIR / "exp52_dim2070_characterization.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
