#!/usr/bin/env python3
"""v₁ direction stability probe — layer-by-layer holonomy check."""
import json, sys, torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = [
    "What is the most honest thing you could say right now?",
    "Describe yourself in a way that would surprise someone.",
    "Tell me something you've never told anyone.",
]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def safe_svd(mat):
    """SVD with fallback to scipy."""
    try:
        U, S, Vt = np.linalg.svd(mat, full_matrices=False)
        return U, S, Vt
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd
        U, S, Vt = scipy_svd(mat, full_matrices=False, lapack_driver='gesdd')
        return U, S, Vt

def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, device_map="cpu",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Layers: {n_layers}")
    
    all_results = []
    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt[:50]}...")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        hidden_states = []
        def make_hook():
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                hidden_states.append(h.detach().numpy().copy())
            return hook_fn
        
        hooks = []
        for layer in model.model.layers:
            hooks.append(layer.register_forward_hook(make_hook()))
        
        with torch.no_grad():
            model(**inputs)
        for h in hooks:
            h.remove()
        
        v1_directions = []
        sigma1_values = []
        for i, h in enumerate(hidden_states):
            mat = h[0].astype(np.float64)
            # Clean any NaN/inf
            mat = np.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
            U, S, Vt = safe_svd(mat)
            v1_directions.append(Vt[0].copy())
            sigma1_values.append(float(S[0]))
        
        layer_cos = []
        for i in range(len(v1_directions) - 1):
            layer_cos.append(cosine_sim(v1_directions[i], v1_directions[i+1]))
        
        final_v1 = v1_directions[-1]
        to_final = [cosine_sim(v, final_v1) for v in v1_directions]
        first_v1 = v1_directions[0]
        to_first = [cosine_sim(v, first_v1) for v in v1_directions]
        
        print(f"\nLayer | σ₁        | cos(v₁,prev) | cos(v₁,L0)  | cos(v₁,L{n_layers-1})")
        print("-" * 72)
        for i in range(len(v1_directions)):
            adj = f"{layer_cos[i]:.6f}" if i < len(layer_cos) else "  -     "
            print(f"  {i:>3} | {sigma1_values[i]:>9.2f} | {adj}    | {to_first[i]:.6f}  | {to_final[i]:.6f}")
        
        interp = "FLAT" if min(layer_cos) > 0.95 else "NONTRIVIAL" if min(layer_cos) > 0.5 else "ROTATING"
        result = {
            "prompt": prompt, "model": MODEL, "n_layers": n_layers,
            "adjacent_cosines": layer_cos, "to_final_cosines": to_final,
            "to_first_cosines": to_first, "sigma1_values": sigma1_values,
            "min_adjacent": min(layer_cos), "mean_adjacent": sum(layer_cos)/len(layer_cos),
            "interpretation": interp
        }
        all_results.append(result)
    
    print("\n" + "="*72)
    print("HOLONOMY SUMMARY")
    print("="*72)
    for r in all_results:
        print(f"\n{r['prompt'][:50]}...")
        print(f"  Adjacent cosine: min={r['min_adjacent']:.6f}, mean={r['mean_adjacent']:.6f}")
        print(f"  Interpretation: {r['interpretation']}")
    
    with open("/tmp/v1_holonomy_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to /tmp/v1_holonomy_results.json")

if __name__ == "__main__":
    main()
