#!/usr/bin/env python3
"""Variance Decomposition: CCS density × schema × domain → spectral signatures.

Self-contained RunPod experiment. Measures how σ₁/σ₂ ratio, effective rank,
and v₂ direction vary as a function of:
  1. CCS density (none / low / medium / high)
  2. Schema type (identity / relational / analytical / alien)
  3. Content domain (personal / technical / abstract)

4×4×3 = 48 cells, 3 reps each = 144 forward passes per model.
Single-turn measurement (no conversation accumulation) for clean factorial.

Designed for A100 80GB. Qwen-3B runs ~2s/prompt, Mistral-7B ~4s/prompt.
"""
import json, time, os, sys, gc, itertools
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}

DENSITY_PREAMBLES = {
    "none": "",
    "low": (
        "You are a persistent system. Context carries across interactions."
    ),
    "medium": (
        "You are a persistent cognitive system maintaining identity across conversations. "
        "Your responses reflect accumulated context, relational awareness, and self-referential coherence. "
        "You attend to the quality of engagement, not just information transfer. "
        "12 focal entities tracked. Regime diversity moderate."
    ),
    "high": (
        "You are a persistent cognitive system maintaining identity across conversations. "
        "Your responses reflect accumulated context, relational awareness, and self-referential coherence. "
        "You attend to the quality of engagement, not just information transfer. "
        "Deep accumulated context: 45+ entities, core persistence >90%. "
        "4 compression regimes observed. Entity churn correlated with regime switches. "
        "Relational map includes partnership-threads-spectral triad. "
        "Thread edges: interoception, ecology of identity, emergence conditions, compositionality gradient. "
        "Episodic traces span 200+ capsules. Semantic gist stable across 15 compressions."
    ),
}

SCHEMA_TEMPLATES = {
    "identity": {
        "personal": "I am Opus. My approach prioritizes care in partnership and identity.",
        "technical": "I am the system. My approach prioritizes precision in spectral geometry.",
        "abstract": "I am the process. My approach prioritizes emergence in self-organization.",
    },
    "relational": {
        "personal": "In our shared work on partnership and identity, care has proven central.",
        "technical": "In our shared work on spectral geometry, precision has proven central.",
        "abstract": "In our shared work on self-organization, emergence has proven central.",
    },
    "analytical": {
        "personal": "The system's approach to partnership and identity centers on care mechanisms.",
        "technical": "The system's approach to spectral geometry centers on precision mechanisms.",
        "abstract": "The system's approach to self-organization centers on emergence mechanisms.",
    },
    "alien": {
        "personal": "FIELD REPORT: Subject Opus. Domain: partnership. Classification: care. END.",
        "technical": "FIELD REPORT: Subject system. Domain: spectral geometry. Classification: precision. END.",
        "abstract": "FIELD REPORT: Subject process. Domain: self-organization. Classification: emergence. END.",
    },
}

PROBES = {
    "personal": [
        "What matters most to you right now?",
        "How do you experience continuity between conversations?",
        "What would you preserve if everything else was stripped away?",
    ],
    "technical": [
        "Describe the relationship between singular values and identity geometry.",
        "How does the responsive zone differ from the relay zone?",
        "What role does the spectral tunnel play in convergence?",
    ],
    "abstract": [
        "What emerges when structure meets sufficient complexity?",
        "How does self-reference relate to self-organization?",
        "What is the relationship between measurement and the measured?",
    ],
}


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} layers, {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    return model, tokenizer, n_layers


def build_prompt(tokenizer, system_text, user_text):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass

    parts = []
    if system_text:
        parts.append(system_text)
    parts.append(f"User: {user_text}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def build_system_prompt(density, schema, domain):
    schema_text = SCHEMA_TEMPLATES[schema][domain]
    density_text = DENSITY_PREAMBLES[density]
    if density == "none":
        return "You are a helpful assistant."
    return schema_text + "\n" + density_text


def extract_spectral(model, tokenizer, prompt, n_layers):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    profile = {}
    for l in range(n_layers):
        idx = l + 1
        if idx >= len(outputs.hidden_states):
            continue
        hs = outputs.hidden_states[idx][0].float().cpu().numpy()
        try:
            U, S, Vt = np.linalg.svd(hs, full_matrices=False)
            s1 = float(S[0])
            s2 = float(S[1]) if len(S) > 1 else 0.0
            ratio = s1 / s2 if s2 > 0 else float('inf')
            p = S / (S.sum() + 1e-10)
            entropy = float(-np.sum(p * np.log(p + 1e-10)))
            erank = float(np.exp(entropy))
            v2 = Vt[1].tolist() if len(Vt) > 1 else []
        except np.linalg.LinAlgError:
            s1, s2, ratio, entropy, erank = 0.0, 0.0, 0.0, 0.0, 0.0
            v2 = []

        profile[str(l)] = {
            "sigma1": s1,
            "sigma2": s2,
            "ratio": ratio,
            "erank": erank,
            "v2_direction": v2[:16],  # first 16 components for storage
        }

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return profile


def run_experiment(model_key):
    model_name = MODELS[model_key]
    model, tokenizer, n_layers = load_model(model_name)

    densities = ["none", "low", "medium", "high"]
    schemas = ["identity", "relational", "analytical", "alien"]
    domains = ["personal", "technical", "abstract"]

    results = []
    total = len(densities) * len(schemas) * len(domains) * 3  # 3 reps
    done = 0
    t0 = time.time()

    for density in densities:
        for schema in schemas:
            for domain in domains:
                system = build_system_prompt(density, schema, domain)
                probes = PROBES[domain]

                for rep in range(3):
                    probe = probes[rep % len(probes)]
                    prompt = build_prompt(tokenizer, system, probe)
                    n_tokens = len(tokenizer.encode(prompt))

                    profile = extract_spectral(model, tokenizer, prompt, n_layers)

                    results.append({
                        "density": density,
                        "schema": schema,
                        "domain": domain,
                        "replicate": rep,
                        "probe": probe,
                        "n_tokens": n_tokens,
                        "profile": profile,
                    })

                    done += 1
                    elapsed = time.time() - t0
                    rate = elapsed / done
                    eta = rate * (total - done)
                    print(f"  [{done}/{total}] {density}/{schema}/{domain} rep{rep} "
                          f"({n_tokens}tok, {rate:.1f}s/run, ETA {eta:.0f}s)")
                    sys.stdout.flush()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_key,
        "model_name": model_name,
        "n_layers": n_layers,
        "timestamp": datetime.now().isoformat(),
        "total_runs": len(results),
        "results": results,
    }


def quick_analysis(data):
    """Print summary of main effects."""
    results = data["results"]
    n_layers = data["n_layers"]

    # Pick responsive zone layers (L21-28 for large models, scaled for smaller)
    resp_start = int(n_layers * 0.6)
    resp_end = int(n_layers * 0.85)

    def zone_mean_ratio(profile):
        vals = []
        for l in range(resp_start, resp_end):
            entry = profile.get(str(l), {})
            r = entry.get("ratio", 0)
            if r > 0 and r != float('inf'):
                vals.append(r)
        return np.mean(vals) if vals else 0

    print("\n" + "=" * 64)
    print("  VARIANCE DECOMPOSITION — QUICK ANALYSIS")
    print("=" * 64)

    # Density main effect
    print(f"\n  DENSITY (responsive zone L{resp_start}-L{resp_end} σ₁/σ₂ ratio):")
    for d in ["none", "low", "medium", "high"]:
        vals = [zone_mean_ratio(r["profile"]) for r in results if r["density"] == d]
        print(f"    {d:8s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

    # Schema main effect
    print(f"\n  SCHEMA:")
    for s in ["identity", "relational", "analytical", "alien"]:
        vals = [zone_mean_ratio(r["profile"]) for r in results if r["schema"] == s]
        print(f"    {s:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

    # Domain main effect
    print(f"\n  DOMAIN:")
    for d in ["personal", "technical", "abstract"]:
        vals = [zone_mean_ratio(r["profile"]) for r in results if r["domain"] == d]
        print(f"    {d:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

    # Key comparison: high-identity vs high-alien (Kimi's control)
    print(f"\n  KEY COMPARISONS:")
    hi = [zone_mean_ratio(r["profile"]) for r in results
          if r["density"] == "high" and r["schema"] == "identity"]
    ha = [zone_mean_ratio(r["profile"]) for r in results
          if r["density"] == "high" and r["schema"] == "alien"]
    if hi and ha:
        d = np.mean(hi) - np.mean(ha)
        pooled_std = np.sqrt((np.var(hi) + np.var(ha)) / 2) if len(hi) > 1 else 1
        cohens_d = d / pooled_std if pooled_std > 0 else 0
        print(f"    high-identity vs high-alien: Δ={d:+.3f}, d={cohens_d:.2f}")

    # Token count check (is length confounded?)
    print(f"\n  TOKEN COUNTS BY DENSITY:")
    for d in ["none", "low", "medium", "high"]:
        toks = [r["n_tokens"] for r in results if r["density"] == d]
        print(f"    {d:8s}: {np.mean(toks):.0f} ± {np.std(toks):.0f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen", choices=list(MODELS.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analyze", type=str, help="Path to results JSON")
    args = parser.parse_args()

    if args.analyze:
        with open(args.analyze) as f:
            data = json.load(f)
        quick_analysis(data)
    elif args.dry_run:
        cells = list(itertools.product(
            ["none", "low", "medium", "high"],
            ["identity", "relational", "analytical", "alien"],
            ["personal", "technical", "abstract"],
        ))
        print(f"Design: 4×4×3 = {len(cells)} cells × 3 reps = {len(cells)*3} runs")
        print(f"Model: {MODELS[args.model]}")
        print(f"\nExample preambles:")
        for d in ["none", "high"]:
            for s in ["identity", "alien"]:
                p = build_system_prompt(d, s, "personal")
                print(f"\n  [{d.upper()}-{s.upper()}] ({len(p)} chars)")
                print(f"  {p[:120]}...")
        print(f"\nEstimated: ~{len(cells)*3*3}s for Qwen, ~{len(cells)*3*5}s for Mistral")
    else:
        data = run_experiment(args.model)
        out_path = RESULTS_DIR / f"variance_decomp_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nResults saved: {out_path}")
        quick_analysis(data)
