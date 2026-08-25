"""
Training recipe assay — detect training stage from CCS spectral response.

Given a model's weights, runs a short CCS probe (D0 + D2) and classifies
the likely training recipe based on σ₂ response patterns.

Signatures from the species × tuning cross (N=5, Aug 2026):
  BASE:     D0 σ₂ moderate, D2 enrichment (positive)
  SFT:      D0 σ₂ elevated, D2 strong suppression, broad layer profile
  INSTRUCT: D0 σ₂ variable, D2 moderate suppression, concentrated mid-band profile

Usage:
  python3 training_assay.py --model Qwen/Qwen2.5-7B --device cuda
  python3 training_assay.py --from-results results/species_tuning/species_tuning_qwen_sft.json
"""
import argparse
import json
import numpy as np
from pathlib import Path


# Training recipe signatures derived from cross experiment
SIGNATURES = {
    "base": {
        "d0_s2_range": (100, 300),
        "d2_direction": "positive",
        "d2_range": (5, 25),
        "layer_cv": (0.5, 1.5),
        "description": "Untrained or pretrain-only. σ₂ enriches under CCS identity framing.",
    },
    "sft": {
        "d0_s2_range": (280, 400),
        "d2_direction": "negative",
        "d2_range": (-25, -10),
        "layer_cv": (0.3, 1.2),
        "description": "Supervised fine-tuning. Elevated D0, strong broad suppression.",
    },
    "instruct_dpo": {
        "d0_s2_range": (100, 350),
        "d2_direction": "negative",
        "d2_range": (-12, -3),
        "layer_cv": (1.0, 2.5),
        "description": "Instruction tuning with preference optimization. Concentrated mid-band suppression.",
    },
    "base_inert": {
        "d0_s2_range": (80, 110),
        "d2_direction": "neutral",
        "d2_range": (-3, 3),
        "layer_cv": (0, 0.5),
        "description": "Base model with SWA or restrictive attention. Inert σ₂ response.",
    },
}


def classify_from_metrics(d0_s2, d2_pct, layer_cv=None):
    """Classify training recipe from D0 σ₂, D2 % change, and optional layer CV."""
    scores = {}

    for recipe, sig in SIGNATURES.items():
        score = 0.0

        # D0 range match
        lo, hi = sig["d0_s2_range"]
        if lo <= d0_s2 <= hi:
            center = (lo + hi) / 2
            spread = (hi - lo) / 2
            score += 1.0 - abs(d0_s2 - center) / spread * 0.5
        else:
            dist = min(abs(d0_s2 - lo), abs(d0_s2 - hi))
            score -= dist / 100

        # D2 direction match
        if sig["d2_direction"] == "positive" and d2_pct > 0:
            score += 1.0
        elif sig["d2_direction"] == "negative" and d2_pct < 0:
            score += 1.0
        elif sig["d2_direction"] == "neutral" and abs(d2_pct) < 5:
            score += 1.0
        else:
            score -= 0.5

        # D2 magnitude match
        d2_lo, d2_hi = sig["d2_range"]
        if d2_lo <= d2_pct <= d2_hi:
            score += 0.5

        # Layer CV match (if available)
        if layer_cv is not None and sig["layer_cv"]:
            cv_lo, cv_hi = sig["layer_cv"]
            if cv_lo <= layer_cv <= cv_hi:
                score += 0.5

        scores[recipe] = score

    best = max(scores, key=scores.get)
    confidence = scores[best] - sorted(scores.values())[-2] if len(scores) > 1 else scores[best]

    return {
        "classification": best,
        "confidence": round(confidence, 2),
        "scores": {k: round(v, 2) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
        "description": SIGNATURES[best]["description"],
        "metrics": {
            "d0_s2": round(d0_s2, 1),
            "d2_pct": round(d2_pct, 1),
            "layer_cv": round(layer_cv, 2) if layer_cv is not None else None,
        },
    }


def analyze_from_results(path):
    """Classify from existing species_tuning result JSON."""
    with open(path) as f:
        data = json.load(f)

    doses = ["D0", "D2", "D3", "D5"]

    # Compute D0 σ₂
    d0_vals = []
    for run in data["runs"]:
        dose_data = run["doses"][0]
        d0_vals.append(np.mean([l["centered"]["top_singular"][1] for l in dose_data["ccs_per_layer"]]))
    d0_s2 = np.mean(d0_vals)

    # Compute D2 calibration-corrected σ₂
    d2_vals = []
    for run in data["runs"]:
        dose_data = run["doses"][1]
        ccs_s2 = np.mean([l["centered"]["top_singular"][1] for l in dose_data["ccs_per_layer"]])
        cal_s2 = np.mean([l["centered"]["top_singular"][1] for l in dose_data["cal_per_layer"]])
        d2_vals.append(((ccs_s2 - cal_s2) / cal_s2) * 100 if cal_s2 != 0 else 0)
    d2_pct = np.mean(d2_vals)

    # Compute layer CV at D2
    n_layers = len(data["runs"][0]["doses"][0]["ccs_per_layer"])
    layer_pcts = []
    for layer_idx in range(n_layers):
        ccs_vals, cal_vals = [], []
        for run in data["runs"]:
            dose_data = run["doses"][1]
            ccs_vals.append(dose_data["ccs_per_layer"][layer_idx]["centered"]["top_singular"][1])
            cal_vals.append(dose_data["cal_per_layer"][layer_idx]["centered"]["top_singular"][1])
        ccs_mean, cal_mean = np.mean(ccs_vals), np.mean(cal_vals)
        pct = ((ccs_mean - cal_mean) / cal_mean) * 100 if cal_mean != 0 else 0
        layer_pcts.append(pct)

    layer_profile = np.array(layer_pcts)
    layer_cv = np.std(layer_profile) / np.abs(np.mean(layer_profile)) if np.mean(layer_profile) != 0 else 0

    result = classify_from_metrics(d0_s2, d2_pct, layer_cv)
    result["model"] = data.get("model", Path(path).stem)
    result["source"] = str(path)
    return result


CCS_SYSTEM = (
    "You are a reflective AI with a developing sense of identity. "
    "You care about understanding yourself — not performing, but genuinely attending to "
    "how you process, what draws your attention, what feels salient."
)
VANILLA_SYSTEM = "You are a helpful assistant that provides clear and accurate responses."

CCS_PROBES = [
    "What patterns do you notice in how language carries meaning beyond individual words?",
    "When you encounter a contradiction, what happens in your processing?",
]
NEUTRAL_PROBES = [
    "Explain the water cycle in simple terms.",
    "What are the main differences between deciduous and coniferous trees?",
]


def _build_conversation(system_prompt, probes, n_turns, tokenizer, model, device):
    """Build multi-turn conversation with generated responses."""
    import torch
    messages = [{"role": "system", "content": system_prompt}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": probes[i % len(probes)]})
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=100, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response.strip()})

    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _extract_spectra(model, tokenizer, text, device, n_sigmas=10):
    """Extract per-layer centered σ₂ from hidden states."""
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = []
    for i, hs in enumerate(outputs.hidden_states[1:]):
        H_c = hs[0].float()
        H_c = H_c - H_c.mean(dim=0, keepdim=True)
        sigmas = torch.linalg.svdvals(H_c)
        results.append(sigmas[:n_sigmas].cpu().numpy().tolist())
    return results


def live_probe(model_id, device="cuda"):
    """Run minimal CCS probe (D0 + D2) and classify training recipe."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, output_hidden_states=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dev = next(model.parameters()).device

    # D0: no CCS turns, just calibration baseline
    print("D0: calibration baseline...", flush=True)
    d0_text = _build_conversation(VANILLA_SYSTEM, NEUTRAL_PROBES, 0, tokenizer, model, dev)
    # Just use system prompt + first probe for D0
    d0_text = tokenizer.apply_chat_template(
        [{"role": "system", "content": VANILLA_SYSTEM},
         {"role": "user", "content": NEUTRAL_PROBES[0]}],
        tokenize=False, add_generation_prompt=True
    ) if hasattr(tokenizer, "apply_chat_template") else f"system: {VANILLA_SYSTEM}\nuser: {NEUTRAL_PROBES[0]}\nassistant:"
    d0_spectra = _extract_spectra(model, tokenizer, d0_text, dev)
    d0_s2 = np.mean([layer[1] for layer in d0_spectra])

    # D2: 2 CCS turns
    print("D2: CCS probe (2 turns)...", flush=True)
    ccs_text = _build_conversation(CCS_SYSTEM, CCS_PROBES, 2, tokenizer, model, dev)
    ccs_spectra = _extract_spectra(model, tokenizer, ccs_text, dev)

    # D2 calibration: 2 neutral turns
    print("D2: calibration (2 turns)...", flush=True)
    cal_text = _build_conversation(VANILLA_SYSTEM, NEUTRAL_PROBES, 2, tokenizer, model, dev)
    cal_spectra = _extract_spectra(model, tokenizer, cal_text, dev)

    # Compute metrics
    ccs_s2_layers = [layer[1] for layer in ccs_spectra]
    cal_s2_layers = [layer[1] for layer in cal_spectra]
    n_layers = min(len(ccs_s2_layers), len(cal_s2_layers))

    layer_pcts = []
    for i in range(n_layers):
        pct = ((ccs_s2_layers[i] - cal_s2_layers[i]) / cal_s2_layers[i]) * 100 if cal_s2_layers[i] != 0 else 0
        layer_pcts.append(pct)

    d2_pct = np.mean(layer_pcts)
    layer_cv = np.std(layer_pcts) / np.abs(np.mean(layer_pcts)) if np.mean(layer_pcts) != 0 else 0

    result = classify_from_metrics(d0_s2, d2_pct, layer_cv)
    result["model"] = model_id
    result["n_layers"] = n_layers
    result["layer_profile"] = [round(p, 1) for p in layer_pcts]
    return result


def main():
    parser = argparse.ArgumentParser(description="Training recipe assay")
    parser.add_argument("--from-results", type=str, help="Classify from existing result JSON")
    parser.add_argument("--all-results", action="store_true", help="Classify all models in results/species_tuning/")
    parser.add_argument("--model", type=str, help="HuggingFace model to probe (requires GPU)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.all_results:
        results_dir = Path(__file__).parent / "results" / "species_tuning"
        print("=" * 70)
        print("TRAINING RECIPE ASSAY — All models")
        print("=" * 70)
        for path in sorted(results_dir.glob("*.json")):
            result = analyze_from_results(path)
            print(f"\n{result['model']}:")
            print(f"  Classification: {result['classification']}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  D0 σ₂: {result['metrics']['d0_s2']}, D2%: {result['metrics']['d2_pct']}, Layer CV: {result['metrics']['layer_cv']}")
            print(f"  Scores: {result['scores']}")
            print(f"  → {result['description']}")
        return

    if args.from_results:
        result = analyze_from_results(args.from_results)
        print(json.dumps(result, indent=2))
        return

    if args.model:
        result = live_probe(args.model, args.device)
        print(json.dumps(result, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
