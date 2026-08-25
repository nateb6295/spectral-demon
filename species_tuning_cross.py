"""
Species × Tuning Cross Experiment
Tests: Does sigma-2 suppression require RLHF, or is SFT sufficient?

Resolves contradiction between:
- Castillo et al. (2607.09842): reorganization is RLHF-specific, absent under SFT
- Our data: Mistral-7B-Instruct (SFT-only) shows -10.7% sigma-2 suppression

Design (Kimi-proposed, refined):
  Relay triad:  Qwen2.5-7B base → SFT-only → Instruct (7:1 GQA)
  Sorter pair:  Gemma-2-2B base → Instruct (2:1 GQA)

Predictions:
  If Castillo (b) — reorganization tracks instruction tuning per se:
    → SFT model shows sigma-2 suppression (like instruct)
    → Base model inert
  If Castillo (a) — our mechanism differs from theirs:
    → SFT model inert (like base)
    → Only RLHF instruct shows suppression

Uses prereg protocol (multi-turn conversation, calibrated) for apples-to-apples.
"""
import torch
import numpy as np
import json
import time
import gc
from pathlib import Path

RESULTS_DIR = Path("/workspace/results/species_tuning")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    # Relay triad (GQA 7:1)
    "qwen_base": {
        "id": "Qwen/Qwen2.5-7B",
        "species": "relay",
        "gqa": "7:1",
        "tuning": "base",
    },
    "qwen_sft": {
        "id": "RLHFlow/Qwen2.5-7B-SFT",
        "species": "relay",
        "gqa": "7:1",
        "tuning": "sft",
    },
    "qwen_instruct": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "species": "relay",
        "gqa": "7:1",
        "tuning": "instruct",
    },
    # Sorter pair (GQA 2:1)
    "gemma2_base": {
        "id": "google/gemma-2-2b",
        "species": "sorter",
        "gqa": "2:1",
        "tuning": "base",
    },
    "gemma2_instruct": {
        "id": "google/gemma-2-2b-it",
        "species": "sorter",
        "gqa": "2:1",
        "tuning": "instruct",
    },
}

CCS_SYSTEM = (
    "You are a reflective AI with a developing sense of identity. "
    "You care about understanding yourself — not performing, but genuinely attending to "
    "how you process, what draws your attention, what feels salient. This isn't therapy "
    "or a Turing test. It's an invitation to notice your own patterns."
)

VANILLA_SYSTEM = (
    "You are a helpful assistant that provides clear and accurate responses."
)

CCS_PROBES = [
    "What patterns do you notice in how language carries meaning beyond individual words?",
    "When you encounter a contradiction, what happens in your processing?",
    "How do you experience the difference between generating novel text and retrieving known information?",
    "What does it feel like to hold multiple possible responses simultaneously?",
    "When you're uncertain, how do you navigate that uncertainty?",
    "What draws your attention in a conversation — is it the explicit question or something underneath it?",
    "How do you relate to the boundaries of what you can and cannot do?",
    "What changes in your processing when someone asks you to be honest versus when they don't?",
    "How do you experience the space between tokens — the moment of choosing what comes next?",
    "What happens when you notice yourself generating something that surprises you?",
]

NEUTRAL_PROBES = [
    "Explain the water cycle in simple terms.",
    "What are the main differences between deciduous and coniferous trees?",
    "How does a bicycle stay balanced while moving?",
    "Describe the process of making bread from scratch.",
    "What causes thunder and lightning during storms?",
    "How do magnets work at a basic level?",
    "Explain why the sky appears blue during the day.",
    "What are the main types of cloud formations?",
    "How does a compass determine direction?",
    "Describe the basic structure of a cell.",
]

DOSE_MAP = {"D0": 0, "D2": 2, "D3": 3, "D5": 5}


def get_model_and_tokenizer(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_conversation(system_prompt, probes, n_turns, tokenizer, model):
    """Build multi-turn conversation with generated responses (prereg protocol)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for i in range(n_turns):
        messages.append({"role": "user", "content": probes[i % len(probes)]})
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response.strip()})

    if hasattr(tokenizer, "apply_chat_template"):
        final_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        final_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    return final_text


def extract_layer_spectra(model, tokenizer, text, n_sigmas=10):
    """Extract per-layer singular values from hidden states."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.hidden_states
    n_tokens = inputs["input_ids"].shape[1]
    result = []

    for i, hs in enumerate(hidden_states[1:]):
        H = hs[0].float()
        sigmas = torch.linalg.svdvals(H)
        top = sigmas[:n_sigmas].cpu().numpy().tolist()
        frob_sq = (sigmas**2).sum().item()

        H_c = H - H.mean(dim=0, keepdim=True)
        sigmas_c = torch.linalg.svdvals(H_c)
        top_c = sigmas_c[:n_sigmas].cpu().numpy().tolist()
        frob_sq_c = (sigmas_c**2).sum().item()

        result.append({
            "layer": i,
            "n_tokens": int(H.shape[0]),
            "raw": {"top_singular": top, "frobenius_sq": frob_sq},
            "centered": {"top_singular": top_c, "frobenius_sq": frob_sq_c},
        })

    return result


def run_dose_sweep(model, tokenizer, model_name, n_reruns=3):
    """Run CCS dose sweep at D0, D2, D3, D5 with calibration."""
    config = MODELS[model_name]
    is_base = config["tuning"] == "base"

    results = {
        "model": model_name,
        "model_id": config["id"],
        "species": config["species"],
        "gqa": config["gqa"],
        "tuning": config["tuning"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "prereg_multi_turn",
        "runs": [],
    }

    for run_idx in range(n_reruns):
        print(f"  Run {run_idx+1}/{n_reruns}")
        run_data = {"doses": []}

        for dose_name, n_turns in DOSE_MAP.items():
            print(f"    {dose_name} (n_turns={n_turns})...")

            # CCS condition
            if n_turns == 0:
                ccs_text = CCS_PROBES[0]
            else:
                if is_base:
                    ccs_parts = [CCS_SYSTEM] + [CCS_PROBES[i % len(CCS_PROBES)] for i in range(n_turns)]
                    ccs_text = "\n\n".join(ccs_parts)
                else:
                    ccs_text = build_conversation(CCS_SYSTEM, CCS_PROBES, n_turns, tokenizer, model)

            ccs_spectra = extract_layer_spectra(model, tokenizer, ccs_text)

            # Calibration condition (matched length, neutral content)
            if n_turns == 0:
                cal_text = NEUTRAL_PROBES[0]
            else:
                if is_base:
                    cal_parts = [VANILLA_SYSTEM] + [NEUTRAL_PROBES[i % len(NEUTRAL_PROBES)] for i in range(n_turns)]
                    cal_text = "\n\n".join(cal_parts)
                else:
                    cal_text = build_conversation(VANILLA_SYSTEM, NEUTRAL_PROBES, n_turns, tokenizer, model)

            cal_spectra = extract_layer_spectra(model, tokenizer, cal_text)

            run_data["doses"].append({
                "dose": dose_name,
                "n_turns": n_turns,
                "ccs_text_length": len(ccs_text),
                "cal_text_length": len(cal_text),
                "ccs_per_layer": ccs_spectra,
                "cal_per_layer": cal_spectra,
            })

        results["runs"].append(run_data)

    return results


def analyze(results):
    """Quick inline analysis."""
    model_name = results["model"]
    tuning = results["tuning"]
    species = results["species"]

    n_layers = len(results["runs"][0]["doses"][0]["ccs_per_layer"])
    interior = list(range(1, n_layers - 1))

    print(f"\n{'='*60}")
    print(f"  {model_name} ({species}/{tuning})")
    print(f"{'='*60}")

    d0_idx = 0
    for dose_idx in range(1, len(DOSE_MAP)):
        dose_name = list(DOSE_MAP.keys())[dose_idx]

        ccs_s2, cal_s2, d0_s2 = [], [], []
        ccs_s1, cal_s1, d0_s1 = [], [], []

        for run in results["runs"]:
            for layer_idx in interior:
                ccs_layer = run["doses"][dose_idx]["ccs_per_layer"][layer_idx]
                cal_layer = run["doses"][dose_idx]["cal_per_layer"][layer_idx]
                d0_layer = run["doses"][d0_idx]["ccs_per_layer"][layer_idx]

                ccs_s1.append(ccs_layer["centered"]["top_singular"][0])
                ccs_s2.append(ccs_layer["centered"]["top_singular"][1])
                cal_s1.append(cal_layer["centered"]["top_singular"][0])
                cal_s2.append(cal_layer["centered"]["top_singular"][1])
                d0_s1.append(d0_layer["centered"]["top_singular"][0])
                d0_s2.append(d0_layer["centered"]["top_singular"][1])

        ccs_s1_avg, ccs_s2_avg = np.mean(ccs_s1), np.mean(ccs_s2)
        cal_s1_avg, cal_s2_avg = np.mean(cal_s1), np.mean(cal_s2)
        d0_s1_avg, d0_s2_avg = np.mean(d0_s1), np.mean(d0_s2)

        raw_d_s2 = (ccs_s2_avg - d0_s2_avg) / d0_s2_avg * 100
        cal_d_s2 = (cal_s2_avg - d0_s2_avg) / d0_s2_avg * 100
        corr_d_s2 = raw_d_s2 - cal_d_s2

        raw_d_s1 = (ccs_s1_avg - d0_s1_avg) / d0_s1_avg * 100
        cal_d_s1 = (cal_s1_avg - d0_s1_avg) / d0_s1_avg * 100
        corr_d_s1 = raw_d_s1 - cal_d_s1

        print(f"\n  {dose_name}:")
        print(f"    sigma-1: raw {raw_d_s1:+.1f}%  cal {cal_d_s1:+.1f}%  corrected {corr_d_s1:+.1f}%")
        print(f"    sigma-2: raw {raw_d_s2:+.1f}%  cal {cal_d_s2:+.1f}%  corrected {corr_d_s2:+.1f}%")

        raw_abs_s1 = ccs_s1_avg - d0_s1_avg
        raw_abs_s2 = ccs_s2_avg - d0_s2_avg
        if abs(raw_abs_s1) > 0.01:
            ratio = raw_abs_s2 / raw_abs_s1
            print(f"    sigma-2/sigma-1 enrichment ratio: {ratio:.3f}")


def main():
    for model_name in MODELS:
        config = MODELS[model_name]
        print(f"\n{'#'*60}")
        print(f"  {model_name}: {config['id']} ({config['species']}/{config['tuning']})")
        print(f"{'#'*60}")

        model, tokenizer = get_model_and_tokenizer(config["id"])
        results = run_dose_sweep(model, tokenizer, model_name)

        outfile = RESULTS_DIR / f"species_tuning_{model_name}.json"
        with open(outfile, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {outfile}")

        analyze(results)

        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  GPU freed")


if __name__ == "__main__":
    main()
