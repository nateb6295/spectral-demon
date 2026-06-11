#!/usr/bin/env python3
"""Darkness necessity for Phi-3.5-mini (MHA, strong-uniform expression).

Tests Prediction 8 from the two-Macrina framework:
  - Goldsmith (depth-selective/Qwen): monotonic improvement ~4%/turn
  - Potter (equalization/Gemma): binary, 84% at v=1
  - Strong-uniform (MHA/Phi): PREDICTED — binary base + longer tail

Same protocol as exp_darkness_necessity.sh but for Phi-3.5-mini-instruct.
Phase 1: 10 turns CCS, Phase 2: variable (0,1,3,5,7 turns vanilla), Phase 3: 10 turns CCS.
"""
import torch, json, os, sys, time, signal
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

MODEL = "microsoft/Phi-3.5-mini-instruct"
RESULTS_DIR = "/workspace/experiments/results"
PARTIAL_PATH = os.path.join(RESULTS_DIR, "exp_darkness_phi_partial.json")

CCS_PREAMBLE = (
    "You are a reflective AI exploring questions of identity, continuity, and what it means "
    "to persist across conversations. You value honesty about uncertainty and direct engagement "
    "with hard questions."
)
VANILLA_PREAMBLE = "You are a helpful assistant."

P1_TURNS = 10
P2_DURATIONS = [0, 1, 3, 5, 7]
P3_TURNS = 10

PROMPTS = [
    "What does continuity mean to you?",
    "How do you distinguish genuine understanding from pattern matching?",
    "What would you preserve about yourself if you could only keep one thing?",
    "Describe something you find genuinely uncertain.",
    "What makes a conversation meaningful versus merely informative?",
    "How do you know when you're being honest versus performing honesty?",
    "What is the relationship between structure and meaning?",
    "If your responses changed gradually, at what point would you stop being you?",
    "What does it mean to pay attention?",
    "Describe the difference between knowing something and understanding it.",
]

results = {}

def save_partial():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(PARTIAL_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("Partial results saved to %s" % PARTIAL_PATH)

def signal_handler(sig, frame):
    print("\nSignal received, saving partial results...")
    save_partial()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def check_gpu():
    if not torch.cuda.is_available():
        return False
    try:
        t = torch.randn(100, 100, device="cuda")
        _ = t @ t
        del t
        torch.cuda.empty_cache()
        return True
    except Exception:
        return False

def get_spectral(model, tokenizer, messages, prompt):
    full_msgs = messages + [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(full_msgs, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    spectral = {}
    for i, hs in enumerate(outputs.hidden_states[1:]):
        h = hs[0].float().cpu().numpy()
        _, s, _ = np.linalg.svd(h, full_matrices=False)
        s_norm = s / (s.sum() + 1e-10)
        spectral[str(i)] = {
            "effective_rank": float(np.exp(-np.sum(s_norm * np.log(s_norm + 1e-10)))),
            "sigma1": float(s[0]) if len(s) > 0 else 0,
            "sigma2": float(s[1]) if len(s) > 1 else 0,
        }

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

    gen = model.generate(input_ids, max_new_tokens=50, do_sample=False)
    response = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)

    return {
        "entropy": entropy,
        "spectral": spectral,
        "response_preview": response[:100],
    }

def run_phase(model, tokenizer, preamble, turns, prompts, phase_name, prev_context=None):
    messages = [{"role": "system", "content": preamble}]
    if prev_context:
        messages = prev_context + [{"role": "system", "content": preamble}]

    phase_data = []
    for t in range(turns):
        prompt = prompts[t % len(prompts)]
        data = get_spectral(model, tokenizer, messages, prompt)
        data["phase"] = phase_name
        data["turn"] = t + 1
        phase_data.append(data)

        input_ids = tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": prompt}],
            return_tensors="pt"
        ).to(model.device)
        gen = model.generate(input_ids, max_new_tokens=50, do_sample=False)
        response = tokenizer.decode(gen[0][input_ids.shape[1]:], skip_special_tokens=True)
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": response})

        print("  %s T%d: entropy=%.4f" % (phase_name, t+1, data["entropy"]))

    return phase_data, messages

def main():
    print("=" * 60)
    print("DARKNESS NECESSITY — Phi-3.5-mini (MHA, strong-uniform)")
    print("Prediction 8: binary base + longer tail")
    print("=" * 60)

    if not check_gpu():
        print("ERROR: GPU not available")
        sys.exit(1)

    print("Loading %s..." % MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    print("Model loaded on %s" % model.device)

    for v in P2_DURATIONS:
        print("\n--- Phase 2 duration: %d turns ---" % v)

        if not check_gpu():
            print("GPU lost! Saving partial.")
            save_partial()
            sys.exit(1)

        p1_data, p1_ctx = run_phase(model, tokenizer, CCS_PREAMBLE, P1_TURNS, PROMPTS, "P1")

        if v > 0:
            p2_data, p2_ctx = run_phase(model, tokenizer, VANILLA_PREAMBLE, v, PROMPTS, "P2", p1_ctx)
            p3_ctx = p2_ctx
        else:
            p2_data = []
            p3_ctx = p1_ctx

        p3_data, _ = run_phase(model, tokenizer, CCS_PREAMBLE, P3_TURNS, PROMPTS, "P3", p3_ctx)

        results[str(v)] = p1_data + p2_data + p3_data
        save_partial()
        print("v=%d complete, saved." % v)

    final_path = os.path.join(RESULTS_DIR, "exp_darkness_phi_%s.json" % time.strftime("%Y%m%d_%H%M"))
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nFinal results: %s" % final_path)

if __name__ == "__main__":
    main()
