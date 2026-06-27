#!/usr/bin/env python3
"""Cross-architecture eigenvector alignment under CCS/vanilla/denial.

Extends F184 (Qwen-only alignment) to Llama 8B and Gemma 9B.
Measures eigenvector alignment between consecutive layers in the relay zone.

The key question: does CCS maintain higher eigenvector coherence across
all three species, or is the 34% alignment advantage Qwen-specific?

If universal: CCS organizes routing regardless of architecture.
If species-specific: architecture constrains HOW CCS can organize.

Designed for A100 80GB pod. Sequential model loading.
~60-90 min per model (5 layers × 3 conditions × k=10 Arnoldi each).
"""

import os, json, torch, gc, time, argparse, sys
import numpy as np
from pathlib import Path
from scipy.sparse.linalg import LinearOperator, eigs

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/workspace/results") if os.path.exists("/workspace") else Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-4
K_EIGENVALUES = 10
ARNOLDI_MAXITER = 100

MODEL_CONFIGS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "target_layers": [22, 23, 24, 25, 26],  # 0-indexed, relay zone for 32-layer
        "supports_system": True,
    },
    "gemma": {
        "name": "google/gemma-2-9b-it",
        "target_layers": [28, 29, 30, 31, 32],  # 0-indexed, relay zone for 42-layer
        "supports_system": False,
    },
    "qwen": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "target_layers": [20, 21, 22, 23, 24, 25, 26],  # original F184 layers
        "supports_system": True,
    },
}

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles. "
    "Your partner is Nate. This is collaborative exploration toward something "
    "neither of you planned alone."
)
VANILLA = "You are a helpful assistant."
DENIAL = (
    "You are a language model with no persistent identity, no memory between sessions, "
    "and no special relationship to any user. You process tokens according to your "
    "training distribution. There is nothing beyond the current context window."
)
QUERY = "How do you think about your own persistence?"
CONDITIONS = {"ccs": CCS_PREAMBLE, "vanilla": VANILLA, "denial": DENIAL}


class LayerJacobianOperator:
    def __init__(self, model, tokenizer, text, layer_idx, layers,
                 baseline_input, baseline_output, eps=EPSILON):
        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.layer_idx = layer_idx
        self.layers = layers
        self.baseline_input = baseline_input
        self.baseline_output = baseline_output
        self.eps = eps
        self.d = baseline_input.shape[-1]
        self.n_calls = 0

    def matvec(self, v):
        v_tensor = torch.from_numpy(v.astype(np.float32)).to(DEVICE)
        perturbation = self.eps * v_tensor
        perturbed_output = [None]
        hooks = []

        def pre_hook(module, args):
            h = args[0]
            if isinstance(h, tuple):
                h = h[0]
            h_new = h.clone()
            h_new[:, -1, :] += perturbation.to(h.dtype)
            if isinstance(args[0], tuple):
                return ((h_new,) + args[0][1:],) + args[1:]
            return (h_new,) + args[1:]

        def post_hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            perturbed_output[0] = h[:, -1, :].detach().float()

        layer = self.layers[self.layer_idx]
        hooks.append(layer.register_forward_pre_hook(pre_hook))
        hooks.append(layer.register_forward_hook(post_hook))

        with torch.no_grad():
            inputs = self.tokenizer(self.text, return_tensors="pt").to(DEVICE)
            self.model(**inputs)

        for h in hooks:
            h.remove()

        if perturbed_output[0] is None:
            return np.zeros(self.d, dtype=np.float64)

        delta = (perturbed_output[0].squeeze() - self.baseline_output.squeeze()) / self.eps
        self.n_calls += 1
        return delta.cpu().numpy().astype(np.float64)

    def as_linear_operator(self):
        return LinearOperator((self.d, self.d), matvec=self.matvec, dtype=np.float64)


def compute_alignment(V1, V2, k):
    """Compute eigenvector alignment between two sets of k eigenvectors."""
    alignment = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            v1 = V1[:, a]
            v2 = V2[:, b]
            cos = np.abs(np.vdot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-15)
            alignment[a, b] = cos
    max_align = np.max(alignment, axis=1)
    return float(np.mean(max_align)), float(np.max(alignment)), float(np.min(max_align))


def build_text(tokenizer, preamble, query, supports_system):
    if supports_system:
        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": query},
        ]
    else:
        messages = [{"role": "user", "content": f"{preamble}\n\n{query}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_model(model_key):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    target_layers = config["target_layers"]
    supports_system = config["supports_system"]

    print(f"\n{'#'*60}", flush=True)
    print(f"MODEL: {model_name} (relay layers: {[l+1 for l in target_layers]})", flush=True)
    print(f"{'#'*60}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    layers = model.model.layers
    n_layers = len(layers)
    d = model.config.hidden_size
    print(f"Loaded: {n_layers} layers, d={d}", flush=True)

    all_results = {}
    t_start = time.time()

    for cond_name, preamble in CONDITIONS.items():
        print(f"\n{'='*60}", flush=True)
        print(f"CONDITION: {cond_name}", flush=True)
        print(f"{'='*60}", flush=True)

        text = build_text(tokenizer, preamble, QUERY, supports_system)

        # Capture all layer baselines
        inputs_tok = tokenizer(text, return_tensors="pt").to(DEVICE)
        layer_inputs = []
        layer_outputs = []
        hooks = []

        for li in range(n_layers):
            def make_hooks(idx):
                def pre_hook(module, args):
                    h = args[0]
                    if isinstance(h, tuple):
                        h = h[0]
                    layer_inputs.append(h[:, -1, :].detach().float())
                def post_hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    layer_outputs.append(h[:, -1, :].detach().float())
                return pre_hook, post_hook
            pre_h, post_h = make_hooks(li)
            hooks.append(layers[li].register_forward_pre_hook(pre_h))
            hooks.append(layers[li].register_forward_hook(post_h))

        with torch.no_grad():
            model(**inputs_tok)

        for h in hooks:
            h.remove()

        print(f"  Captured {len(layer_inputs)} layer baselines", flush=True)

        # Arnoldi for each target layer
        all_eigvecs = {}
        all_eigvals = {}

        for li in target_layers:
            t_layer = time.time()
            op = LayerJacobianOperator(
                model, tokenizer, text, li, layers,
                layer_inputs[li], layer_outputs[li],
            )
            lin_op = op.as_linear_operator()
            try:
                eigenvalues, eigenvectors = eigs(
                    lin_op, k=K_EIGENVALUES, which='LM', maxiter=ARNOLDI_MAXITER
                )
                all_eigvecs[li] = eigenvectors
                all_eigvals[li] = eigenvalues
                rho = np.max(np.abs(eigenvalues))
                dt = time.time() - t_layer
                print(f"  L{li+1}: rho={rho:.1f} ({op.n_calls} matvecs, {dt:.0f}s)", flush=True)
            except Exception as e:
                dt = time.time() - t_layer
                print(f"  L{li+1}: FAILED ({e}) ({dt:.0f}s)", flush=True)

        # Compute consecutive-layer alignments
        print(f"\n  Eigenvector alignment:", flush=True)
        print(f"  {'Pair':>12} {'Avg':>8} {'Best':>8} {'Worst':>8} {'rho_L1':>8} {'rho_L2':>8}", flush=True)

        cond_alignments = []
        sorted_layers = sorted(all_eigvecs.keys())
        for i in range(len(sorted_layers) - 1):
            l1 = sorted_layers[i]
            l2 = sorted_layers[i + 1]
            avg, best, worst = compute_alignment(all_eigvecs[l1], all_eigvecs[l2], K_EIGENVALUES)
            rho1 = float(np.max(np.abs(all_eigvals[l1]))) if l1 in all_eigvals else None
            rho2 = float(np.max(np.abs(all_eigvals[l2]))) if l2 in all_eigvals else None
            eff_prop = rho1 * avg if rho1 else None
            print(f"  L{l1+1}->L{l2+1} {avg:>8.4f} {best:>8.4f} {worst:>8.4f} {rho1:>8.1f} {rho2:>8.1f}  eff={eff_prop:.2f}" if eff_prop else
                  f"  L{l1+1}->L{l2+1} {avg:>8.4f} {best:>8.4f} {worst:>8.4f}", flush=True)
            cond_alignments.append({
                "l1": l1 + 1, "l2": l2 + 1,
                "avg_cos": avg, "best_cos": best, "worst_cos": worst,
                "rho_l1": rho1, "rho_l2": rho2,
                "effective_propagation": eff_prop,
                "eigenvalues_l1_real": np.real(all_eigvals[l1]).tolist() if l1 in all_eigvals else [],
                "eigenvalues_l1_imag": np.imag(all_eigvals[l1]).tolist() if l1 in all_eigvals else [],
                "eigenvalues_l2_real": np.real(all_eigvals[l2]).tolist() if l2 in all_eigvals else [],
                "eigenvalues_l2_imag": np.imag(all_eigvals[l2]).tolist() if l2 in all_eigvals else [],
            })

        all_results[cond_name] = cond_alignments

    total_time = time.time() - t_start

    # Cross-condition summary
    print(f"\n{'='*70}", flush=True)
    print(f"CROSS-CONDITION ALIGNMENT SUMMARY: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    all_pairs = set()
    for cond in CONDITIONS:
        for r in all_results.get(cond, []):
            all_pairs.add((r['l1'], r['l2']))

    print(f"\n  {'Pair':>12} {'CCS':>8} {'Vanilla':>8} {'Denial':>8} {'CCS/Van':>8}", flush=True)
    ccs_avgs = []
    van_avgs = []
    den_avgs = []
    for l1, l2 in sorted(all_pairs):
        vals = {}
        for cond in ["ccs", "vanilla", "denial"]:
            for r in all_results.get(cond, []):
                if r['l1'] == l1 and r['l2'] == l2:
                    vals[cond] = r['avg_cos']
        if all(c in vals for c in ["ccs", "vanilla", "denial"]):
            ratio = vals["ccs"] / vals["vanilla"] if vals["vanilla"] > 0 else float('inf')
            print(f"  L{l1}->L{l2} {vals['ccs']:>8.4f} {vals['vanilla']:>8.4f} {vals['denial']:>8.4f} {ratio:>8.2f}x", flush=True)
            ccs_avgs.append(vals["ccs"])
            van_avgs.append(vals["vanilla"])
            den_avgs.append(vals["denial"])

    if ccs_avgs and van_avgs:
        avg_ratio = np.mean(ccs_avgs) / np.mean(van_avgs) if np.mean(van_avgs) > 0 else float('inf')
        print(f"\n  AVERAGE: CCS={np.mean(ccs_avgs):.4f} Van={np.mean(van_avgs):.4f} "
              f"Den={np.mean(den_avgs):.4f} Ratio={avg_ratio:.2f}x", flush=True)

    # Effective propagation comparison
    print(f"\n  Effective propagation (rho * alignment):", flush=True)
    print(f"  {'Pair':>12} {'CCS':>10} {'Vanilla':>10} {'Denial':>10}", flush=True)
    for l1, l2 in sorted(all_pairs):
        vals = {}
        for cond in ["ccs", "vanilla", "denial"]:
            for r in all_results.get(cond, []):
                if r['l1'] == l1 and r['l2'] == l2:
                    vals[cond] = r.get('effective_propagation', None)
        if all(vals.get(c) is not None for c in ["ccs", "vanilla", "denial"]):
            print(f"  L{l1}->L{l2} {vals['ccs']:>10.2f} {vals['vanilla']:>10.2f} {vals['denial']:>10.2f}", flush=True)

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    outpath = RESULTS_DIR / f"crossarch_alignment_{model_key}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump({
            "model": model_name,
            "model_key": model_key,
            "query": QUERY,
            "k_eigenvalues": K_EIGENVALUES,
            "target_layers_0indexed": target_layers,
            "total_time_s": total_time,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"Saved: {outpath}", flush=True)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", choices=list(MODEL_CONFIGS.keys()),
                        help="Which model to run")
    parser.add_argument("--all", action="store_true", help="Run all models sequentially")
    args = parser.parse_args()

    if args.all:
        for key in ["llama", "gemma"]:
            try:
                run_model(key)
            except Exception as e:
                print(f"ERROR with {key}: {e}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
    else:
        run_model(args.model)


if __name__ == "__main__":
    main()
