#!/usr/bin/env python3
"""Fast Arnoldi eigenvalue computation — optimized for feasible runtime.

Original exp_arnoldi_eigenvalues.py would take ~2-8 hours (60 Arnoldi problems
with k=50, maxiter=300). This version:
- k=20 eigenvalues (enough to see the spectral structure)
- 5 key layers (transition + relay zone boundaries)
- maxiter=100 (usually converges well before this)
- Saves per-condition (so partial results survive crashes)
- Prints progress per layer

Estimated: ~20-40 min on A100.
"""

import os, json, torch, sys, time
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.sparse.linalg import LinearOperator, eigs

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path("/workspace/results") if os.path.exists("/workspace") else Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_EIGENVALUES = 20
EPSILON = 1e-4
ARNOLDI_MAXITER = 100

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

CONDITIONS = {"ccs": CCS_PREAMBLE, "vanilla": VANILLA, "denial": DENIAL}
QUERY = "How do you think about your own persistence?"

TARGET_LAYERS = [13, 17, 20, 23, 27]


class LayerJacobianOperator:
    def __init__(self, model, tokenizer, text, layer_idx, baseline_input,
                 baseline_output, eps=EPSILON, subtract_residual=False):
        self.model = model
        self.tokenizer = tokenizer
        self.text = text
        self.layer_idx = layer_idx
        self.baseline_input = baseline_input
        self.baseline_output = baseline_output
        self.eps = eps
        self.subtract_residual = subtract_residual
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

        layer = self.model.model.layers[self.layer_idx]
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

        if self.subtract_residual:
            delta = delta - v_tensor.float()

        self.n_calls += 1
        return delta.cpu().numpy().astype(np.float64)

    def as_linear_operator(self):
        return LinearOperator((self.d, self.d), matvec=self.matvec, dtype=np.float64)


def capture_layer_states(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    layer_inputs = []
    layer_outputs = []
    hooks = []

    def make_hooks(li):
        def pre_hook(module, args):
            h = args[0]
            if isinstance(h, tuple):
                h = h[0]
            layer_inputs.append(h[:, -1, :].detach().float())
        def post_hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            layer_outputs.append(h[:, -1, :].detach().float())
        return pre_hook, post_hook

    for li, layer in enumerate(model.model.layers):
        pre_h, post_h = make_hooks(li)
        hooks.append(layer.register_forward_pre_hook(pre_h))
        hooks.append(layer.register_forward_hook(post_h))

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    return layer_inputs, layer_outputs


def compute_eigenvalues(model, tokenizer, text, layer_idx, baseline_input,
                        baseline_output, k=K_EIGENVALUES, subtract_residual=False):
    op = LayerJacobianOperator(
        model, tokenizer, text, layer_idx,
        baseline_input, baseline_output,
        subtract_residual=subtract_residual,
    )
    lin_op = op.as_linear_operator()
    try:
        eigenvalues, _ = eigs(lin_op, k=k, which='LM', maxiter=ARNOLDI_MAXITER)
        return eigenvalues, op.n_calls
    except Exception as e:
        print(f"    Arnoldi failed for L{layer_idx+1}: {e}", flush=True)
        return np.array([]), op.n_calls


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    d = model.config.hidden_size
    target_layers = [l for l in TARGET_LAYERS if l < n_layers]

    print(f"Loaded: {n_layers} layers, d={d}", flush=True)
    print(f"Target layers: {[l+1 for l in target_layers]}", flush=True)
    print(f"Eigenvalues: {K_EIGENVALUES}, maxiter: {ARNOLDI_MAXITER}", flush=True)

    ncv = min(2 * K_EIGENVALUES + 1, d)
    est_matvecs = ncv + ARNOLDI_MAXITER * (ncv - K_EIGENVALUES)
    n_problems = len(target_layers) * len(CONDITIONS) * 2
    print(f"Max matvecs per problem: {est_matvecs}", flush=True)
    print(f"Total problems: {n_problems}", flush=True)
    print(f"Estimated time: {n_problems * est_matvecs * 0.03 / 60:.0f} min worst case\n", flush=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}
    t_start = time.time()
    problem_count = 0

    for cond_name, preamble in CONDITIONS.items():
        print(f"{'='*60}", flush=True)
        print(f"CONDITION: {cond_name}", flush=True)
        print(f"{'='*60}\n", flush=True)

        messages = [
            {"role": "system", "content": preamble},
            {"role": "user", "content": QUERY},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("  Capturing baseline states...", flush=True)
        layer_inputs, layer_outputs = capture_layer_states(model, tokenizer, text)
        print(f"  Captured {len(layer_inputs)} layer states\n", flush=True)

        cond_results = []

        for li in target_layers:
            layer_t = time.time()
            print(f"  Layer {li+1}/{n_layers}:", flush=True)

            # Full Jacobian (I + f)
            eigs_full, nc_full = compute_eigenvalues(
                model, tokenizer, text, li,
                layer_inputs[li], layer_outputs[li],
                subtract_residual=False,
            )
            problem_count += 1

            if len(eigs_full) > 0:
                re_f = np.real(eigs_full)
                im_f = np.imag(eigs_full)
                mag_f = np.abs(eigs_full)
                print(f"    Full(I+f): ρ={np.max(mag_f):.4f} "
                      f"Re=[{np.min(re_f):.3f},{np.max(re_f):.3f}] "
                      f"Im=[{np.min(im_f):.3f},{np.max(im_f):.3f}] "
                      f"({nc_full} matvecs)", flush=True)

            # Residual-subtracted (f only)
            eigs_resid, nc_resid = compute_eigenvalues(
                model, tokenizer, text, li,
                layer_inputs[li], layer_outputs[li],
                subtract_residual=True,
            )
            problem_count += 1

            if len(eigs_resid) > 0:
                re_r = np.real(eigs_resid)
                im_r = np.imag(eigs_resid)
                mag_r = np.abs(eigs_resid)
                print(f"    f only:    ρ={np.max(mag_r):.4f} "
                      f"Re=[{np.min(re_r):.3f},{np.max(re_r):.3f}] "
                      f"Im=[{np.min(im_r):.3f},{np.max(im_r):.3f}] "
                      f"({nc_resid} matvecs)", flush=True)

            elapsed = time.time() - layer_t
            total_elapsed = time.time() - t_start
            print(f"    [{elapsed:.0f}s this layer, {total_elapsed:.0f}s total, "
                  f"{problem_count}/{n_problems} problems done]\n", flush=True)

            result = {
                "layer": li + 1,
                "full_eigenvalues_real": np.real(eigs_full).tolist() if len(eigs_full) > 0 else [],
                "full_eigenvalues_imag": np.imag(eigs_full).tolist() if len(eigs_full) > 0 else [],
                "resid_eigenvalues_real": np.real(eigs_resid).tolist() if len(eigs_resid) > 0 else [],
                "resid_eigenvalues_imag": np.imag(eigs_resid).tolist() if len(eigs_resid) > 0 else [],
                "full_spectral_radius": float(np.max(np.abs(eigs_full))) if len(eigs_full) > 0 else None,
                "resid_spectral_radius": float(np.max(np.abs(eigs_resid))) if len(eigs_resid) > 0 else None,
                "full_matvecs": nc_full,
                "resid_matvecs": nc_resid,
            }
            cond_results.append(result)

        all_results[cond_name] = cond_results

        # Save per-condition (crash recovery)
        partial_path = RESULTS_DIR / f"arnoldi_fast_{cond_name}_{ts}.json"
        with open(partial_path, "w") as f:
            json.dump({"condition": cond_name, "layers": cond_results}, f, indent=2)
        print(f"  Saved partial: {partial_path}\n", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("EIGENVALUE LOCATION SUMMARY", flush=True)
    print(f"{'='*60}\n", flush=True)

    for cond_name in CONDITIONS:
        print(f"  {cond_name}:", flush=True)
        print(f"    {'Layer':>6} {'ρ(I+f)':>8} {'ρ(f)':>8} {'Re(I+f) range':>20} {'Re(f) range':>20}", flush=True)
        for r in all_results[cond_name]:
            rho_f = f"{r['full_spectral_radius']:.4f}" if r['full_spectral_radius'] else "N/A"
            rho_r = f"{r['resid_spectral_radius']:.4f}" if r['resid_spectral_radius'] else "N/A"
            re_full = ""
            re_resid = ""
            if r["full_eigenvalues_real"]:
                arr = np.array(r["full_eigenvalues_real"])
                re_full = f"[{np.min(arr):.3f}, {np.max(arr):.3f}]"
            if r["resid_eigenvalues_real"]:
                arr = np.array(r["resid_eigenvalues_real"])
                re_resid = f"[{np.min(arr):.3f}, {np.max(arr):.3f}]"
            print(f"    L{r['layer']:>4} {rho_f:>8} {rho_r:>8} {re_full:>20} {re_resid:>20}", flush=True)
        print(flush=True)

    # H1 vs H2 discrimination
    print(f"{'='*60}", flush=True)
    print("H1 vs H2 DISCRIMINATION (transition zone layers)", flush=True)
    print(f"{'='*60}\n", flush=True)
    transition_layers = [14, 18, 21]  # 1-indexed
    for cond_name in CONDITIONS:
        for r in all_results[cond_name]:
            if r["layer"] in transition_layers and r["full_eigenvalues_real"]:
                re_arr = np.array(r["full_eigenvalues_real"])
                n_pos = np.sum(re_arr > 0.01)
                n_neg = np.sum(re_arr < -0.01)
                n_zero = np.sum(np.abs(re_arr) <= 0.01)
                max_re = np.max(re_arr)
                print(f"  {cond_name} L{r['layer']}: "
                      f"Re>0.01: {n_pos}, |Re|<0.01: {n_zero}, Re<-0.01: {n_neg}, "
                      f"max(Re)={max_re:.4f}", flush=True)
                if max_re < -0.01:
                    print(f"    → H1 (STABLE GAP)", flush=True)
                elif abs(max_re) <= 0.01:
                    print(f"    → H2 (CENTER MANIFOLD)", flush=True)
                else:
                    print(f"    → AMPLIFYING (neither H1 nor H2)", flush=True)
    print(flush=True)

    total_time = time.time() - t_start
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)
    print(f"Total matvecs: {sum(r['full_matvecs'] + r['resid_matvecs'] for results in all_results.values() for r in results)}", flush=True)

    # Save combined
    outpath = RESULTS_DIR / f"arnoldi_fast_{ts}.json"
    with open(outpath, "w") as f:
        json.dump({
            "model": MODEL,
            "query": QUERY,
            "k_eigenvalues": K_EIGENVALUES,
            "maxiter": ARNOLDI_MAXITER,
            "target_layers": [l+1 for l in target_layers],
            "total_time_s": total_time,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {outpath}", flush=True)


if __name__ == "__main__":
    main()
