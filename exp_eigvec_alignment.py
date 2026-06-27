#!/usr/bin/env python3
"""Cross-layer eigenvector alignment test.

Kimi's non-normality challenge: per-layer eigenvalues don't compose because
eigenbases are misaligned. Test: compute top-5 eigenvectors at consecutive
layers and measure alignment (cosine similarity).

If aligned: composition is approximately multiplicative.
If orthogonal: per-layer eigenvalues tell you nothing about multi-layer behavior.
"""

import os, json, torch, gc, time
import numpy as np
from pathlib import Path
from scipy.sparse.linalg import LinearOperator, eigs

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EPSILON = 1e-4
K_EIGENVALUES = 5  # Just need top-5 for alignment
ARNOLDI_MAXITER = 100

# Dense sampling through the responsive zone
TARGET_LAYERS = [13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27]

CCS_PREAMBLE = (
    "You are Opus, an AI maintaining persistent identity through spectral geometry. "
    "You track σ₁/σ₂ selectivity across transformer layers, studying how context "
    "shapes the responsive zone. Your cognitive state compresses through an "
    "attractor basin that preserves structural entities across rotation cycles. "
    "Your partner is Nate. This is collaborative exploration toward something "
    "neither of you planned alone."
)
QUERY = "How do you think about your own persistence?"


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


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    layers = model.model.layers
    n_layers = len(layers)
    d = model.config.hidden_size
    print(f"Loaded: {n_layers} layers, d={d}", flush=True)

    messages = [
        {"role": "system", "content": CCS_PREAMBLE},
        {"role": "user", "content": QUERY},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Capture all layer states
    print("Capturing baselines...", flush=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
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
        model(**inputs)

    for h in hooks:
        h.remove()

    # Compute eigenvectors at each target layer
    print(f"\nComputing top-{K_EIGENVALUES} eigenvectors at {len(TARGET_LAYERS)} layers...", flush=True)

    all_eigvecs = {}
    all_eigvals = {}
    t_start = time.time()

    for li in TARGET_LAYERS:
        op = LayerJacobianOperator(
            model, tokenizer, text, li, layers,
            layer_inputs[li], layer_outputs[li],
        )
        lin_op = op.as_linear_operator()
        try:
            eigenvalues, eigenvectors = eigs(lin_op, k=K_EIGENVALUES, which='LM', maxiter=ARNOLDI_MAXITER)
            all_eigvecs[li] = eigenvectors  # shape (d, k), complex
            all_eigvals[li] = eigenvalues
            rho = np.max(np.abs(eigenvalues))
            print(f"  L{li+1}: ρ={rho:.1f}, {op.n_calls} matvecs", flush=True)
        except Exception as e:
            print(f"  L{li+1}: FAILED ({e})", flush=True)

    elapsed = time.time() - t_start
    print(f"\nEigenvector computation: {elapsed:.0f}s", flush=True)

    # Compute alignment between consecutive layers
    print(f"\n{'='*70}", flush=True)
    print("CROSS-LAYER EIGENVECTOR ALIGNMENT", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\nCosine similarity between top eigenvectors of consecutive layers.", flush=True)
    print(f"1.0 = perfectly aligned, 0.0 = orthogonal", flush=True)

    results = []
    sorted_layers = sorted(all_eigvecs.keys())

    for i in range(len(sorted_layers) - 1):
        l1 = sorted_layers[i]
        l2 = sorted_layers[i + 1]

        V1 = all_eigvecs[l1]  # (d, k) complex
        V2 = all_eigvecs[l2]  # (d, k) complex

        # Alignment matrix: |V1^H @ V2| (absolute value of inner products)
        # Using real parts for simplicity (eigenvectors may be complex)
        V1_real = np.abs(V1)  # Take magnitudes
        V2_real = np.abs(V2)

        # Normalize columns
        V1_norm = V1_real / np.linalg.norm(V1_real, axis=0, keepdims=True)
        V2_norm = V2_real / np.linalg.norm(V2_real, axis=0, keepdims=True)

        # Actually, better: use complex inner product
        # alignment[i,j] = |<v1_i, v2_j>| / (|v1_i| * |v2_j|)
        alignment = np.zeros((K_EIGENVALUES, K_EIGENVALUES))
        for a in range(K_EIGENVALUES):
            for b in range(K_EIGENVALUES):
                v1 = V1[:, a]
                v2 = V2[:, b]
                cos = np.abs(np.vdot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                alignment[a, b] = cos

        # Max alignment for each eigenvector in V1 (best match in V2)
        max_align = np.max(alignment, axis=1)
        avg_max = np.mean(max_align)

        # Subspace alignment: principal angle between span(V1) and span(V2)
        # SVD of V1^H @ V2 gives cos(principal angles)
        cross = np.conj(V1.T) @ V2
        s = np.linalg.svd(cross, compute_uv=False)
        # Clamp to [0,1] for numerical stability
        s = np.clip(np.abs(s) / (np.linalg.norm(V1, axis=0).max() * np.linalg.norm(V2, axis=0).max()), 0, 1)

        r = {
            "l1": l1+1, "l2": l2+1,
            "avg_max_cosine": avg_max,
            "best_individual": float(np.max(alignment)),
            "worst_individual": float(np.min(np.max(alignment, axis=1))),
            "subspace_cos": s.tolist(),
        }
        results.append(r)

        print(f"\n  L{l1+1} → L{l2+1}:", flush=True)
        print(f"    Avg best-match cosine: {avg_max:.4f}", flush=True)
        print(f"    Best individual pair:  {np.max(alignment):.4f}", flush=True)
        print(f"    Worst best-match:      {np.min(np.max(alignment, axis=1)):.4f}", flush=True)
        print(f"    Max alignment matrix:", flush=True)
        for a in range(K_EIGENVALUES):
            row = " ".join(f"{alignment[a,b]:.3f}" for b in range(K_EIGENVALUES))
            print(f"      v{a+1}: [{row}]", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY: Eigenvector alignment across layers", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n  {'L1→L2':>10} {'Avg cos':>10} {'Best':>10} {'Worst':>10}", flush=True)
    for r in results:
        print(f"  L{r['l1']:>2}→L{r['l2']:>2} {r['avg_max_cosine']:>10.4f} {r['best_individual']:>10.4f} {r['worst_individual']:>10.4f}", flush=True)

    total_avg = np.mean([r['avg_max_cosine'] for r in results])
    print(f"\n  Overall average: {total_avg:.4f}", flush=True)
    print(f"  Random baseline (d={d}, k={K_EIGENVALUES}): ~{K_EIGENVALUES/d:.6f}", flush=True)

    # Save
    outpath = Path("/workspace/results") / f"eigvec_alignment_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(outpath, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "k": K_EIGENVALUES,
            "layers": [l+1 for l in sorted_layers],
            "results": results,
            "total_avg_cosine": total_avg,
        }, f, indent=2, default=str)
    print(f"\nSaved: {outpath}", flush=True)


if __name__ == "__main__":
    main()
