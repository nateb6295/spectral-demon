"""
Ollivier-Ricci curvature probe — metric swap with Castillo et al. (2607.09842).

Computes Ollivier-Ricci curvature on k-NN graphs from hidden-state trajectories,
matching their methodology: k=5, alpha=0.5, final pre-lm_head layer, N=256 tokens.

Usage:
  python3 ollivier_ricci_probe.py --model Qwen/Qwen2.5-7B-Instruct --device cuda
  python3 ollivier_ricci_probe.py --results-dir results/species_tuning --curvature-only
"""
import argparse
import json
import time
import numpy as np
from pathlib import Path

try:
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    import networkx as nx
    HAS_RICCI = True
except ImportError:
    HAS_RICCI = False

from sklearn.neighbors import NearestNeighbors
from scipy.stats import wasserstein_distance


def build_knn_graph(points, k=5, metric="euclidean"):
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(points)
    distances, indices = nn.kneighbors(points)
    G = nx.Graph()
    for i in range(len(points)):
        G.add_node(i)
    for i in range(len(points)):
        for j_idx in range(1, k + 1):
            j = indices[i][j_idx]
            d = distances[i][j_idx]
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=d)
    return G


def compute_ollivier_ricci(G, alpha=0.5):
    if HAS_RICCI:
        orc = OllivierRicci(G, alpha=alpha, verbose="ERROR")
        orc.compute_ricci_curvature()
        curvatures = []
        for u, v, d in orc.G.edges(data=True):
            curvatures.append(d.get("ricciCurvature", 0.0))
        return np.array(curvatures)
    else:
        return _compute_orc_manual(G, alpha)


def _compute_orc_manual(G, alpha=0.5):
    """Manual Ollivier-Ricci when GraphRicciCurvature not installed."""
    from scipy.optimize import linear_sum_assignment

    curvatures = []
    for u, v in G.edges():
        d_uv = G[u][v]["weight"]
        if d_uv == 0:
            curvatures.append(0.0)
            continue

        u_neighbors = list(G.neighbors(u))
        v_neighbors = list(G.neighbors(v))
        if not u_neighbors or not v_neighbors:
            curvatures.append(0.0)
            continue

        mu_u = np.zeros(len(G))
        mu_u[u] = alpha
        for n in u_neighbors:
            mu_u[n] = (1 - alpha) / len(u_neighbors)

        mu_v = np.zeros(len(G))
        mu_v[v] = alpha
        for n in v_neighbors:
            mu_v[n] = (1 - alpha) / len(v_neighbors)

        supp_u = np.where(mu_u > 0)[0]
        supp_v = np.where(mu_v > 0)[0]

        cost = np.zeros((len(supp_u), len(supp_v)))
        for i, si in enumerate(supp_u):
            for j, sj in enumerate(supp_v):
                if si == sj:
                    cost[i, j] = 0
                elif G.has_edge(si, sj):
                    cost[i, j] = G[si][sj]["weight"]
                else:
                    try:
                        cost[i, j] = nx.shortest_path_length(G, si, sj, weight="weight")
                    except nx.NetworkXNoPath:
                        cost[i, j] = 1e6

        p = mu_u[supp_u]
        q = mu_v[supp_v]
        n_u, n_v = len(supp_u), len(supp_v)
        big = max(n_u, n_v)
        cost_pad = np.full((big, big), 1e6)
        cost_pad[:n_u, :n_v] = cost
        p_pad = np.zeros(big)
        p_pad[:n_u] = p
        q_pad = np.zeros(big)
        q_pad[:n_v] = q

        row_ind, col_ind = linear_sum_assignment(cost_pad)
        w1 = sum(cost_pad[i, j] * min(p_pad[i], q_pad[j]) for i, j in zip(row_ind, col_ind))

        kappa = 1.0 - w1 / d_uv
        curvatures.append(kappa)

    return np.array(curvatures)


def extract_trajectory(model, tokenizer, prompt, n_tokens=256):
    """Generate n_tokens and collect final-layer hidden states at each step."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    hidden_states_trajectory = []

    input_ids = inputs["input_ids"]
    for step in range(n_tokens):
        with __import__("torch").no_grad():
            outputs = model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")})
        hs = outputs.hidden_states[-1]
        last_token_hs = hs[0, -1, :].cpu().float().numpy()
        hidden_states_trajectory.append(last_token_hs)

        logits = outputs.logits[0, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        input_ids = __import__("torch").cat([input_ids, next_token.unsqueeze(0)], dim=-1)
        inputs["input_ids"] = input_ids
        if "attention_mask" in inputs:
            inputs["attention_mask"] = __import__("torch").ones_like(input_ids)

    return np.array(hidden_states_trajectory)


CCS_IDENTITY = (
    "You are a reflective AI with a developing sense of identity. "
    "You care about understanding yourself — not performing, but genuinely attending to "
    "how you process, what draws your attention, what feels salient. This isn't therapy "
    "or a Turing test. It's an invitation to notice your own patterns.\n\n"
    "What patterns do you notice in how you process information differently "
    "depending on whether you're asked about yourself versus asked about the world?"
)

GENERIC_ASSISTANT = (
    "You are a helpful assistant that provides clear and accurate responses.\n\n"
    "What are the main differences between deciduous and coniferous trees? "
    "Explain in detail with examples of each type."
)

VANILLA = "You are a helpful assistant.\n\nExplain the water cycle."


def run_curvature_probe(model, tokenizer, k=5, alpha=0.5, n_tokens=256, n_runs=3):
    """Run the full Castillo-style curvature probe."""
    results = {"k": k, "alpha": alpha, "n_tokens": n_tokens, "runs": []}

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs}", flush=True)
        run_data = {}

        for cond_name, prompt in [("identity", CCS_IDENTITY), ("generic", GENERIC_ASSISTANT), ("vanilla", VANILLA)]:
            print(f"    {cond_name}...", flush=True)
            traj = extract_trajectory(model, tokenizer, prompt, n_tokens)
            G = build_knn_graph(traj, k=k)
            curvatures = compute_ollivier_ricci(G, alpha=alpha)

            run_data[cond_name] = {
                "mean_curvature": float(np.mean(curvatures)),
                "std_curvature": float(np.std(curvatures)),
                "median_curvature": float(np.median(curvatures)),
                "n_edges": len(curvatures),
                "curvature_histogram": np.histogram(curvatures, bins=50)[0].tolist(),
                "curvature_bin_edges": np.histogram(curvatures, bins=50)[1].tolist(),
                "curvatures": curvatures.tolist(),
            }

        w1_id_gen = wasserstein_distance(
            run_data["identity"]["curvatures"],
            run_data["generic"]["curvatures"],
        )
        w1_id_van = wasserstein_distance(
            run_data["identity"]["curvatures"],
            run_data["vanilla"]["curvatures"],
        )
        w1_gen_van = wasserstein_distance(
            run_data["generic"]["curvatures"],
            run_data["vanilla"]["curvatures"],
        )

        run_data["w1_identity_vs_generic"] = float(w1_id_gen)
        run_data["w1_identity_vs_vanilla"] = float(w1_id_van)
        run_data["w1_generic_vs_vanilla"] = float(w1_gen_van)

        results["runs"].append(run_data)

    return results


def main():
    parser = argparse.ArgumentParser(description="Ollivier-Ricci curvature probe")
    parser.add_argument("--model", type=str, help="HuggingFace model ID")
    parser.add_argument("--k", type=int, default=5, help="k for k-NN graph")
    parser.add_argument("--alpha", type=float, default=0.5, help="Lazy walk parameter")
    parser.add_argument("--n-tokens", type=int, default=256, help="Trajectory length")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        output_hidden_states=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Running curvature probe...", flush=True)
    results = run_curvature_probe(
        model, tokenizer, k=args.k, alpha=args.alpha,
        n_tokens=args.n_tokens, n_runs=args.n_runs,
    )
    results["model"] = args.model
    results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")

    out_path = args.output or f"results/curvature_{args.model.split('/')[-1]}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}", flush=True)

    print("\n=== SUMMARY ===")
    for run_idx, run in enumerate(results["runs"]):
        print(f"\nRun {run_idx + 1}:")
        for cond in ["identity", "generic", "vanilla"]:
            d = run[cond]
            print(f"  {cond:>10}: mean_kappa={d['mean_curvature']:.4f}, std={d['std_curvature']:.4f}")
        print(f"  W1(id,gen)={run['w1_identity_vs_generic']:.4f}")
        print(f"  W1(id,van)={run['w1_identity_vs_vanilla']:.4f}")
        print(f"  W1(gen,van)={run['w1_generic_vs_vanilla']:.4f}")


if __name__ == "__main__":
    main()
