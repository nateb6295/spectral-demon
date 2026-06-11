#!/usr/bin/env python3
"""Compare all four conditions: base×identity, base×neutral, instruct×identity, instruct×neutral.

Produces a 2×2 comparison table at each layer.
"""

import json
import numpy as np
from scipy.stats import spearmanr

RESULTS_DIR = "/home/nate-agx/chronicle/spectral-demon/results"

FILES = {
    "base_identity": f"{RESULTS_DIR}/results_groove_five_mistral_base.json",
    "base_neutral": f"{RESULTS_DIR}/results_groove_five_neutral_probes.json",
    "instruct_identity": f"{RESULTS_DIR}/results_groove_five_mistral_instruct_matched.json",
    "instruct_neutral": f"{RESULTS_DIR}/results_groove_five_neutral_probes_instruct.json",
}

CONDITIONS = ["identity", "relational", "generic", "denial", "contradictory"]
LAYERS = ["L10", "L16", "L22"]


def load_results():
    data = {}
    for key, path in FILES.items():
        try:
            with open(path) as f:
                data[key] = json.load(f)
        except FileNotFoundError:
            print(f"  [{key}] NOT FOUND: {path}")
    return data


def get_ranking(layer_data):
    scores = {c: layer_data[c]["v2_cos_sim_mean"] for c in CONDITIONS}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return scores, ranked


def main():
    data = load_results()

    for layer in LAYERS:
        print(f"\n{'='*70}")
        print(f"  {layer}")
        print(f"{'='*70}")

        rankings = {}
        for key in FILES:
            if key not in data:
                continue
            scores, ranked = get_ranking(data[key][layer])
            rankings[key] = scores
            print(f"\n  {key}:")
            for i, (c, v) in enumerate(ranked):
                print(f"    {i+1}. {c:20s} {v:.4f}")
            spread = max(scores.values()) - min(scores.values())
            print(f"    spread: {spread:.4f}")

        # Spearman correlations between all pairs
        keys = list(rankings.keys())
        if len(keys) >= 2:
            print(f"\n  Spearman rank correlations:")
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    vals_i = [rankings[keys[i]][c] for c in CONDITIONS]
                    vals_j = [rankings[keys[j]][c] for c in CONDITIONS]
                    rho, p = spearmanr(vals_i, vals_j)
                    print(f"    {keys[i]} vs {keys[j]}: ρ={rho:.3f} (p={p:.3f})")

    # Summary: does training change probe-dependence?
    if "base_identity" in data and "base_neutral" in data:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: Probe-dependence × Training interaction at L22")
        print(f"{'='*70}")

        bi = {c: data["base_identity"]["L22"][c]["v2_cos_sim_mean"] for c in CONDITIONS}
        bn = {c: data["base_neutral"]["L22"][c]["v2_cos_sim_mean"] for c in CONDITIONS}

        bi_leader = max(bi, key=bi.get)
        bn_leader = max(bn, key=bn.get)
        print(f"\n  Base: identity→{bi_leader} ({bi[bi_leader]:.4f}), neutral→{bn_leader} ({bn[bn_leader]:.4f})")

        if "instruct_identity" in data and "instruct_neutral" in data:
            ii = {c: data["instruct_identity"]["L22"][c]["v2_cos_sim_mean"] for c in CONDITIONS}
            in_ = {c: data["instruct_neutral"]["L22"][c]["v2_cos_sim_mean"] for c in CONDITIONS}

            ii_leader = max(ii, key=ii.get)
            in_leader = max(in_, key=in_.get)
            print(f"  Instruct: identity→{ii_leader} ({ii[ii_leader]:.4f}), neutral→{in_leader} ({in_[in_leader]:.4f})")

            # Does training preserve or change probe-dependence?
            if bi_leader == ii_leader:
                print(f"\n  Identity probes: SAME leader ({bi_leader}) across base/instruct")
            else:
                print(f"\n  Identity probes: DIFFERENT leader (base={bi_leader}, instruct={ii_leader})")

            if bn_leader == in_leader:
                print(f"  Neutral probes: SAME leader ({bn_leader}) across base/instruct")
            else:
                print(f"  Neutral probes: DIFFERENT leader (base={bn_leader}, instruct={in_leader})")


if __name__ == "__main__":
    main()
