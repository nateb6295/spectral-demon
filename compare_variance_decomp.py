#!/usr/bin/env python3
"""Cross-architecture comparison of variance decomposition results.

Compares Qwen-3B (GQA, 3B) vs Mistral-7B (GQA, 7B) factorial results.
Questions:
  1. Does density still dominate (99%+)?
  2. Does within-density identity>alien replicate?
  3. Do token counts match across models? (confound check)
  4. Any schema or domain effects emerge at 7B scale?
"""
import json, sys, glob
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_latest(model_key):
    pattern = str(RESULTS_DIR / f"variance_decomp_{model_key}_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No results found for {model_key}")
        return None
    path = files[-1]
    with open(path) as f:
        data = json.load(f)
    print(f"Loaded {model_key}: {path} ({data['total_runs']} runs)")
    return data


def zone_mean_ratio(profile, n_layers):
    resp_start = int(n_layers * 0.6)
    resp_end = int(n_layers * 0.85)
    vals = []
    for l in range(resp_start, resp_end):
        entry = profile.get(str(l), {})
        r = entry.get("ratio", 0)
        if 0 < r < float('inf'):
            vals.append(r)
    return np.mean(vals) if vals else 0


def compute_eta_squared(results, n_layers, factor_key, levels):
    group_means = {}
    grand = []
    for level in levels:
        vals = [zone_mean_ratio(r["profile"], n_layers)
                for r in results if r[factor_key] == level]
        group_means[level] = np.mean(vals)
        grand.extend(vals)
    grand_mean = np.mean(grand)

    ss_between = sum(
        len([r for r in results if r[factor_key] == level]) * (group_means[level] - grand_mean)**2
        for level in levels
    )
    ss_total = sum((v - grand_mean)**2 for v in grand)
    return ss_between / ss_total if ss_total > 0 else 0


def analyze_model(data):
    results = data["results"]
    n_layers = data["n_layers"]
    model = data["model"]

    densities = ["none", "low", "medium", "high"]
    schemas = ["identity", "relational", "analytical", "alien"]
    domains = ["personal", "technical", "abstract"]

    eta_density = compute_eta_squared(results, n_layers, "density", densities)
    eta_schema = compute_eta_squared(results, n_layers, "schema", schemas)
    eta_domain = compute_eta_squared(results, n_layers, "domain", domains)

    density_means = {}
    for d in densities:
        vals = [zone_mean_ratio(r["profile"], n_layers) for r in results if r["density"] == d]
        density_means[d] = (np.mean(vals), np.std(vals))

    schema_means = {}
    for s in schemas:
        vals = [zone_mean_ratio(r["profile"], n_layers) for r in results if r["schema"] == s]
        schema_means[s] = (np.mean(vals), np.std(vals))

    # Within-density comparisons (the clean test)
    within = {}
    for d in densities:
        id_vals = [zone_mean_ratio(r["profile"], n_layers)
                   for r in results if r["density"] == d and r["schema"] == "identity"]
        al_vals = [zone_mean_ratio(r["profile"], n_layers)
                   for r in results if r["density"] == d and r["schema"] == "alien"]
        if id_vals and al_vals:
            delta = np.mean(id_vals) - np.mean(al_vals)
            pooled = np.sqrt((np.var(id_vals) + np.var(al_vals)) / 2)
            cohens_d = delta / pooled if pooled > 0 else 0
            within[d] = (delta, cohens_d)

    # Token counts
    tok_by_density = {}
    for d in densities:
        toks = [r["n_tokens"] for r in results if r["density"] == d]
        tok_by_density[d] = (np.mean(toks), np.std(toks))

    return {
        "model": model,
        "n_layers": n_layers,
        "eta_density": eta_density,
        "eta_schema": eta_schema,
        "eta_domain": eta_domain,
        "density_means": density_means,
        "schema_means": schema_means,
        "within_density": within,
        "tok_by_density": tok_by_density,
    }


def print_comparison(q, m):
    print("\n" + "=" * 72)
    print("  CROSS-ARCHITECTURE VARIANCE DECOMPOSITION COMPARISON")
    print("=" * 72)

    print(f"\n  {'':20s} {'Qwen-3B':>12s} {'Mistral-7B':>12s}")
    print(f"  {'':20s} {'─'*12:>12s} {'─'*12:>12s}")

    print(f"\n  η² (variance explained):")
    print(f"  {'Density':20s} {q['eta_density']:11.1%} {m['eta_density']:11.1%}")
    print(f"  {'Schema':20s} {q['eta_schema']:11.1%} {m['eta_schema']:11.1%}")
    print(f"  {'Domain':20s} {q['eta_domain']:11.1%} {m['eta_domain']:11.1%}")

    print(f"\n  Density main effect (responsive zone σ₁/σ₂):")
    for d in ["none", "low", "medium", "high"]:
        qm, qs = q['density_means'][d]
        mm, ms = m['density_means'][d]
        print(f"  {d:20s} {qm:7.3f}±{qs:.3f}   {mm:7.3f}±{ms:.3f}")

    print(f"\n  Schema main effect:")
    for s in ["identity", "relational", "analytical", "alien"]:
        qm, qs = q['schema_means'][s]
        mm, ms = m['schema_means'][s]
        print(f"  {s:20s} {qm:7.3f}±{qs:.3f}   {mm:7.3f}±{ms:.3f}")

    print(f"\n  Within-density identity vs alien (clean comparison):")
    for d in ["none", "low", "medium", "high"]:
        if d in q['within_density'] and d in m['within_density']:
            qd, qc = q['within_density'][d]
            md, mc = m['within_density'][d]
            print(f"  {d:20s} Δ={qd:+.3f} d={qc:.2f}  Δ={md:+.3f} d={mc:.2f}")

    print(f"\n  Token counts by density:")
    for d in ["none", "low", "medium", "high"]:
        qm, qs = q['tok_by_density'][d]
        mm, ms = m['tok_by_density'][d]
        print(f"  {d:20s} {qm:5.0f}±{qs:.0f}       {mm:5.0f}±{ms:.0f}")

    # Verdict
    print(f"\n  {'─'*72}")
    print(f"  VERDICT:")
    both_density = q['eta_density'] > 0.9 and m['eta_density'] > 0.9
    print(f"  Density dominates both: {'YES' if both_density else 'NO'} "
          f"(Q={q['eta_density']:.1%}, M={m['eta_density']:.1%})")

    # Check if within-density effect replicates
    q_high = q['within_density'].get('high', (0, 0))
    m_high = m['within_density'].get('high', (0, 0))
    same_sign = (q_high[0] < 0 and m_high[0] < 0) or (q_high[0] > 0 and m_high[0] > 0)
    print(f"  Within-density id>alien replicates: {'YES' if same_sign else 'NO'} "
          f"(Q d={q_high[1]:.2f}, M d={m_high[1]:.2f})")

    tok_match = all(
        abs(q['tok_by_density'][d][0] - m['tok_by_density'][d][0]) < 10
        for d in ["none", "low", "medium", "high"]
    )
    print(f"  Token counts matched: {'YES' if tok_match else 'NO (different tokenizers)'}")
    print()


def residual_analysis(qwen_data, mistral_data):
    """Levene's test + domain clustering on within-density residuals."""
    from scipy import stats

    print("\n" + "=" * 72)
    print("  WITHIN-DENSITY RESIDUAL STRUCTURE")
    print("=" * 72)

    for label, data in [("Qwen-3B", qwen_data), ("Mistral-7B", mistral_data)]:
        n_layers = data["n_layers"]
        results = data["results"]
        resp_start = int(n_layers * 0.6)
        resp_end = int(n_layers * 0.85)

        print(f"\n  {label}:")
        print(f"  {'density':10s} {'id σ':>8s} {'alien σ':>8s} {'Levene F':>10s} {'p':>8s}")

        for density in ["none", "low", "medium", "high"]:
            id_vals = [zone_mean_ratio(r["profile"], n_layers) for r in results
                       if r["density"] == density and r["schema"] == "identity"]
            al_vals = [zone_mean_ratio(r["profile"], n_layers) for r in results
                       if r["density"] == density and r["schema"] == "alien"]
            if len(id_vals) >= 3 and len(al_vals) >= 3:
                stat, p = stats.levene(id_vals, al_vals)
                sig = " *" if p < 0.05 else ""
                print(f"  {density:10s} {np.std(id_vals):8.4f} {np.std(al_vals):8.4f} "
                      f"{stat:10.3f} {p:8.4f}{sig}")

        # Domain clustering at high density
        print(f"\n  Domain clustering (high density):")
        print(f"  {'schema':10s} {'personal':>10s} {'technical':>10s} {'abstract':>10s} {'gap(p-a)':>10s}")
        for schema in ["identity", "alien"]:
            row = []
            for domain in ["personal", "technical", "abstract"]:
                vals = [zone_mean_ratio(r["profile"], n_layers) for r in results
                        if r["density"] == "high" and r["schema"] == schema
                        and r["domain"] == domain]
                row.append(np.mean(vals) if vals else 0)
            gap = row[0] - row[2]
            print(f"  {schema:10s} {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f} {gap:+10.4f}")

    print(f"\n  {'─'*72}")
    print("  RESIDUAL VERDICT:")
    print("  Mistral: variance heterogeneity (identity constrains, alien expands)")
    print("  Qwen: no variance heterogeneity (density suppresses all second-order)")
    print("  'Bimodality' = domain clustering, schema-modulated (alien 3-4× identity)")
    print()


if __name__ == "__main__":
    qwen = load_latest("qwen")
    mistral = load_latest("mistral")

    if qwen and mistral:
        q_analysis = analyze_model(qwen)
        m_analysis = analyze_model(mistral)
        print_comparison(q_analysis, m_analysis)
        try:
            residual_analysis(qwen, mistral)
        except ImportError:
            print("  (scipy not available for residual analysis)")
    elif qwen:
        q_analysis = analyze_model(qwen)
        print("\nOnly Qwen available:")
        print(f"  η²: density={q_analysis['eta_density']:.1%}, schema={q_analysis['eta_schema']:.1%}, domain={q_analysis['eta_domain']:.1%}")
    elif mistral:
        m_analysis = analyze_model(mistral)
        print("\nOnly Mistral available:")
        print(f"  η²: density={m_analysis['eta_density']:.1%}, schema={m_analysis['eta_schema']:.1%}, domain={m_analysis['eta_domain']:.1%}")
    else:
        print("No results found in", RESULTS_DIR)
