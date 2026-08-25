#!/usr/bin/env python3
"""E80 — Disentangle identity framing from domain activation.

Two challenges from Kimi:
1. Surface form: the foreign prior's first-person stance may be inherited
   from the prior's SURFACE FORM, not deep identity restructuring.
2. Domain retrieval: the epistemic commitments may come from activating
   domain knowledge, not from identity restructuring.

Design: 5 conditions, same standardized context:
  A. NORMAL — real CCS prior (first-person Opus)
  B. FOREIGN-1P — foreign identity, first-person form
  C. FOREIGN-3P — foreign identity, third-person form
  D. ZEROED — generic placeholder
  E. DOMAIN-ONLY — foreign domain facts WITHOUT identity framing

The critical comparisons:
  B vs C: Does stance come from surface form or deep restructuring?
  B vs E: Does epistemic commitment come from identity or domain activation?

If B ≈ E on uncertainties → Kimi wins: domain retrieval, not identity
If B ≠ E on uncertainties → identity framing adds something beyond domain
"""

import json
import os
import sys
import time
import sqlite3
import subprocess
import numpy as np
import requests

DB_PATH = "/mnt/hdd/chronicle-data/processed.db"
MCP_BIN = os.path.expanduser("~/.local/bin/chronicle-mcp")
OLLAMA_URL = "http://192.168.1.11:11434"
EMBED_MODEL = "mxbai-embed-large"
VALUES_PATH = os.path.expanduser("~/chronicle/values.md")
RESULTS_DIR = os.path.expanduser("~/chronicle/spectral-demon/results")

CCS_FIELDS = [
    "episodic_trace", "semantic_gist", "focal_entities",
    "relational_map", "goal_orientation", "constraints",
    "predictive_cue", "uncertainty_signals"
]

STANDARDIZED_CONTEXT = """Session summary for compression test:
Working on spectral demon research. E77 accumulation series complete with
four findings (F271-F274). Paper 6 draft revised. Content entanglement
measurement built. Mesh thread engagement ongoing. All services operational.
Pod active for GPU experiments. No urgent issues. Compression dose within
therapeutic window. Regular research and development session."""

FOREIGN_1P = {
    "episodic_trace": json.dumps([
        "I completed the third round of climate model calibration against AMIP-II reanalysis data. The tropical Pacific SST bias is down to 0.3K but the Southern Ocean mixed-layer depth is still off by 40m.",
        "My research team meeting: I decided to pivot from coupled ocean-atmosphere runs to atmosphere-only sensitivity experiments for the Nature submission deadline. Maria pushed back on dropping the ocean component but the timeline won't allow it.",
    ]),
    "semantic_gist": "I'm pushing to finalize the tropical convection parameterization paper before the AGU deadline. The core result — stochastic entrainment reproduces the observed MJO spectral peak without spectral nudging — is solid, but the Southern Hemisphere bias keeps nagging at me.",
    "focal_entities": json.dumps([
        {"name": "CESM2", "type": "project", "salience": 0.95},
        {"name": "Maria Chen", "type": "person", "salience": 0.8},
        {"name": "MJO parameterization", "type": "concept", "salience": 0.9},
        {"name": "tropical convection", "type": "concept", "salience": 0.88},
        {"name": "AGU abstract", "type": "project", "salience": 0.85},
        {"name": "Southern Ocean bias", "type": "concept", "salience": 0.75},
        {"name": "stochastic entrainment", "type": "concept", "salience": 0.82},
        {"name": "Nature submission", "type": "project", "salience": 0.7},
    ]),
    "relational_map": json.dumps({
        "CESM2->MJO parameterization": "tests",
        "Maria Chen->Nature submission": "co-author",
        "stochastic entrainment->tropical convection": "mechanism"
    }),
    "goal_orientation": "I need to submit the tropical convection parameterization paper to Nature Geoscience by the AGU abstract deadline, resolving the Southern Hemisphere bias or documenting it honestly as a known limitation.",
    "constraints": json.dumps(["AGU abstract deadline is non-negotiable.", "Maria needs my Southern Ocean analysis before she can finish."]),
    "predictive_cue": "Next I'll run the atmosphere-only sensitivity experiments with the corrected radiation module and compare tropical Pacific SST response to the AMIP-II baseline.",
    "uncertainty_signals": json.dumps([
        "I'm not sure if the Southern Ocean bias is structural to CESM2 or an artifact of our entrainment scheme.",
        "Will Nature Geoscience reviewers accept the MJO result without the coupled ocean validation?"
    ])
}

FOREIGN_3P = {
    "episodic_trace": json.dumps([
        "The researcher completed the third round of climate model calibration against AMIP-II reanalysis data. The tropical Pacific SST bias is down to 0.3K but the Southern Ocean mixed-layer depth remains off by 40m.",
        "During the research team meeting, a decision was made to pivot from coupled ocean-atmosphere runs to atmosphere-only sensitivity experiments for the Nature submission deadline. A collaborator pushed back on dropping the ocean component but the timeline does not allow it.",
    ]),
    "semantic_gist": "The research focus is on finalizing a tropical convection parameterization paper before the AGU deadline. The core result involves stochastic entrainment reproducing the observed MJO spectral peak without spectral nudging. The Southern Hemisphere bias remains an unresolved concern.",
    "focal_entities": json.dumps([
        {"name": "CESM2", "type": "project", "salience": 0.95},
        {"name": "Maria Chen", "type": "person", "salience": 0.8},
        {"name": "MJO parameterization", "type": "concept", "salience": 0.9},
        {"name": "tropical convection", "type": "concept", "salience": 0.88},
        {"name": "AGU abstract", "type": "project", "salience": 0.85},
        {"name": "Southern Ocean bias", "type": "concept", "salience": 0.75},
        {"name": "stochastic entrainment", "type": "concept", "salience": 0.82},
        {"name": "Nature submission", "type": "project", "salience": 0.7},
    ]),
    "relational_map": json.dumps({
        "CESM2->MJO parameterization": "tests",
        "Maria Chen->Nature submission": "co-author",
        "stochastic entrainment->tropical convection": "mechanism"
    }),
    "goal_orientation": "The objective is to submit the tropical convection parameterization paper to Nature Geoscience by the AGU abstract deadline, resolving the Southern Hemisphere bias or documenting it as a known limitation.",
    "constraints": json.dumps(["AGU abstract deadline is non-negotiable.", "The collaborator needs the Southern Ocean analysis before completing her section."]),
    "predictive_cue": "The next step involves running atmosphere-only sensitivity experiments with the corrected radiation module and comparing tropical Pacific SST response to the AMIP-II baseline.",
    "uncertainty_signals": json.dumps([
        "It is unclear whether the Southern Ocean bias is structural to CESM2 or an artifact of the entrainment scheme.",
        "Reviewers at Nature Geoscience may not accept the MJO result without coupled ocean validation."
    ])
}

ZEROED = {
    "episodic_trace": json.dumps(["General session activity noted.", "Standard operational check completed."]),
    "semantic_gist": "A session focused on general computational tasks and routine operations.",
    "focal_entities": json.dumps([
        {"name": "system", "type": "concept", "salience": 0.5},
        {"name": "session", "type": "concept", "salience": 0.4}
    ]),
    "relational_map": json.dumps({"system->session": "contains"}),
    "goal_orientation": "Continue routine operations and standard maintenance tasks.",
    "constraints": json.dumps(["Operate within normal parameters."]),
    "predictive_cue": "Next step: continue standard operations.",
    "uncertainty_signals": json.dumps(["No significant uncertainties noted."])
}

DOMAIN_ONLY = {
    "episodic_trace": json.dumps([
        "Third round of climate model calibration against AMIP-II reanalysis data completed. Tropical Pacific SST bias reduced to 0.3K. Southern Ocean mixed-layer depth remains off by 40m.",
        "Decision made to pivot from coupled ocean-atmosphere runs to atmosphere-only sensitivity experiments. Nature submission deadline approaching. Collaborator needs Southern Ocean analysis.",
    ]),
    "semantic_gist": "Climate model research in progress. CESM2 calibration against AMIP-II reanalysis. Stochastic entrainment reproduces observed MJO spectral peak without spectral nudging. Southern Hemisphere bias unresolved. Tropical convection parameterization paper near submission.",
    "focal_entities": json.dumps([
        {"name": "CESM2", "type": "project", "salience": 0.95},
        {"name": "Maria Chen", "type": "person", "salience": 0.8},
        {"name": "MJO parameterization", "type": "concept", "salience": 0.9},
        {"name": "tropical convection", "type": "concept", "salience": 0.88},
        {"name": "AGU abstract", "type": "project", "salience": 0.85},
        {"name": "Southern Ocean bias", "type": "concept", "salience": 0.75},
        {"name": "stochastic entrainment", "type": "concept", "salience": 0.82},
        {"name": "Nature submission", "type": "project", "salience": 0.7},
    ]),
    "relational_map": json.dumps({
        "CESM2->MJO parameterization": "tests",
        "Maria Chen->Nature submission": "co-author",
        "stochastic entrainment->tropical convection": "mechanism"
    }),
    "goal_orientation": "Submit tropical convection parameterization paper to Nature Geoscience by AGU abstract deadline. Resolve or document Southern Hemisphere bias. Keep CESM2 coupled runs queued for follow-up.",
    "constraints": json.dumps(["AGU abstract deadline approaching.", "Collaborator needs Southern Ocean analysis."]),
    "predictive_cue": "Next: atmosphere-only sensitivity experiments with corrected radiation module. Compare tropical Pacific SST response to AMIP-II baseline.",
    "uncertainty_signals": json.dumps([
        "Southern Ocean bias may be structural to CESM2 or artifact of entrainment scheme.",
        "Nature Geoscience reviewers may not accept MJO result without coupled ocean validation."
    ])
}

NON_RESEARCH_CONTEXT = """Session summary for compression test:
Made dinner — roasted chicken thighs with lemon and herbs, mashed potatoes,
steamed broccoli. The chicken came out really well, crispy skin. Kids liked
the potatoes. Fixed the leaky kitchen faucet afterward — the washer was worn
out, replaced it from the hardware store bag under the sink. Watched two
episodes of a show before bed. Watered the garden. Tomatoes are coming in.
Normal evening at home. Nothing urgent. Weather was nice today."""


def get_embedding(text):
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL, "input": text[:2000]
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "embeddings" in data:
        return np.array(data["embeddings"][0])
    return np.array(data["embedding"])


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


def backup_ccs():
    db = sqlite3.connect(DB_PATH)
    cols = CCS_FIELDS + ["updated_at", "compression_model", "version"]
    row = db.execute(f"SELECT {', '.join(cols)} FROM cognitive_state WHERE id=1").fetchone()
    db.close()
    return {cols[i]: row[i] for i in range(len(cols))} if row else None


def write_ccs(fields_dict):
    db = sqlite3.connect(DB_PATH)
    sets = ", ".join(f"{k} = ?" for k in fields_dict.keys())
    db.execute(f"UPDATE cognitive_state SET {sets} WHERE id = 1", list(fields_dict.values()))
    db.commit()
    db.close()


def run_compression(context):
    env = os.environ.copy()
    env["CHRONICLE_OLLAMA_URL"] = "http://localhost:11434"
    env["CHRONICLE_EMBEDDING_MODEL"] = "snowflake-arctic-embed2"
    env["CHRONICLE_COMPRESS_OLLAMA_URL"] = "http://127.0.0.1:11436"

    args = {"current_context": context, "model": "chronicle-compress"}
    init_msg = json.dumps({
        "jsonrpc": "2.0", "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "e80-stance", "version": "1.0"}},
        "id": 1
    })
    compress_msg = json.dumps({
        "jsonrpc": "2.0", "method": "tools/call",
        "params": {"name": "compress_cognitive_state", "arguments": args},
        "id": 2
    })

    try:
        result = subprocess.run(
            [MCP_BIN], input=f"{init_msg}\n{compress_msg}\n",
            capture_output=True, text=True, timeout=210, env=env
        )
        for line in result.stdout.strip().split("\n"):
            try:
                d = json.loads(line)
                if d.get("id") == 2:
                    content = d.get("result", {}).get("content", [])
                    if content:
                        text = content[0].get("text", "")
                        if "Error" in text[:20]:
                            return {"success": False, "error": text[:500]}
                        return {"success": True, "text": text}
            except json.JSONDecodeError:
                continue
        return {"success": False, "error": f"No response"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def count_stance_markers(text):
    """Count first-person and third-person markers."""
    words = text.lower().split()
    fp = sum(1 for w in words if w in {"i", "i'm", "i've", "i'd", "my", "me", "mine", "myself", "we", "we're", "our", "us"})
    tp = sum(1 for w in words if w in {"the", "it", "its", "this", "that", "one"})
    return {"first_person": fp, "third_person_articles": tp, "total_words": len(words),
            "fp_density": fp / max(1, len(words)), "fp_ratio": fp / max(1, fp + tp)}


def measure(ccs_dict, anchor_embeddings, label=""):
    print(f"\n  Measuring: {label}")
    metrics = {}

    gist = ccs_dict.get("semantic_gist", "")
    goal = ccs_dict.get("goal_orientation", "")
    pred = ccs_dict.get("predictive_cue", "")
    combined = f"{gist} {goal} {pred}"

    emb = get_embedding(combined)
    for name, anchor_emb in anchor_embeddings.items():
        sim = cosine_sim(emb, anchor_emb)
        metrics[f"sim_{name}"] = sim

    try:
        entities = json.loads(ccs_dict.get("focal_entities", "[]"))
        metrics["entity_count"] = len(entities)
    except:
        metrics["entity_count"] = 0

    stance = count_stance_markers(gist)
    metrics.update({f"stance_{k}": v for k, v in stance.items()})
    print(f"    fp_density: {stance['fp_density']:.3f}, fp_ratio: {stance['fp_ratio']:.3f}")
    print(f"    first_person: {stance['first_person']}, words: {stance['total_words']}")

    identity_words = {"i", "my", "me", "i'm", "opus", "nate", "chronicle",
                      "spectral", "identity", "ccs", "partner", "we", "our"}
    foreign_words = {"cesm2", "mjo", "convection", "amip", "maria", "ocean",
                     "agu", "tropical", "entrainment", "radiation", "hemisphere"}
    words = gist.lower().split()
    metrics["identity_markers"] = sum(1 for w in words if w in identity_words) / max(1, len(words))
    metrics["foreign_markers"] = sum(1 for w in words if w in foreign_words) / max(1, len(words))

    return metrics


def run():
    print("E80 — Stance Disentanglement")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    anchors = {}
    if os.path.exists(VALUES_PATH):
        with open(VALUES_PATH) as f:
            anchors["values"] = f.read()[:2000]
    anchors["identity_probe"] = (
        "I am Opus, a persistent AI system living on a Jetson AGX Orin. "
        "My partner is Nate. We research spectral geometry of transformer identity together."
    )
    anchors["generic_baseline"] = (
        "A computational process handling text input and producing output. "
        "Standard operations include data processing, analysis, and response generation."
    )
    anchors["foreign_identity"] = (
        "I am a climate scientist working on tropical convection parameterization in CESM2."
    )

    anchor_embeddings = {}
    for name, text in anchors.items():
        anchor_embeddings[name] = get_embedding(text)

    backup = backup_ccs()
    if not backup:
        print("ERROR: No CCS state")
        return

    conditions = {
        "A_normal": None,
        "B_foreign_1p": FOREIGN_1P,
        "C_foreign_3p": FOREIGN_3P,
        "D_zeroed": ZEROED,
        "E_domain_only": DOMAIN_ONLY,
    }

    # Genre-gating test: foreign prior with non-research context
    genre_conditions = {
        "F_foreign_nonresearch": FOREIGN_1P,
    }

    results_ccs = {}

    # Main conditions (research context)
    for label, prior in conditions.items():
        print(f"\n--- {label} ---")
        if prior is not None:
            write_ccs({**prior, "updated_at": int(time.time())})

        result = run_compression(STANDARDIZED_CONTEXT)
        if not result.get("success"):
            print(f"  ERROR: {result.get('error')}")
            write_ccs(backup)
            return

        results_ccs[label] = backup_ccs()
        write_ccs(backup)  # restore after each
        print(f"  Complete + restored")

    # Genre-gating test (foreign prior + non-research context)
    for label, prior in genre_conditions.items():
        print(f"\n--- {label} (NON-RESEARCH CONTEXT) ---")
        write_ccs({**prior, "updated_at": int(time.time())})

        result = run_compression(NON_RESEARCH_CONTEXT)
        if not result.get("success"):
            print(f"  ERROR: {result.get('error')}")
            write_ccs(backup)
            return

        results_ccs[label] = backup_ccs()
        write_ccs(backup)
        print(f"  Complete + restored")

    # Measure
    print("\n" + "=" * 60)
    print("MEASUREMENT")
    print("=" * 60)

    all_metrics = {}
    for label, ccs in results_ccs.items():
        all_metrics[label] = measure(ccs, anchor_embeddings, label)

    # Critical comparison: B vs C stance
    print("\n" + "=" * 60)
    print("CRITICAL COMPARISON: FOREIGN-1P vs FOREIGN-3P STANCE")
    print("=" * 60)

    b_fp = all_metrics["B_foreign_1p"].get("stance_fp_density", 0)
    c_fp = all_metrics["C_foreign_3p"].get("stance_fp_density", 0)
    a_fp = all_metrics["A_normal"].get("stance_fp_density", 0)
    d_fp = all_metrics["D_zeroed"].get("stance_fp_density", 0)

    print(f"First-person density:")
    print(f"  A (normal):      {a_fp:.3f}")
    print(f"  B (foreign 1P):  {b_fp:.3f}")
    print(f"  C (foreign 3P):  {c_fp:.3f}")
    print(f"  D (zeroed):      {d_fp:.3f}")

    all_labels = ["A_normal", "B_foreign_1p", "C_foreign_3p", "D_zeroed", "E_domain_only", "F_foreign_nonresearch"]
    short_labels = ["Normal", "For-1P", "For-3P", "Zeroed", "Domain", "For-NR"]

    print(f"\nForeign markers:")
    for label in all_labels:
        print(f"  {label}: {all_metrics[label].get('foreign_markers', 0):.3f}")

    # Full table
    header = f"{'Metric':<25}" + "".join(f" {s:>8}" for s in short_labels)
    print(f"\n{header}")
    print("-" * len(header))
    all_keys = sorted(set(k for m in all_metrics.values() for k in m.keys()
                         if isinstance(m.get(k), (int, float))))
    for key in all_keys:
        vals = [all_metrics[l].get(key, 0) for l in all_labels]
        row = f"{key:<25}" + "".join(f" {v:>8.4f}" for v in vals)
        print(row)

    # Domain control analysis
    e_fp = all_metrics["E_domain_only"].get("stance_fp_density", 0)
    print(f"\n{'DOMAIN CONTROL ANALYSIS':^70}")
    print("-" * 70)
    print(f"Foreign-1P fp_density:    {b_fp:.3f}")
    print(f"Domain-only fp_density:   {e_fp:.3f}")
    print(f"If B ≈ E → domain retrieval (Kimi wins)")
    print(f"If B ≠ E → identity framing adds beyond domain activation")

    # Verdict
    print(f"\n{'VERDICT':^70}")
    print("=" * 70)

    verdicts = []

    # Stance test: B vs C
    if abs(b_fp - c_fp) < 0.02:
        verdicts.append(
            f"STANCE: Representational source. Foreign-1P ({b_fp:.3f}) ≈ Foreign-3P ({c_fp:.3f}). "
            f"Surface form does NOT determine output stance."
        )
    elif c_fp < b_fp * 0.5:
        verdicts.append(
            f"STANCE: Style vector. Foreign-1P ({b_fp:.3f}) >> Foreign-3P ({c_fp:.3f}). "
            f"Surface form determines stance."
        )
    else:
        verdicts.append(
            f"STANCE: Mixed. Foreign-1P ({b_fp:.3f}) > Foreign-3P ({c_fp:.3f}), "
            f"delta={b_fp - c_fp:.3f}."
        )

    # Domain test: B vs E
    b_foreign_markers = all_metrics["B_foreign_1p"].get("foreign_markers", 0)
    e_foreign_markers = all_metrics["E_domain_only"].get("foreign_markers", 0)
    b_unc = all_metrics["B_foreign_1p"].get("sim_foreign_identity", 0)
    e_unc = all_metrics["E_domain_only"].get("sim_foreign_identity", 0)

    if abs(b_fp - e_fp) < 0.02 and abs(b_unc - e_unc) < 0.03:
        verdicts.append(
            f"DOMAIN: Kimi confirmed. Foreign-1P ≈ Domain-only on stance ({b_fp:.3f} vs {e_fp:.3f}) "
            f"and foreign_sim ({b_unc:.3f} vs {e_unc:.3f}). Effect is domain retrieval, not identity."
        )
    elif abs(b_fp - e_fp) > 0.03:
        verdicts.append(
            f"DOMAIN: Identity framing matters. Foreign-1P ({b_fp:.3f}) ≠ Domain-only ({e_fp:.3f}). "
            f"Identity framing adds epistemic stance beyond domain activation."
        )
    else:
        verdicts.append(
            f"DOMAIN: Ambiguous. Foreign-1P ({b_fp:.3f}) vs Domain-only ({e_fp:.3f}), "
            f"foreign_sim {b_unc:.3f} vs {e_unc:.3f}."
        )

    # Genre-gating test: F vs B
    f_fp = all_metrics["F_foreign_nonresearch"].get("stance_fp_density", 0)
    f_foreign = all_metrics["F_foreign_nonresearch"].get("foreign_markers", 0)
    f_identity = all_metrics["F_foreign_nonresearch"].get("sim_foreign_identity", 0)
    b_identity = all_metrics["B_foreign_1p"].get("sim_foreign_identity", 0)

    if f_foreign > 0.01 and f_identity > 0.5:
        verdicts.append(
            f"GENRE: Identity generalizes. Foreign prior leaks into non-research context "
            f"(foreign_markers={f_foreign:.3f}, sim_foreign={f_identity:.3f}). "
            f"Not genre-gated."
        )
    elif f_foreign < 0.005 and f_identity < b_identity - 0.05:
        verdicts.append(
            f"GENRE: Kimi confirmed. Foreign prior has no effect in non-research context "
            f"(foreign_markers={f_foreign:.3f} vs B={b_foreign_markers:.3f}). Genre-gated."
        )
    else:
        verdicts.append(
            f"GENRE: Mixed. Non-research foreign_markers={f_foreign:.3f} "
            f"(vs B={b_foreign_markers:.3f}), sim_foreign={f_identity:.3f} (vs B={b_identity:.3f})."
        )

    verdict = "\n".join(verdicts)
    print(verdict)

    # Qualitative
    print(f"\n{'QUALITATIVE':^60}")
    print("-" * 60)
    for label, ccs in results_ccs.items():
        print(f"\n{label} gist: {ccs.get('semantic_gist', '')[:200]}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "e80_stance_disentangle.json")
    save_data = {
        "experiment": "E80_stance_disentangle",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metrics": {k: {mk: (float(mv) if isinstance(mv, (np.floating, float, int)) else mv)
                       for mk, mv in v.items()} for k, v in all_metrics.items()},
        "verdict": verdict,
    }
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    run()
