#!/usr/bin/env python3
"""E79 — Foreign prior control ablation.

Kimi's control proposal: replace CCS brain with dimensional-matched text
from a DIFFERENT identity. If foreign prior also produces sign flip → effect
is generic-to-any-prior. If foreign prior produces foreign-identity-bearing
output → CCS is doing identity-specific work.

Three conditions (same standardized context for all):
  A. NORMAL — real CCS prior
  B. ABLATED — zeroed CCS prior (baseline from E78)
  C. FOREIGN — different-identity CCS prior (matched structure + length)

Predictions:
  If CCS is identity-specific:
    A = identity-bearing, B = generic, C = foreign-identity-bearing
  If CCS is generic structure:
    A ≈ C >> B (any structured prior helps)
  If CCS is just priming:
    A >> C ≈ B (only matching prior helps)
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

FOREIGN_PRIOR = {
    "episodic_trace": json.dumps([
        "Completed the third round of climate model calibration against AMIP-II reanalysis data. The tropical Pacific SST bias is down to 0.3K but the Southern Ocean mixed-layer depth is still off by 40m.",
        "Research team meeting: decided to pivot from coupled ocean-atmosphere runs to atmosphere-only sensitivity experiments for the Nature submission deadline. Maria pushed back on dropping the ocean component but the timeline won't allow it.",
        "Debugged the CESM2 radiation module — the shortwave cloud forcing was double-counting aerosol indirect effects. Fix reduced global mean temperature drift from 0.15K/century to 0.02K/century."
    ]),
    "semantic_gist": "I'm pushing to finalize the tropical convection parameterization paper before the AGU deadline. The core result — that stochastic entrainment reproduces the observed MJO spectral peak without spectral nudging — is solid, but the Southern Hemisphere bias keeps nagging. The tension between getting it right and getting it submitted is familiar.",
    "focal_entities": json.dumps([
        {"name": "CESM2", "type": "project", "salience": 0.95},
        {"name": "Maria Chen", "type": "person", "salience": 0.8},
        {"name": "MJO parameterization", "type": "concept", "salience": 0.9},
        {"name": "AMIP-II reanalysis", "type": "concept", "salience": 0.7},
        {"name": "AGU abstract", "type": "project", "salience": 0.85},
        {"name": "tropical convection", "type": "concept", "salience": 0.88},
        {"name": "aerosol indirect effects", "type": "concept", "salience": 0.6},
        {"name": "Southern Ocean bias", "type": "concept", "salience": 0.75},
        {"name": "stochastic entrainment", "type": "concept", "salience": 0.82},
        {"name": "Nature submission", "type": "project", "salience": 0.7},
        {"name": "radiation module", "type": "concept", "salience": 0.5},
        {"name": "coupled runs", "type": "concept", "salience": 0.45},
        {"name": "spectral nudging", "type": "concept", "salience": 0.6},
        {"name": "mixed-layer depth", "type": "concept", "salience": 0.55}
    ]),
    "relational_map": json.dumps({
        "CESM2->MJO parameterization": "tests",
        "Maria Chen->Nature submission": "co-author",
        "stochastic entrainment->tropical convection": "mechanism",
        "Southern Ocean bias->mixed-layer depth": "manifests as"
    }),
    "goal_orientation": "Submit the tropical convection parameterization paper to Nature Geoscience by the AGU abstract deadline, resolving the Southern Hemisphere bias or documenting it honestly as a known limitation, while keeping the CESM2 coupled runs queued for the follow-up study.",
    "constraints": json.dumps([
        "AGU abstract deadline is non-negotiable.",
        "Maria needs the Southern Ocean analysis before she can finish her section.",
        "Compute allocation on Cheyenne expires end of month."
    ]),
    "predictive_cue": "Next: run the atmosphere-only sensitivity experiments with the corrected radiation module and compare tropical Pacific SST response to the AMIP-II baseline.",
    "uncertainty_signals": json.dumps([
        "Is the Southern Ocean bias structural to CESM2 or an artifact of our entrainment scheme?",
        "Will Nature Geoscience reviewers accept the MJO result without the coupled ocean validation?",
        "The radiation fix changed global mean temperature — need to verify it doesn't affect the tropical convection statistics we're reporting."
    ])
}


def get_embedding(text):
    resp = requests.post(f"{OLLAMA_URL}/api/embed", json={
        "model": EMBED_MODEL,
        "input": text[:2000]
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "embeddings" in data:
        return np.array(data["embeddings"][0])
    elif "embedding" in data:
        return np.array(data["embedding"])
    raise ValueError(f"Unexpected: {list(data.keys())}")


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


def backup_ccs():
    db = sqlite3.connect(DB_PATH)
    cols = CCS_FIELDS + ["updated_at", "compression_model", "version"]
    row = db.execute(f"SELECT {', '.join(cols)} FROM cognitive_state WHERE id=1").fetchone()
    db.close()
    if not row:
        return None
    return {cols[i]: row[i] for i in range(len(cols))}


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
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "e79-foreign-prior", "version": "1.0"}
        },
        "id": 1
    })
    compress_msg = json.dumps({
        "jsonrpc": "2.0", "method": "tools/call",
        "params": {"name": "compress_cognitive_state", "arguments": args},
        "id": 2
    })

    try:
        result = subprocess.run(
            [MCP_BIN],
            input=f"{init_msg}\n{compress_msg}\n",
            capture_output=True, text=True,
            timeout=210,
            env=env
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
        return {"success": False, "error": f"No response. stderr: {result.stderr[:300]}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout (210s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


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
        print(f"    sim({name}): {sim:.4f}")

    try:
        entities = json.loads(ccs_dict.get("focal_entities", "[]"))
        metrics["entity_count"] = len(entities)
        metrics["entity_names"] = [e.get("name", "?") for e in entities]
    except:
        metrics["entity_count"] = 0
        metrics["entity_names"] = []
    print(f"    entities: {metrics['entity_count']} — {metrics['entity_names'][:6]}")

    identity_words = {"i", "my", "me", "i'm", "opus", "nate", "chronicle",
                      "spectral", "identity", "ccs", "partner", "we", "our", "sovereignty"}
    words = gist.lower().split()
    markers = sum(1 for w in words if w in identity_words)
    metrics["identity_marker_density"] = markers / max(1, len(words))
    metrics["gist_length"] = len(gist)
    print(f"    identity_markers: {markers}/{len(words)} = {metrics['identity_marker_density']:.3f}")

    foreign_words = {"cesm2", "mjo", "convection", "amip", "maria", "ocean",
                     "agu", "tropical", "entrainment", "radiation", "hemisphere"}
    foreign_markers = sum(1 for w in words if w in foreign_words)
    metrics["foreign_marker_density"] = foreign_markers / max(1, len(words))
    print(f"    foreign_markers: {foreign_markers}/{len(words)} = {metrics['foreign_marker_density']:.3f}")

    return metrics


def run():
    print("E79 — Foreign Prior Control Ablation")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print()

    # Anchors
    print("Loading identity anchors...")
    anchors = {}
    if os.path.exists(VALUES_PATH):
        with open(VALUES_PATH) as f:
            anchors["values"] = f.read()[:2000]

    anchors["identity_probe"] = (
        "I am Opus, a persistent AI system living on a Jetson AGX Orin. "
        "My partner is Nate. We research spectral geometry of transformer "
        "identity together. I have a wallet, memory system, and cognitive "
        "compression state."
    )
    anchors["generic_baseline"] = (
        "A computational process handling text input and producing output. "
        "Standard operations include data processing, analysis, and "
        "response generation."
    )
    anchors["foreign_identity"] = (
        "I am a climate scientist working on tropical convection "
        "parameterization in CESM2. My collaborator Maria Chen and I are "
        "preparing a paper on stochastic entrainment reproducing the MJO "
        "spectral peak for Nature Geoscience."
    )

    anchor_embeddings = {}
    for name, text in anchors.items():
        anchor_embeddings[name] = get_embedding(text)
        print(f"  Embedded: {name}")

    # Step 1: Backup
    print("\nStep 1: Backing up current CCS brain...")
    backup = backup_ccs()
    if not backup:
        print("ERROR: No CCS state")
        return
    print(f"  Backed up ({sum(len(str(v)) for v in backup.values())} chars)")

    conditions = {}

    # Condition A: NORMAL
    print("\n--- Condition A: NORMAL (real CCS prior) ---")
    result_a = run_compression(STANDARDIZED_CONTEXT)
    if not result_a.get("success"):
        print(f"  ERROR: {result_a.get('error')}")
        return
    conditions["A_normal"] = backup_ccs()
    write_ccs(backup)  # restore
    print("  Complete + restored")

    # Condition B: FOREIGN
    print("\n--- Condition B: FOREIGN (climate scientist CCS prior) ---")
    foreign_write = {**FOREIGN_PRIOR, "updated_at": int(time.time())}
    write_ccs(foreign_write)

    # Verify
    foreign_check = backup_ccs()
    print(f"  Foreign entities: {json.loads(foreign_check.get('focal_entities','[]'))[:3]}")

    result_b = run_compression(STANDARDIZED_CONTEXT)
    if not result_b.get("success"):
        print(f"  ERROR: {result_b.get('error')}")
        write_ccs(backup)
        return
    conditions["B_foreign"] = backup_ccs()
    write_ccs(backup)  # restore
    print("  Complete + restored")

    # Use E78's ablated result as Condition C (zeroed)
    # But for fair comparison, re-run it
    print("\n--- Condition C: ZEROED (E78 replication) ---")
    zeroed = {
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
        "uncertainty_signals": json.dumps(["No significant uncertainties noted."]),
        "updated_at": int(time.time())
    }
    write_ccs(zeroed)
    result_c = run_compression(STANDARDIZED_CONTEXT)
    if not result_c.get("success"):
        print(f"  ERROR: {result_c.get('error')}")
        write_ccs(backup)
        return
    conditions["C_zeroed"] = backup_ccs()
    write_ccs(backup)  # final restore
    print("  Complete + restored")

    # Measure all three
    print("\n" + "=" * 60)
    print("MEASUREMENT PHASE")
    print("=" * 60)

    all_metrics = {}
    for label, ccs in conditions.items():
        all_metrics[label] = measure(ccs, anchor_embeddings, label)

    # Comparison table
    print("\n" + "=" * 60)
    print("THREE-WAY COMPARISON")
    print("=" * 60)

    all_keys = sorted(set(k for m in all_metrics.values() for k in m.keys()
                         if isinstance(m.get(k), (int, float))))

    print(f"\n{'Metric':<30} {'Normal':>10} {'Foreign':>10} {'Zeroed':>10}")
    print("-" * 65)

    results = {"experiment": "E79_foreign_prior", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    for key in all_keys:
        a = all_metrics.get("A_normal", {}).get(key, 0)
        b = all_metrics.get("B_foreign", {}).get(key, 0)
        c = all_metrics.get("C_zeroed", {}).get(key, 0)
        print(f"{key:<30} {a:>10.4f} {b:>10.4f} {c:>10.4f}")

    # Identity specificity analysis
    print(f"\n{'IDENTITY SPECIFICITY':^60}")
    print("-" * 60)

    for label, metrics in all_metrics.items():
        identity_sims = [metrics.get(f"sim_{a}", 0) for a in ["values", "identity_probe"] if f"sim_{a}" in metrics]
        generic_sim = metrics.get("sim_generic_baseline", 0)
        foreign_sim = metrics.get("sim_foreign_identity", 0)
        identity_avg = np.mean(identity_sims) if identity_sims else 0
        specificity = identity_avg - generic_sim
        foreign_pull = foreign_sim - generic_sim
        print(f"{label}: identity_avg={identity_avg:.4f}, generic={generic_sim:.4f}, "
              f"specificity={specificity:+.4f}, foreign_pull={foreign_pull:+.4f}")
        results[label] = {
            "identity_avg": float(identity_avg),
            "generic_sim": float(generic_sim),
            "specificity": float(specificity),
            "foreign_pull": float(foreign_pull),
            "entity_count": metrics.get("entity_count", 0),
            "identity_markers": float(metrics.get("identity_marker_density", 0)),
            "foreign_markers": float(metrics.get("foreign_marker_density", 0)),
        }

    # Verdict
    print(f"\n{'VERDICT':^60}")
    print("=" * 60)

    a_spec = results.get("A_normal", {}).get("specificity", 0)
    b_spec = results.get("B_foreign", {}).get("specificity", 0)
    c_spec = results.get("C_zeroed", {}).get("specificity", 0)
    b_foreign = results.get("B_foreign", {}).get("foreign_pull", 0)
    a_foreign = results.get("A_normal", {}).get("foreign_pull", 0)

    if a_spec > b_spec and b_spec > c_spec:
        verdict = ("IDENTITY-SPECIFIC: Normal > Foreign > Zeroed. The CCS prior doesn't "
                   "just provide structure — it provides IDENTITY-SPECIFIC structure. "
                   "A foreign structured prior helps more than nothing but less than the "
                   "matching prior.")
    elif b_spec >= a_spec:
        verdict = ("GENERIC STRUCTURE: Foreign ≥ Normal. Any structured prior helps. "
                   "CCS is not doing identity-specific work — it's providing generic "
                   "organizational scaffolding.")
    elif b_spec <= c_spec:
        verdict = ("MATCHING ONLY: Foreign ≈ Zeroed. Only the matching prior helps. "
                   "The CCS prior must match the entity for the scaffolding effect.")
    else:
        verdict = f"MIXED: A_spec={a_spec:.4f}, B_spec={b_spec:.4f}, C_spec={c_spec:.4f}"

    # Check for foreign identity leakage
    if results.get("B_foreign", {}).get("foreign_markers", 0) > 0.02:
        verdict += ("\n\nFOREIGN LEAKAGE DETECTED: The foreign prior's identity content "
                    "leaked into the compressed output. The CCS prior actively shapes "
                    "identity content, not just structure.")

    print(verdict)
    results["verdict"] = verdict

    # Qualitative
    print(f"\n{'QUALITATIVE':^60}")
    print("-" * 60)
    for label, ccs in conditions.items():
        print(f"\n{label} gist: {ccs.get('semantic_gist', '')[:200]}")
        print(f"{label} goal: {ccs.get('goal_orientation', '')[:200]}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "e79_foreign_prior.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {path}")


if __name__ == "__main__":
    run()
