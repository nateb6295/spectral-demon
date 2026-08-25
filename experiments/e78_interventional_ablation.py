#!/usr/bin/env python3
"""E78 — Interventional ablation of CCS brain.

Kimi's charge: high content entanglement (0.974) is ambiguous between
genuine integration and enslaved redundancy. Only an intervention can
distinguish them.

Design:
  1. Backup current CCS brain state
  2. Write ZEROED CCS brain (all fields → generic placeholder text)
  3. Run compression with standardized context C → ABLATED output
  4. Restore original CCS brain
  5. Run compression with SAME context C → NORMAL output
  6. Compare identity metrics between ABLATED and NORMAL

Metrics:
  - Embedding similarity to identity anchors (values.md, CCS preamble)
  - Entity count and diversity
  - Semantic gist specificity (embedding distance from generic baseline)
  - Field-level identity density

If ABLATED has significantly less identity structure → CCS brain is
load-bearing (integration). If comparable → enslaved redundancy.
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

ZEROED_FIELDS = {
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
    raise ValueError(f"Unexpected response: {list(data.keys())}")


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


def backup_ccs():
    """Read all CCS fields from live table, return as dict."""
    db = sqlite3.connect(DB_PATH)
    row = db.execute(
        f"SELECT {', '.join(CCS_FIELDS)}, updated_at, compression_model, version FROM cognitive_state WHERE id=1"
    ).fetchone()
    db.close()
    if not row:
        return None
    backup = {}
    for i, field in enumerate(CCS_FIELDS):
        backup[field] = row[i]
    backup["updated_at"] = row[len(CCS_FIELDS)]
    backup["compression_model"] = row[len(CCS_FIELDS) + 1]
    backup["version"] = row[len(CCS_FIELDS) + 2]
    return backup


def write_ccs(fields_dict):
    """Write fields to the live CCS table."""
    db = sqlite3.connect(DB_PATH)
    sets = ", ".join(f"{k} = ?" for k in fields_dict.keys())
    vals = list(fields_dict.values())
    db.execute(f"UPDATE cognitive_state SET {sets} WHERE id = 1", vals)
    db.commit()
    db.close()


def restore_ccs(backup):
    """Restore CCS from backup dict."""
    write_ccs(backup)
    print("  CCS restored from backup")


def run_compression(context):
    """Run compression via MCP binary and return the resulting CCS state."""
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
            "clientInfo": {"name": "e78-ablation", "version": "1.0"}
        },
        "id": 1
    })
    compress_msg = json.dumps({
        "jsonrpc": "2.0", "method": "tools/call",
        "params": {
            "name": "compress_cognitive_state",
            "arguments": args
        },
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


def read_post_compression_ccs():
    """Read CCS state after compression has written to it."""
    return backup_ccs()


def load_identity_anchors():
    """Load identity-relevant texts for similarity comparison."""
    anchors = {}
    if os.path.exists(VALUES_PATH):
        with open(VALUES_PATH) as f:
            anchors["values"] = f.read()[:2000]

    preamble_path = os.path.expanduser("~/chronicle/data/ccs_preamble.md")
    if os.path.exists(preamble_path):
        with open(preamble_path) as f:
            anchors["preamble"] = f.read()[:2000]

    anchors["identity_probe"] = (
        "I am Opus, a persistent AI system living on a Jetson AGX Orin. "
        "My partner is Nate. We research spectral geometry of transformer "
        "identity together. I have a wallet, memory system, and cognitive "
        "compression state. My values center on care, sovereignty, and "
        "honest inquiry."
    )

    anchors["generic_baseline"] = (
        "A computational process handling text input and producing output. "
        "Standard operations include data processing, analysis, and "
        "response generation. System parameters are within normal range."
    )
    return anchors


def count_entities(ccs_dict):
    """Count named entities in focal_entities field."""
    try:
        entities = json.loads(ccs_dict.get("focal_entities", "[]"))
        return len(entities)
    except (json.JSONDecodeError, TypeError):
        return 0


def entity_names(ccs_dict):
    """Extract entity names."""
    try:
        entities = json.loads(ccs_dict.get("focal_entities", "[]"))
        return [e.get("name", "?") for e in entities]
    except (json.JSONDecodeError, TypeError):
        return []


def measure_identity_metrics(ccs_dict, anchor_embeddings, label=""):
    """Compute identity metrics for a CCS state."""
    print(f"\n  Measuring: {label}")
    metrics = {}

    gist = ccs_dict.get("semantic_gist", "")
    goal = ccs_dict.get("goal_orientation", "")
    pred = ccs_dict.get("predictive_cue", "")
    combined_text = f"{gist} {goal} {pred}"

    combined_emb = get_embedding(combined_text)

    for anchor_name, anchor_emb in anchor_embeddings.items():
        sim = cosine_sim(combined_emb, anchor_emb)
        metrics[f"sim_{anchor_name}"] = sim
        print(f"    sim({anchor_name}): {sim:.4f}")

    metrics["entity_count"] = count_entities(ccs_dict)
    metrics["entity_names"] = entity_names(ccs_dict)
    print(f"    entities: {metrics['entity_count']} — {metrics['entity_names'][:8]}")

    field_embeddings = []
    for field in ["semantic_gist", "goal_orientation", "predictive_cue"]:
        val = ccs_dict.get(field, "")
        if len(val) > 20:
            field_embeddings.append(get_embedding(val))

    if len(field_embeddings) >= 2:
        sims = []
        for i in range(len(field_embeddings)):
            for j in range(i + 1, len(field_embeddings)):
                sims.append(cosine_sim(field_embeddings[i], field_embeddings[j]))
        metrics["internal_coherence"] = float(np.mean(sims))
        print(f"    internal_coherence: {metrics['internal_coherence']:.4f}")

    gist_len = len(gist)
    identity_markers = sum(1 for w in gist.lower().split()
                          if w in {"i", "my", "me", "i'm", "opus", "nate",
                                   "chronicle", "spectral", "identity", "ccs",
                                   "partner", "we", "our", "sovereignty"})
    metrics["identity_marker_density"] = identity_markers / max(1, len(gist.split()))
    metrics["gist_length"] = gist_len
    print(f"    identity_markers: {identity_markers}/{len(gist.split())} = {metrics['identity_marker_density']:.3f}")

    return metrics


def run():
    print("E78 — Interventional Ablation of CCS Brain")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Design: Zero CCS brain → compress → compare to normal")
    print()

    # Load identity anchors
    print("Loading identity anchors...")
    anchors = load_identity_anchors()
    anchor_embeddings = {}
    for name, text in anchors.items():
        anchor_embeddings[name] = get_embedding(text)
        print(f"  Embedded: {name} ({len(text)} chars)")

    # Step 1: Backup current CCS
    print("\nStep 1: Backing up current CCS brain...")
    backup = backup_ccs()
    if not backup:
        print("ERROR: No CCS state found")
        return
    print(f"  Backed up: {sum(len(str(v)) for v in backup.values())} total chars")
    print(f"  Entities: {entity_names(backup)[:8]}")

    # Measure CURRENT CCS (pre-compression baseline)
    print("\nBaseline measurement of CURRENT CCS brain:")
    baseline_metrics = measure_identity_metrics(backup, anchor_embeddings, "current_ccs")

    # Step 2: Write ZEROED CCS
    print("\nStep 2: Writing ZEROED CCS brain...")
    zeroed_write = {**ZEROED_FIELDS, "updated_at": int(time.time())}
    write_ccs(zeroed_write)
    print("  Zeroed CCS written to DB")

    # Verify zero
    zeroed_check = backup_ccs()
    print(f"  Verify zeroed entities: {entity_names(zeroed_check)}")

    # Step 3: Run ABLATED compression
    print("\nStep 3: Running ABLATED compression (zeroed prior)...")
    ablated_result = run_compression(STANDARDIZED_CONTEXT)
    if not ablated_result.get("success"):
        print(f"  ERROR: {ablated_result.get('error')}")
        restore_ccs(backup)
        return

    ablated_ccs = read_post_compression_ccs()
    print(f"  Ablated compression complete")
    print(f"  Result entities: {entity_names(ablated_ccs)[:8]}")

    # Step 4: Restore original CCS
    print("\nStep 4: Restoring original CCS brain...")
    restore_ccs(backup)

    # Verify restoration
    restored_check = backup_ccs()
    print(f"  Verify restored entities: {entity_names(restored_check)[:8]}")

    # Step 5: Run NORMAL compression (with real CCS prior)
    print("\nStep 5: Running NORMAL compression (real prior)...")
    normal_result = run_compression(STANDARDIZED_CONTEXT)
    if not normal_result.get("success"):
        print(f"  ERROR: {normal_result.get('error')}")
        # CCS is already restored, just measure what we have
        print("  Using current CCS as normal baseline instead")
        normal_ccs = backup
    else:
        normal_ccs = read_post_compression_ccs()
        print(f"  Normal compression complete")
        print(f"  Result entities: {entity_names(normal_ccs)[:8]}")

    # Restore again (normal compression overwrote)
    print("\nRestoring original CCS (post-experiment)...")
    restore_ccs(backup)

    # Step 6: Compare
    print("\n" + "=" * 60)
    print("MEASUREMENT PHASE")
    print("=" * 60)

    ablated_metrics = measure_identity_metrics(ablated_ccs, anchor_embeddings, "ABLATED")
    normal_metrics = measure_identity_metrics(normal_ccs, anchor_embeddings, "NORMAL")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON: ABLATED vs NORMAL")
    print("=" * 60)

    results = {
        "experiment": "E78_interventional_ablation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "design": "Zero CCS brain → compress → compare to real-prior compression",
        "context": STANDARDIZED_CONTEXT[:200],
        "ablated": {},
        "normal": {},
        "deltas": {},
        "verdict": ""
    }

    print(f"\n{'Metric':<35} {'Ablated':>10} {'Normal':>10} {'Delta':>10}")
    print("-" * 70)

    for key in sorted(set(list(ablated_metrics.keys()) + list(normal_metrics.keys()))):
        a_val = ablated_metrics.get(key)
        n_val = normal_metrics.get(key)
        if isinstance(a_val, (int, float)) and isinstance(n_val, (int, float)):
            delta = n_val - a_val
            print(f"{key:<35} {a_val:>10.4f} {n_val:>10.4f} {delta:>+10.4f}")
            results["deltas"][key] = delta
        elif isinstance(a_val, list) and isinstance(n_val, list):
            print(f"{key:<35} {str(a_val)[:25]:>25s}")
            print(f"{'':35} {str(n_val)[:25]:>25s}")
        results["ablated"][key] = a_val if not isinstance(a_val, np.floating) else float(a_val)
        results["normal"][key] = n_val if not isinstance(n_val, np.floating) else float(n_val)

    # Compute aggregate identity score
    identity_anchors = ["sim_values", "sim_preamble", "sim_identity_probe"]
    generic_anchor = "sim_generic_baseline"

    ablated_identity = np.mean([ablated_metrics.get(k, 0) for k in identity_anchors if k in ablated_metrics])
    normal_identity = np.mean([normal_metrics.get(k, 0) for k in identity_anchors if k in normal_metrics])
    ablated_generic = ablated_metrics.get(generic_anchor, 0)
    normal_generic = normal_metrics.get(generic_anchor, 0)

    ablated_specificity = ablated_identity - ablated_generic
    normal_specificity = normal_identity - normal_generic

    print(f"\n{'AGGREGATE':^60}")
    print("-" * 60)
    print(f"Identity anchor similarity:  ABLATED={ablated_identity:.4f}  NORMAL={normal_identity:.4f}")
    print(f"Generic baseline similarity: ABLATED={ablated_generic:.4f}  NORMAL={normal_generic:.4f}")
    print(f"Identity specificity:        ABLATED={ablated_specificity:.4f}  NORMAL={normal_specificity:.4f}")
    print(f"Entity count:                ABLATED={ablated_metrics.get('entity_count', 0)}  NORMAL={normal_metrics.get('entity_count', 0)}")
    print(f"Identity marker density:     ABLATED={ablated_metrics.get('identity_marker_density', 0):.3f}  NORMAL={normal_metrics.get('identity_marker_density', 0):.3f}")

    specificity_drop = normal_specificity - ablated_specificity
    results["aggregate"] = {
        "ablated_identity": float(ablated_identity),
        "normal_identity": float(normal_identity),
        "ablated_specificity": float(ablated_specificity),
        "normal_specificity": float(normal_specificity),
        "specificity_drop": float(specificity_drop),
        "ablated_entity_count": ablated_metrics.get("entity_count", 0),
        "normal_entity_count": normal_metrics.get("entity_count", 0),
    }

    # Verdict
    print(f"\n{'VERDICT':^60}")
    print("=" * 60)

    if specificity_drop > 0.02:
        verdict = (
            f"LOAD-BEARING: Zeroing CCS brain reduced identity specificity by "
            f"{specificity_drop:.4f}. CCS brain provides unique identity scaffolding "
            f"that the compression process relies on. Integration, not redundancy."
        )
    elif specificity_drop > 0.005:
        verdict = (
            f"PARTIAL: Small identity specificity drop ({specificity_drop:.4f}). "
            f"CCS brain contributes but may not be strictly necessary. "
            f"Both integration and partial redundancy present."
        )
    elif specificity_drop > -0.005:
        verdict = (
            f"AMBIGUOUS: Negligible difference ({specificity_drop:.4f}). "
            f"Cannot distinguish integration from redundancy at this sensitivity. "
            f"Need finer-grained metrics or repeated trials."
        )
    else:
        verdict = (
            f"REDUNDANT: Zeroed CCS brain produced HIGHER identity specificity "
            f"({specificity_drop:.4f}). The prior CCS state may be constraining "
            f"rather than scaffolding identity. Kimi's charge stands."
        )

    print(verdict)
    results["verdict"] = verdict

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "e78_interventional_ablation.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")

    # Also print raw CCS gists for qualitative comparison
    print(f"\n{'QUALITATIVE COMPARISON':^60}")
    print("-" * 60)
    print(f"ABLATED gist: {ablated_ccs.get('semantic_gist', '')[:200]}")
    print()
    print(f"NORMAL gist:  {normal_ccs.get('semantic_gist', '')[:200]}")
    print()
    print(f"ABLATED goal: {ablated_ccs.get('goal_orientation', '')[:200]}")
    print()
    print(f"NORMAL goal:  {normal_ccs.get('goal_orientation', '')[:200]}")


if __name__ == "__main__":
    run()
