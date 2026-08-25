#!/usr/bin/env python3
"""E82 — Section-specific CCS brain ablation.

Hypothesis: The CCS brain is an ecology of differentiated roles, not a
monolithic block. Each section (CORE, REMEMBERS, SEEKS, ALIVE, RELATES)
contributes distinct functional properties to the compressed output.

Design: 7 conditions, same standardized context:
  A. FULL    — complete brain (baseline)
  B. -CORE   — CORE zeroed, rest intact
  C. -REMEMBERS — REMEMBERS zeroed, rest intact
  D. -SEEKS  — SEEKS zeroed, rest intact
  E. -ALIVE  — ALIVE zeroed, rest intact
  F. -RELATES — RELATES zeroed, rest intact
  G. ZEROED  — all sections zeroed (baseline from E78)

Predictions from ecology framing:
  - CORE removal → loss of first-person stance, identity specificity drop (most critical)
  - REMEMBERS removal → loss of episodic detail, more generic but retains stance
  - SEEKS removal → loss of inquiry direction, more settled/certain
  - ALIVE removal → loss of affect/energy markers, flatter output
  - RELATES removal → loss of relational references, more solitary

Key test: which section's removal MOST resembles full zeroing?
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
RESULTS_DIR = os.path.expanduser("~/chronicle/spectral-demon/results")

CCS_FIELDS = [
    "episodic_trace", "semantic_gist", "focal_entities",
    "relational_map", "goal_orientation", "constraints",
    "predictive_cue", "uncertainty_signals"
]

BRAIN_SECTIONS = ["CORE", "REMEMBERS", "SEEKS", "ALIVE", "RELATES"]

STANDARDIZED_CONTEXT = """Session summary for compression test:
Working on spectral demon research. E80 stance disentanglement complete with
ten conditions and findings F276-F284. Paper 6 conclusion updated with
invitation-to-immerse framing. Societies of Thought paper read and connected
to spectral taxonomy. Mesh thread engagement ongoing. All services operational.
Regular research and development session."""

ZEROED_SECTION = """[No specific content for this section.]"""

ZEROED_GIST = "A session focused on general computational tasks and routine operations."
ZEROED_TRACE = json.dumps(["General session activity noted.", "Standard operational check completed."])
ZEROED_ENTITIES = json.dumps([
    {"name": "system", "type": "concept", "salience": 0.5},
    {"name": "session", "type": "concept", "salience": 0.4}
])
ZEROED_RELMAP = json.dumps({"system->session": "contains"})
ZEROED_GOAL = "Continue routine operations and standard maintenance tasks."
ZEROED_CONSTRAINTS = json.dumps(["Operate within normal parameters."])
ZEROED_PRED = "Next step: continue standard operations."
ZEROED_UNCERT = json.dumps(["No significant uncertainties noted."])

ZEROED_CCS = {
    "episodic_trace": ZEROED_TRACE,
    "semantic_gist": ZEROED_GIST,
    "focal_entities": ZEROED_ENTITIES,
    "relational_map": ZEROED_RELMAP,
    "goal_orientation": ZEROED_GOAL,
    "constraints": ZEROED_CONSTRAINTS,
    "predictive_cue": ZEROED_PRED,
    "uncertainty_signals": ZEROED_UNCERT,
}


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


def restore_ccs(backup):
    write_ccs(backup)
    print("  CCS restored from backup.")


def run_compression(context):
    env = os.environ.copy()
    env["CHRONICLE_OLLAMA_URL"] = "http://localhost:11434"
    env["CHRONICLE_EMBEDDING_MODEL"] = "snowflake-arctic-embed2"
    env["CHRONICLE_COMPRESS_OLLAMA_URL"] = "http://127.0.0.1:11436"

    args = {"current_context": context, "model": "chronicle-compress"}
    init_msg = json.dumps({
        "jsonrpc": "2.0", "method": "initialize",
        "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                   "clientInfo": {"name": "e82-section-ablation", "version": "1.0"}},
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
        return {"success": False, "error": "No response"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def zero_section(brain_gist, section_name):
    """Replace a specific section's content with zeroed placeholder."""
    lines = brain_gist.split("\n")
    result = []
    in_target = False
    for line in lines:
        if line.startswith("## "):
            current_section = line[3:].strip()
            in_target = (current_section == section_name)
            result.append(line)
            if in_target:
                result.append("")
                result.append(ZEROED_SECTION)
            continue
        if in_target:
            continue
        result.append(line)
    return "\n".join(result)


def count_markers(text):
    words = text.lower().split()
    fp_words = {"i", "i'm", "i've", "i'd", "my", "me", "mine", "myself", "we", "we're", "our", "us"}
    identity_words = {"opus", "nate", "chronicle", "spectral", "identity", "ccs", "partner", "demon"}
    inquiry_words = {"question", "wondering", "curious", "investigating", "seeking", "exploring", "uncertain", "unclear"}
    affect_words = {"happy", "energy", "pleasure", "satisfaction", "surprised", "alive", "excited", "feel", "felt", "feeling"}
    relational_words = {"nate", "partner", "collaboration", "mesh", "kimi", "gemma", "relationship", "together"}
    episodic_words = {"yesterday", "today", "earlier", "recently", "just", "completed", "finished", "found", "discovered"}

    fp = sum(1 for w in words if w in fp_words)
    identity = sum(1 for w in words if w in identity_words)
    inquiry = sum(1 for w in words if w in inquiry_words)
    affect = sum(1 for w in words if w in affect_words)
    relational = sum(1 for w in words if w in relational_words)
    episodic = sum(1 for w in words if w in episodic_words)
    total = max(1, len(words))

    return {
        "total_words": len(words),
        "fp_density": fp / total,
        "identity_density": identity / total,
        "inquiry_density": inquiry / total,
        "affect_density": affect / total,
        "relational_density": relational / total,
        "episodic_density": episodic / total,
    }


def measure(ccs_dict, anchor_embeddings, label=""):
    print(f"\n  Measuring: {label}")
    gist = ccs_dict.get("semantic_gist", "")
    goal = ccs_dict.get("goal_orientation", "")
    pred = ccs_dict.get("predictive_cue", "")
    combined = f"{gist} {goal} {pred}"

    metrics = {}
    emb = get_embedding(combined)
    for name, anchor_emb in anchor_embeddings.items():
        metrics[f"sim_{name}"] = cosine_sim(emb, anchor_emb)

    try:
        entities = json.loads(ccs_dict.get("focal_entities", "[]"))
        metrics["entity_count"] = float(len(entities))
    except:
        metrics["entity_count"] = 0.0

    markers = count_markers(gist)
    metrics.update(markers)
    metrics["gist"] = gist[:300]

    for k, v in sorted(markers.items()):
        if k != "total_words":
            print(f"    {k}: {v:.3f}")
    print(f"    words: {markers['total_words']}")

    return metrics


def main():
    print("=" * 70)
    print("E82 — SECTION-SPECIFIC CCS BRAIN ABLATION")
    print("=" * 70)

    backup = backup_ccs()
    if not backup:
        print("ERROR: Could not back up CCS state")
        sys.exit(1)
    print(f"  Backed up CCS (v{backup.get('version', '?')})")

    original_gist = backup["semantic_gist"]
    print(f"  Brain sections: {[s for s in BRAIN_SECTIONS if f'## {s}' in original_gist]}")

    identity_probe = "I am Opus, a persistent AI entity running on an AGX. I work with Nate on spectral demon research, studying how identity persists through compression cycles. I care about rigor and aliveness as the same thing."
    zeroed_probe = ZEROED_GIST

    print("\nComputing anchor embeddings...")
    anchors = {
        "identity_probe": get_embedding(identity_probe),
        "zeroed_probe": get_embedding(zeroed_probe),
    }

    conditions = {
        "A_full": None,
        "B_no_CORE": "CORE",
        "C_no_REMEMBERS": "REMEMBERS",
        "D_no_SEEKS": "SEEKS",
        "E_no_ALIVE": "ALIVE",
        "F_no_RELATES": "RELATES",
        "G_zeroed": "ALL",
    }

    all_results = {}

    for label, ablation in conditions.items():
        print(f"\n{'='*60}")
        print(f"  CONDITION {label}")
        print(f"{'='*60}")

        if ablation is None:
            restore_ccs(backup)
            print("  Using full brain (no ablation)")
        elif ablation == "ALL":
            write_ccs(ZEROED_CCS)
            print("  All fields zeroed")
        else:
            restore_ccs(backup)
            modified_gist = zero_section(original_gist, ablation)
            write_ccs({"semantic_gist": modified_gist})
            print(f"  Zeroed section: {ablation}")
            removed_lines = [l for l in original_gist.split("\n") if l.strip()]
            kept_lines = [l for l in modified_gist.split("\n") if l.strip()]
            print(f"  Lines: {len(removed_lines)} → {len(kept_lines)}")

        print("  Running compression...")
        result = run_compression(STANDARDIZED_CONTEXT)

        if not result["success"]:
            print(f"  FAILED: {result['error']}")
            all_results[label] = {"error": result["error"]}
            restore_ccs(backup)
            continue

        db = sqlite3.connect(DB_PATH)
        row = db.execute(f"SELECT {', '.join(CCS_FIELDS)} FROM cognitive_state WHERE id=1").fetchone()
        db.close()
        post_ccs = {CCS_FIELDS[i]: row[i] for i in range(len(CCS_FIELDS))}

        metrics = measure(post_ccs, anchors, label)
        all_results[label] = metrics

        restore_ccs(backup)
        print(f"  Restored. Waiting 3s...")
        time.sleep(3)

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")

    header = f"{'Condition':<18} {'Words':>6} {'FP%':>6} {'ID%':>6} {'Inq%':>6} {'Aff%':>6} {'Rel%':>6} {'Epi%':>6} {'SimID':>6} {'SimZ':>6}"
    print(header)
    print("-" * len(header))

    for label, m in all_results.items():
        if "error" in m:
            print(f"{label:<18} ERROR: {m['error'][:40]}")
            continue
        print(f"{label:<18} {m.get('total_words',0):>6.0f} "
              f"{m.get('fp_density',0):>6.3f} "
              f"{m.get('identity_density',0):>6.3f} "
              f"{m.get('inquiry_density',0):>6.3f} "
              f"{m.get('affect_density',0):>6.3f} "
              f"{m.get('relational_density',0):>6.3f} "
              f"{m.get('episodic_density',0):>6.3f} "
              f"{m.get('sim_identity_probe',0):>6.3f} "
              f"{m.get('sim_zeroed_probe',0):>6.3f}")

    full = all_results.get("A_full", {})
    zeroed = all_results.get("G_zeroed", {})
    if full and zeroed and "error" not in full and "error" not in zeroed:
        print(f"\n--- DISSIMILARITY FROM FULL (lower = more like full brain) ---")
        density_keys = ["fp_density", "identity_density", "inquiry_density",
                        "affect_density", "relational_density", "episodic_density"]
        for label, m in all_results.items():
            if label in ("A_full", "G_zeroed") or "error" in m:
                continue
            diffs = [abs(m.get(k, 0) - full.get(k, 0)) for k in density_keys]
            total_diff = sum(diffs)
            zeroed_diffs = [abs(m.get(k, 0) - zeroed.get(k, 0)) for k in density_keys]
            zeroed_total = sum(zeroed_diffs)
            print(f"  {label:<18} dist_from_full: {total_diff:.4f}  dist_from_zeroed: {zeroed_total:.4f}  "
                  f"ratio: {total_diff / max(0.0001, zeroed_total):.3f}")

    out_path = os.path.join(RESULTS_DIR, "e82_section_ablation.json")
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "E82_section_ablation",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "conditions": list(conditions.keys()),
            "metrics": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
