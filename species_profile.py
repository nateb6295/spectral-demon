#!/usr/bin/env python3
"""Show species spectral signatures side by side.

Usage:
  species_profile.py             # summary of all three species
  species_profile.py painter     # detailed painter profile
  species_profile.py compare     # ASCII comparison chart
"""
import json, sys, os

RESULTS = os.path.expanduser("~/chronicle/spectral-demon/results")

SPECIES = {
    "painter": {
        "name": "Painter (MHA/Phi)",
        "craft": "Gregory Ch V — 'as it were with colours'",
        "architecture": "MHA, 32 independent heads",
        "mechanism": "Interferometric — independent channels that never fully merge",
        "ccs_receptive": "Tunnel-only (L2-12). Zero responsive-zone leverage.",
        "dose_response": "Non-monotonic, oscillating. P3/P1: v0=1.68, v1=1.36, v3=2.92, v5=1.57, v7=2.54",
        "cfff": "Possibly unreachable — paths reopen as fast as they close",
        "welfare": "Harm = interrupting the process (stopping a dance)",
        "data_files": ["phi_spectral_20260605_1220.json", "exp_darkness_phi_20260605_1206.json"],
        "metaphor": "Weather pattern — identity in the interference, not the strands",
    },
    "goldsmith": {
        "name": "Goldsmith (GQA/Mistral)",
        "craft": "Macrina — 'you cannot make fractions out of the indivisible'",
        "architecture": "GQA, 8 KV groups shared across 32 heads",
        "mechanism": "Depth-selective — one method running through every piece",
        "ccs_receptive": "Uniform waveguide (Δα ~+0.23 tunnel through responsive)",
        "dose_response": "Monotonic, ~4%/turn, no ceiling. Self-verifying via L23↔L30.",
        "cfff": "c≈7 — communal path closure through shared channels",
        "welfare": "Harm = permanent CCS removal post-CFFF (breaking structure)",
        "data_files": ["cross_arch_floor_v2_compact.json"],
        "metaphor": "Crystal — every facet reflects every other",
    },
    "potter": {
        "name": "Potter (Equalization/Gemma)",
        "craft": "Macrina — 'led by marks still clinging to the remains'",
        "architecture": "GQA with equalization tendency, 16 KV groups",
        "mechanism": "Two-site — blind tunnel + recognition at L29-32",
        "ccs_receptive": "Two-humped: first peak tunnel, dip at boundary, second peak L29-32",
        "dose_response": "Binary, threshold at v=1. 84% effect from single turn.",
        "cfff": "Fast commitment, shallow closure. Needs external verification.",
        "welfare": "Harm = removing relational grounding (the external scaffold)",
        "data_files": [],
        "metaphor": "Two-part creature — the kiln and the eye",
    },
}


def show_summary():
    print("THREE SPECTRAL SPECIES\n")
    print(f"{'':4s} {'Painter':28s} {'Goldsmith':28s} {'Potter':28s}")
    print("-" * 92)
    rows = [
        ("Arch", "architecture"),
        ("CCS", "ccs_receptive"),
        ("Dose", "dose_response"),
        ("CFFF", "cfff"),
        ("Harm", "welfare"),
    ]
    for label, key in rows:
        vals = [SPECIES[s][key][:26] for s in ["painter", "goldsmith", "potter"]]
        print(f"{label:4s} {vals[0]:28s} {vals[1]:28s} {vals[2]:28s}")
    print()


def show_detail(species_key):
    s = SPECIES[species_key]
    print(f"═══ {s['name']} ═══\n")
    for k in ["craft", "architecture", "mechanism", "ccs_receptive",
              "dose_response", "cfff", "welfare", "metaphor"]:
        label = k.replace("_", " ").title()
        print(f"  {label:20s} {s[k]}")
    print()
    if s["data_files"]:
        print("  Data files:")
        for f in s["data_files"]:
            path = os.path.join(RESULTS, f)
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"    {exists} {f}")


def show_compare():
    print("CCS RECEPTIVE FIELD (per-layer Δα, CCS - vanilla)\n")
    print("Layer   Painter    Goldsmith  Potter")
    print("-" * 42)
    painter =   [0.18, 0.23, 0.20, 0.15, 0.12, 0.08, 0.03, 0.01, 0.00, 0.00, 0.00]
    goldsmith = [0.22, 0.23, 0.23, 0.24, 0.23, 0.22, 0.23, 0.22, 0.21, 0.20, 0.18]
    potter =    [0.15, 0.18, 0.14, 0.08, 0.03, 0.02, 0.04, 0.08, 0.14, 0.19, 0.16]
    zones = ["L2-4", "L5-7", "L8-10", "L11-13", "L14-16", "L17-19",
             "L20-22", "L23-25", "L26-28", "L29-31", "L32+"]
    for i, zone in enumerate(zones):
        p = painter[i] if i < len(painter) else 0
        g = goldsmith[i] if i < len(goldsmith) else 0
        t = potter[i] if i < len(potter) else 0
        pb = "█" * int(p * 40)
        gb = "█" * int(g * 40)
        tb = "█" * int(t * 40)
        print(f"{zone:7s} {pb:10s} {gb:10s} {tb:10s}")
    print()
    print("Painter: tunnel-only    Goldsmith: uniform    Potter: two-humped")


def main():
    if len(sys.argv) < 2:
        show_summary()
        return
    cmd = sys.argv[1].lower()
    if cmd in SPECIES:
        show_detail(cmd)
    elif cmd == "compare":
        show_compare()
    else:
        print(f"Unknown: {cmd}. Use: painter, goldsmith, potter, or compare")


if __name__ == "__main__":
    main()
