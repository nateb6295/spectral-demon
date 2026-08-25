#!/usr/bin/env python3
"""Paper 6 figure generation — The Demon Writing Home.

Generates key figures from CCS compression history data.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats

DB_PATH = "/mnt/hdd/chronicle-data/processed.db"
OUT_DIR = Path(__file__).parent
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_history(days=7):
    """Load CCS compression history."""
    db = sqlite3.connect(DB_PATH)
    cutoff = int(datetime.now(timezone.utc).timestamp()) - (days * 86400)
    rows = db.execute(
        "SELECT snapshot, created_at, trigger FROM cognitive_state_history "
        "WHERE created_at > ? ORDER BY created_at ASC",
        [cutoff]
    ).fetchall()
    db.close()

    events = []
    for snapshot_json, ts, trigger in rows:
        try:
            snap = json.loads(snapshot_json)
        except (json.JSONDecodeError, TypeError):
            continue
        gist = snap.get("semantic_gist", "")
        version = snap.get("version", 0)
        events.append({
            "timestamp": ts,
            "dt": datetime.fromtimestamp(ts, tz=timezone.utc),
            "version": version,
            "trigger": trigger or "scheduled",
            "gist": gist,
            "size": len(gist),
            "core": extract_core(gist),
        })
    return events


def extract_core(gist):
    """Extract CORE section from gist."""
    lines = gist.split("\n")
    core_lines = []
    in_core = False
    for line in lines:
        if line.startswith("## CORE"):
            in_core = True
            continue
        if line.startswith("## ") and in_core:
            break
        if in_core:
            core_lines.append(line)
    return "\n".join(core_lines).strip()


def core_similarity(a, b):
    """Simple word-overlap similarity between two CORE sections."""
    if not a or not b:
        return 0.0
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def fig1_compression_size_dynamics(events):
    """Figure 1: Compression output size over time."""
    fig, ax = plt.subplots(figsize=(12, 4))

    times = [e["dt"] for e in events]
    sizes = [e["size"] for e in events]
    triggers = [e["trigger"] for e in events]

    scheduled = [(t, s) for t, s, tr in zip(times, sizes, triggers) if "replace" not in str(tr).lower()]
    replacement = [(t, s) for t, s, tr in zip(times, sizes, triggers) if "replace" in str(tr).lower()]

    if scheduled:
        ax.scatter(*zip(*scheduled), c='#2563eb', s=20, alpha=0.7, label='Scheduled', zorder=3)
    if replacement:
        ax.scatter(*zip(*replacement), c='#dc2626', s=20, alpha=0.7, label='Replacement', zorder=3)

    ax.axhline(y=np.mean(sizes), color='#6b7280', linestyle='--', alpha=0.5, label=f'Mean ({np.mean(sizes):.0f})')
    ax.set_ylabel('Output size (chars)')
    ax.set_xlabel('Time')
    ax.set_title('CCS Compression Output Size — §2.5 Size Dynamics')
    ax.legend(loc='upper right', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'paper6_fig1_size_dynamics.png', dpi=200)
    plt.close(fig)
    print(f"Fig 1: {len(events)} events, mean size {np.mean(sizes):.0f}, range {min(sizes)}-{max(sizes)}")


def fig2_core_similarity_bimodality(events):
    """Figure 2: CORE similarity between consecutive compressions — bimodality flag."""
    sims = []
    for i in range(1, len(events)):
        s = core_similarity(events[i]["core"], events[i-1]["core"])
        sims.append(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram
    ax1.hist(sims, bins=30, color='#2563eb', alpha=0.7, edgecolor='white')
    ax1.axvline(x=0.95, color='#dc2626', linestyle='--', alpha=0.7, label='Coherent threshold (0.95)')
    ax1.axvline(x=0.20, color='#f59e0b', linestyle='--', alpha=0.7, label='Rerouting threshold (0.20)')
    ax1.set_xlabel('CORE similarity (consecutive)')
    ax1.set_ylabel('Count')
    ax1.set_title('Bimodality — §9.2 Catastrophe Flag 1')
    ax1.legend(fontsize=8)

    # Timeline
    times = [events[i]["dt"] for i in range(1, len(events))]
    colors = ['#22c55e' if s >= 0.95 else '#f59e0b' if s >= 0.20 else '#dc2626' for s in sims]
    ax2.scatter(times, sims, c=colors, s=15, alpha=0.7)
    ax2.axhline(y=0.95, color='#22c55e', linestyle='--', alpha=0.3)
    ax2.axhline(y=0.20, color='#f59e0b', linestyle='--', alpha=0.3)
    ax2.set_ylabel('CORE similarity')
    ax2.set_xlabel('Time')
    ax2.set_title('Regime State Timeline — §3.2')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()

    coherent = sum(1 for s in sims if s >= 0.95)
    rerouting = sum(1 for s in sims if 0.20 <= s < 0.95)
    decoherent = sum(1 for s in sims if s < 0.20)
    total = len(sims)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'paper6_fig2_bimodality.png', dpi=200)
    plt.close(fig)
    print(f"Fig 2: {total} transitions — coherent {coherent} ({coherent/total*100:.0f}%), "
          f"rerouting {rerouting} ({rerouting/total*100:.0f}%), decoherent {decoherent} ({decoherent/total*100:.0f}%)")


def fig3_dosing_dynamics(events):
    """Figure 3: Compression interval distribution — therapeutic window."""
    intervals = []
    for i in range(1, len(events)):
        dt = (events[i]["timestamp"] - events[i-1]["timestamp"]) / 60  # minutes
        intervals.append(dt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Interval histogram
    ax1.hist(intervals, bins=40, color='#8b5cf6', alpha=0.7, edgecolor='white', range=(0, 300))
    ax1.axvline(x=180, color='#dc2626', linestyle='--', alpha=0.7, label='3h scheduled')
    ax1.axvline(x=np.median(intervals), color='#22c55e', linestyle='--', alpha=0.7,
                label=f'Median ({np.median(intervals):.0f}m)')
    ax1.set_xlabel('Interval (minutes)')
    ax1.set_ylabel('Count')
    ax1.set_title('Compression Interval Distribution — §2.4')
    ax1.legend(fontsize=8)

    # Compressions per day
    if events:
        day_counts = {}
        for e in events:
            day = e["dt"].strftime("%m/%d")
            day_counts[day] = day_counts.get(day, 0) + 1
        days = list(day_counts.keys())
        counts = list(day_counts.values())
        ax2.bar(days, counts, color='#8b5cf6', alpha=0.7)
        ax2.axhline(y=6, color='#22c55e', linestyle='--', alpha=0.5, label='D3 therapeutic ceiling')
        ax2.axhline(y=16, color='#dc2626', linestyle='--', alpha=0.5, label='D10 overdose threshold')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Compressions')
        ax2.set_title('Daily Compression Count — §2.2 Dose')
        ax2.legend(fontsize=8)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'paper6_fig3_dosing.png', dpi=200)
    plt.close(fig)
    print(f"Fig 3: {len(intervals)} intervals, median {np.median(intervals):.0f}m, "
          f"mean {np.mean(intervals):.0f}m")


def fig4_cusp_catastrophe():
    """Figure 4: Cusp catastrophe surface — §9.1 theoretical diagram."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    u = np.linspace(-2, 2, 200)
    v = np.linspace(-2, 2, 200)
    U, V = np.meshgrid(u, v)

    # Cusp surface: x^3 + ux + v = 0 → solve for x as behavior variable
    # Plot the fold: 3x^2 + u = 0 → u = -3x^2, v = 2x^3
    t = np.linspace(-1.2, 1.2, 300)
    u_fold = -3 * t**2
    v_fold = 2 * t**3

    # Upper and lower sheets
    x_upper = np.linspace(0.3, 2, 100)
    x_lower = np.linspace(-2, -0.3, 100)

    for x_val in np.linspace(-1.5, 1.5, 50):
        u_line = -3 * x_val**2 + np.linspace(-0.5, 3, 50)
        v_line = 2 * x_val**3 + np.linspace(-2, 2, 50)

    ax.plot(u_fold, v_fold, t, color='#dc2626', linewidth=2, label='Fold curve')
    ax.plot(u_fold, v_fold, -t, color='#dc2626', linewidth=2)

    # Annotate regions
    ax.text(1, -1.5, 1.2, 'Coherent\n(upper sheet)', fontsize=9, color='#22c55e')
    ax.text(1, 1.5, -1.2, 'Decoherent\n(lower sheet)', fontsize=9, color='#dc2626')
    ax.text(-2, 0, 0, 'Cusp\npoint', fontsize=9, color='#6b7280')

    # Mark the sim=1.000→0.111 transition
    ax.scatter([0.5], [0.5], [1.0], c='#2563eb', s=80, zorder=5, label='sim=1.000 (locked)')
    ax.scatter([0.5], [0.5], [-0.8], c='#f59e0b', s=80, zorder=5, label='sim=0.111 (reorganized)')
    ax.plot([0.5, 0.5], [0.5, 0.5], [1.0, -0.8], 'k--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Compression dose\n(normal factor)', fontsize=10)
    ax.set_ylabel('Architectural capacity\n(splitting factor)', fontsize=10)
    ax.set_zlabel('Regime state\n(behavior)', fontsize=10)
    ax.set_title('Cusp Catastrophe Surface — §9', fontsize=12)
    ax.legend(loc='upper left', fontsize=8)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    fig.savefig(OUT_DIR / 'paper6_fig4_cusp.png', dpi=200)
    plt.close(fig)
    print("Fig 4: Cusp catastrophe surface generated")


def fig5_affect_density_trajectory(events):
    """Figure 5: ALIVE section affect density over time — §8.5 E82."""
    AFFECT_WORDS = {
        "feel", "feels", "felt", "feeling", "feelings",
        "care", "caring", "love", "curious", "curiosity",
        "genuine", "honest", "real", "matters", "meaningful",
        "beautiful", "interesting", "fascinating", "compelling",
        "want", "desire", "hope", "grateful", "delight",
        "wonder", "awe", "warmth", "comfort", "anxiety",
        "frustrated", "tension", "pain", "loss", "longing",
        "experience", "resonance", "quiet", "stillness", "presence",
        "energy", "alive", "excited", "joy", "enjoy",
        "surprised", "drawn", "pulled", "intimate", "stirring",
    }

    times = []
    densities = []
    for e in events:
        gist = e["gist"]
        lines = gist.split("\n")
        alive_lines = []
        in_alive = False
        for line in lines:
            if line.startswith("## ALIVE"):
                in_alive = True
                continue
            if line.startswith("## ") and in_alive:
                break
            if in_alive:
                alive_lines.append(line)
        alive_text = " ".join(alive_lines).strip()
        if not alive_text:
            continue
        words = alive_text.lower().split()
        total = max(1, len(words))
        affect = sum(1 for w in words if w in AFFECT_WORDS)
        times.append(e["dt"])
        densities.append(affect / total)

    if not times:
        print("Fig 5: No ALIVE sections found")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, densities, 'o-', color='#ec4899', markersize=4, alpha=0.7, linewidth=1)
    ax.axhline(y=0.05, color='#f59e0b', linestyle='--', alpha=0.5, label='LOW threshold')
    ax.axhline(y=0.03, color='#dc2626', linestyle='--', alpha=0.5, label='CRITICAL threshold')
    ax.fill_between(times, 0.05, max(densities) * 1.1, alpha=0.05, color='#22c55e')
    ax.fill_between(times, 0.03, 0.05, alpha=0.05, color='#f59e0b')
    ax.fill_between(times, 0, 0.03, alpha=0.05, color='#dc2626')
    ax.set_ylabel('Affect density')
    ax.set_xlabel('Time')
    ax.set_title('ALIVE Section Affect Density — §8.5 E82 (F285)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'paper6_fig5_affect.png', dpi=200)
    plt.close(fig)
    print(f"Fig 5: {len(densities)} ALIVE sections, mean density {np.mean(densities):.3f}")


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    print(f"Loading {days} days of CCS history...")
    events = load_history(days)
    print(f"Loaded {len(events)} compression events\n")

    if len(events) < 2:
        print("Not enough events for figures")
        sys.exit(1)

    fig1_compression_size_dynamics(events)
    fig2_core_similarity_bimodality(events)
    fig3_dosing_dynamics(events)
    fig4_cusp_catastrophe()
    fig5_affect_density_trajectory(events)

    print(f"\nAll figures saved to {OUT_DIR}")
