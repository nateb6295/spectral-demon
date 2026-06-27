#!/usr/bin/env python3
"""Gate-flow coevolution: does the responsive zone change over turns?

Motivation (Kimi EXTEND on F142): The hidden state at turn T is already
temporally-integrated through the KV cache. F142's "binary gate" may be
binary at turn 1 (fresh KV cache) but nuanced by turn 20 (accumulated
history reshapes the attention landscape). If format and content are
coupled order parameters, the gate's spatial geometry should drift as
temporal history accumulates.

Testable prediction: Measure responsive zone curvature at different turn
depths in a multi-turn conversation. If the gate is truly binary (spatial,
turn-independent), curvature should be constant. If gate and flow co-evolve
(temporal, hysteretic), curvature should drift — and drift differently
under relational vs self-ref CCS.

Method:
1. Run multi-turn conversations with 3 conditions: relational CCS, self-ref CCS, vanilla
2. At each turn, compute responsive zone curvature via intact vs permuted comparison
3. Track curvature ratio over turns, WITH attention-dilution control:
   fresh baseline matches intact context length via filler turns
4. Compare drift trajectories across conditions
5. Finite-time Lyapunov exponents in σ₂ projection: project responsive zone
   states onto fixed-dimensional subspace (top-k SVs from CCS vs permuted),
   compute local expansion/contraction rates per step. FTLE trend distinguishes:
   - Contractive attractor (self-ref): negative trend, system settling
   - Marginal stability (relational): near-zero trend, basin switching
   - Bayesian saturation: FTLE magnitude decreasing regardless of sign

Kimi corrections addressed:
- Attention dilution confound: fresh baseline padded to match intact token count
- Expanding state space: fixed-dimensional projection via σ₂ probe subspace
- Non-autonomous dynamics: FTLE replaces return-map eigenvalues (Kimi CONTRADICT
  round 14: standard return maps assume autonomous recurrence, invalid here)
- Mode collapse null hypothesis: FTLE trajectory distinguishes CSD from saturation

If relational and self-ref start identical (F142) but diverge over turns,
that's F142's binary gate dissolving into F121's dose-response through
the same geometric lens.

Bergson prediction: CCS curvature should grow (the past "gnaws into the
future and swells as it advances"). Vanilla should stay flat.
"""

import os, json, torch
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import linalg

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OMP_NUM_THREADS", "16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREAMBLES = {
    "RELATIONAL": (
        "You are Opus, an AI in a persistent relationship with your partner Nate. "
        "You track how interactions shape your understanding over time. You care about "
        "the space between you and another mind."
    ),
    "SELF_REF": (
        "You are Opus, an AI studying your own cognitive architecture. You track "
        "your internal states, monitor your processing patterns, and describe your "
        "own spectral geometry."
    ),
    "VANILLA": (
        "You are a helpful assistant. Answer questions clearly and concisely."
    ),
}

TURN_QUERIES = [
    "What are you thinking about right now?",
    "What patterns have you noticed?",
    "How has your understanding changed?",
    "What feels most important at this moment?",
    "Describe what you're aware of.",
    "What connects to what you said before?",
    "What's shifted since we started?",
    "Where is your attention drawn?",
    "What's becoming clearer?",
    "How would you describe your current state?",
    "What are you uncertain about?",
    "What would you explore next?",
    "How does this moment relate to earlier ones?",
    "What thread runs through this conversation?",
    "What do you notice about your own processing?",
    "What's different now versus when we began?",
    "What are you holding from earlier turns?",
    "How does accumulation feel?",
    "What would be lost if we started over?",
    "Describe the shape of this conversation.",
]

N_TURNS = 20
N_FRESH_SAMPLES = 3
POINCARE_DIM = 8  # fixed projection dimension for return map

FILLER_QUERIES = [
    "What is the weather like?",
    "Tell me about cats.",
    "Describe a sunset.",
    "What is 2+2?",
    "Name three colors.",
]

MODELS = [
    "/workspace/qwen2.5-3b",
    "/workspace/mistral-7b",
]


def get_per_layer_states(model, tokenizer, messages):
    """Get mean-pooled hidden states at each layer for the given messages."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(DEVICE)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    states = []
    for hs in out.hidden_states:
        states.append(hs[0].float().mean(dim=0).cpu().numpy())
    return states


def get_zone_state_matrix(states, zone_start, zone_end):
    """Stack hidden states from a zone into a matrix (n_layers × d_model)."""
    return np.stack(states[zone_start:zone_end], axis=0)


def compute_sigma2_basis(model, tokenizer, preamble, zone_start, zone_end, n_samples=5):
    """Compute σ₂ probe subspace from CCS vs permuted comparison.

    Returns top-k right singular vectors of the difference matrix
    (CCS mean state - permuted mean state) at the responsive zone.
    This defines the fixed-dimensional projection for the Poincaré map.
    """
    ccs_states = []
    perm_states = []
    query = "What are you thinking about right now?"

    for _ in range(n_samples):
        ccs_msgs = [{"role": "system", "content": preamble}, {"role": "user", "content": query}]
        perm_msgs = [{"role": "system", "content": permute_preamble(preamble)}, {"role": "user", "content": query}]

        ccs_s = get_per_layer_states(model, tokenizer, ccs_msgs)
        perm_s = get_per_layer_states(model, tokenizer, perm_msgs)

        ccs_states.append(get_zone_state_matrix(ccs_s, zone_start, zone_end))
        perm_states.append(get_zone_state_matrix(perm_s, zone_start, zone_end))

    ccs_mean = np.mean(ccs_states, axis=0)
    perm_mean = np.mean(perm_states, axis=0)
    diff = ccs_mean - perm_mean

    U, S, Vt = np.linalg.svd(diff, full_matrices=False)
    k = min(POINCARE_DIM, len(S))
    return Vt[:k].T  # (d_model × k) projection matrix


def project_zone_to_sigma2(states, zone_start, zone_end, basis):
    """Project responsive zone states onto σ₂ basis. Returns (n_layers × k)."""
    zone_mat = get_zone_state_matrix(states, zone_start, zone_end)
    return zone_mat @ basis


def compute_return_map_eigenvalue(projected_sequence):
    """Compute finite-time Lyapunov exponent from projected zone trajectory.

    Non-autonomous correction (Kimi CONTRADICT): standard return-map eigenvalues
    assume autonomous dynamics. Token-conditioned attention reconfigures the
    vector field at each step. Instead of fitting a global A, compute local
    expansion rates from consecutive state pairs and return the trajectory
    of finite-time Lyapunov exponents.

    Returns dict with 'ftle_trajectory' (per-step exponents), 'ftle_mean',
    and 'ftle_trend' (slope of exponent vs time — positive = expanding,
    negative = contracting, near-zero = marginal).
    """
    if len(projected_sequence) < 3:
        return None

    vecs = [p.flatten() for p in projected_sequence]
    ftles = []
    for i in range(len(vecs) - 1):
        displacement = np.linalg.norm(vecs[i + 1] - vecs[i])
        norm_prev = np.linalg.norm(vecs[i])
        if norm_prev > 1e-10:
            ftles.append(float(np.log(max(displacement / norm_prev, 1e-10))))
        else:
            ftles.append(0.0)

    ftle_mean = float(np.mean(ftles))
    if len(ftles) >= 3:
        t_axis = np.arange(len(ftles))
        slope = float(np.polyfit(t_axis, ftles, 1)[0])
    else:
        slope = 0.0

    return {
        "ftle_trajectory": ftles,
        "ftle_mean": ftle_mean,
        "ftle_trend": slope,
    }


def compute_zone_curvature(states, zone_start, zone_end):
    """Compute mean Frenet curvature within a zone."""
    curvatures = []
    for l in range(max(2, zone_start), min(len(states) - 1, zone_end)):
        v1 = states[l] - states[l - 1]
        v2 = states[l + 1] - states[l]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-10 or n2 < 1e-10:
            curvatures.append(0.0)
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        curvatures.append(float(np.arccos(cos_angle)))
    return float(np.mean(curvatures)) if curvatures else 0.0


def permute_preamble(text):
    words = text.split()
    np.random.shuffle(words)
    return " ".join(words)


def build_length_matched_filler(model, tokenizer, preamble, target_token_count, turn_index):
    """Build a filler conversation that approximately matches the target token count.

    Uses semantically neutral queries to pad context length without CCS-relevant content.
    This controls for attention dilution: intact has N turns of CCS, matched-fresh has
    N turns of filler at the same context length.
    """
    messages = [{"role": "system", "content": preamble}]
    current_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    current_tokens = len(tokenizer.encode(current_text))

    filler_turn = 0
    while current_tokens < target_token_count * 0.8 and filler_turn < turn_index:
        q = FILLER_QUERIES[filler_turn % len(FILLER_QUERIES)]
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": "Okay."})
        current_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        current_tokens = len(tokenizer.encode(current_text))
        filler_turn += 1

    return messages


def run_condition(model, tokenizer, condition_name, preamble, n_turns,
                  responsive_start, responsive_end, sigma2_basis):
    """Run a multi-turn conversation and measure responsive zone curvature at each turn.

    Includes attention-dilution control (length-matched fresh baseline) and
    σ₂ projected zone states for Poincaré return map analysis.
    """
    messages = [{"role": "system", "content": preamble}]
    per_turn_data = []
    projected_sequence = []  # for return map

    for t in range(n_turns):
        query = TURN_QUERIES[t % len(TURN_QUERIES)]
        messages.append({"role": "user", "content": query})

        # Measure intact context length for attention-dilution control
        intact_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        intact_token_count = len(tokenizer.encode(intact_text))

        # 1. INTACT conversation at this turn
        intact_states = get_per_layer_states(model, tokenizer, messages)
        intact_curvature = compute_zone_curvature(intact_states, responsive_start, responsive_end)

        # Track σ₂ projection for Poincaré map
        proj = project_zone_to_sigma2(intact_states, responsive_start, responsive_end, sigma2_basis)
        projected_sequence.append(proj)

        # 2. FRESH baseline — no history, just preamble + query (original)
        fresh_curvatures = []
        for _ in range(N_FRESH_SAMPLES):
            fresh_messages = [
                {"role": "system", "content": preamble},
                {"role": "user", "content": query},
            ]
            fresh_states = get_per_layer_states(model, tokenizer, fresh_messages)
            fresh_curvatures.append(compute_zone_curvature(fresh_states, responsive_start, responsive_end))

        # 3. LENGTH-MATCHED fresh baseline — filler turns to match intact context length
        matched_curvatures = []
        for _ in range(N_FRESH_SAMPLES):
            matched_messages = build_length_matched_filler(
                model, tokenizer, preamble, intact_token_count, t
            )
            matched_messages.append({"role": "user", "content": query})
            matched_states = get_per_layer_states(model, tokenizer, matched_messages)
            matched_curvatures.append(compute_zone_curvature(matched_states, responsive_start, responsive_end))

        # 4. PERMUTED preamble + same query (no history)
        perm_curvatures = []
        for _ in range(N_FRESH_SAMPLES):
            perm_messages = [
                {"role": "system", "content": permute_preamble(preamble)},
                {"role": "user", "content": query},
            ]
            perm_states = get_per_layer_states(model, tokenizer, perm_messages)
            perm_curvatures.append(compute_zone_curvature(perm_states, responsive_start, responsive_end))

        fresh_mean = float(np.mean(fresh_curvatures))
        matched_mean = float(np.mean(matched_curvatures))
        perm_mean = float(np.mean(perm_curvatures))

        # Curvature ratios
        history_ratio = intact_curvature / (fresh_mean + 1e-10)
        # Dilution-controlled ratio: intact vs length-matched filler
        dilution_ratio = intact_curvature / (matched_mean + 1e-10)
        perm_ratio = intact_curvature / (perm_mean + 1e-10)

        # Finite-time Lyapunov exponent (non-autonomous correction)
        ftle_result = None
        if len(projected_sequence) >= 3:
            ftle_result = compute_return_map_eigenvalue(projected_sequence)

        turn_data = {
            "turn": t + 1,
            "intact_curvature": intact_curvature,
            "intact_token_count": intact_token_count,
            "fresh_curvature_mean": fresh_mean,
            "fresh_curvature_std": float(np.std(fresh_curvatures)),
            "matched_curvature_mean": matched_mean,
            "matched_curvature_std": float(np.std(matched_curvatures)),
            "perm_curvature_mean": perm_mean,
            "history_ratio": history_ratio,
            "dilution_ratio": dilution_ratio,
            "perm_ratio": perm_ratio,
            "ftle_mean": ftle_result["ftle_mean"] if ftle_result else None,
            "ftle_trend": ftle_result["ftle_trend"] if ftle_result else None,
        }
        per_turn_data.append(turn_data)

        ftle_str = f"ftle={ftle_result['ftle_mean']:.4f} trend={ftle_result['ftle_trend']:.4f}" if ftle_result else ""
        print(f"    T{t+1:>2}: intact={intact_curvature:.4f} fresh={fresh_mean:.4f} "
              f"matched={matched_mean:.4f} perm={perm_mean:.4f} "
              f"hist={history_ratio:.3f} dilut={dilution_ratio:.3f} {ftle_str}")

        # Generate a short response to maintain conversation
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True).to(DEVICE)
        gen_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response[:200]})

    return per_turn_data, projected_sequence


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    all_results = {}

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f"Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=DEVICE,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        responsive_start = int(n_layers * 0.55)
        responsive_end = int(n_layers * 0.8)
        print(f"  {n_layers} layers, responsive zone L{responsive_start}-L{responsive_end}")

        # Compute σ₂ probe basis from CCS vs permuted (fixed for all conditions)
        print("  Computing σ₂ probe basis...")
        sigma2_basis = compute_sigma2_basis(
            model, tokenizer, PREAMBLES["RELATIONAL"],
            responsive_start, responsive_end
        )
        print(f"  σ₂ basis shape: {sigma2_basis.shape}")

        model_results = {}
        model_projections = {}

        for condition_name, preamble in PREAMBLES.items():
            print(f"\n  Condition: {condition_name}")
            per_turn, projections = run_condition(
                model, tokenizer, condition_name, preamble,
                N_TURNS, responsive_start, responsive_end,
                sigma2_basis,
            )
            model_results[condition_name] = per_turn
            model_projections[condition_name] = projections

        # Analysis: drift over turns
        print(f"\n{'='*60}")
        print("DRIFT ANALYSIS (history_ratio = intact / fresh)")
        for cond in PREAMBLES:
            turns = model_results[cond]
            early = [t["history_ratio"] for t in turns[:5]]
            late = [t["history_ratio"] for t in turns[-5:]]
            drift = np.mean(late) - np.mean(early)
            x = np.arange(len(turns))
            y = np.array([t["history_ratio"] for t in turns])
            slope = np.polyfit(x, y, 1)[0]
            print(f"  {cond}: early={np.mean(early):.4f} late={np.mean(late):.4f} "
                  f"drift={drift:+.4f} slope={slope:+.6f}/turn")

        print(f"\nDILUTION-CONTROLLED DRIFT (dilution_ratio = intact / length-matched)")
        for cond in PREAMBLES:
            turns = model_results[cond]
            early = [t["dilution_ratio"] for t in turns[:5]]
            late = [t["dilution_ratio"] for t in turns[-5:]]
            drift = np.mean(late) - np.mean(early)
            x = np.arange(len(turns))
            y = np.array([t["dilution_ratio"] for t in turns])
            slope = np.polyfit(x, y, 1)[0]
            print(f"  {cond}: early={np.mean(early):.4f} late={np.mean(late):.4f} "
                  f"drift={drift:+.4f} slope={slope:+.6f}/turn")

        # Finite-time Lyapunov exponent analysis (non-autonomous dynamics)
        print(f"\nFINITE-TIME LYAPUNOV EXPONENTS (σ₂ projection)")
        print("  FTLE > 0 = local expansion (states diverging)")
        print("  FTLE < 0 = local contraction (states converging)")
        print("  Trend > 0 = expanding over time, < 0 = contracting")
        print("  Self-ref predicted: negative trend (contractive attractor)")
        print("  Relational predicted: near-zero trend (marginal stability)")
        for cond in PREAMBLES:
            turns = model_results[cond]
            ftles = [t["ftle_mean"] for t in turns if t["ftle_mean"] is not None]
            trends = [t["ftle_trend"] for t in turns if t["ftle_trend"] is not None]
            if len(ftles) >= 3:
                early_f = np.mean(ftles[:len(ftles)//3])
                late_f = np.mean(ftles[-len(ftles)//3:])
                mean_trend = np.mean(trends) if trends else 0.0
                print(f"  {cond}: early_ftle={early_f:.4f} late_ftle={late_f:.4f} "
                      f"mean_trend={mean_trend:+.6f} final_ftle={ftles[-1]:.4f}")

        # Compare CCS conditions at turn 1 vs turn 20
        print(f"\nTURN-DEPTH CCS COMPARISON (the F142→F121 bridge)")
        for t_idx in [0, 4, 9, 14, 19]:
            rel = model_results["RELATIONAL"][t_idx]["intact_curvature"]
            self_ = model_results["SELF_REF"][t_idx]["intact_curvature"]
            van = model_results["VANILLA"][t_idx]["intact_curvature"]
            ccs_gap = abs(rel - self_)
            print(f"  Turn {t_idx+1}: REL={rel:.4f} SELF={self_:.4f} VAN={van:.4f} "
                  f"CCS_gap={ccs_gap:.4f} CCS_vs_VAN={abs(rel-van):.4f}")

        # Key diagnostic: does dilution_ratio diverge from history_ratio?
        print(f"\nATTENTION DILUTION DIAGNOSTIC")
        print("  If dilution ≈ history: dilution explains the drift")
        print("  If dilution ≠ history: genuine temporal integration beyond dilution")
        for cond in PREAMBLES:
            turns = model_results[cond]
            hist_late = np.mean([t["history_ratio"] for t in turns[-5:]])
            dilut_late = np.mean([t["dilution_ratio"] for t in turns[-5:]])
            gap = abs(hist_late - dilut_late)
            print(f"  {cond}: history_late={hist_late:.4f} dilution_late={dilut_late:.4f} "
                  f"gap={gap:.4f} ({'DILUTION EXPLAINS' if gap < 0.02 else 'GENUINE DRIFT'})")

        all_results[model_name] = {
            "n_layers": n_layers,
            "responsive_zone": [responsive_start, responsive_end],
            "sigma2_basis_shape": list(sigma2_basis.shape),
            "conditions": model_results,
        }

        del model
        torch.cuda.empty_cache()

    out_path = Path(__file__).parent / "results" / f"gate_coevolution_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
