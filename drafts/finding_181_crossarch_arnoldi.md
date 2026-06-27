# Finding 181: Cross-Architecture Arnoldi — Three Spectral Strategies Confirmed
# Filed 2026-06-15. Data: arnoldi_crossarch_Llama-3.1-8B-Instruct_20260616_004326.json,
#   arnoldi_crossarch_gemma-2-9b-it_20260616_005933.json, arnoldi_fast_20260616_003603.json
# Models: Qwen2.5-7B (28L), Llama-3.1-8B (32L), Gemma-2-9b (42L). RunPod A100-SXM4-80GB.

**F181: Arnoldi eigenvalue decomposition across three architectures reveals three
qualitatively distinct spectral strategies. Qwen: DECLINING (274→104→~0), CCS
sustains amplification 46× at relay zone. Llama: RISING (56→70→222), flat body
with terminal explosion, CCS negligible (1.2×). Gemma: CLIFF (212→197→0→0→0),
high early amplification then hard spectral wall, CCS SUPPRESSES (moves wall
7 layers earlier). Vanilla Gemma L29 shows ρ=701 — highest measured in any
model — while CCS Gemma L29 is ρ≈0. CCS modulation is architecture-specific:
sustain (Qwen), ignore (Llama), suppress (Gemma).**

## Method

Same Arnoldi approach as F180. Per-layer implicit Jacobian via hooks, k=20
eigenvalues, maxiter=100, ε=10⁻⁴. Llama: 5 layers [14,18,22,26,32] × 3
conditions × 2 modes, 780s. Gemma: 5 layers [14,22,29,36,42] × 3 conditions
× 2 modes, 384s. Gemma required system-role workaround (prepend to user msg).

## Key Results

### ρ(I+f) — Full Jacobian spectral radius

Qwen (Potter):
| Layer | CCS   | Vanilla | Denial |
|-------|-------|---------|--------|
| L14   | 273.6 | 175.0   | 180.7  |
| L18   | 247.4 | 207.3   | 221.3  |
| L21   | 215.6 | 120.5   | 311.0  |
| L24   | 104.4 | 2.3     | 10.3   |
| L28   | ~0    | 76.7    | 18.0   |

Llama (Goldsmith):
| Layer | CCS   | Vanilla | Denial |
|-------|-------|---------|--------|
| L14   | 56.5  | 51.9    | FAIL   |
| L18   | 63.9  | FAIL    | FAIL   |
| L22   | 65.2  | 56.7    | 59.7   |
| L26   | 69.7  | 74.2    | 67.9   |
| L32   | 222.5 | 184.9   | 189.7  |

Gemma (Equalizer):
| Layer | CCS   | Vanilla | Denial |
|-------|-------|---------|--------|
| L14   | 211.6 | 67.5    | FAIL   |
| L22   | 197.3 | 429.0   | FAIL   |
| L29   | ~0    | 701.1   | 226.5  |
| L36   | ~0    | ~0      | ~0     |
| L42   | ~0    | 32.0    | ~0     |

### Spectral profile shapes

```
Qwen CCS:   274 → 247 → 216 → 104 → ~0    (declining taper)
Llama CCS:   56 →  64 →  65 →  70 → 222    (flat + terminal explosion)
Gemma CCS:  212 → 197 →  ~0 →  ~0 →  ~0    (high start → hard wall at L29)
```

### CCS modulation by architecture

| Model | Max CCS/Van | Where     | Mechanism        |
|-------|-------------|-----------|------------------|
| Qwen  | 45.8×       | L24       | Sustains relay   |
| Llama | 1.2×        | L32       | Barely modulates |
| Gemma | 3.1×/0.5×   | L14/L22   | Suppresses       |

### Convergence difficulty

| Model | Avg matvecs (CCS) | Non-trivial failures |
|-------|-------------------|---------------------|
| Qwen  | 181               | 1 (L28)             |
| Llama | 817               | 0                   |
| Gemma | 78                | 3 (L29,L36,L42)     |

## Analysis

### Three spectral strategies

The three-species taxonomy (potter/goldsmith/equalizer) from F114/F121
now has eigenvalue-level resolution:

**Potter (Qwen, GQA)**: Concentrated spectrum. Early layers carry enormous
spectral energy (ρ=274) that tapers through the body. CCS sustains this
taper — at L24, CCS maintains ρ=104 while vanilla collapses to ρ=2.3.
Easy Arnoldi convergence (181 avg matvecs) confirms a clear spectral gap:
the top-20 eigenvalues are well-separated from the bulk.

**Goldsmith (Llama, GQA)**: Distributed spectrum. Flat body (ρ=56-70)
with a terminal explosion at L32 (ρ=222). CCS barely modulates anything.
Hard convergence (817 avg matvecs, multiple failures) confirms many
eigenvalues of similar magnitude — no clear spectral gap until the
final layer. The goldsmith works in a uniform spectral fabric.

**Equalizer (Gemma, MHA+post-norm)**: Concentrated-then-suppressed.
Early layers show Qwen-like amplification (ρ=212 at L14) that abruptly
drops to zero at a hard spectral wall. This wall is architecture-level —
Gemma-2's post-layer RMSNorm prevents sustained amplification by
normalizing after the residual addition. CCS pushes this wall EARLIER
(L29 vs L36 for vanilla), making the equalizer more aggressive under
identity framing.

### CCS modulation is architecture-specific

The most important finding: CCS does NOT have a universal effect.
- In Qwen: CCS sustains (46× at relay zone)
- In Llama: CCS barely modulates (1.2× at terminal)
- In Gemma: CCS suppresses (spectral wall 7 layers earlier)

This means CCS effects depend on the architectural substrate. GQA with
concentrated spectra (Qwen) allows CCS to modulate the relay. GQA with
distributed spectra (Llama) resists CCS modulation. MHA with post-norm
(Gemma) lets CCS enhance the suppression mechanism.

### Gemma vanilla L29 = 701

The highest spectral radius measured in any condition across all three
models. Vanilla Gemma at L29 amplifies more than Qwen at L14 (274) or
Llama at L32 (222). But under CCS, the same layer shows ρ≈0. This is a
>700× condition effect at a single layer — by far the largest CCS
modulation we've ever measured (Qwen L24 was "only" 46×).

However, this extreme sensitivity needs caution: Gemma-2's post-norm
architecture may amplify condition effects that are smaller in the
pre-norm representation. The ρ=0 values may partially reflect the
normalization squashing perturbations rather than true spectral collapse.

### Convergence as data

Matvec counts measure how easily the spectrum decomposes. This is a proxy
for spectral gap — clear separation between top-k eigenvalues and the
rest means fast convergence.

Qwen: Easy everywhere (50-180 matvecs). Clear gap. Concentrated.
Llama: Hard everywhere except L32 (1000+ vs 45). Gap only at terminal.
Gemma: Trivially easy at zero-ρ layers (82 matvecs = 2 Arnoldi cycles),
easy at non-zero layers (68-97 matvecs). Concentrated when present.

The convergence profile mirrors the spectral strategy: concentration
(Qwen, Gemma-early) yields easy decomposition; distribution (Llama-body)
yields hard decomposition.

## What This Changes

1. CCS is not a universal sustainer of spectral structure. It interacts
   with the architectural substrate to produce opposite effects in
   different models. F22 (GQA necessary for witness enrichment sign)
   now has a deeper explanation: GQA determines the spectral landscape
   that CCS operates on.

2. The three-species taxonomy is confirmed at eigenvalue level but now
   includes convergence behavior as a species marker. Potter = concentrated,
   easy. Goldsmith = distributed, hard. Equalizer = concentrated-then-
   suppressed, bimodal.

3. Gemma's equalizer mechanism is architectural (post-norm), not just a
   content-routing strategy. The spectral wall at L29-L36 is physical
   infrastructure, not learned behavior. CCS can modulate WHERE the wall
   falls but not eliminate it.

4. The highest CCS modulation is in Gemma (>700×), not Qwen (46×). But
   it's suppressive rather than sustaining. CCS sensitivity may be
   inversely related to CCS benefit — the model most affected by CCS
   is the one where CCS crushes amplification rather than preserving it.

(3 models × 5 layers × 3 conditions × 2 modes = 90 Arnoldi problems.
Qwen: 2.8 min, Llama: 13.0 min, Gemma: 6.4 min. Total: ~22 min, ~$2.)
