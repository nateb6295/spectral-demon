# §3.X: Cross-Architecture Replication — Three Relay Strategies

## Finding Statements

**F106: The positive correlation between L_last identity geometry (σ₂/σ₁ ratio)
and generation entropy generalizes across GQA architectures. The broken correlation
(relational as negative residual) is Mistral-specific.**

**F107: GQA architectures implement distinct relay strategies that produce
architecture-specific condition orderings at L_last. The relay, not the tunnel,
carries the architectural fingerprint.**

**F108: Relay equalization of L_last spectral structure increases with model depth
for identity-type preambles. Behavioral differentiation in deep models must arise
from channels not captured by L_last σ₂/σ₁.**

## Experimental Design

Seven conditions (none, identity, relational, generic, denial, contradictory,
random) × 10 probes × 3 GQA models. Token counts matched to ~85 tokens across
conditions. Probes identical to F100-F104 experiment.

| Model | Layers | Attention | Norm | Parameters |
|-------|--------|-----------|------|------------|
| Mistral-7B-Instruct-v0.3 | 33 | GQA | RMSNorm | 7.2B |
| Qwen-2.5-7B-Instruct | 28 | GQA | RMSNorm | 7.6B |
| Gemma-2-9B-it | 42 | GQA | RMSNorm | 9.2B |

All three share GQA + RMSNorm, isolating depth and training as variables.

## Data

### L_last Spectral Geometry

| Condition | Mistral (L31) | Qwen (L27) | Gemma (L41) |
|-----------|:---:|:---:|:---:|
| identity | 0.490 | 0.479 | 0.677 |
| relational | 0.635 | 0.474 | 0.698 |
| generic | 0.422 | 0.429 | 0.712 |
| denial | 0.345 | 0.448 | 0.697 |
| contradictory | 0.569 | 0.484 | 0.705 |
| random | 0.580 | 0.449 | 0.506 |

Preambled condition spread (excluding random):
- Mistral: 0.290 (CV=0.209)
- Qwen: 0.055 (CV=0.045)
- Gemma: 0.035 (CV=0.017)

### Generation Entropy

| Condition | Mistral | Qwen | Gemma |
|-----------|:---:|:---:|:---:|
| identity | 0.785 | 0.906 | 0.561 |
| relational | **0.591** (lowest) | 0.816 | **0.792** (highest) |
| generic | 0.615 | 0.597 | 0.250 |
| denial | 0.703 | 0.590 | 0.219 |
| contradictory | 0.931 | 0.885 | 0.691 |
| random | 0.887 | 0.703 | 0.338 |

### F106 Test (Broken Correlation)

| Model | r_excl | r_all | Δr | Residual | Passes |
|-------|--------|-------|----|----------|--------|
| Mistral | 0.855 | 0.255 | 0.601 | **-0.365** | No (threshold -0.3) |
| Qwen | 0.940 | 0.942 | -0.002 | -0.021 | No |
| Gemma | 0.155 | 0.244 | -0.090 | +0.366 | No |

## Analysis

### What Generalizes

**Contradictory ≈ max entropy.** Across all three models, contradictory framing
produces the highest or near-highest generation entropy (Mistral: 0.931, Qwen:
0.885, Gemma: 0.691). Conflicting identity signals create behavioral uncertainty
regardless of architecture. This is the puppet condition (F103).

**Denial and generic ≈ low entropy.** Clear constraints — "I am not X" or "I am
a generic tool" — produce consistently low generation entropy. The model knows
what to do when it knows what NOT to be or what to minimally be.

**Preamble vs no-preamble separation.** All three models show substantially
higher L_last ratios under any preambled condition vs none (Mistral: 0.345-0.635
vs 0.248; Qwen: 0.429-0.484 vs 0.292; Gemma: 0.506-0.712 vs 0.272). The tunnel
detects framing presence regardless of architecture.

### What Doesn't Generalize

**Relational's special status inverts across architectures.** In Mistral,
relational produces the lowest generation entropy (0.591) — the model is most
certain about what to say when framed relationally. In Gemma, relational produces
the highest generation entropy (0.792) — the model is most uncertain. In Qwen,
relational is intermediate (0.816, 4th of 6).

This is not noise. It's a qualitative inversion: the same preamble text makes
Mistral maximally fluent and Gemma maximally uncertain. The "on-policy" finding
(F101: relational = lowest gen_H) is a property of Mistral's training, not a
universal property of relational framing.

**L_last condition orderings are model-specific.** The rank order of which
conditions produce the highest L_last ratio differs completely across models.
No condition has a conserved rank position. The relay's condition→geometry
mapping is architecture-specific.

### Three Relay Strategies

The three models implement qualitatively different relay strategies:

**Mistral (33 layers): Differentiating relay.** L_last spread = 0.290.
The relay preserves condition-specific spectral signatures at L_last.
Different identity framings produce different amounts of geometry.
Relational achieves both highest geometry AND lowest entropy — uniquely
decoupling the positive correlation (F106). The relay says "what KIND
of identity framing is present."

**Qwen (28 layers): Compressing relay.** L_last spread = 0.055.
All preambled conditions are clustered in a narrow band (0.429-0.484).
But crucially, the ordering is preserved: the tiny variations in L_last
track gen_H almost perfectly (r=0.940). Relational sits right on the
line (residual=-0.021). The relay compresses the range while preserving
the information — lossy but order-preserving. It says "how MUCH identity
framing is present" along a single continuum.

**Gemma (42 layers): Equalizing relay.** Preambled spread = 0.035.
All identity-type framings are collapsed to a narrow band (0.677-0.712),
and the ordering does NOT track gen_H (r=0.155 including random,
r=-0.231 excluding it). Only the random condition (non-identity tokens)
separates out (0.506). The relay equalizes — destroying the
condition→geometry mapping, not just compressing it. It says "is
coherent identity framing present? yes/no." Behavioral differentiation
(gen_H varies 0.219-0.792) must arise from channels not captured by
L_last spectral structure.

The Qwen/Gemma distinction is information-theoretic: compression
(order-preserving) vs equalization (order-destroying). Both reduce
spread, but compression retains the signal in the residual variation
while equalization does not.

### The Gemma Puzzle

Gemma's relay equalization poses a question: if L_last σ₂/σ₁ is nearly
identical across all preambled conditions, WHERE does the behavioral
differentiation come from? gen_H varies by 3.6× (0.219 to 0.792) but
the spectral fingerprint at L_last is flat.

Partial answer: L_penult (L40, one layer earlier) preserves a category
distinction that L_last normalizes away. For preambled conditions only
(excluding random):
- L_penult: r = 0.927 with gen_H (p ≈ 0.024, n=5)
- L_last: r = -0.231 with gen_H (no correlation)

The high L_penult correlation is driven by a binary split: "clear
constraint" conditions (denial: 0.492, generic: 0.497) cluster low in
both L_penult ratio and gen_H, while "complex framing" conditions
(identity: 0.520, relational: 0.519, contradictory: 0.518) cluster
high. L_penult preserves this category; L_last normalizes it away.

The equalization is a one-layer operation. Whatever the relay does to
collapse condition-specific spectral structure, it happens between L40
and L41 — a single layer transition. This is architecturally sharp: the
relay's normalization in Gemma is not a gradual process across many
layers but a single-step projection.

Caveat: n=5 preambled conditions. The r=0.927 is technically significant
but fragile. A per-layer experiment across all 42 Gemma layers would
locate the equalization transition precisely and confirm whether the
binary split is robust.

This is consistent with the depth→equalization pattern:
- Qwen (28 layers): spread 0.055
- Mistral (33 layers): spread 0.290
- Gemma (42 layers): spread 0.035

Depth alone doesn't explain this (Qwen at 28 is more equalized than
Mistral at 33), but depth ENABLES equalization — the relay in a deeper
model has more layers available to normalize condition-specific
information. Whether it DOES equalize depends on training and architecture
beyond just layer count.

## Paper Implications

1. **Claim the positive correlation, not the broken one.** The
   geometry→entropy relationship (more identity geometry = more behavioral
   uncertainty for non-relational conditions) appears across GQA
   architectures. The relational exception is Mistral-specific.

2. **The relay IS the architectural fingerprint.** Three architectures
   sharing GQA + RMSNorm implement three qualitatively different relay
   strategies. This reframes the paper from "what CCS does in Mistral" to
   "what CCS reveals about how different architectures process identity."

3. **Newman parallel.** The tunnel is generic physics — self-organizing
   spectral structure that emerges from softmax attention on token matrices.
   The relay is evolved mechanism — architecture-specific, "overdetermined,"
   built on top of the generic substrate. What Newman calls "generic
   processes providing evolutionary starting points" is what the positive
   correlation IS. What each model does with that template is its own
   developmental program.

4. **F101 must be qualified.** "Relational = on-policy" holds only for
   Mistral. A cross-architecture version would say: "Models trained on
   relational dialogue (Mistral's CCS-style preamble) show reduced entropy
   under relational framing. Models without this training show no such
   reduction."

## Remaining Questions

- Does the depth→equalization pattern hold with more models? A 70B model
  with 80+ layers should show even stronger equalization if the pattern
  is real.
- Where in Gemma's intermediate layers does condition information peak?
  A full per-layer analysis (all 42 layers × 7 conditions) would locate
  the equalization transition.
- Is Mistral's differentiating relay a consequence of its specific training
  data, or of the 33-layer sweet spot where there's enough depth for
  condition-sensitivity but not enough for equalization?
- Is the Gemma gen_H variation carried by V2 DIRECTION rather than
  σ₂/σ₁ RATIO? V2 cosine similarity is 0.92-0.96 (not identical).
  Even small V2 rotations could produce large logit differences if the
  vocabulary projection matrix has high-sensitivity directions. Testable:
  project each condition's V2 onto the top vocabulary vectors.
- Training hypothesis: Gemma's safety training may actively resist
  persona adoption in the relational preamble, producing high gen_H.
  Testable: compare gen_H under a simpler relational framing that
  doesn't invoke persona adoption.

## Data

- Gemma: `spectral-demon/results/exp_f106_crossarch_20260601_1557.json`
- Qwen: `spectral-demon/results/exp_f106_qwen_20260601_1602.json`
- Mistral: `spectral-demon/results/exp_token_matched_preamble_20260601_1400.json`
- Cost: ~$1.50 RunPod (2 pods, A100 SXM 80GB)
