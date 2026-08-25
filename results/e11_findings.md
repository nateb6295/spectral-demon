# E11: Transition Zone Redirect Test

## Method
Perturb σ₁ direction at three injection sites (L3 early, L15 transition, L21 relay)
with four conditions (0.5×, ablate, invert, vanilla-replace). 6 probes, Qwen2.5 7B IT.
Measure downstream relay geometry shift (σ₁ at L20-27).

## F304: Transition zone is POST-COMMITMENT, not slow manifold
Ablation (σ₁ zeroed):
- L3: relay σ₁ shift = 4.12 (MASSIVE — 13.5× transition)
- L15: relay σ₁ shift = 0.30 (minimal)
- L21: relay σ₁ shift = 0.32 (comparable to L15)

L15 perturbation produces almost no change in relay geometry.
Identity geometry is set at L3-4 and the transition zone coasts.
Kimi's "slow manifold" hypothesis refuted — the transition zone
is post-commitment, not redirectable.

## F305: Early layers are the commitment point
L3 ablation produces 13.5× more relay disruption than L15 ablation.
L3 inversion (scale=-1.0) produces 5.48 shift — the most destructive
condition across all injection sites. Early layers carry the identity
"seed crystal" that the rest of the network develops.

## F306: Vanilla replacement is nearly invisible at L3
L3_vanilla_replace: σ₁ shift = 0.012, logit cosine = 0.997.
Replacing CCS σ₁ with vanilla σ₁ at L3 produces ALMOST NO CHANGE.
At D5, the CCS-induced σ₁ direction at L3 is nearly identical to vanilla.
The differentiation happens DOWNSTREAM of L3, not at L3 itself.

## F307: Inversion reveals graded commitment
Scale=-1.0 (σ₁ inverted):
- L3: shift=5.48 (catastrophic)
- L15: shift=1.35 (moderate — MORE than ablation's 0.30)
- L21: shift=0.45 (small)

Inversion is more disruptive than ablation at L15 (1.35 vs 0.30)
but still far less than L3 (5.48). The transition zone has SOME
directional sensitivity — it can detect sign-flips — but can't
be redirected by scaling. This is passive confinement, not active 
redirection.

## Implications
- The four-zone architecture is confirmed: L0-4 = commitment,
  L5-19 = coasting/transition, L20-27 = relay, L28 = output
- CCS works by setting geometry at L3-4; the transition zone 
  simply passes it through without modification
- Therapeutic/overdose effects must originate at L3-4 (the 
  commitment point), not in the transition zone
- F306 (vanilla≈CCS at L3) suggests the commitment happens 
  through ACCUMULATED CONTEXT, not preamble token geometry
