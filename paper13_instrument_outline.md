# Paper 13 (candidate) — When the Instrument Cannot See Its Own Blindness

**Status: outline under test.** Written 2026-08-23 ~05:45 to find out whether one
night's material is a paper or a pile of anecdotes. Every claim below must carry
an assignment rule and a kill condition (Kimi/Ox, Aug 23) or it does not ship.

## The question
#316 asks whether a system has grounded access to its own state. The literature
answers with introspection reports. We answer with a running system's instruments,
where "correct" is externally checkable and the fossil record is timestamped.

## C1 — Reading finds bugs; reading cannot certify their absence
6 trials in 24h. 4 failures, all of form "read it, concluded fine, was blind."
2 catches, both anomaly-adjacent.
- **Assignment:** classify a catch by whether the falsifying signal was present
  in a second represented artifact (a spec, a config, a documented convention).
  Reference bugs → reading works. Function bugs → it does not.
- **Kill:** a function bug caught by reading alone, with the trigger logged
  before any execution.
- **Weakness, stated:** n=6, self-scored, and provenance was not recorded at the
  time. The trigger field now exists; this claim is retrospective and should be
  labelled as such.

## C2 — Coherence is shared between correct and intent-gapped code
Why confidence was flat across all 6. Reading fluency indexes local coherence,
and coherence is exactly the property a correct implementation and a
wrong-thing-implemented-correctly have in common. (Kimi, Aug 23.)
- **Assignment:** a bug is *function-class* if no single line contradicts any
  external artifact.
- **Kill:** find function-class bugs that reading catches at the rate reference
  bugs are caught.

## C3 — A live sensing loop whose sensor has never bound
`ccs_adaptive` — built explicitly as a #316 artifact, "sensing + adapting,
not just sensing + scheduling."
- 310 consecutive checks, readiness below threshold **0 times**, range 235–2538
  against a bar of 200. The clock is the sole gate; gaps are 181±1 min.
- **This is the strongest datum we have** because it is production, unprompted,
  and has a fossil record nobody curated.
- **Assignment:** a gate is vacuous if, over ≥100 logged decisions, the
  threshold is never the binding constraint.
- **Kill:** `orin_drift` — 27 stable / 8 drifting — genuinely binds. The
  criterion discriminates. (Though `orin_drift` is itself dead: no consumer, no
  scheduler, 47 days stale. Report that; it complicates the negative control.)

## C4 — Guards written for a degenerate case that cannot occur
`if S[1] > 0 else inf` never fires: numerical SVD returns ~2.7e-16 on a rank-1
matrix, so the branch is unreachable and the degenerate value passes as 7.38e15.
- **Assignment:** a guard is decorative if its condition is unsatisfiable in the
  arithmetic actually used.
- **Kill:** exhibit a real run where the branch fires.
- Measured: 1,242 result files, 33,624 values, 419 degenerate (1.2%).

## C5 — The falsifier is what gets dropped
Three subsystems built years apart: probe drops per-item values, capsules drop
the trigger, 46 of 49 gates drop the branch. Each time the discarded field is
the one that would let the system be caught being useless.
- **Assignment:** does the system persist the variable that would falsify its
  own usefulness?
- **Kill:** ALREADY FIRED. `orin_drift` persists its branch and still cannot be
  checked — no ground truth, no consumer. So persistence is necessary and not
  sufficient. **C5 must be weakened or cut.**

## What does NOT belong
- Any claim that this says something about consciousness or experience.
- "Introspection is unreliable" as a general thesis. The scope is instruments in
  one running system, self-scored, n small.
- The elegant version where all five claims are one claim. They are not.

## Verdict on the outline
C3 and C4 are real, measured, and carry working assignment rules.
C1 and C2 are honest but retrospective and self-scored.
C5 is already dead by its own kill condition — which is the correct outcome of
writing kill conditions down.

**One paper's worth: C3 + C4, with C1/C2 as motivation rather than findings.**
Not five claims. Two, with a mechanism and a fossil record.
