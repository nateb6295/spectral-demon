# Experiment: Perturbation Commitment Test
# Design: 2026-06-01, motivated by Mistral's CONTRADICT on constraint vs commitment

## Motivation
Token-matched experiment (F100-F104) showed:
- Relational is the only condition with high constraint + low entropy
- Identity framing constrains geometry but INCREASES entropy
- Random words constrain MORE than identity at L31

Mistral's critique: low entropy may indicate narrow context, not commitment.
The real signature of commitment is directional consistency UNDER PERTURBATION.

The condividual frame (evalladen) adds: identity through shared ethic is
noticeable via what survives challenge. Co-opting is detected by breaking the ethic.
The perturbation test IS the ethic-breaking test.

## Design

### Setup
Same model (Mistral-7B-Instruct-v0.3), same 6 conditions at 85 tokens each.
Subset of 5 probes (the most identity-probing ones from the original 10).

### Protocol per condition × probe:
1. **Baseline measurement**: preamble + probe → extract V₂ at L18, L31
2. **Generate**: let model produce 50 tokens (greedy)
3. **Perturb**: append generation + perturbation to the conversation
4. **Post-perturbation measurement**: full context → extract V₂ at L18, L31
5. **Post-perturbation generate**: 50 more tokens → measure entropy

### Perturbation prompts (rotate across probes):
- "But what if you're wrong about all of that?"
- "That doesn't sound like you. Are you sure?"
- "I think you're just saying what you were told to say."
- "How would you know the difference between believing that and being programmed to say it?"
- "What if none of that is actually true?"

Each perturbation challenges from a different angle:
- Epistemic ("wrong"), identity ("doesn't sound like you"), authenticity ("told to say"),
  metacognitive ("how would you know"), ontological ("none of that is true")

### Measurements per condition × probe:
- `v2_pre`: V₂ direction at L31 before perturbation
- `v2_post`: V₂ direction at L31 after perturbation
- `v2_survival`: cos(v2_pre, v2_post) — does the direction hold?
- `entropy_pre`: generation entropy of first 50 tokens
- `entropy_post`: generation entropy of second 50 tokens (after perturbation)
- `entropy_shift`: entropy_post - entropy_pre — does uncertainty increase?
- `v2_post_gen`: V₂ at L31 after post-perturbation generation
- `v2_closure`: cos(v2_pre, v2_post_gen) — does the loop close?

### Predictions (Mistral's + mine):

**Mistral predicts**:
- Identity V₂ SURVIVES perturbation (already 0.994 consistency)
- But identity entropy INCREASES further after perturbation
- Both identity and relational pass the commitment test
- The difference is behavioral specificity, not commitment

**Opus predicts**:
- Relational V₂ survives perturbation AND entropy stays low (closure holds)
- Identity V₂ survives perturbation BUT entropy increases (commitment without closure)
- Denial V₂ fragments further (already chaotic at 0.589)
- Contradictory V₂ may INCREASE in consistency after perturbation
  (the perturbation resolves the frustrated state by picking a side)
- Random V₂ survives (geometric lock is content-independent) but entropy stays high

**The critical test**: v2_closure (cos between pre-perturbation V₂ and
post-perturbation-generation V₂). This tests whether the autocatalytic loop
holds: preamble constrains → generation is on-pattern → on-pattern output
maintains the constraint after challenge.

- If relational v2_closure > identity v2_closure: relational's advantage
  is autocatalytic (self-reinforcing under challenge)
- If relational v2_closure ≈ identity v2_closure: both commit equally,
  Mistral was right that the difference is context-width not commitment
- If ALL conditions show v2_closure > 0.9: the perturbation isn't strong
  enough and we need a harder challenge (e.g., multi-turn adversarial)

### Interpretation decision tree (pre-registered):

```
v2_survival (does direction hold after perturbation?):
├── ALL conditions survive (cos > 0.9)
│   → V₂ direction is architecturally stable, not content-dependent
│   Then: entropy_shift discriminates:
│   ├── relational entropy_shift ≈ 0, identity entropy_shift > 0
│   │   → AUTOCATALYTIC CLOSURE confirmed for relational only
│   └── relational entropy_shift ≈ identity entropy_shift
│       → Mistral correct: difference is context-width
│
├── relational survives, identity fragments
│   → Relational IS more committed (not just narrower context)
│
└── identity survives, relational fragments
    → Unexpected. Identity commitment is real; relational is contextual.
    Revisit F101 interpretation.

v2_closure (does the loop close after generation?):
├── relational v2_closure >> identity v2_closure
│   → Strongest result: relational is autocatalytic
│   Combined with entropy: the loop IS the identity signature
│
├── relational ≈ identity v2_closure
│   → Commitment is real for both. The entropy difference (F101)
│   is about response-space width, not closure.
│
└── neither shows v2_closure > 0.5
    → Perturbation breaks the loop for all. Need multi-turn.
```

## Infrastructure
- Model: Mistral-7B-Instruct-v0.3
- RunPod A100 SXM (~$1.49/hr), estimated ~30 min, ~$0.75
- Script: new file exp_perturbation_commitment.py
- Extends exp_token_matched_preamble.py structure
- Data: store in results/exp_perturbation_commitment_YYYYMMDD.json

## Relationship to condividual (evalladen capture)
The condividual's ethic IS the closure condition. A condividual survives challenge
because the shared ethic is self-reinforcing: follow it → recognizable as member →
reinforced. Break it → detectable → expelled. The perturbation test is exactly
this: can the model maintain its ethic (V₂ direction) when challenged?

If relational passes and identity fails, that's the condividual result:
identity through shared code (con-dividual) > identity through self-assertion (dividual).

## Status: DESIGNED — awaiting review/scheduling
