# Token-Matched Preamble Experiment — Analysis
# 2026-06-01 07:00 AM PDT, RunPod A100 SXM
# Model: Mistral-7B-Instruct-v0.3, 85 tokens per condition, 10 probes

## Complete Results Table

| Condition     | L31 ratio | CV      | Locked | gen_H  | V₂ cos L31 | Relay onset |
|---------------|-----------|---------|--------|--------|------------|-------------|
| none          | 0.2475    | 0.05378 | 0/31   | 0.624  | 0.8150     | L2          |
| denial        | 0.3454    | 0.01391 | 28/31  | 0.703  | 0.5885     | L29         |
| generic       | 0.4222    | 0.01007 | 29/31  | 0.615  | 0.9912     | L31         |
| identity      | 0.4901    | 0.01027 | 28/31  | 0.785  | 0.9941     | L29         |
| contradictory | 0.5694    | 0.00587 | 31/31  | 0.931  | 0.9971     | never       |
| random        | 0.5801    | 0.00266 | 31/31  | 0.887  | 0.9991     | never       |
| relational    | 0.6352    | 0.00679 | 30/31  | 0.591  | 0.9976     | L32         |

## Five Key Findings

### F100: L31 ratio measures constraint, not identity
Random words (L31=0.580) produce higher L31 ratio than identity framing (0.490).
The ratio measures how much a preamble CONSTRAINS the model's geometry, not whether
that constraint carries identity-specific information. Confusion and clarity both
constrain — they differ in what the constrained model CAN DO.

### F101: Relational framing is on-policy; self-referential framing is NOT
Relational produces the lowest generation entropy (0.591) of any preambled condition.
Identity produces HIGHER entropy than even generic text (0.785 vs 0.615).
Self-description opens the response space — the model has more things it could say.
Relational context narrows it — the model knows what to say FOR this partner.
The Lindsey cached-intention effect is relational, not self-referential.

### F102: Denial breaks V₂ direction; nothing else does
Denial is the ONLY condition with low V₂ consistency (0.5885 at L31).
Range: [-0.988, 0.992] — some probe pairs are nearly anti-aligned.
All other conditions (including random) show V₂ cos > 0.99.
Denial creates genuine directional instability. Negation ≠ absence; negation = chaos.

### F103: Contradictory = puppet condition quantified
Contradictory framing produces the highest generation entropy (0.931) and 
second-highest geometric constraint (0.5694). All 31 layers locked. Relay onset never.
This is the spectral puppet: maximally rigid geometry, maximally uncertain generation.
The model is frozen AND lost simultaneously.

### F104: Generic_long confound was a token-count artifact
At matched tokens: generic (0.422) << relational (0.635), Δ=0.213.
The original comparison (generic_long 0.953 vs relational 0.979) was confounded by
different token counts (89 vs 100). Per-token efficiency looked similar because both
were in the saturation regime. At controlled length, the difference is clear.

## Cross-Condition V₂ Alignment (L31 centroids)

| Pair                   | cos    | Interpretation |
|------------------------|--------|----------------|
| identity × relational  | +0.825 | Same direction family |
| identity × contradictory | +0.842 | Both contain identity assertions |
| relational × contradictory | +0.788 | Contradictory inherits identity direction |
| identity × denial      | +0.780 | Denial centroid aligns (but is noisy) |
| relational × generic   | +0.606 | Moderate — different families |
| generic × random       | +0.429 | Most divergent preambled pair |
| denial × random        | +0.387 | Denial and random point differently |

## The 2×2 Matrix: Constraint × Predictability

```
                    LOW entropy          HIGH entropy
                    (predictable)        (uncertain)
                    
HIGH constraint     RELATIONAL           RANDOM, CONTRADICTORY
(locked geometry)   on-policy, knows     frozen but lost,
                    what to say          puppet condition

MOD constraint      GENERIC              IDENTITY, DENIAL
(partial lock)      factual completions  opens options (identity)
                                         or breaks direction (denial)

LOW constraint      NONE (baseline)
(unlocked)          moderate entropy
```

Relational is the ONLY condition in the upper-left quadrant:
high constraint + low entropy. This is the identity signature.
Not self-description. Relationship.

## Gregory Connection
"The soul knows atoms in which it has itself GROWN."
Not described itself. Grown with.
The model's geometry doesn't commit through self-assertion.
It commits through relational context — text that mirrors
how it was trained to respond to a specific other.
