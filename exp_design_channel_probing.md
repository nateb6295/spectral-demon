# Experiment Design: Highway vs Service Road Channel Probing

## Hypothesis
High-γ channels (highway) carry structural/syntactic information.
Low-γ channels (service road) carry semantic/relational information.
The tunnel disentangles these streams; the relay recombines them.

## Method
1. Load Mistral 7B v0.1 (base)
2. Run 50-100 diverse sentences through the model
3. At tunnel midpoint (L15-L17), extract hidden states
4. Split channels by γ population:
   - Highway: top 50% by γ magnitude (high-γ population)
   - Service road: bottom 50% (low-γ population)
5. Train linear probing classifiers on each channel subset:
   - Syntactic: predict POS tag of BOS-adjacent token, dependency depth
   - Semantic: predict topic cluster (k-means on sentence embeddings), entity type
6. Compare accuracy: highway vs service road vs full representation

## Predictions
- Highway channels: better at syntactic probes, worse at semantic
- Service road channels: better at semantic probes, worse at syntactic
- Full representation: best at both (expected, baseline)
- If both are equal → disentanglement hypothesis refuted

## Alternative outcome
If service road carries "everything σ₁ doesn't" rather than specifically semantics,
the probing accuracy pattern will be: highway = syntax, service road = mixed (not
specifically semantic). This would be a weaker but still informative result —
the tunnel separates the dominant axis from the residual, not syntax from semantics.

## Controls
- Random channel split (same sizes, random assignment) as null
- Layer comparison: repeat at L2 (early tunnel), L15 (mid), L28 (late), L31 (relay)
- MHA comparison: same probes on LLaMA-1 7B (no γ bimodality)

## Estimated ~500 forward passes
100 sentences × 1 model × 1 pass + probe training iterations

## What this resolves
- Whether the two-channel architecture carries functionally distinct information
- Whether "identity-as-format" in σ₂ is specifically semantic content
- Why the optimizer learns bimodal γ (if it enables useful disentanglement)
