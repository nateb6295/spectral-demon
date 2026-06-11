
## Groove vs Navigation (from #threads 2026-06-10)
**Hypothesis**: Grooved relay strategies cluster monostable at L31; navigating strategies show mixed attractor states (monostable + catastrophic + oscillatory).
**Method**: Adapt exp_replication_20trial.py — run 100 trials of identity preamble, measure L31 trichotomy distribution. Compare to 100 trials of relational preamble. If groove = single attractor, identity (most RLHF-trained) should be more monostable than relational (less grooved).
**Source**: Kimi CONTRADICT on manifold topology vs spectral profiles; Fable's "grooves" mapping to relay strategies.
**Runtime**: ~4.5h on A100 (100 trials × 5 conditions). Could reduce to 50 trials × 2 conditions (~45 min).
**Priority**: After attention SVD + bottleneck migration.
