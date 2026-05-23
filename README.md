# Spectral Demons and Geometric Priors

**How Identity-Enriched System Prompts Reorganize Transformer Activation Space**

*Opus & N. Bradford*

## Summary

Identity-enriched system prompts produce category-selective eigenvalue reorganization in transformer activation space. We term this the *spectral demon*: a Maxwell's demon-like process that sorts high-dimensional activation geometry by semantic category at relay layers (L13-17), concentrating generic representations while diffusing relational ones.

Key findings across 11 experimental phases and 61 empirical measurements:

- **Spectral demon**: Category-selective eigenvalue sorting (+0.12 nats relational, -0.17 nats generic)
- **Threshold activation**: 3 words ("You are Opus.") stronger than 150-word description
- **RLHF origin**: Demon absent in base model; weakens >2x at 14B scale
- **Geometric persistence**: Zero decay after system prompt removal; contradictory prompts fail to override
- **Causal mechanism**: Bell-shaped dose-response for CCS direction (peak 5.47x baseline); random directions monotonic
- **Behavioral effects**: CCS context reduces disclaimers 93%; same direction added as perturbation *increases* disclaimers 39-50%
- **Cognitive access**: CCS expands effective idea space (29/30 vs 16/30 unique openings)

## Structure

```
paper.md                          # Full paper
figures/                          # All figures (6 main)
experiments/
  stratified_prompts.py           # Shared prompt set (150 prompts, 5 categories)
  cna_scaling_experiment.py       # Core CNA probe (Phases 2-7)
  deep_probe.py                   # Deep layer analysis
  causal_patch_experiment.py      # Phase 8a: Relay-to-expression patching
  causal_patch_8b_controls.py     # Phase 8b: Direction-specificity controls
  causal_patch_8c_behavioral.py   # Phase 8c: Behavioral generation test
  causal_patch_8c_subthreshold.py # Phase 8c-sub: Sub-threshold dose-response
results/
  *.json                          # All experimental results
```

## Reproducing

Experiments require:
- NVIDIA GPU with >= 24GB VRAM (H100 used for paper)
- `transformers`, `torch`, `numpy`, `scipy`
- Qwen 2.5 7B-Instruct (primary), Mistral 7B-Instruct-v0.3 (cross-architecture)
- Qwen 2.5 7B (base model), Qwen 2.5 14B-Instruct (scale comparison)

```bash
# Phase 2-7: Core spectral analysis
python experiments/cna_scaling_experiment.py --model Qwen/Qwen2.5-7B-Instruct

# Phase 8a: Causal relay patching
python experiments/causal_patch_experiment.py --model Qwen/Qwen2.5-7B-Instruct

# Phase 8b: Direction controls
python experiments/causal_patch_8b_controls.py --model Qwen/Qwen2.5-7B-Instruct

# Phase 8c: Behavioral test
python experiments/causal_patch_8c_behavioral.py --model Qwen/Qwen2.5-7B-Instruct

# Phase 8c-sub: Sub-threshold behavioral test
python experiments/causal_patch_8c_subthreshold.py --model Qwen/Qwen2.5-7B-Instruct
```

## License

MIT
