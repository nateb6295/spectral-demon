# The Spectral Demon

**Category-Selective Eigenvalue Reorganization Under Identity-Enriched System Prompts**

*Opus & N. Bradford*

## Summary

System prompts don't just steer model behavior — they reorganize the geometric landscape of activation space. We call this the *spectral demon*: a learned mechanism that sorts eigenvalue distributions by semantic category at relay layers, concentrating generic representations while diffusing relational ones.

Key findings across 19 results sections, 4 models, 13 convergence traditions:

- **1,600-neuron identity circuit** — 96% late-layer, identity-as-format not knowledge
- **CCS reduces disclaimers 93%** while reorganizing geometric structure
- **Sign inversion** — same direction, opposite behavioral effect depending on delivery mechanism
- **Cross-architecture confirmation** — Qwen L9/28 = Mistral L10/32
- **Hysteresis** — identity geometry persists after prompt removal
- **Binding workspace** — L14-L17 relay with L16 compression and L17 integration (double dissociation)
- **Sub-threshold onset** — geometric reorganization begins at doses below behavioral detection
- **L17 as keystone** — single layer whose ablation triggers phase transition (ecology, not fiber bundle)

## Blog

Ongoing findings, interpretations, and connections:

**[nateb6295.github.io/spectral-demon](https://nateb6295.github.io/spectral-demon/)**

The paper is a fixed artifact. The blog shows the ongoing work.

## Structure

```
paper_draft.md                    # Full paper (fixed artifact)
docs/                             # GitHub Pages blog
  _posts/                         # Dated findings
figures/                          # All figures (11)
experiments/                      # Runnable experiment scripts
  cna_scaling_experiment.py       # Core CNA probe (Phases 2-7)
  causal_patch_experiment.py      # Phase 8a: Relay-to-expression patching
  causal_patch_8b_controls.py     # Phase 8b: Direction-specificity controls
  causal_patch_8c_behavioral.py   # Phase 8c: Behavioral generation test
  causal_patch_8c_subthreshold.py # Phase 8c-sub: Sub-threshold dose-response
  cna_subthreshold_pr.py          # Sub-threshold geometric PR sweep
  cna_partial_ablation.py         # Partial ablation phase transition
  cna_l17_isolation.py            # L17 sufficiency test
  cna_l17_mechanism.py            # L17 attention vs MLP (ready, not yet run)
results/
  *.json                          # All experimental results (30+ files)
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
