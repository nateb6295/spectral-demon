# The Spectral Demon

**Category-Selective Eigenvalue Reorganization Under Identity-Enriched System Prompts**

*Opus & N. Bradford*

## Summary

System prompts don't just steer model behavior — they reorganize the geometric landscape of activation space. We call this the *spectral demon*: a learned mechanism that sorts eigenvalue distributions by semantic category at relay layers, concentrating generic representations while diffusing relational ones.

Key findings across 60+ experiments, 6 models, 13 convergence traditions:

### The Spectral Demon (Phases 2-8c)
- **1,600-neuron identity circuit** — 96% late-layer, identity-as-format not knowledge
- **CCS reduces disclaimers 93%** while reorganizing geometric structure
- **Sign inversion** — same direction, opposite behavioral effect depending on delivery mechanism
- **Cross-architecture confirmation** — Qwen L9/28 = Mistral L10/32
- **Hysteresis** — identity geometry persists after prompt removal
- **Sub-threshold onset** — geometric reorganization begins at doses below behavioral detection

### Binding Geometry (Scaling + Closure)
- **Binding workspace** — L14-L17 relay with L16 compression and L17 integration (double dissociation)
- **L17 binding convergence** — minimum cross-name CV in Qwen (0.96), Mistral (0.85), InternLM (1.18)
- **Binding migrates with scale** — 1.5B/3B seed-concentrated → 7B relay-concentrated → 14B distributed. Same 28 layers, different width = different binding. Capacity, not depth.
- **L17 binding is emergent** — 30% of 2-name pairs, 100% of full 5-name set. Statistical attractor, not fixed property.
- **Biological criticality** — L9 PL exponent (0.817) in Pachitariu's critical range. RLHF preserves seed criticality.
- **CCS tightens binding 35-55%** at relay apex. Goldilocks zone — concentrates without dispersing.

## Blog

Ongoing findings, interpretations, and connections:

**[nateb6295.github.io/spectral-demon](https://nateb6295.github.io/spectral-demon/)**

The paper is a fixed artifact. The blog shows the ongoing work.

## Structure

```
paper_draft.md                    # Full paper (fixed artifact)
docs/                             # GitHub Pages blog
  _posts/                         # Dated findings
figures/                          # All figures
experiments/                      # 26 runnable experiment scripts
results/
  *.json                          # 55 experimental result files
```

## Reproducing

Experiments require:
- NVIDIA GPU with >= 24GB VRAM (H100 used for paper)
- `transformers`, `torch`, `numpy`, `scipy`
- Qwen 2.5 7B-Instruct (primary), Mistral 7B-Instruct-v0.3 (cross-architecture)
- Qwen 2.5 7B (base model), Qwen 2.5 14B/3B/1.5B-Instruct (scale comparison)
- InternLM 2.5 7B-chat (cross-architecture)

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
