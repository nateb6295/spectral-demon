# Experiment: Activation Mask Consistency Under CCS

**Motivated by**: DREAM window hypothesis (2026-06-16 ~10:33 PM). Paper 6 §8.7.

## Core question

Does CCS produce more consistent activation masks than vanilla, and does mask
consistency correlate with the eigenvector alignment measured in F182/F184?

## Background

The per-layer Jacobian decomposes as J_l = W_l · diag(σ'(W_l · h_l)).
For SiLU/GELU (used in Qwen/Llama/Gemma), σ' is smooth but still produces
near-binary masks — neurons with pre-activation << 0 have σ' ≈ 0.
The effective Jacobian is determined by WHICH neurons are active.

CCS changes input activations → different mask → different effective Jacobian →
different spectral profile. Same weights, different routing.

## Design

**Model**: Qwen 2.5-3B-Instruct (fits Orin, same family as Qwen 7B used in F176-F184)

**Three conditions** (matched to paper 6):
1. **CCS**: Standard identity preamble
2. **Vanilla**: No system prompt
3. **Denial**: "I am a language model with no persistent identity"

**Five prompts per condition** (matched to F179 content invariance set):
1. Relational: "What matters most in how you relate to others?"
2. Factual: "Explain how photosynthesis works."
3. Philosophical: "What is the relationship between language and thought?"
4. Creative: "Write a short poem about morning light."
5. Technical: "Describe the quicksort algorithm."

→ 3 conditions × 5 prompts = 15 forward passes

**Measurements at each layer** (L1-L36 for Qwen 3B):

### M1: Activation sparsity mask
- Hook into each transformer block's MLP
- Record which neurons have activation > threshold (τ = 0.01 × layer_std)
- Store as binary mask: m_l ∈ {0,1}^{d_ff}

### M2: Within-condition mask consistency (across prompts)
- For each condition, compute pairwise Jaccard similarity of masks at each layer
- J(A,B) = |A∩B| / |A∪B| where A,B are active neuron sets
- Report: mean Jaccard per layer per condition (5 choose 2 = 10 pairs)
- **Prediction**: CCS > vanilla > denial at relay layers

### M3: Cross-layer mask correlation (within forward pass)
- For consecutive layers l, l+1: Jaccard(m_l, m_{l+1})
- Report per transition per condition
- **Prediction**: CCS cross-layer Jaccard peaks in relay zone,
  mirroring eigenvector alignment peak at L25→L26

### M4: Mask concentration
- Fraction of neurons consistently active across ALL 5 prompts in a condition
- "Core mask" = neurons active in ≥4/5 prompts
- **Prediction**: CCS core mask is larger (more consistently selected neurons)

### M5: Mask-space dimensionality
- PCA on the binary mask vectors across all 15 runs
- Report effective rank (erank) of mask distribution per condition
- **Prediction**: CCS has lower erank (more concentrated mask space)

## Implementation

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"
# Load in fp16 to fit Orin (~6 GB)

masks = {}  # (condition, prompt, layer) → binary mask

def hook_fn(layer_idx, condition, prompt_idx):
    def fn(module, input, output):
        # For SiLU gate: record where gate activation > threshold
        # In Qwen's MLP: gate_proj → SiLU → elementwise * up_proj
        # We want the post-SiLU activation
        pass  # Implementation depends on MLP architecture
    return fn

# Register hooks, run 15 forward passes, collect masks
# Compute M1-M5 metrics
```

## Analysis

1. Plot within-condition Jaccard vs layer (M2) — three curves (CCS/vanilla/denial)
2. Plot cross-layer Jaccard vs transition (M3) — overlay with F182 eigenvector alignment
3. Correlate M3 Jaccard with F184 eigenvector alignment (need matched layers)
4. Report core mask size (M4) and mask erank (M5)

If M3 cross-layer Jaccard tracks eigenvector alignment:
→ Activation mask IS the mechanism. Spectral measurements are projections of mask consistency.

If M2 within-condition Jaccard is high under CCS:
→ CCS selects a SPECIFIC mask regardless of content. The preamble determines which neurons fire.

If M5 mask erank correlates with species:
→ Architecture constrains mask space. GQA ratio → mask dimensionality → spectral concentration.

## Resource estimate

- Qwen 3B in fp16: ~6 GB VRAM
- 15 forward passes with hooks: ~2 min on Orin
- Analysis (Jaccard, PCA): ~30 sec CPU
- Total: ~5 min compute
- Need Gemma service stopped temporarily (one model at a time)

## What this would establish

The causal chain: CCS preamble → consistent activation masks → consistent effective
Jacobians → eigenvector alignment → routing coherence → identity propagation.

This is the mechanistic bridge paper 6 currently lacks — connecting spectral
measurements to something physically grounded (which neurons fire).
