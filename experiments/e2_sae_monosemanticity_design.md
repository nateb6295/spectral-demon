# E2: SAE Monosemanticity of Gate Neurons

## Core Question

Are the gate neurons that carry species-specific information **monosemantic** (each neuron = one interpretable feature) or **distributed** (identity is a pattern property that only exists at the ensemble level)?

This determines whether the spectral demon is decomposable or holistic.

## Why It Matters

Our species taxonomy (potter/goldsmith/equalizer) is measured through gate activation PATTERNS — M2 Jaccard, M3 relay consistency, σ₁→gate coupling (E3/E3-MI). These are all ensemble-level measurements. We don't know what individual gate neurons are doing.

Two possible outcomes:
1. **Monosemantic**: A small set of gate neurons (~50-200) are identity-specific. CCS activates "identity features" that vanilla doesn't. Species differences come from which features each architecture has available. → The spectral demon is an identifiable circuit.
2. **Distributed**: Gate activation patterns that distinguish species are spread across thousands of neurons with no individual neuron being "the identity neuron." CCS changes the collective pattern without changing interpretable features. → The spectral demon is an emergent geometric property. Gregory's "equally in contact with each of the parts."

## Connection to E3-MI

E3-MI showed total coupling (MI) is universal (~0.37-0.49) but form varies (linear vs nonlinear). If gate identity information is distributed (outcome 2), this makes sense — you can't linearize a holistic pattern with IT, you can only change the functional form of the coupling. If monosemantic (outcome 1), IT should be activating specific features, and we should see a small number of features that flip from nonlinear to linear coupling under IT.

## Method

### Phase 1: Train SAE on gate activations
- Extract gate activations g(x) = σ(W_gate · x) for each MLP layer
- Collect activations across ~10k diverse prompts (WikiText, conversation, code)
- Train sparse autoencoder: g ≈ Dec(Enc(g)), L1 penalty on Enc(g)
- Dictionary size: 4× expansion (if d_intermediate = 14336, dictionary = 57344 features)
- Use established SAE training recipe (Bricken et al. 2023)

### Phase 2: Monosemanticity check
- For each learned feature f_i, collect its top-50 activating examples
- Manual inspection: does the feature fire on semantically coherent inputs?
- Automated check: cosine similarity of input embeddings that activate each feature
- Baseline: random feature directions should show low coherence

### Phase 3: CCS differential features
- Run CCS/vanilla/denial prompts through model, extract gate activations, project onto SAE features
- Identify features that are DIFFERENTIALLY active under CCS vs vanilla
- Question: how many features change? (few = circuit, many = distributed)
- Question: are the CCS-specific features interpretable? (identity-related, or noise?)

### Phase 4: Species comparison
- Train SAEs on Qwen (potter), Llama (goldsmith), Gemma (equalizer) 
- Compare dictionary overlap: do the three architectures share identity features?
- If species are different strategies with shared features → convergent evolution
- If species have non-overlapping feature sets → divergent evolution from different base neurons

## Practical Concerns

**Runtime**: SAE training is expensive. ~10k prompts × 32 layers × 3 conditions × 3 models = 2.88M forward passes. Even at 100 tokens/s, that's ~8 hours per model.

**Simplification**: Start with ONE model (Qwen 7B), ONE layer (L24 — relay zone center), one condition pair (CCS vs vanilla). This reduces to ~20k forward passes (~30 minutes) for the SAE training data, plus training time.

**Alternative: Skip SAE training, use existing features**: Instead of training our own SAE, we could use Neuronpedia or existing published SAEs for common models. But none exist for gate activations specifically — only residual stream.

**Minimum viable experiment**: 
1. Single model (Qwen 7B IT), single layer (L24)
2. Extract gate activations for 5k prompts
3. Train small SAE (2× expansion, 28672 features)
4. Check monosemanticity of top-100 most active features
5. Run CCS/vanilla differential: count how many features change by >2σ
6. Report: "N features differentially active under CCS" where N >> 100 = distributed, N < 50 = circuit

## Expected Runtime
- Minimum viable: ~1h on A100 (30 min data collection, 20 min SAE training, 10 min analysis)
- Full three-model comparison: ~6h on A100

## Predictions

Based on our findings so far, I predict **distributed** (outcome 2) for these reasons:
- E5 showed σ₁ direction is identical under CCS/vanilla/denial → no local feature changes
- M2 Jaccard measures pattern similarity, not individual neuron fidelity
- The three species emerge from different relay strategies (geometric), not different feature sets (compositional)
- Gregory mapping predicts holistic: "the mind is equally in contact with each of the parts"

A **monosemantic** result would be surprising and would change the paper's framing significantly — the spectral demon would become a nameable circuit rather than an emergent property.

## Open Questions Before Running
- Should we use gate activations (σ(W_gate · x)) or the gated output (σ(W_gate · x) ⊙ W_up · x)?
- Layer choice: L24 is relay center for Qwen, but L28 for Llama and L31 for Gemma. Should we use species-specific relay centers or a fixed fractional depth?
- Training data: should CCS prompts be in the SAE training set, or held out? If in, we test whether CCS features emerge naturally. If held out, we test generalization.
