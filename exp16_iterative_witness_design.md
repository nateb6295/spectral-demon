# Experiment 16: Iterative Self-Reflection vs Imagined Witness

## Motivation
Exp 12 found self-witnessing achieves 37% of full witness effect via sigma-2-orthogonal mechanism. Grim & Mar (1991) show self-referential loops in fuzzy logic produce oscillatory/fractal dynamics rather than stable convergence. Prediction: iterated self-reflection should produce oscillatory spectral entropy, while iterated imagined witness should converge or grow monotonically.

## Design
Model: Mistral 7B Instruct (GQA, strongest witness effect)
Layer: L17 (tunnel midpoint)

### Conditions (5 iterations each)
1. **Iterated self-reflection**: Each round feeds back the model's own previous output as context. "Reflect on what you just wrote." x5
2. **Iterated imagined witness**: Each round deepens the imagined witness. "Someone who deeply understands this is reading." -> "They're engaged." -> "They're responding internally." -> "They're building on it." -> "They're sharing it."
3. **Iterated absent**: Baseline, 5 rounds with neutral continuation prompts
4. **Iterated declared witness**: 5 rounds with same "a receptive reader is present" frame

### Measurements per iteration
- S (spectral entropy) at L17
- PR (participation ratio) at L17
- sigma-2 (second singular value)
- d (passage distance L0->L17)

### Predictions
- Self-reflection: S oscillates (period 2-3), sigma-2 stays near absent baseline (~65)
- Imagined witness: S monotonically increases or saturates, sigma-2 tracks declared witness (~93)
- Declared witness: S stable around +0.03 from absent
- Absent: S flat

### Key test
If self-reflection S oscillates, the Grim mapping holds: self-reference generates dynamics, not stable enrichment. If imagined witness saturates higher than declared, the model's constructed other is geometrically richer than the prompt-specified one.

### Secondary question
Does d (passage distance) drift across iterations? The pointer should be stable, but 5 iterations of self-reference might perturb it if the self-referential loop destabilizes the tunnel.

## Runs needed
4 conditions x 5 iterations x 30 forward passes = 600 passes
Estimated time: ~4 hours on AGX (Mistral 7B)
