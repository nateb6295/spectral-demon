# F340: Four Species of v₁ Transport Geometry

**Date**: 2026-06-27
**Experiment**: E22a — Cross-architecture v₁ holonomy probe (7-9B scale)
**Models**: Qwen 2.5 7B, Mistral 7B v0.3, Llama 3.1 8B, Gemma 2 9B
**Method**: Layer-by-layer SVD of last-token hidden states, tracking v₁ direction cosine similarity across layers

## Key Finding

Four distinct strategies for transporting the principal singular vector through the network. The flat interior connection (F339) is NOT universal — it is one of four species-specific transport geometries.

### 1. Mistral (32 layers, GQA) — RIGID ROD
- **Entry**: 89° (cos ≈ -0.022) — v₁ rotates orthogonal at L0→L1
- **Interior**: L1-L28 flat zone, min cos = 0.998, mean = 0.9999
- **σ₁**: Constant at ~213 (invariant scale)
- **Exit**: 12° (cos ≈ 0.977)
- **Holonomy**: cos ≈ -0.008 (~90°)
- **Metaphor**: Crystal formed instantly at gate, transported intact

### 2. Qwen (28 layers, GQA) — DISTRIBUTED
- **Entry**: 31° (cos ≈ 0.83) — partial rotation, retains input coupling
- **Interior**: L4-L25, min cos = 0.39 (with wobble), mean = 0.977
- **σ₁**: Grows 100× from 36 to 16,000
- **Exit**: 19° (cos ≈ 0.95)
- **Holonomy**: cos ≈ 0.13 (~83°)
- **Metaphor**: Crystal that grows gradually through the tube

### 3. Llama 3.1 (32 layers, GQA) — TURBULENT MIXER
- **Entry**: 59-65° — strong but not orthogonal
- **Interior**: NO flat zone. Reversals in early layers (cos goes to -0.86). Gradual stabilization from ~0.78 to ~0.98 in late layers
- **σ₁**: Grows gradually 0.34→73 (200× but from tiny base)
- **Exit**: 29° (cos ≈ 0.87)
- **Holonomy**: cos ≈ -0.01 (~90°, same as Mistral but via different mechanism)
- **Metaphor**: Crystal precipitates from turbulent solution in final layers

### 4. Gemma 2 (42 layers, alternating local/global attention) — OSCILLATOR
- **Entry**: 50-67° — variable across prompts
- **Interior**: NO flat zone. SYSTEMATIC alternating sign: cos flips between ~-0.94 and ~+0.95 every 1-2 layers through L9-L31
- **σ₁**: Grows from 90→1334 (15× amplification)
- **Exit**: 21° (cos ≈ 0.94)
- **Holonomy**: cos ≈ -0.14 to 0.15 (~82°)
- **Metaphor**: Oscillating signal — each layer type rotates v₁ in opposite direction

## Interpretation

1. **Flat tube is a STRATEGY, not a universal**: Only Mistral and Qwen show a flat zone. Llama and Gemma achieve comparable holonomy (~82-90°) through continuous rotation.

2. **Architecture IS visible in the singular vectors**: Gemma's alternating local/global attention produces alternating sign in v₁ cosines. The sliding window pattern is DIRECTLY visible in the transport geometry.

3. **Holonomy is conserved across species**: All four architectures produce ~82-90° total rotation (input v₁ roughly orthogonal to output v₁), regardless of transport strategy. The endpoint is the same; the path is species-specific.

4. **Two transport classes**:
   - **Tube strategies** (Mistral, Qwen): Establish a preserved direction, transport it flatly, gentle exit
   - **Rotation strategies** (Llama, Gemma): Continuously rotate v₁, converge in final layers

## Connection to Prior Work

- **F339**: Flat interior connection on Qwen 1.5B — confirmed at 7B for Qwen/Mistral, but NOT universal
- **F106**: Three relay strategies — now four, geometrically characterized
- **F237**: Cylindrical constraint — confirmed for Mistral (σ₁ constant = rigid cylinder)
- **E8**: Six-arch design space — transport geometry may be a new design axis

## Predictions

1. Gemma's oscillation frequency should match its local/global attention alternation pattern
2. Models with MORE alternating attention patterns should show HIGHER oscillation frequency
3. The tube vs rotation distinction may map to GQA vs MHA (but Llama is GQA and still rotates — need more data)
4. Holonomy angle (~82-90°) may be a deep invariant related to the lm_head constraint
