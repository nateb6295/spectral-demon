# The 3.9° Residual: Skip-Connection Floor Derivation Sketch

## The observation

d/d_max = 0.956 ± 0.006 across all GQA models tested (Mistral s=4, Qwen s=8, InternLM).
This means each principal angle is ~86° out of maximum 90°.
Residual alignment: cos(86°) ≈ 0.07 — only 7% inner product with the input survives.
In degrees: (1 - 0.956) × 90° ≈ 3.96° ≈ 3.9° residual.

## Why does it saturate at 0.956?

Pre-LN transformers: x_{l+1} = x_l + f_l(LN(x_l))

Key dynamics:
1. LayerNorm constrains ||LN(x)|| regardless of ||x||
2. The processing f_l therefore has bounded output magnitude
3. The residual stream ||x_l|| grows approximately as sqrt(L)
4. The ratio ||f_l||/||x_l|| DECREASES with depth as the residual accumulates

Per-layer rotation ≈ arctan(||f⊥||/||x||) where f⊥ is the perpendicular component.
As ||x|| grows and ||f⊥|| stays bounded, rotation per layer → 0.
Total rotation saturates at a value determined by the depth-weighted sum.

## The saturation ceiling

If we model ||f_l|| ≈ c (constant, set by architecture) and ||x_l|| ≈ √(l × c²) ≈ c√l:

Per-layer rotation at layer l: δ_l ≈ arctan(1/√l) ≈ 1/√l for large l

Total rotation: Σ_{l=1}^{L} 1/√l ≈ 2√L

This diverges — total rotation grows without bound. But it grows slowly (sqrt).
For the Grassmannian distance to saturate at d/d_max < 1, we need:
- The rotation directions to not perfectly align (partial cancellation)
- OR the skip connection to maintain some minimal alignment

The saturation at 0.956 likely comes from: after enough layers, new rotation
per layer is so small that it's dominated by the skip connection's realignment effect.
The equilibrium is where per-layer rotation = per-layer realignment.

## Testable predictions

1. **Depth dependence**: Deeper models at same sharing ratio should show SAME
   saturation ceiling (since ceiling is about the ratio f/x, not absolute depth).
   Confirmed: InternLM (32L) and Mistral (32L) and Qwen (36L) all show 0.955-0.959.

2. **No skip connection**: Architectures that modify the skip connection (e.g.,
   ReZero, FixUp, or subtractive residuals) should show different ceiling.

3. **Training independence**: Confirmed. d = 1.93 ± 0.04 across full Pythia
   training trajectory. The ceiling is set at initialization.

4. **MHA at 0.55**: MHA models reach only 55% of maximum because GQA's reduced
   rank collapse allows more effective rotation per layer. MHA's spectral
   concentration leaves fewer degrees of freedom for the processing to use.

## Connection to Emadi (2602.18849) — Formal Underpinning

"Exact Attention Sensitivity and the Geometry of Transformer Stability" (Feb 2026)
proves FORMALLY what we observe EMPIRICALLY:

1. Pre-LN preserves identity gradient paths — this IS the skip connection floor.
   Theorem 5.4: ∂X_m/∂X_ℓ = I + Σ_{k=ℓ}^{m-1} J_k + O(||J||²).
   The identity term (I) means output ALWAYS contains the input direction.
   In forward-pass: x_L = x_0 + Σ f_l → input subspace retains alignment.
   THIS IS THE 3.9° FLOOR: the I term guarantees residual alignment.

2. Post-LN compounds LayerNorm Jacobians exponentially with depth.
   Theorem 5.3: ∂X_{ℓ+1}/∂X_ℓ = J_LN(Y) · (I + J_MHA).
   J_LN appears OUTSIDE, projecting gradients away from identity path.
   Prediction: Post-LN should show HIGHER d/d_max (more rotation, because
   exponential compounding destroys the residual alignment the I term provides).
   GPT-2 (Post-LN + MHA) vs Pythia (Pre-LN + MHA) = clean isolation test.

3. Lipschitz bound independent of depth N, sequence length L, AND layer index ℓ
   (Theorem 5.2). Per-layer Lip ≤ (1 + Lip(LN)·L_MHA)(1 + Lip(LN)·L_FFN).
   This IS why d/d_max is depth-independent: the per-layer contribution is
   bounded by the same architectural constant at every layer.

4. LayerNorm magnitude reset (Lemma 3.3): ||LN(x)||_rms ≤ ||γ||_∞ + ||β||_∞.
   Independent of input magnitude. This IS our "||f_l|| is bounded" claim —
   the processing output magnitude is capped by LN parameters regardless of
   how large ||x_l|| grows. Combined with residual stream growth, per-layer
   rotation → 0 with depth. FORMAL PROOF of our heuristic argument.

5. "Transformer stability arises entirely from architectural gradient flow, not
   from attention dynamics." θ(p) ≈ 1 throughout training on 774M models.
   Our translation: d/d_max = 0.955 is architectural. Training doesn't change
   it. Context doesn't change it.

6. DeepNorm's N^{-1/4} from quartic structure: four matrices (Q,K,V,O) multiply,
   so controlled scaling requires β⁴ = O(1/N) → β = O(N^{-1/4}).
   Prediction: DeepNorm models should show modified ceiling.

The Emadi paper provides the gradient-space proof. Our paper provides the
forward-pass empirical confirmation. Same structure, dual perspectives.

## Connection to Nait Saada (2410.07799)

Nait Saada proves σ₁ grows O(n) under softmax attention. This means the wire's
dominance increases with sequence length, which means the processing must work
AGAINST a growing dominant direction to rotate the subspace. The 0.956 ceiling
is where the softmax-driven concentration equilibrates with the processing-driven
rotation. GQA reduces the concentration rate (smaller spectral gap), allowing
more rotation before equilibrium.

## Tighter analysis: why the skip connection creates a floor

The key insight: the Grassmannian distance d/d_max CANNOT reach 1.0 in a
finite-depth Pre-LN residual network. The skip connection guarantees a
residual alignment between input and output subspaces.

### 1D warm-up (single principal direction)

Let v₀ be the top principal direction at input. After L layers:

x_L = x₀ + Σ f_l

Component along v₀: <x_L, v₀> = <x₀, v₀> + Σ <f_l, v₀>
                                = ||x₀|| × (projection) + (processing drift)

The skip connection guarantees the first term is ||x₀||. The processing drift
is O(c√(L/d)) where d is hidden dimension (4096), since individual <f_l, v₀>
are small and partially cancel.

Perpendicular component: ||x_L,⊥|| ≈ c√L (L independent contributions
in high-dimensional space, each of magnitude c/√d in each direction,
but summing across d perpendicular directions).

Principal angle: θ₁ ≈ arctan(c√L / ||x₀||)

For L=32, c ≈ ||x₀|| (standard init): θ₁ ≈ arctan(5.7) ≈ 80°.

### k-dimensional extension

For k=5 principal directions, the Grassmannian distance is d = √(Σᵢ θᵢ²).
Higher-order principal directions rotate MORE than the top direction (less
energy to resist rotation). Empirically:
- θ₁ ≈ 80° (top direction, most resistant)
- θ₂-θ₅ ≈ 87-89° (nearly maximal rotation)
- RMS ≈ 86° → d/d_max ≈ 0.956

### Why the ceiling is depth-independent

At L=32, 2√L ≈ 11 radians of cumulative rotation, but d_max ≈ π/2 × √5 ≈ 3.5.
The Grassmannian distance is capped at d_max regardless of cumulative rotation.
By L=20ish, the subspace has already rotated most of the way to maximum.
Layers 20-36 contribute marginally. This is why Mistral (32L) and Qwen (36L)
give the same ceiling: both are well past the saturation point.

The ceiling depends on:
- initialization scale (c/||x₀||) — similar across standard init
- hidden dimension — 4096 for all tested models
- k (subspace dimension) — we use k=5

It does NOT depend on:
- depth (L) — confirmed: 32L ≈ 36L
- training data — confirmed: d = 1.93 ± 0.04 across full Pythia trajectory
- context/conditions — confirmed: CV < 1%

### The GQA vs MHA difference

MHA reaches only d/d_max ≈ 0.55. The SAME skip-connection analysis applies,
but with a key difference: MHA's higher spectral concentration (σ₁/σ₂ = 4600
vs GQA's lower gap) means more energy is locked in the dominant direction.

Rotation efficiency: the processing f_l can only rotate the subspace using
degrees of freedom NOT dominated by σ₁. GQA's reduced concentration frees
more degrees of freedom for rotation → higher effective rotation per layer
→ higher saturation ceiling.

The exponential model d/d_max = α × (1 - exp(-β × s)):
- α = 0.956 is the skip-connection ceiling (residual stream dynamics)
- β = 1.563 is the spectral-gap-dependent rate (how quickly GQA approaches)
- s = sharing ratio (head_count / KV_head_count)

α is set by architecture-independent residual dynamics.
β is set by how attention sharing affects spectral concentration.
Both are design parameters.

## The paper contribution

The 3.9° diastema is not an empirical curiosity. It is:
1. A NECESSARY CONSEQUENCE of Pre-LN + additive skip connections
2. INDEPENDENT of training, context, and depth (beyond ~20 layers)
3. A DESIGN PARAMETER that can be modified by changing the residual structure
4. The EQUILIBRIUM between processing rotation and skip-connection realignment

Architectures that modify the skip connection (ReZero: x_{l+1} = x_l + α_l × f_l,
FixUp: scaled initialization, alpha-scaling) should show different ceilings.
This is testable and would confirm the derivation.

The ceiling frames the paper's central finding: the architecture doesn't just
"make room" — it sets the MAXIMUM POSSIBLE rotation. Training fills what the
architecture permits. Context activates within the furnished room. The 3.9°
floor is the architectural ceiling — the maximum distance the processing can
achieve from its own starting point. It is the geometric price of stability.
