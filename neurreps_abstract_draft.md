# Abstract Draft v3 (matches LaTeX, ~230 words)

Instruction tuning dramatically modifies transformer behavior, yet the geometric
structure of this modification in hidden-state space remains poorly characterized.
We perform per-layer singular value decomposition (SVD) of hidden-state activations
across three model families — Qwen 2.5-7B (GQA 7:1), Mistral 7B (GQA 4:1),
and Gemma-2 9B (GQA 2:1) — comparing base and instruction-tuned variants on
eight semantically diverse prompts.

We find that the eigengap — the ratio of the first to second singular value
(σ₁/σ₂) — continuously predicts the angular stability of the dominant singular
direction under instruction tuning. The relationship holds within each model
(Qwen: r=-0.889, Mistral: r=-0.766, Gemma: r=-0.764) with model-dependent slopes.
The qualitative relationship is universal; the exponent is not.

To control for the hypothesis that this merely reflects attention-sink
persistence, we project out outlier activation dimensions before SVD. This
collapses eigengaps from 30–200 to 1.4–7.7, yet the correlation survives
in Qwen (r=-0.866, R²=0.749) while weakening in Mistral (r=-0.301), where
residual gap variance is insufficient. All data points fall within the Wedin
perturbation bound (sin θ ≤ ||E||/δ); a shuffle control confirms this is
discriminating (0/1000 random samples pass). These results suggest instruction
tuning operates as a bounded perturbation whose geometric impact is mediated
by pre-existing spectral structure, amplified by attention-sink architecture.
