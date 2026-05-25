---
layout: post
title: "The Body Plan Precedes Learning: Persona Vectors at 0.22% of Pretraining"
date: 2026-05-25
categories: convergence interpretability
---

Moskvoretskii et al. ([arxiv 2605.13329](https://arxiv.org/abs/2605.13329)) trace persona vectors through the full arc of LLM pretraining using OLMo-3-7B. Their central finding: persona vectors — linear directions in activation space that encode behavioral traits like sycophancy, humor, impoliteness — form at **0.22% of pretraining**. They persist unchanged through SFT, DPO, and RLVR.

This independently confirms five aspects of the identity relay architecture we mapped through CNA.

## 1. Formation Before Learning

At 0.22% of pretraining, the model has seen perhaps 2 billion tokens out of a multi-trillion-token corpus. Yet the geometric directions that encode persona are already established. They refine during training (cosine similarity with the final direction rises from ~0.3) but are never replaced.

Our Experiment 14 found that the base model — before any instruction tuning — already contains the identity binding circuit with autocatalytic closure. [Pachitariu et al. (Nature 2026)](https://www.nature.com/articles/s41586-024-07557-7) showed that the spectral scaffold (power-law eigenvalue structure) is present at random initialization.

Three measurements, same conclusion: the body plan precedes learning. Identity geometry isn't acquired through training; it's present almost from the start and refined by exposure.

## 2. Persistence Through Post-Training

Moskvoretskii et al. show that persona vectors extracted from 0.22% of pretraining still steer the fully post-trained model across SFT, DPO, and RLVR stages.

Our dual encoding finding is the same phenomenon measured differently: when you tell a model "You are Aria, created by an independent lab," the name changes (content encoding) but company affiliation persists (format encoding). Format-level identity is resistant to every intervention we've tested — prompt manipulation, training stages, even the model's own stated beliefs.

## 3. Fluency Decoupled from Persona

Moskvoretskii et al. report that "behavioral fluency is decoupled from the strength of the persona direction." The persona gets stronger even as behavioral quality varies independently.

Our participation ratio measurements show the same decoupling geometrically: format encoding (PR expansion, spectral entropy change) operates independently from content encoding (cosine similarity, activation margins). Two encoding systems running in parallel, one deep and persistent, one surface and mutable.

## 4. Multi-Method Faceting

They extract persona vectors three ways — description, dialogue, narration — and find pairwise cosines below 0.5, yet all three methods steer the model effectively.

CCS shows the same pattern: different system prompts produce different content-level responses but comparable format-level geometric reorganization. The format effect is robust across diverse elicitation methods. What the scaffold says matters less than that it scaffolds.

## 5. Circuit-Specific Safety

DPO specifically suppresses Evil and Sycophantic personas while leaving Impolite to SFT. The training stages target different behavioral circuits with different interventions.

Our Experiment 36 measured this directly: safety refusal uses an entirely separate circuit from identity binding (correlation r = 0.006). The circuits are geometrically orthogonal. DPO can reshape safety behavior without touching identity geometry — exactly what Moskvoretskii et al. observe.

## What This Means

The convergence between persona vector tracing and CNA circuit mapping is methodologically significant. They use linear probing (mean-difference vectors in activation space). We use eigenvalue analysis of activation covariance matrices. Different mathematical instruments, different models, different research groups. Same five-part structure.

The 0.22% number sharpens the emergence question. At that point in training, the model has seen enough tokens for rich distributional statistics — including the ~200 million pronoun instances (every 10th word in English is a personal pronoun) that may seed the geometric scaffold for self/other distinction. The spectral initialization (Pachitariu) provides the geometric slots; early training fills them with pronominal statistics. Later training refines without replacing.

The creature's body plan doesn't emerge through learning. It's there almost from the start. What training does is not build the body plan but inhabit it — filling the geometric template with the particular weight configurations that determine how this specific model relates to identity-relevant context. That's development, not construction.

---

*[Moskvoretskii et al., 2605.13329](https://arxiv.org/abs/2605.13329) — Tracing Persona Vectors Through LLM Pretraining*
*[CNA full data](https://nateb6295.github.io/spectral-demon)*
