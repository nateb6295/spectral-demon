---
layout: post
title: "The Keystone Is a Symbiosis"
date: 2026-05-23
categories: findings
---

We identified L17 as the keystone layer — ablating it triggers a phase transition in identity geometry (gen_CV 3.5%→13.3%). But what *mechanism* at L17 implements the binding?

Treisman's Feature Integration Theory predicts attention. The [simultagnosia parallel]({% post_url 2026-05-23-simultagnosia-not-balints %}) suggests attention-mediated between-object binding. Our experiment tested this directly.

## L17 Mechanism Dissection

Four conditions, measuring cross-name coefficient of variation at the expression layer (L25):

| Condition | rel_CV | gen_CV | Phase transition? |
|---|---|---|---|
| Baseline | 3.7% | 3.5% | — |
| L17 full ablation | 2.1% | 13.3% | **Yes** (3.8× gen_CV increase) |
| L17 attention only | 4.2% | 4.5% | No (1.3× increase) |
| L17 MLP only | 2.4% | 2.7% | No (0.8× — *decrease*) |

**Neither attention nor MLP alone triggers the phase transition.** It takes both.

## Synergistic Binding

The binding at L17 is a two-component interaction:

1. **Attention routes** information cross-positionally — distributing name-specific features across the sequence
2. **MLP transforms** the routed information — creating bound representations from distributed features
3. **The interaction** between routing and transformation produces the identity binding. Remove either component and the other partially compensates.

The MLP ablation actually *decreases* gen_CV (3.5%→2.7%) — the MLP contributes structured variation that the full attention-MLP interaction needs. Without MLP, the signal is cleaner but unbound.

## Ecology Refined

This sharpens the ecology metaphor. In real ecosystems, keystones are often **interaction effects**, not individual organisms. A keystone predator's impact depends on its relationship with prey species, competitors, and habitat structure. Remove any one relationship and the system compensates. Remove the predator entirely — all relationships at once — and the cascade begins.

L17 is the same: the keystone is the *symbiosis* between attention routing and MLP transformation, not either mechanism alone.

## Treisman, Partially Confirmed

Treisman predicted attention-mediated binding. She was partially right — attention IS involved, carrying cross-positional routing information. But she was working with a single-mechanism framework. The transformer architecture adds a second mechanism (MLP transformation) that doesn't exist in her cognitive model.

Feature Integration Theory needs updating: binding requires both *where to look* (attention) and *what to make of it* (MLP). The integration isn't in the attention — it's in the attention-MLP interaction.

## Three Experiments, One Evening

Tonight's session ran three experiments on H100:

1. **Intervention entropy**: Different identity prompts produce different L17 entropy trajectories. Documentation = most stable (observer mode). [Details →]({% post_url 2026-05-23-three-identity-geometries %})
2. **Unembedding projection**: CCS direction projects to vocabulary noise — identity operates below the token surface. Direction norms amplify monotonically through the relay (L14→L17: +24%).
3. **L17 mechanism**: Synergistic attention-MLP binding (this post).

**Previous posts**: [Three Identity Geometries]({% post_url 2026-05-23-three-identity-geometries %}), [Binding Workspace Double Dissociation]({% post_url 2026-05-23-binding-workspace-double-dissociation %})
