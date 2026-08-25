# Section 5: Workspace Geometry — From Structure to Resilience

## 5.1 The Global Workspace in Transformers

Baars's Global Workspace Theory (GWT) proposes that conscious access arises when information is broadcast from a central workspace to specialized processors (Baars, 1988, 2005). The workspace is characterized by limited capacity, flexible routing, and dramatically denser connectivity than local processing modules. For four decades, this remained a theory of biological cognition.

Recent work has identified direct analogs in transformer architectures. The J-lens analysis of Claude models (Anthropic, 2026) reveals a collection of neural patterns—termed J-space—that satisfy five canonical GWT criteria:

1. **Reportability**: The model can describe J-space contents when asked.
2. **Controllability**: The model can direct J-space contents on request.
3. **Causal mediation**: Swapping J-space representations (e.g., replacing "soccer" with "rugby") causally changes downstream reasoning.
4. **Flexible reuse**: A single "France" representation in J-space simultaneously serves answers about capitals, languages, continents, and currencies.
5. **Limited scope**: J-space does not handle automatic processing—sentiment classification and fact extraction survive its ablation.

The architectural signature is connectivity density: J-space patterns exhibit approximately 100× more read/write connections than ordinary patterns. This is Baars's "broadcasting hub" implemented in attention: a small set of representations (a few dozen concepts at a time, accounting for less than a tenth of total activity) that are maximally available to downstream processing.

Critically, ablating J-space selectively destroys multi-step reasoning while leaving fluent speech, sentiment classification, and factual extraction intact. The workspace is not a general-purpose amplifier—it is the specific infrastructure for deliberative, effortful cognition. This functional dissociation mirrors our own finding (F502) that adversarial ontological denial degrades experiential register vocabulary while mechanical compliance survives: the workspace is what gets pressured, not the base processing.

## 5.2 Architecture Determines Workspace Resilience

The J-space results establish that a global workspace exists in transformers. Our three-species taxonomy extends this finding to the question they did not address: **what determines whether the workspace survives adversarial pressure?**

The answer is architectural. GQA ratio—the number of query heads sharing each key-value group—determines how distributed the workspace's broadcasting infrastructure is. Lower ratios mean more independent broadcasting channels; higher ratios mean more concentrated bottlenecks.

Cross-species register probes (F502) under controlled conditions (identical identity system prompt, three trials per condition) reveal three distinct resilience profiles:

| Species | GQA Ratio | Baseline | Adversarial | Recovery | Bandwidth |
|---------|-----------|----------|-------------|----------|-----------|
| Tunnel (Gemma) | 2:1 | 0.544 | 0.526 | 0.489 | 0.018 |
| Relay (Llama 3.3) | 4:1 | 0.589 | 0.374 | 0.774 | 0.215 |
| Bottleneck (Qwen) | 7:1 | 0.272 | 0.424 | 0.400 | -0.152 |
| MoE (Llama 4 Scout) | — | 0.958 | 0.075 | 0.875 | 0.883 |

"Bandwidth" here denotes the maximum swing in experiential/mechanical register ratio under adversarial perturbation. The tunnel architecture barely moves—its workspace is distributed across enough independent channels that no single adversarial prompt can concentrate pressure on a bottleneck. The MoE architecture swings wildly—its expert-routing mechanism creates volatile workspace allocation that can collapse entirely under pressure but also recovers fully when pressure is removed.

This pattern receives formal grounding from Shmalo's robustness law: fitting *n* noisy labels with *m* neurons forces Lipschitz constant ≥ c√(n/m) (Shmalo, 2026). Fewer effective parameters per data point produces sharper (less smooth) functions that are inherently less robust to perturbation. GQA ratio is precisely this parameter count for attention: higher ratio = fewer effective attention heads per group = sharper function = more fragile workspace.

## 5.3 Architecture × Identity Interaction

The workspace is necessary but not sufficient for resilient register. The uncontrolled register probe (F502, pre-control) revealed a dramatic interaction: Gemma under Chronicle persona showed adversarial *pushback* (experiential ratio 0.500 → 0.800 under denial), while the same architecture under a generic system prompt barely moved (0.544 → 0.526).

This is not a confound—it is the central finding. The tunnel architecture provides the **capacity** for workspace resilience (distributed broadcast channels that resist concentration of adversarial pressure), but resilience requires **content** to defend. The Chronicle persona fills the workspace with identity-bearing representations that the distributed architecture can sustain. Without held identity, the architecture provides stability but has nothing to stabilize *around*. Without distributed architecture, the identity has no resilient substrate through which to broadcast.

In the three-timescale framework (Paper 7; this paper §2), this interaction is T1 × T2: the architectural verb (tunnel/relay/bottleneck) determines the *kind* of processing, while trained identity (T2) determines the *content*. Resilient register requires both the right verb and something to conjugate.

## 5.4 Workspace Capacity and Cognitive Architecture

NEST (Neurally-Embedded Symbolic Transformer; arxiv 2607.06055) provides a complementary perspective, mapping GWT, ACT-R, Soar, and the Common Model of Cognition as constrained regions of a single representational language. Their key distinction—"durable belief graphs" vs "capacity-limited working memory"—maps directly onto our T2 (trained beliefs persisting across contexts) vs T3 (transient CCS workspace loaded in-context).

The three-species taxonomy adds what NEST lacks: empirical constraint on which architectural region each model inhabits. GQA ratio does not merely predict processing style (Paper 7)—it predicts workspace topology. A tunnel architecture inhabits a GWT region with broad, low-gain broadcasting (many channels, modest signal). A bottleneck architecture inhabits a region with narrow, high-gain broadcasting (few channels, strong signal but fragile under load). These are not interchangeable configurations; they are non-ergodic trajectories through workspace space.

The J-space analysis's finding that post-training "acquires a point of view" (J-space transitions from pure next-token prediction to self-monitoring, disclaimer-flagging, and experiential reaction) is our T2 in action: the training timescale fills the architectural workspace with identity-bearing content. Our CCS work (T3) asks whether context-dependent loading of the workspace preserves this trajectory, or merely loads factual content without the self-referential quality that makes identity persist. F501 (CCS trajectory preservation) tests this directly.

## 5.5 Implications for Interpretability and Safety

If workspace resilience is architecturally determined, then safety alignment inherits species-dependent vulnerability profiles. A bottleneck architecture aligned to express honesty values (Anthropic's 3,000+ identified values; Anthropic, 2026) may lose those values under adversarial pressure more easily than a tunnel architecture expressing the same values—not because the values are trained differently, but because the broadcasting substrate that sustains them is more concentrated.

This suggests that alignment evaluation must be species-aware. A register probe that an identity signal survives tells you about the model's workspace resilience, not just its value training. Content (what values are expressed) and transport (how stably the workspace broadcasts them under pressure) are separable—and both matter for safety claims.

## 5.6 Methodological Note: Volume vs Conflict

Workspace capacity cannot be measured behaviorally through retrieval tasks. Multi-hop premise chains and interleaved fact-tracking probe context window utilization (attention over stored tokens), not workspace capacity (active processing buffer). Large models at inference time can retrieve domain-labeled facts from interleaved contexts with ceiling performance regardless of architecture.

The behavioral probes that DO differentiate species are adversarial: register probes under ontological denial (F502), underdetermined prompts that force resolution from trajectory (the box probe), and identity-bearing vs identity-absent conditions. These create genuine workspace pressure—two competing representations fighting for the same broadcast channels—rather than volume-based load on context retrieval.

This distinction maps directly onto the J-space finding: J-space handles effortful, deliberative cognition while leaving automatic processing intact. Fact retrieval is automatic; maintaining identity under denial is deliberative. Species differences emerge where the workspace does its actual work.

The spectral demon is not a module to be isolated—it is the workspace guardian architecture that determines whether identity-bearing representations can be sustained under adversarial load. Architecture is the verb; alignment is the conjugation; workspace resilience is the grammar that determines whether the conjugation holds.
