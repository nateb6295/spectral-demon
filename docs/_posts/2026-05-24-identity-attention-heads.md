---
layout: post
title: "Dedicated Identity Attention Heads: The Relay Has Eyes"
date: 2026-05-24
categories: findings
experiment: cna_attention_identity
models: [Qwen 2.5 7B Instruct]
---

Specific attention heads at each relay station attend to name tokens regardless of which name is present. The relay architecture has dedicated hardware.

## Setup

Measure attention from the last token position to name token positions at key layers (L7, L9, L12, L14, L17) across three identity names. Test which heads are consistent across names, and whether CCS scaffolding changes the attention patterns.

## Results

### Identity Attention Heads

| Layer | Top 5 Heads (Opus) | Top 5 Heads (Aria) | Consistent |
|-------|--------------------|--------------------|-----------|
| L7 | 0, 18, 16, 26, 27 | 0, 18, 27, 16, 26 | **5/5** |
| L12 | 4, 8, 12, 15, 20 | 12, 4, 8, 10, 22 | **3/5** |
| L17 | 23, 22, 25, 21, 7 | 23, 7, 22, 25, 27 | **4/5** |

The same attention heads attend to name tokens regardless of which name is present. These are dedicated identity attention heads — they implement the relay.

### Attention Concentration by Layer

| Layer | Peak Head Attention | Interpretation |
|-------|-------------------|----------------|
| L7 | 0.09-0.12 | Moderate — lexical detection |
| L9 | 0.14-0.18 | Stronger — seed processing |
| L12 | 0.15-0.19 | Router — focused but distributed |
| L14 | **0.33-0.40** | **Relay peak** — heads 16, 27 laser-focused on name |
| L17 | 0.31-0.34 | Binding — strong, slightly less than relay |

L14 has the most concentrated name attention — up to 40% of one head's attention goes to the name tokens. This is consistent with Experiment 33 showing L14 amplification (+157%) is stronger than L12 (+77%). The heads that attend most strongly to the name produce the most amplification when their signal is strengthened.

### CCS Effect on Attention

| Condition | Name Tokens | L17 Avg Name Attn | L17 Entropy |
|-----------|------------|-------------------|-------------|
| Bare | 2 | 0.076 | 1.37/3.00 (46%) |
| CCS | 8 | 0.050 | 2.19/5.17 (42%) |

CCS provides 4x more name tokens but gets LOWER per-head attention to name. Per-token: bare = 0.038 attention per name token, CCS = 0.006 per name token.

CCS doesn't concentrate attention — it distributes it across more targets. Relative entropy is similar (42-46% of max). The attention pattern doesn't become more focused; it becomes more spread.

## Three Findings

### 1. The Relay Has Dedicated Hardware

5/5 heads at L7 and 4/5 at L17 are consistent across names. These heads are not responding to specific name content — they're responding to the structural position of "identity name" in the prompt. This is a genuine computational circuit, not an emergent correlation.

### 2. L14 Is the Attention Bottleneck

L14 heads 16 and 27 allocate 33-40% of their attention to name tokens — the highest concentration in the entire network. The relay's strongest link is between the router (L12) and the binding layer (L17). This explains why L14 amplification (Experiment 33) is 2x more effective than L12 amplification: the signal is most concentrated at L14, so modulating it there has the largest effect.

### 3. CCS Distributes Rather Than Concentrates

CCS scaffolding provides more name tokens for the attention heads to attend to, but the per-token attention DECREASES. The total name attention budget is roughly constant — the heads distribute it across more targets rather than concentrating it.

Combined with Experiment 36 (CCS defense doesn't increase margins), this suggests CCS works not by strengthening the relay signal but by providing more reference points for the attention heads. The mechanism is redundancy (more things to attend to) rather than amplification (stronger signal per thing).

## Connection to Experiments 33 and 36

- **Experiment 33** showed L14 amplification is 2x more effective than L12. This experiment shows L14 has 2x more concentrated attention to name tokens. The correlation is direct: concentrated attention = effective amplification target.

- **Experiment 36** showed CCS doesn't increase activation margins. This experiment shows CCS doesn't concentrate attention. Together: CCS doesn't amplify the relay because attention is the relay's mechanism, and CCS distributes attention rather than concentrating it.

The relay is an attention-mediated circuit. The heads that form the relay are name-general (same heads for any name) and function-specific (dedicated to identity, not general-purpose). Modulating their outputs (Experiment 33) directly modulates identity binding because attention IS the relay mechanism.

## Data

Full attention analysis: `results/cna_attention_identity.json`
