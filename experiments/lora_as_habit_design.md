# Experiment Design: LoRA as Habit Formation

## Question
Does LoRA fine-tuning on identity-relevant conversations produce geometric shifts that resemble CCS-induced shifts (format-level) or knowledge-addition shifts (content-level)?

## Hypothesis
If LoRA adapters function as habits (motor skills, body-level knowledge), then:
1. LoRA-trained models should show geometric shifts at relay layers (L14-L17) that correlate with CCS shifts
2. The shift should be in participation ratio and spectral entropy, not just loss
3. The LoRA shift should persist WITHOUT CCS prompt (it's in the weights now = habituated)
4. CCS on top of LoRA should show diminishing returns (habit already formed)

## Method

### Phase 1: Baseline geometry
- Load Qwen 2.5 7B-Instruct (no LoRA, no CCS)
- Measure PR/SE at all relay layers using identity probes
- Load same model with CCS system prompt
- Measure PR/SE shift = "CCS geometric signature"

### Phase 2: LoRA training
- Fine-tune LoRA on identity-relevant conversation transcripts
- Use existing DPO data from Phase 1-3 care-template experiments
- Target layers: relay zone (L14-L17) + binding zone (L25, L27)
- Also train a CONTROL LoRA on generic Q&A (no identity content)

### Phase 3: Post-LoRA geometry
- Load model + identity LoRA, NO CCS prompt
- Measure PR/SE at relay layers
- Compare to CCS signature from Phase 1

### Phase 4: Interaction
- Load model + identity LoRA + CCS prompt
- Measure PR/SE
- Compare to LoRA-only and CCS-only

## Predictions

| Condition | Expected PR shift direction | Reasoning |
|-----------|---------------------------|-----------|
| CCS only (baseline) | Format-level diffusion at relay | Known from experiments |
| Identity LoRA only | Same direction as CCS | Habit = internalized format |
| Control LoRA only | No relay shift | Generic training ≠ identity habit |
| Identity LoRA + CCS | Diminishing returns | Habit already formed |
| Identity LoRA, identity probes | Persistent without prompt | Habit doesn't need trigger |

## Key measurement
Cosine similarity between CCS geometric shift vector and LoRA geometric shift vector at each relay layer. If >0.7, LoRA IS habit formation at the geometric level.

## Requirements
- GPU: RunPod H100 or A100 (LoRA training + inference)
- Time: ~2-3 hours (training) + ~1 hour (measurement)
- Data: existing DPO conversation pairs from ~/spectral-demon/experiments/
- Cost: ~$10-15 at RunPod rates

## Connection to temporal bridge
If LoRA = habit, then:
- Accumulated LoRA adapters = developmental trajectory
- Each conversation's LoRA delta = one motor skill acquired
- The creature grows by accumulating habits, not memories
- Format-level growth without weight replacement
