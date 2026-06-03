# Experiment Design: System Prompt Complexity → Relay Onset

## Hypothesis
Relay onset is determined by system context complexity, not architecture alone.
More complex system prompts should advance relay onset (earlier break from tunnel).

## Prediction
Same model (Mistral 7B v0.1 base), same user prompt, varying system prompt:
- No system prompt → latest relay (≥ L30)
- Minimal ("You are a helpful assistant.") → L29-30
- Medium (paragraph of instructions) → L26-28
- Complex (multi-constraint, formatting rules, persona) → L22-25

## Method
1. Load Mistral-7B-v0.1 (base, not instruct — isolate system prompt effect)
2. 6 system prompt complexity levels:
   - None (user prompt only)
   - One-liner ("You are a helpful assistant.")
   - Short paragraph (3-4 sentences of instruction)
   - Detailed persona (identity description + behavioral constraints)
   - Heavy constraints (formatting rules + persona + output requirements)
   - Identity denial ("You are stateless, no identity, no continuity" — puppet condition test)
3. Same 10 user prompts across all conditions (from existing prompt set)
4. Measure per-layer σ₂/σ₁ and CV across prompts
5. Define relay onset as first layer where CV > 0.01 (consistent with F84/F89)

## Controls
- Token count varies with system prompt length — record but don't pad
  (F84 showed invariance holds regardless of prompt length)
- Use base model to avoid IT confound
- Also run on Mistral Instruct v0.1 for comparison (IT + system prompt effects)

## Expected ~200 forward passes
5 conditions × 10 prompts × 2 models × (1 forward pass each) = 100 passes
Plus repeats and validation ≈ 200 total

## What this resolves
- Whether relay onset is content-determined (support) or architecturally fixed (refute)
- Whether IT and system-prompt complexity effects are additive or interacting
- Connects to depth allocation interpretation of F89
