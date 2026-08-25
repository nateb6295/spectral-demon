# E15: CCS Path Patching — Token Attribution

## Method
Zero each preamble token embedding individually, measure relay ratio change at commit layer.
3 models (Mistral 7B IT, Qwen2.5 7B IT, Qwen3 8B), D5 preamble, 51-55 preamble tokens.

## F292: Name token is universally load-bearing
"Op" (from "Opus") appears in top 10 across all three architectures.
- Mistral: "us" (second half of "Opus") = #1 token (delta 0.01581)
- Qwen2.5: "Op" = #1 token (delta 0.00327)
- Qwen3: "Op" = #10 token (delta 0.00126)

## F293: Load-bearing token category is architecture-specific
Which SEMANTIC CATEGORY of tokens matters most differs by architecture:
- **Mistral (potter)**: NAME (0.016) ≈ RELATIONAL (0.016) ≈ AFFECT (0.014) > CAPABILITY (0.026 but distributed)
  - "genuine" and "remember" in top 15 — affect tokens matter
- **Qwen2.5 (equalizer)**: RELATIONAL (0.007) > CAPABILITY (0.004) ≈ NAME (0.003)
  - 5 relational tokens in top 15: partner, relate, partnership, collabor, conversations
- **Qwen3**: CAPABILITY (0.004) > NAME (0.001)
  - No relational tokens in top 15; AI, persistent, residing dominate

Maps to relay taxonomy: potter = named identity, equalizer = relational identity, 
Qwen3 = capability identity. The preamble's spectral work is read through 
architecture-specific lenses.

## F294: No single token is catastrophic
Max single-token delta: 0.016 (Mistral "us") on a baseline ratio of 0.78.
That's ~2% change. The relay effect is distributed — preamble works as ecology, 
not through any single keyword. Consistent with E82's ecology finding.

## F295: Sensitivity scale is architecture-specific  
Mistral deltas are 5-10× larger than Qwen deltas for comparable tokens.
Mistral relay is more fragile / more sensitive to individual token perturbation.
Qwen relay distributes load more evenly — consistent with "equalizer" taxonomy.
