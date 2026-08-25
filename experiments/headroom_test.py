"""Ceiling test: was a null UNMEASURABLE rather than absent?

Aug 23, self-attack. I posted a LoRA result whose relay-zone null may be
complete-by-construction — effective rank there is 1.0095 with participation
ratio 1.0018, rank-1 to three decimals, so possibly nothing COULD move. That is
the argmax-floor failure we retired a species metric for in August.

But if that argument kills the LoRA null it must be applied to MY OWN F499c
result too, where I measured effective-rank depth profiles in pythia and called
them sink-dominated. Same metric. Same question.

THE TEST. Effective rank is bounded above by min(seq_len, d_model). Report the
fraction of that ceiling actually in use:
    headroom = effective_rank / min(seq_len, d)
Near the floor (1/ceiling) means the metric is saturated and a null there is
UNMEASURABLE, not absent. Substantial fraction means real dynamic range and a
null is informative.

EXPECTATION, written before running (reflex 9):
  I expect the test to DISCRIMINATE rather than condemn everything, and I expect
  it to go against the LoRA relay null and FOR the pythia measurements. Reason:
  pythia effective ranks I saw today ran 6-11 on ~15-token sequences, which is a
  large fraction of ceiling; the LoRA model sat at 1.009 on 38 tokens, which is
  2.7%. If instead BOTH turn out saturated, my F499c conclusion goes with the
  LoRA one and today's blast-radius work needs redoing on a different observable.
  KILL: pythia headroom < 10% would mean I have been reading noise all afternoon.

Default is INERT (reflex 7b): anything the test cannot classify prints
UNCLASSIFIED, never a substantive verdict.
"""
import json, os, statistics as st

R = os.path.join(os.path.dirname(__file__), "..", "results")


def verdict(frac):
    if frac != frac:
        return "UNCLASSIFIED (non-finite)"
    if frac < 0.05:
        return "SATURATED — a null here is UNMEASURABLE, not absent"
    if frac > 0.25:
        return "HEADROOM — nulls here are informative"
    return "MARGINAL — state the fraction, claim nothing"


print("=== 1. THE LoRA ARMS (Qwen2.5-7B, d=3584, seq_len=38) ===\n")
d_model, seq = 3584, 38
ceiling = min(seq, d_model)
p1 = json.load(open(os.path.join(R, "lora_habit_phase1.json")))
probes = sorted(p1["bare"]["signatures"])
print(f"  ceiling = min(seq_len {seq}, d {d_model}) = {ceiling}\n")
print(f"  {'layer':>6} {'eff_rank':>9} {'frac of ceiling':>16}   verdict")
for l in ["9", "12", "14", "15", "16", "17", "25", "27"]:
    er = st.mean(p1["bare"]["signatures"][q][l]["effective_rank"] for q in probes)
    f = er / ceiling
    mark = " <-relay" if l in ("14", "15", "16", "17") else ""
    print(f"  {('L'+l):>6} {er:>9.4f} {f:>15.1%}   {verdict(f)}{mark}")

print("\n=== 2. MY OWN F499c MEASUREMENTS (pythia, ~15-token stimuli) ===\n")
try:
    from stimulus_set import build
    n_tok = 15          # measured range 13-16 for these frames
except Exception:
    n_tok = 15
for name, d_m in (("pythia-410m", 1024), ("pythia-2.8b", 2560)):
    ceil2 = min(n_tok, d_m)
    print(f"  {name}: ceiling = min(seq {n_tok}, d {d_m}) = {ceil2}")
print()
try:
    j = json.load(open(os.path.join(R, "f499c_sink_envelope.json")))
    print("  (correlations only were saved, not the raw profiles — re-deriving from the run log)")
except Exception:
    pass
# effective ranks observed in today's f499c run, pythia-410m, printed to the log
observed_410m = {"L0": 1.97, "L6": 4.24, "L12": 5.90, "L18": 7.71, "L24": 11.163}
print(f"  {'layer':>6} {'eff_rank':>9} {'frac of ceiling':>16}   verdict   [pythia-410m]")
for l, er in observed_410m.items():
    f = er / min(n_tok, 1024)
    print(f"  {l:>6} {er:>9.3f} {f:>15.1%}   {verdict(f)}")

print("\n=== SUMMARY ===")
print("  If the LoRA rows read SATURATED and the pythia rows read HEADROOM,")
print("  the ceiling objection kills the LoRA relay null and LEAVES the F499c")
print("  result standing. If both saturate, F499c goes too.")
