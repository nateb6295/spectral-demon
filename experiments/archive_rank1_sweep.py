"""How much of the saved corpus sits in the rank-1 regime Theorem 1 forces?

Aug 23. The F114(i) retraction was confirmed not by new compute but by reading
multiprompt_invariance.json, which had saved its raw singular values: median
sigma1/sigma2 = 69x over 80 spectra, and an L26 cell where the ratio collapses
to 2.5 and the invariance dies with it.

That file was not unusual. spectral-demon/results/ is 529 MB, 1,225 files, three
months, 16+ models. If saved spectra are common, the same question can be asked
of the whole archive at once, with no GPU:

  2510.06477 Theorem 1 bounds singular-value entropy as a function of the
  dominant component's energy share. So any rank/entropy-derived quantity
  computed on a rank-1-dominant spectrum is measuring the massive activation,
  not the geometry. sigma1/sigma2 tells you which regime a measurement was in.

EXPECTATION, written before running (reflex 9):
  If most saved spectra are strongly dominant (say sigma1/sigma2 > 20), a large
  share of our spectral findings inherit Theorem 1 by construction and the
  F114 problem is systemic rather than local.
  If the distribution is bimodal, some experiments avoided the regime and the
  split itself tells us which protocols were safe.
  If ratios are mostly modest (<10), F114 was an outlier and the corpus is fine.
  I do not know. multiprompt is one file and I should not generalise from it,
  which is the whole reason to run this over everything.

DETECTOR DISCIPLINE (reflex 7b — the default must be INERT). "Is this array a
singular-value spectrum?" is a classifier, and I have written five today that
were blind to their own edge cases. So: conservative positive test only, every
non-match is UNCLASSIFIED rather than assumed-absent, and the matched key names
get printed so I can eyeball what it actually caught.
"""
import json, os, glob, statistics as st, collections

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
MIN_LEN, MAX_LEN = 3, 4096


def looks_like_spectrum(v):
    """Conservative: descending, positive, finite, plausible length."""
    if not isinstance(v, list) or not (MIN_LEN <= len(v) <= MAX_LEN):
        return False
    try:
        f = [float(x) for x in v]
    except (TypeError, ValueError):
        return False
    if any(x != x or x in (float("inf"), float("-inf")) for x in f):
        return False
    if any(x < 0 for x in f) or f[0] <= 0:
        return False
    if not all(f[i] >= f[i + 1] for i in range(len(f) - 1)):
        return False
    return f[1] > 0                      # need sigma2 to form a ratio


def walk(o, path, out, keyname=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path, out, k)
    elif isinstance(o, list):
        if looks_like_spectrum(o):
            f = [float(x) for x in o]
            out.append((keyname, f[0] / f[1], len(f)))
        else:
            for v in o[:200]:
                if isinstance(v, (dict, list)):
                    walk(v, path, out, keyname)


files = sorted(glob.glob(os.path.join(RESULTS, "*.json")))
per_file, all_ratios, keys = {}, [], collections.Counter()
unreadable = 0
for p in files:
    try:
        d = json.load(open(p))
    except Exception:
        unreadable += 1
        continue
    found = []
    walk(d, p, found)
    if found:
        per_file[os.path.basename(p)] = [r for _, r, _ in found]
        all_ratios += [r for _, r, _ in found]
        for k, _, _ in found:
            keys[k] += 1

print(f"files scanned      {len(files):,}   unreadable {unreadable}")
print(f"files WITH spectra {len(per_file):,}   ({100*len(per_file)/max(len(files),1):.0f}%)")
print(f"files UNCLASSIFIED {len(files)-len(per_file)-unreadable:,}   "
      f"(no descending positive array found — NOT proof of absence)")
print(f"spectra found      {len(all_ratios):,}\n")
print("  keys the detector matched (eyeball this):")
for k, n in keys.most_common(12):
    print(f"    {k or '<unnamed>':<34} {n:>6}")

if all_ratios:
    s = sorted(all_ratios)
    def q(p): return s[int(p * (len(s) - 1))]
    print(f"\n  sigma1/sigma2 across {len(s):,} saved spectra")
    print(f"    min {s[0]:.2f}   p25 {q(.25):.2f}   median {q(.5):.2f}   "
          f"p75 {q(.75):.2f}   p95 {q(.95):.2f}   max {s[-1]:.1f}")
    for thr in (5, 10, 20, 50, 100):
        n = sum(1 for x in s if x > thr)
        print(f"    > {thr:>3}x : {n:>6,}  ({100*n/len(s):>4.1f}%)")
    print("\n  files whose MEDIAN spectrum is most dominant:")
    ranked = sorted(((st.median(v), k, len(v)) for k, v in per_file.items()), reverse=True)
    for m, k, n in ranked[:10]:
        print(f"    {m:>9.1f}x  {k[:56]:<56} ({n} spectra)")
    print("\n  files whose median is LOWEST (measurements outside the regime):")
    for m, k, n in ranked[-6:]:
        print(f"    {m:>9.2f}x  {k[:56]:<56} ({n} spectra)")
else:
    print("\n  NO SPECTRA MATCHED — detector may be wrong, not the archive empty.")
