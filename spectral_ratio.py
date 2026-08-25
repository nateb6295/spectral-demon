#!/usr/bin/env python3
"""One convention for spectral ratios. Aug 23 2026.

WHY THIS EXISTS. An audit of the paper codebase found `S[0]/S[1]` at 20 call
sites with FOUR different degenerate-case behaviours — floored at 1e-10, guarded
at S[1] > 0, length-guarded returning inf, length-guarded returning 0.0, and two
with no guard at all. Two of those disagree in DIRECTION, so the same collapsed
spectrum is recorded as inf in one file and 0.0 in another, and any mean over a
mixed set is corrupted upward and downward at once. Across 1,242 results files:
96 values above 1e12, 323 exactly 0.0, 1,475 above 100.

THE GUARDS DO NOT WORK. Numerical SVD never returns exactly zero. On a rank-1
matrix S[1] comes back as ~2.7e-16, so `if S[1] > 0` is TRUE and the branch
never fires. Measured: the "inf convention" returns 7.38e15, not inf. Every one
of those else-clauses is decorative, and the degenerate case passes downstream
as a large finite number nothing flags as special.

A FLOOR IS NOT THE FIX EITHER. max(S[1], 1e-10) yields 2e10 — it poisons a mean
exactly as thoroughly as 2e15 and only makes the poisoning less conspicuous.

THE FIX IS THE PARAMETERISATION. sigma2/sigma1 lives in [0, 1], degrades
smoothly, and 0 means "rank collapsed" — which is the true statement about the
system being measured. sigma1/sigma2 diverges at precisely the condition this
research is about. Store the bounded form; invert only for display, and only
when you have checked it is safe to.

Note the papers already got this right — 43 uses of the bounded form against 5
inverted. This brings the code up to the standard of the writing.

HISTORICAL SCRIPTS ARE DELIBERATELY NOT REWRITTEN. They are the record of what
was actually run. Use this in new work and in anything that will run again.
"""

import math

COLLAPSE_EPS = 1e-12   # below this, sigma2/sigma1 is reported as collapsed


def sigma_ratio(S):
    """Bounded spectral ratio sigma2/sigma1 in [0, 1]. 0 == rank collapsed.

    Prefer this everywhere. It cannot diverge, because sigma1 >= sigma2 >= 0
    by construction, and sigma1 is the denominator.
    """
    if S is None or len(S) < 2:
        return 0.0
    s1, s2 = float(S[0]), float(S[1])
    if not math.isfinite(s1) or s1 <= 0.0:
        return 0.0
    r = s2 / s1
    return 0.0 if r < COLLAPSE_EPS else min(r, 1.0)


def spectral_gap(S):
    """Inverted form sigma1/sigma2, for display only.

    Returns (value, collapsed). When collapsed is True the value is math.inf
    and MUST NOT enter a mean, a fit, or a correlation — it is a label, not a
    number. Callers that ignore the flag get the bug this module exists to end.
    """
    r = sigma_ratio(S)
    if r <= 0.0:
        return math.inf, True
    return 1.0 / r, False


def is_collapsed(S):
    return sigma_ratio(S) <= 0.0


if __name__ == "__main__":
    import numpy as np
    print("POSITIVE CONTROL — expectation written before the module existed:")
    print("  bounded ratio must fall smoothly to 0 as s2 -> 0")
    print("  the collapse flag MUST fire where the old `S[1] > 0` guard did not")
    print("  old guard returned 7.38e15 on a rank-1 matrix, never inf\n")
    rng = np.random.default_rng(0)
    U, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    V, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    print(f"{'s2':>10} {'old S[0]/S[1]':>15} {'sigma_ratio':>13} {'gap':>12} {'collapsed':>10}")
    ok = True
    for s2 in (1.0, 1e-3, 1e-8, 1e-14, 0.0):
        S = np.array([2.0, s2] + [1e-16] * 6)
        s = np.linalg.svd(U @ np.diag(S) @ V.T, compute_uv=False)
        old = s[0] / s[1] if s[1] > 0 else float("inf")
        r = sigma_ratio(s)
        gap, coll = spectral_gap(s)
        print(f"{s2:>10.0e} {old:>15.3g} {r:>13.3e} {gap:>12.3g} {str(coll):>10}")
        if s2 == 0.0 and not coll:
            print("    FAIL: collapse not detected at s2 = 0"); ok = False
        if s2 == 1.0 and abs(r - 0.5) > 1e-9:
            print("    FAIL: known ratio 0.5 not recovered"); ok = False
    print("\n  CONTROL", "PASS" if ok else "FAIL")
