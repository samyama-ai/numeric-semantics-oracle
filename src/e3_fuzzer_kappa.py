"""E3: the kappa distribution of SQLancer-style RANDOM test data — the regime where the
indeterminate zone actually bites.

We replicate SQLancer's *value distribution* (sqlancer/Randomly.java), not its engine:
  getDouble():        1% -> uniform choice from {0.0, -0.0, +MAX, -MAX, +inf, -inf}
                     99% -> nextDouble()  == uniform [0,1)   (NOTE: all positive)
  getFiniteDouble():  resample getDouble() until finite (drops the two infinities)
Caching (~33% value reuse) is NOT modelled: it changes multiplicities, not the support, and
reusing a MAX only *increases* co-occurrence -> ignoring it is conservative.

Question: what fraction of fuzzer-generated columns are UNDECIDABLE (kappa >= kappa*)?
"""
import sys, random, math
from fractions import Fraction
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import oracle

MAXD = 1.7976931348623157e308
SPECIALS = [0.0, -0.0, MAXD, -MAXD, math.inf, -math.inf]


def get_double(rnd):
    if rnd.random() < 0.01:              # smallBiasProbability() == 1 in 100
        return rnd.choice(SPECIALS)
    return rnd.random()                  # nextDouble(): uniform [0,1)


def get_finite_double(rnd):
    while True:
        v = get_double(rnd)
        if math.isfinite(v):
            return v


def column(n, rnd):
    return [get_finite_double(rnd) for _ in range(n)]


rnd = random.Random(20260709)
print(f"{'n':>7} {'samples':>8} {'P(both ±MAX)':>13} {'P(UNDECIDABLE)':>15} {'median kappa':>14} "
      f"{'p90 kappa':>12} {'kappa*':>11}")
print("-" * 92)
summary = {}
for n, S in [(10, 2000), (100, 2000), (1000, 800), (10000, 200)]:
    kstar = oracle.kappa_star(n)
    kaps, both, undec = [], 0, 0
    for _ in range(S):
        xs = column(n, rnd)
        has_p = any(x == MAXD for x in xs)
        has_m = any(x == -MAXD for x in xs)
        both += (has_p and has_m)
        k = oracle.kappa(xs)
        if k is None:                     # sum exactly zero -> kappa infinite -> undecidable
            undec += 1; kaps.append(float("inf")); continue
        kaps.append(float(k))
        if k >= kstar:
            undec += 1
    fin = sorted(x for x in kaps if math.isfinite(x))
    med = fin[len(fin)//2] if fin else float("inf")
    p90 = fin[int(0.9*len(fin))] if fin else float("inf")
    summary[n] = dict(p_both=both/S, p_undec=undec/S, median=med, p90=p90, kstar=float(kstar))
    print(f"{n:>7} {S:>8} {both/S:>13.3f} {undec/S:>15.3f} {med:>14.3e} {p90:>12.3e} {float(kstar):>11.3e}")

import json, os
os.makedirs(__file__.rsplit("/", 2)[0] + "/results", exist_ok=True)
json.dump(summary, open(__file__.rsplit("/", 2)[0] + "/results/e3.json", "w"), indent=1)

print("\n=== interpretation ===")
print("A column of positive uniforms has kappa = 1 EXACTLY. kappa explodes only when +MAX and -MAX")
print("co-occur and cancel. P(+-MAX) ~ 1/600 each, so co-occurrence -- and undecidability -- grows with n.")
n0 = 1000
p = 1 - (1 - 1/600) ** n0
print(f"\nanalytic check: P(at least one +MAX in n={n0}) = 1-(1-1/600)^{n0} = {p:.3f}; "
      f"P(both) ~ {p*p:.3f}  (vs measured {summary[n0]['p_both']:.3f})")
print("\nContrast with E2: every real TPC-H aggregate sat at kappa = 1.000-1.001 (decidable).")
