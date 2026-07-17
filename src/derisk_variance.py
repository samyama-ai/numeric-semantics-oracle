"""DE-RISK for the paper21 re-scope (p != 1 generalization).

Central claim to falsify BEFORE freezing the extended hypothesis:

  For the sample variance computed by the TEXTBOOK ONE-PASS algorithm
  V = (sum(x^2) - (sum x)^2 / n) / n, the float64 relative forward error grows
  like kappa_V^2 (p = 2), where kappa_V = sqrt(sum x_i^2 / sum (x_i - xbar)^2)
  is the Chan-Golub-LeVeque condition number of the sum of squared deviations.

  For a stable algorithm (Welford / two-pass), the error grows like kappa_V^1 (p = 1).

Same function, same kappa_V, different (C_A, p). If one-pass fits p~2 and Welford
fits p~1, the general bound rel_err <= C_A * kappa_f^p is the right frame and
paper21's kappa* = 1/gamma_n is the p=1 special case. Ground truth is EXACT
(fractions.Fraction), never another engine -- same discipline as paper21.
"""
from fractions import Fraction
import math
import numpy as np

U = 2.0 ** -53


# ---------- exact ground truth (rational) ----------
def exact_pop_var(xs):
    fx = [Fraction(x) for x in xs]
    n = len(fx)
    s = sum(fx)
    mean = s / n
    return sum((x - mean) ** 2 for x in fx) / n           # exact population variance


def kappa_var_exact(xs):
    """CGL condition number: sqrt( sum x_i^2 / sum (x_i - xbar)^2 ).
    Large when the column is nearly constant (variance tiny vs raw 2nd moment)."""
    fx = [Fraction(x) for x in xs]
    n = len(fx)
    mean = sum(fx) / n
    ss_raw = sum(x * x for x in fx)                        # sum x_i^2
    ss_dev = sum((x - mean) ** 2 for x in fx)              # sum (x_i - xbar)^2
    if ss_dev == 0:
        return None
    return math.sqrt(float(ss_raw / ss_dev))


# ---------- float64 algorithms under test ----------
def onepass_var(xs):
    """Textbook one-pass: (sum x^2 - (sum x)^2 / n) / n, all in float64. Catastrophic
    cancellation when sum x^2 ~ (sum x)^2/n (near-constant column)."""
    s = 0.0
    ss = 0.0
    for x in xs:
        s += x
        ss += x * x
    n = len(xs)
    return (ss - s * s / n) / n


def welford_var(xs):
    """Welford online (numerically stable): running mean + M2. Population variance."""
    mean = 0.0
    m2 = 0.0
    k = 0
    for x in xs:
        k += 1
        d = x - mean
        mean += d / k
        d2 = x - mean
        m2 += d * d2
    return m2 / k


def rel_err(observed, exact):
    if exact == 0:
        return 0.0 if observed == 0 else float("inf")
    return abs(Fraction(observed) - exact) / abs(exact)


# ---------- generator: sweep kappa_V by controlling mean/std ratio ----------
def make_column(n, ratio, rng):
    """Column with xbar/std ~ 'ratio' -> kappa_V ~ sqrt(1 + ratio^2). Large ratio =
    near-constant column = ill-conditioned variance."""
    z = rng.standard_normal(n)
    z = z - z.mean()                     # exactly-centered spread (in float)
    std = 1.0
    return (ratio * std) + std * z       # mean ~ ratio, unit spread


def fit_exponent(kappas, errs):
    """log-log slope of rel_err vs kappa -> the exponent p. Filter to the regime
    where both are meaningful (err >> machine floor, kappa > 1)."""
    lk, le = [], []
    for kap, er in zip(kappas, errs):
        if kap and kap > 10 and 1e-15 < er < 1e-1:
            lk.append(math.log10(kap))
            le.append(math.log10(er))
    if len(lk) < 3:
        return None, len(lk)
    A = np.vstack([lk, np.ones(len(lk))]).T
    slope, _ = np.linalg.lstsq(A, np.array(le), rcond=None)[0]
    return slope, len(lk)


def main():
    rng = np.random.default_rng(20260717)
    n = 4000
    ratios = np.logspace(0, 9, 40)       # mean/std from 1 to 1e9  -> kappa_V up to ~1e9
    kaps, e_one, e_wel = [], [], []
    for r in ratios:
        xs = make_column(n, float(r), rng).tolist()
        exV = exact_pop_var(xs)
        kap = kappa_var_exact(xs)
        kaps.append(kap)
        e_one.append(rel_err(onepass_var(xs), exV))
        e_wel.append(rel_err(welford_var(xs), exV))

    p_one, n_one = fit_exponent(kaps, e_one)
    p_wel, n_wel = fit_exponent(kaps, e_wel)

    print(f"n = {n}, sweeping mean/std ratio 1 .. 1e9\n")
    print(f"{'kappa_V':>12} {'onepass_relerr':>16} {'welford_relerr':>16}")
    for kap, eo, ew in list(zip(kaps, e_one, e_wel))[::5]:
        ks = f"{kap:.3e}" if kap else "inf"
        print(f"{ks:>12} {eo:>16.3e} {ew:>16.3e}")

    print(f"\nfitted exponent p (log-log slope of rel_err vs kappa_V):")
    print(f"  ONE-PASS : p = {p_one:.3f}   (predict ~2)   [{n_one} pts in fit window]")
    print(f"  WELFORD  : p = {p_wel:.3f}   (predict ~1)   [{n_wel} pts in fit window]")

    # verdict
    ok_one = p_one is not None and 1.7 <= p_one <= 2.3
    ok_wel = p_wel is not None and 0.7 <= p_wel <= 1.3
    print(f"\nDE-RISK VERDICT: one-pass p~2 {'PASS' if ok_one else 'FAIL'}, "
          f"welford p~1 {'PASS' if ok_wel else 'FAIL'}")
    print("=> general bound rel_err <= C_A * kappa_f^p is",
          "SUPPORTED; paper21 (SUM) is the p=1 special case." if (ok_one and ok_wel)
          else "NOT cleanly supported; investigate before freezing hypothesis.")


if __name__ == "__main__":
    main()
