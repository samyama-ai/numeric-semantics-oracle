"""The numeric oracle: exact ground truth, condition number, forward-error bounds, classification.

Ground truth is the EXACT rational value of the stored float64s (a float IS an exact rational),
so the yardstick is mathematics, not another engine.

GENERAL FRAME (paper21 re-scope, HYPOTHESIS-v2-variance.md). For an aggregate f computed by an
algorithm A, the float64 relative forward error obeys

    rel_err  <=  C_A(n, u) * kappa_f ** p_A

and the testability boundary (where the bound admits any discrepancy, so no oracle can decide) is

    kappa*_{f,A} = (1 / C_A) ** (1 / p_A).

SUM is the linear (p=1) instance: f=SUM, kappa_f = sum|x|/|sum x|, C_A = gamma_n, p = 1, so
kappa* = 1/gamma_n -- paper21's boundary is this special case. Variance is a p=2 instance for the
textbook one-pass algorithm and a p=1 instance for Welford/two-pass: SAME function, different (C_A, p),
so the boundary is a property of the ALGORITHM, not the query. See `AGG` below.
"""
from fractions import Fraction
import math

U = Fraction(1, 2**53)          # unit roundoff, IEEE binary64

EXACT, BOUNDED, INDETERMINATE, ANOMALY = "exact", "bounded", "indeterminate", "ANOMALY"


def gamma(n):
    """Higham's gamma_n = n*u/(1-n*u) for recursive summation (Higham, ASNA Thm 4.1)."""
    nu = n * U
    return nu / (1 - nu) if nu < 1 else None      # None => bound is meaningless


def exact_sum(xs):
    return sum(map(Fraction, xs))


def kappa(xs):
    """Condition number of summation: sum|x| / |sum x|, computed exactly."""
    num = sum(abs(Fraction(x)) for x in xs)
    den = abs(exact_sum(xs))
    return None if den == 0 else num / den        # None => infinite


def bound_rel(n, kap, compensated=False):
    """Relative forward-error bound.
      plain recursive:   gamma_n * kappa            (Higham)
      compensated (KBN): (2u + O((n u)^2)) * kappa  (Neumaier / Higham ASNA ch.4)
    Returns Fraction, or None if meaningless (n*u >= 1)."""
    if kap is None:
        return None
    if compensated:
        nu = n * U
        return (2 * U + nu * nu) * kap
    g = gamma(n)
    return None if g is None else g * kap


def rel_error(observed, exact):
    if exact == 0:
        return None if observed == 0 else Fraction(float("inf"))
    return abs(Fraction(observed) - exact) / abs(exact)


def classify(observed, exact, n, kap, compensated=False):
    """exact / bounded / indeterminate / ANOMALY.

    ANOMALY = the discrepancy EXCEEDS what floating-point summation can explain.
    Per NC2 this is a hard gate: it is a candidate engine bug or a modelling error,
    and must be triaged, never dismissed.
    """
    err = rel_error(observed, exact)
    if err is None or err == 0:
        return EXACT, err, bound_rel(n, kap, compensated)
    B = bound_rel(n, kap, compensated)
    if B is None or B >= 1:
        # the bound admits any discrepancy: no oracle can decide here
        return INDETERMINATE, err, B
    return (BOUNDED if err <= B else ANOMALY), err, B


def kappa_star(n, compensated=False):
    """A1: the testability boundary. B(n,kappa) >= 1  <=>  kappa >= kappa*.
    Beyond kappa*, NO differential oracle over float64 SUM can separate a bug from rounding."""
    if compensated:
        nu = n * U
        c = 2 * U + nu * nu
    else:
        c = gamma(n)
    return None if not c else 1 / c


def kappa_eps(n, eps, compensated=False):
    """A2: the epsilon crossover. Below it a fixed eps is too LAX (false negatives);
    above it, too STRICT (false positives). A single eps is sound at exactly one kappa."""
    if compensated:
        nu = n * U
        c = 2 * U + nu * nu
    else:
        c = gamma(n)
    return None if not c else Fraction(eps).limit_denominator(10**18) / c


# ======================================================================================
# GENERAL FRAME: rel_err <= C_A(n,u) * kappa_f ** p   (paper21 re-scope, p != 1)
# ======================================================================================
# An algorithm A is (C_A(n) as a Fraction, exponent p). SUM is the p=1 instance; the
# textbook one-pass variance is p=2; Welford/two-pass variance is p=1. Constants are
# SOUND upper bounds (verified never-exceeded by the NC2v hard gate), not fits.

def _c_sum_plain(n):      return gamma(n)                       # Higham Thm 4.1
def _c_sum_comp(n):       nu = n * U; return 2 * U + nu * nu    # Neumaier/KBN
def _c_var_onepass(n):    return gamma(n)                       # (sum x^2 - (sum x)^2/n)/n
def _c_var_welford(n):    return gamma(n)                       # stable online / two-pass

ALGO = {
    # name          C_A(n)            p    human label
    "sum_plain":    (_c_sum_plain,    1,  "SUM (recursive)"),
    "sum_comp":     (_c_sum_comp,     1,  "SUM (compensated/KBN)"),
    "var_onepass":  (_c_var_onepass,  2,  "VARIANCE (textbook one-pass)"),
    "var_welford":  (_c_var_welford,  1,  "VARIANCE (Welford/two-pass)"),
}


def exact_var(xs, sample=False):
    """Exact population (or sample) variance as a Fraction. A float64 is an exact rational,
    so this is the mathematically correct answer -- not another engine's opinion."""
    fx = [Fraction(x) for x in xs]
    n = len(fx)
    if n == 0 or (sample and n == 1):
        return None
    mean = sum(fx) / n
    ss = sum((x - mean) ** 2 for x in fx)
    return ss / (n - 1 if sample else n)


def kappa_var(xs):
    """Chan-Golub-LeVeque condition number of the variance:
        kappa_V = sqrt( sum x_i^2 / sum (x_i - xbar)^2 ) = sqrt(1 + xbar^2 / V_pop).
    Large exactly when the column is NEAR-CONSTANT (variance tiny vs raw 2nd moment).
    The ratio is exact rational; only the final sqrt is inexact (of an exact rational)."""
    fx = [Fraction(x) for x in xs]
    n = len(fx)
    mean = sum(fx) / n
    ss_raw = sum(x * x for x in fx)
    ss_dev = sum((x - mean) ** 2 for x in fx)
    if ss_dev == 0:
        return None                                   # infinite conditioning
    return math.sqrt(float(ss_raw / ss_dev))


def bound_general(n, kappa_f, algo):
    """Relative forward-error bound C_A(n) * kappa_f^p for the named algorithm.
    Returns Fraction, or None if meaningless (n*u >= 1 or kappa_f is None)."""
    if kappa_f is None:
        return None
    c_fn, p, _ = ALGO[algo]
    c = c_fn(n)
    if c is None:
        return None
    # kappa_f is a float (has a sqrt in it for variance); keep the bound as a Fraction
    return c * (Fraction(kappa_f) ** p)


def classify_general(observed, exact, n, kappa_f, algo):
    """exact / bounded / indeterminate / ANOMALY against the algorithm's own bound.
    ANOMALY (per NC2v) = discrepancy exceeds what algorithm A can produce -> candidate
    engine bug or modelling error, must be triaged."""
    err = rel_error(observed, exact)
    if err is None or err == 0:
        return EXACT, err, bound_general(n, kappa_f, algo)
    B = bound_general(n, kappa_f, algo)
    if B is None or B >= 1:
        return INDETERMINATE, err, B
    return (BOUNDED if err <= B else ANOMALY), err, B


def kappa_star_general(n, algo):
    """The testability boundary kappa*_{f,A} = (1/C_A)^(1/p). SUM plain reproduces 1/gamma_n;
    var_onepass gives (1/gamma_n)^(1/2) -- a QUADRATICALLY narrower decidable zone."""
    c_fn, p, _ = ALGO[algo]
    c = c_fn(n)
    if not c:
        return None
    return float((1 / c) ** (1.0 / p)) if p != 1 else float(1 / c)
