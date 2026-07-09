"""The numeric oracle: exact ground truth, condition number, Higham bounds, classification.

Ground truth is the EXACT rational sum of the stored float64s (a float IS an exact rational),
so the yardstick is mathematics, not another engine.
"""
from fractions import Fraction

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
