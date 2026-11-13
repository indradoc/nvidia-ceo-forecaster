#!/usr/bin/env python3
"""Quantile forecaster for Polymarket "Next CEO of NVIDIA?" markets.

Reads a case: /input/<id>.in — every trading market's Yes price at every
minute of the 72 hours up to a checkpoint. Writes /output/<id>.out:
for each market, five quantiles (5/25/50/75/95%) of the price at each of
the 240 minutes after the checkpoint.

Model: each market's recent minute prices give a drift and a volatility
estimate (robust, from the last K minutes). Future price paths are
modelled as a damped random walk with mean reversion toward the recent
level and dispersion growing with sqrt(horizon), clipped to [0,1] with
mass pushed to the boundaries near 0/1 (prices rarely move far when close
to an edge). Quantiles come from a discretised Gaussian on the transformed
logit scale, which keeps quantiles monotone and inside [0,1] by
construction. Everything is stdlib-only, deterministic, and fast.
"""
import json
import math
import os
import sys

QS = (0.05, 0.25, 0.50, 0.75, 0.95)
HORIZON = 240  # minutes ahead
EPS = 1e-6


def logit(p):
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def expit(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def quantiles_of_normal(mu, sigma, qs):
    # inverse normal CDF at qs, computed with a rational approximation
    # (Acklam) — stdlib only, good to ~1e-9.
    out = []
    for q in qs:
        out.append(mu + sigma * _norm_ppf(q))
    return out


_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def _norm_ppf(p):
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    elif p <= 1 - plow:
        q = p - 0.5
        r = q * q
        x = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
            (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
            ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    return x


def forecast_market(prices):
    """prices: list of minute Yes prices (0..1) up to checkpoint, oldest first.
    Returns list of HORIZON entries, each a 5-tuple of quantile prices."""
    n = len(prices)
    if n == 0:
        return [(0.0,) * len(QS)] * HORIZON
    last = prices[-1]
    if n < 30:
        # almost no history: price stays, tiny spread
        mu0, sig0 = logit(last), 0.02
        spread = sig0
    else:
        # drift over last K minutes, robust (median of diffs over windows)
        K = min(120, n)
        recent = prices[-K:]
        diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        diffs.sort()
        m = len(diffs)
        med = diffs[m // 2] if m else 0.0
        # volatility: MAD-based scale, on logit scale
        ld = [logit(max(min(p, 1 - EPS), EPS)) for p in recent]
        ldiffs = sorted(ld[i + 1] - ld[i] for i in range(len(ld) - 1))
        lmed = ldiffs[len(ldiffs) // 2]
        mad = ldiffs[len(ldiffs) // 2 + len(ldiffs) // 4] - ldiffs[len(ldiffs) // 4]
        sigma1 = max(1.4826 * abs(mad) * 0.5, 1e-4)
        # long-run vol from the whole window (scaled comparison)
        Kw = min(720, n)
        wd = sorted(prices[i + 1] - prices[i] for i in range(n - Kw, n - 1))
        long_range = (wd[int(len(wd) * 0.95)] - wd[int(len(wd) * 0.05)]) if len(wd) > 20 else 0.0
        mu0 = logit(last) + med * 3.0 * (1.0 if abs(med) > 2.5 * (sigma1 + 1e-5) else 0.0)
        spread = max(sigma1, 0.004)
        if long_range > 0.5:
            spread = max(spread, 0.02)
    out = []
    for h in range(1, HORIZON + 1):
        # dispersion grows like sqrt(h) but damped: prices don't random-walk forever
        t = h / 60.0
        sig = spread * math.sqrt(t) * (1.0 + 0.5 * math.sqrt(t))
        # edge effects: near 0 or 1 the price tends to stay
        near_edge = 0.0
        if last < 0.03 or last > 0.97:
            near_edge = 0.6
        sig *= (1.0 - near_edge * min(0.6, abs(math.log(max(last, EPS)) if last < 0.5 else abs(math.log(max(1 - last, EPS)))) * 0.5))
        sig = max(sig, 0.001)
        mu = mu0 - 0.10 * sig * sig * mu0  # slight mean reversion in logit space
        qs = quantiles_of_normal(mu, sig, QS)
        out.append(tuple(min(max(expit(q), 0.0), 1.0) for q in qs))
    # enforce monotone quantiles (they are by construction, but clip to [0,1] can
    # collapse them): sort each row
    return [tuple(sorted(r)) for r in out]


def main():
    import os as _os
    in_dir = _os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = _os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    if len(sys.argv) == 3:
        in_dir, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    forecasts = []
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name)) as f:
            case = json.load(f)
        for m in case.get("markets", []):
            prices = m.get("prices") or []
            q = forecast_market(prices)
            forecasts.append({"id": m["id"], "q": [[round(v, 6) for v in row] for row in q]})
        case_id = name[:-3]
        with open(os.path.join(out_dir, case_id + ".out"), "w") as f:
            json.dump({"forecasts": forecasts}, f, separators=(",", ":"))
        forecasts = []


if __name__ == "__main__":
    main()
