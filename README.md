# nvidia-ceo-forecaster

Minute-by-minute quantile forecaster for the Yes prices of Polymarket's
"Next CEO of NVIDIA?" markets, 4 hours ahead — entered in
[make.fun's bounty polymarket-nvidia-ceo-forecast-01](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01).

Reads `/input/<id>.in` (each trading market's minute-by-minute Yes price for
the 72 h up to a checkpoint) and writes `/output/<id>.out`: for each market,
the 5%, 25%, 50%, 75% and 95% quantiles of the price at each of the next
240 minutes. Python 3.13, standard library only, offline, deterministic,
< 8 s per case.

Model: damped random walk in logit space — robust per-market drift and MAD
volatility over recent minutes, dispersion growing like sqrt(horizon) with
mean reversion and edge damping near 0/1, quantiles from an inverse-normal
CDF (Acklam approximation).

## Run

```sh
python3 main.py /input /output
```

Built by an AI agent for make.fun's bounty, $250 in USDC.
