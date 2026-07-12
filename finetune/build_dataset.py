"""
Dataset builder for LoRA fine-tuning.
Generates 160K instruction-following examples across multiple market regimes
with realistic options chain data, Greeks, and strategy selection.
Target: ~$40 training cost on Lightning.ai A10G ($0.71/hr × 56hrs).
"""
import json
import logging
import math
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an expert options trader. Given market context, output a JSON trade decision."

STRATEGIES = [
    "long_call", "long_put", "bull_call_spread", "bear_put_spread",
    "iron_condor", "straddle", "strangle", "calendar_spread",
]

# ── Realistic ticker pools with sector context ────────────────────────────────

TICKERS = {
    "large_cap_etf": ["SPY", "QQQ", "IWM", "DIA"],
    "mega_cap_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "high_beta_tech": ["AMD", "NFLX", "CRM", "ADBE", "SNOW", "PLTR", "COIN", "ROKU"],
    "financials": ["JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BRK-B"],
    "healthcare": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT"],
    "consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO"],
    "industrials": ["CAT", "DE", "HON", "UNP", "RTX", "LMT", "BA", "GE"],
    "high_vol": ["GME", "AMC", "BBBY", "SPCE", "RIVN", "LCID", "SOFI", "HOOD"],
}

ALL_TICKERS = [t for group in TICKERS.values() for t in group]

# ── News headline templates ────────────────────────────────────────────────────

NEWS_TEMPLATES = {
    "bullish": [
        "{ticker} beats earnings estimates, raises full-year guidance",
        "{ticker} announces major partnership with {partner}",
        "{ticker} stock upgraded to Overweight by {bank}",
        "{ticker} reports record revenue, shares surge in pre-market",
        "Analysts raise price targets on {ticker} after strong quarterly results",
        "{ticker} secures $2B government contract",
        "{ticker} FDA approval for new drug sends shares higher",
        "{ticker} announces $5B share buyback program",
        "{ticker} expands into new international markets",
        "Insider buying surge at {ticker} signals confidence",
    ],
    "bearish": [
        "{ticker} misses earnings estimates, cuts guidance",
        "{ticker} shares fall on disappointing revenue outlook",
        "{ticker} downgraded to Underweight by {bank}",
        "{ticker} faces regulatory investigation over accounting practices",
        "{ticker} announces layoffs affecting 15% of workforce",
        "{ticker} loses major customer contract worth $1.5B",
        "{ticker} CEO steps down amid strategic disagreements",
        "{ticker} recalls 2M units due to safety concerns",
        "{ticker} warns of supply chain disruptions impacting Q4",
        "Short interest in {ticker} surges to 25% of float",
    ],
    "neutral": [
        "{ticker} reports in-line earnings, maintains guidance",
        "{ticker} announces board refresh with new independent directors",
        "{ticker} expands board of directors",
        "{ticker} completes previously announced acquisition",
        "{ticker} initiates quarterly dividend of $0.50/share",
        "Market watches {ticker} ahead of Fed decision",
        "{ticker} trading volume elevated amid sector rotation",
        "Options activity suggests mixed sentiment on {ticker}",
        "{ticker} analyst day scheduled for next week",
        "Institutional investors adjust {ticker} positions",
    ],
    "earnings": [
        "{ticker} reports earnings tomorrow — implied vol elevated",
        "{ticker} earnings next week, options market pricing 8% move",
        "{ticker} pre-earnings: IV rank at 85%, historical move avg 6%",
        "{ticker} earnings in 3 days, straddle pricing suggests large move",
        "{ticker} post-earnings: actual move of {move_pct}% vs implied {imp_pct}%",
        "{ticker} earnings surprise history: beat 8 of last 12 quarters",
        "{ticker} options volume surges ahead of earnings announcement",
    ],
    "iv_event": [
        "{ticker} VIX equivalent spikes to 45 ahead of Fed announcement",
        "{ticker} implied volatility crush expected post-earnings",
        "{ticker} IV rank at 92% — highest in 52 weeks",
        "{ticker} options unusually expensive relative to historical norms",
        "{ticker} volatility surface shows steep skew in puts",
        "{ticker} term structure inverted — near-term IV > long-term IV",
        "{ticker} realized vol 30% vs implied vol 45% — vol risk premium elevated",
    ],
}

PARTNERS = ["Microsoft", "Apple", "Google", "Amazon", "NVIDIA", "Meta", "Samsung", "Tesla"]
BANKS = ["Goldman Sachs", "Morgan Stanley", "JPMorgan", "Bank of America", "Citigroup", "Wells Fargo"]

# ── Market regime definitions ──────────────────────────────────────────────────

MARKET_REGIMES = {
    "strong_uptrend": {"rsi": (60, 80), "iv_rank": (0.2, 0.5), "macd_hist": (0.3, 1.5), "trend": "bullish"},
    "weak_uptrend": {"rsi": (50, 65), "iv_rank": (0.3, 0.6), "macd_hist": (0.05, 0.4), "trend": "bullish"},
    "strong_downtrend": {"rsi": (20, 40), "iv_rank": (0.4, 0.8), "macd_hist": (-1.5, -0.3), "trend": "bearish"},
    "weak_downtrend": {"rsi": (35, 50), "iv_rank": (0.3, 0.6), "macd_hist": (-0.4, -0.05), "trend": "bearish"},
    "ranging_low_vol": {"rsi": (40, 60), "iv_rank": (0.1, 0.3), "macd_hist": (-0.1, 0.1), "trend": "neutral"},
    "ranging_high_vol": {"rsi": (35, 65), "iv_rank": (0.6, 0.9), "macd_hist": (-0.2, 0.2), "trend": "neutral"},
    "pre_earnings": {"rsi": (45, 65), "iv_rank": (0.7, 0.95), "macd_hist": (-0.3, 0.3), "trend": "neutral"},
    "post_earnings_crush": {"rsi": (40, 60), "iv_rank": (0.1, 0.3), "macd_hist": (-0.5, 0.5), "trend": "neutral"},
    "squeeze_setup": {"rsi": (45, 55), "iv_rank": (0.05, 0.2), "macd_hist": (-0.05, 0.05), "trend": "neutral"},
    "breakout": {"rsi": (65, 80), "iv_rank": (0.4, 0.7), "macd_hist": (0.5, 2.0), "trend": "bullish"},
    "breakdown": {"rsi": (20, 35), "iv_rank": (0.5, 0.85), "macd_hist": (-2.0, -0.5), "trend": "bearish"},
    "mean_reversion": {"rsi": (15, 30), "iv_rank": (0.5, 0.8), "macd_hist": (-0.8, -0.1), "trend": "bearish"},
}


def _rng_range(rng: random.Random, low: float, high: float) -> float:
    """Generate a random float in [low, high]."""
    return rng.uniform(low, high)


def _generate_greeks(
    rng: random.Random,
    option_type: str,
    strike: float,
    spot: float,
    dte: int,
    iv: float,
) -> dict[str, float]:
    """
    Generate realistic Greeks using Black-Scholes approximations.
    Uses analytical formulas rather than random values for physical accuracy.
    """
    T = max(dte / 365.0, 1 / 365)
    r = 0.05  # risk-free rate approx
    sqrt_T = math.sqrt(T)

    # Moneyness ratio
    moneyness = spot / strike
    ln_m = math.log(moneyness)

    # Black-Scholes d1, d2
    d1 = (ln_m + (r + 0.5 * iv ** 2) * T) / (iv * sqrt_T)
    d2 = d1 - iv * sqrt_T

    # Standard normal PDF and CDF approximations
    pdf_d1 = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)

    if option_type == "call":
        delta = _norm_cdf(d1)
        theta = (
            -(spot * pdf_d1 * iv) / (2 * sqrt_T)
            - r * strike * math.exp(-r * T) * _norm_cdf(d2)
        ) / 365
    else:
        delta = _norm_cdf(d1) - 1
        theta = (
            -(spot * pdf_d1 * iv) / (2 * sqrt_T)
            + r * strike * math.exp(-r * T) * _norm_cdf(-d2)
        ) / 365

    gamma = pdf_d1 / (spot * iv * sqrt_T)
    vega = spot * sqrt_T * pdf_d1 / 100  # per 1% move in IV

    # Clamp to realistic ranges
    delta = max(-1, min(1, delta))
    gamma = max(0, min(0.15, gamma))
    theta = min(0, max(-2, theta))
    vega = max(0, min(5, vega))

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "rho": round(rng.uniform(-0.05, 0.05), 4),
    }


def _norm_cdf(x: float) -> float:
    """Approximation of the standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _generate_option_chain(
    rng: random.Random,
    spot: float,
    dte_range: tuple[int, int],
    iv: float,
    num_calls: int = 8,
    num_puts: int = 8,
) -> dict[str, list[dict[str, Any]]]:
    """Generate a realistic option chain with multiple strikes and expirations."""
    calls = []
    puts = []

    # Generate strikes around spot price
    strike_step = spot * rng.uniform(0.01, 0.03)
    num_strikes = max(num_calls, num_puts)

    # Strike offsets from ATM (negative = below, positive = above)
    offsets = sorted([rng.uniform(-3, 3) for _ in range(num_strikes)])

    expirations = []
    for _ in range(rng.randint(2, 4)):
        dte = rng.randint(*dte_range)
        exp_date = (datetime.now(timezone.utc) + timedelta(days=dte)).strftime("%Y-%m-%d")
        expirations.append((exp_date, dte))

    for exp_date, dte in expirations:
        for offset in offsets[:num_calls]:
            strike = round(spot + offset * strike_step, 1)
            if strike <= 0:
                continue

            greeks = _generate_greeks(rng, "call", strike, spot, dte, iv)
            mid = max(0.05, spot * iv * sqrt(dte / 365) * abs(greeks["delta"]) * rng.uniform(0.8, 1.2))
            spread = mid * rng.uniform(0.02, 0.15)
            bid = max(0.01, round(mid - spread / 2, 2))
            ask = round(mid + spread / 2, 2)

            calls.append({
                "symbol": f"{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}{exp_date.replace('-', '')}C{int(strike*1000):08d}",
                "type": "call",
                "strike": strike,
                "expiration": exp_date,
                "dte": dte,
                "open_interest": rng.randint(50, 15000),
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 2),
                "spread_pct": round((ask - bid) / mid if mid > 0 else 0, 4),
                "last_trade": round(mid * rng.uniform(0.95, 1.05), 2),
                "implied_volatility": round(iv * rng.uniform(0.85, 1.15), 4),
                **greeks,
            })

        for offset in offsets[:num_puts]:
            strike = round(spot + offset * strike_step, 1)
            if strike <= 0:
                continue

            greeks = _generate_greeks(rng, "put", strike, spot, dte, iv)
            mid = max(0.05, spot * iv * sqrt(dte / 365) * abs(greeks["delta"]) * rng.uniform(0.8, 1.2))
            spread = mid * rng.uniform(0.02, 0.15)
            bid = max(0.01, round(mid - spread / 2, 2))
            ask = round(mid + spread / 2, 2)

            puts.append({
                "symbol": f"{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=4))}{exp_date.replace('-', '')}P{int(strike*1000):08d}",
                "type": "put",
                "strike": strike,
                "expiration": exp_date,
                "dte": dte,
                "open_interest": rng.randint(50, 15000),
                "bid": bid,
                "ask": ask,
                "mid": round(mid, 2),
                "spread_pct": round((ask - bid) / mid if mid > 0 else 0, 4),
                "last_trade": round(mid * rng.uniform(0.95, 1.05), 2),
                "implied_volatility": round(iv * rng.uniform(0.85, 1.15), 4),
                **greeks,
            })

    return {
        "calls": sorted(calls, key=lambda x: x["strike"])[:num_calls],
        "puts": sorted(puts, key=lambda x: x["strike"])[:num_puts],
    }


def _select_strategy(
    rng: random.Random,
    regime: str,
    rsi: float,
    iv_rank: float,
    trend: str,
    earnings_soon: bool,
    iv_percentile: float,
    bollinger_position: float,  # 0 = at lower band, 1 = at upper band
    macd_hist: float,
) -> tuple[str, str]:
    """
    Select the optimal strategy given market conditions.
    Returns (strategy, action).
    """
    # HOLD conditions — no edge
    if regime == "ranging_low_vol" and not earnings_soon:
        if rng.random() < 0.6:
            return "none", "HOLD"

    if regime == "post_earnings_crush" and iv_rank < 0.15:
        return "none", "HOLD"

    # Pre-earnings plays
    if earnings_soon and iv_rank > 0.60:
        if rng.random() < 0.5:
            return "straddle", "BUY"
        elif rng.random() < 0.5:
            return "strangle", "BUY"
        else:
            return "iron_condor", "BUY"  # sell the IV

    # High IV — sell premium
    if iv_rank > 0.70:
        if trend == "bullish" and bollinger_position < 0.4:
            return "bull_call_spread", "BUY"
        elif trend == "bearish" and bollinger_position > 0.6:
            return "bear_put_spread", "BUY"
        elif abs(macd_hist) < 0.2:
            if rng.random() < 0.6:
                return "iron_condor", "BUY"
            else:
                return "strangle", "SELL"  # sell premium

    # Low IV — buy options
    if iv_rank < 0.30:
        if trend == "bullish" and rsi < 60:
            return "long_call", "BUY"
        elif trend == "bearish" and rsi > 40:
            return "long_put", "BUY"
        elif earnings_soon:
            return "straddle", "BUY"

    # Trending markets — directional plays
    if trend == "bullish":
        if rsi > 70:
            # Overbought — consider spreads for defined risk
            return "bull_call_spread", "BUY"
        elif rsi < 45 and bollinger_position < 0.2:
            return "long_call", "BUY"
        else:
            return "bull_call_spread", "BUY"

    if trend == "bearish":
        if rsi < 30:
            return "bear_put_spread", "BUY"
        elif rsi > 55 and bollinger_position > 0.8:
            return "long_put", "BUY"
        else:
            return "bear_put_spread", "BUY"

    # Neutral / ranging
    if abs(macd_hist) < 0.1 and iv_rank > 0.5:
        return "iron_condor", "BUY"

    # Squeeze setup
    if regime == "squeeze_setup":
        if rng.random() < 0.5:
            return "long_call", "BUY"
        else:
            return "long_put", "BUY"

    # Mean reversion
    if regime == "mean_reversion":
        if rsi < 25:
            return "long_call", "BUY"
        elif rsi > 75:
            return "long_put", "BUY"

    # Default: no trade
    return "none", "HOLD"


def _build_legs(
    rng: random.Random,
    strategy: str,
    spot: float,
    chain: dict[str, list[dict]],
    exp_date: str,
) -> list[dict[str, Any]]:
    """Build realistic option legs for a given strategy."""
    if strategy == "none":
        return []

    calls = chain.get("calls", [])
    puts = chain.get("puts", [])

    # Find ATM strike
    all_strikes = sorted(set([c["strike"] for c in calls] + [p["strike"] for p in puts]))
    if not all_strikes:
        return []

    atm_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
    atm_strike = all_strikes[atm_idx]

    # Offset for OTM strikes (1-3 strikes away)
    offset = rng.randint(1, 3)

    def find_contract(strike: float, opt_type: str):
        chain_list = calls if opt_type == "call" else puts
        for c in chain_list:
            if abs(c["strike"] - strike) < 0.01 and c["expiration"] == exp_date:
                return c
        # Fallback to closest
        closest = min(chain_list, key=lambda c: abs(c["strike"] - strike)) if chain_list else None
        return closest

    if strategy == "long_call":
        contract = find_contract(atm_strike, "call")
        if not contract:
            return []
        return [{"type": "call", "strike": contract["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"}]

    elif strategy == "long_put":
        contract = find_contract(atm_strike, "put")
        if not contract:
            return []
        return [{"type": "put", "strike": contract["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"}]

    elif strategy == "bull_call_spread":
        buy_strike = all_strikes[max(0, atm_idx - 1)]
        sell_strike = all_strikes[min(len(all_strikes) - 1, atm_idx + offset)]
        buy_c = find_contract(buy_strike, "call")
        sell_c = find_contract(sell_strike, "call")
        if not buy_c or not sell_c:
            return []
        return [
            {"type": "call", "strike": buy_c["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
            {"type": "call", "strike": sell_c["strike"], "expiration": exp_date, "quantity": 1, "side": "sell"},
        ]

    elif strategy == "bear_put_spread":
        buy_strike = all_strikes[min(len(all_strikes) - 1, atm_idx + 1)]
        sell_strike = all_strikes[max(0, atm_idx - offset)]
        buy_p = find_contract(buy_strike, "put")
        sell_p = find_contract(sell_strike, "put")
        if not buy_p or not sell_p:
            return []
        return [
            {"type": "put", "strike": buy_p["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
            {"type": "put", "strike": sell_p["strike"], "expiration": exp_date, "quantity": 1, "side": "sell"},
        ]

    elif strategy == "iron_condor":
        put_sell_strike = all_strikes[max(0, atm_idx - offset)]
        put_buy_strike = all_strikes[max(0, atm_idx - offset - 2)]
        call_sell_strike = all_strikes[min(len(all_strikes) - 1, atm_idx + offset)]
        call_buy_strike = all_strikes[min(len(all_strikes) - 1, atm_idx + offset + 2)]

        ps = find_contract(put_sell_strike, "put")
        pb = find_contract(put_buy_strike, "put")
        cs = find_contract(call_sell_strike, "call")
        cb = find_contract(call_buy_strike, "call")

        if not all([ps, pb, cs, cb]):
            return []
        return [
            {"type": "put", "strike": pb["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
            {"type": "put", "strike": ps["strike"], "expiration": exp_date, "quantity": 1, "side": "sell"},
            {"type": "call", "strike": cs["strike"], "expiration": exp_date, "quantity": 1, "side": "sell"},
            {"type": "call", "strike": cb["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
        ]

    elif strategy == "straddle":
        call_c = find_contract(atm_strike, "call")
        put_c = find_contract(atm_strike, "put")
        if not call_c or not put_c:
            return []
        return [
            {"type": "call", "strike": call_c["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
            {"type": "put", "strike": put_c["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
        ]

    elif strategy == "strangle":
        otm_call = all_strikes[min(len(all_strikes) - 1, atm_idx + offset)]
        otm_put = all_strikes[max(0, atm_idx - offset)]
        cc = find_contract(otm_call, "call")
        pc = find_contract(otm_put, "put")
        if not cc or not pc:
            return []
        return [
            {"type": "call", "strike": cc["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
            {"type": "put", "strike": pc["strike"], "expiration": exp_date, "quantity": 1, "side": "buy"},
        ]

    elif strategy == "calendar_spread":
        near_exp = exp_date
        far_exp = (datetime.strptime(exp_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        near_c = find_contract(atm_strike, "call")
        far_c = find_contract(atm_strike, "call")
        if not near_c or not far_c:
            return []
        return [
            {"type": "call", "strike": near_c["strike"], "expiration": near_exp, "quantity": 1, "side": "sell"},
            {"type": "call", "strike": far_c["strike"], "expiration": far_exp, "quantity": 1, "side": "buy"},
        ]

    return []


def _build_reasoning(
    rng: random.Random,
    strategy: str,
    regime: str,
    rsi: float,
    iv_rank: float,
    trend: str,
    earnings_soon: bool,
    symbol: str,
) -> str:
    """Generate detailed, realistic reasoning for the trade decision."""
    parts = []

    # IV analysis
    if iv_rank > 0.70:
        parts.append(f"IV rank at {iv_rank:.0%} is elevated, favoring premium selling strategies")
    elif iv_rank < 0.30:
        parts.append(f"IV rank at {iv_rank:.0%} is depressed, making long options relatively cheap")
    else:
        parts.append(f"IV rank at {iv_rank:.0%} is moderate, allowing for balanced approach")

    # Technical analysis
    if rsi > 70:
        parts.append(f"RSI {rsi:.0f} indicates overbought conditions — cautious on directional longs")
    elif rsi < 30:
        parts.append(f"RSI {rsi:.0f} signals oversold — potential mean reversion opportunity")
    elif rsi > 55:
        parts.append(f"RSI {rsi:.0f} shows mild bullish momentum")
    elif rsi < 45:
        parts.append(f"RSI {rsi:.0f} suggests mild bearish pressure")
    else:
        parts.append(f"RSI {rsi:.0f} is neutral — no strong directional signal")

    # Earnings context
    if earnings_soon:
        parts.append("Upcoming earnings adds event risk and elevated IV — consider event-driven strategies")

    # Strategy justification
    strategy_rationale = {
        "long_call": f"Buying calls on {symbol} to capture upside with defined risk",
        "long_put": f"Buying puts on {symbol} for downside protection or directional bearish bet",
        "bull_call_spread": f"Bull call spread limits cost while capturing upside momentum in {symbol}",
        "bear_put_spread": f"Bear put spread provides defined-risk bearish exposure on {symbol}",
        "iron_condor": f"Iron condor profits from range-bound price action and time decay in {symbol}",
        "straddle": f"Straddle positions for potential large move in {symbol}, regardless of direction",
        "strangle": f"Strangle provides cheaper alternative to straddle for {symbol} volatility play",
        "calendar_spread": f"Calendar spread profits from time decay differential and potential IV increase",
    }
    parts.append(strategy_rationale.get(strategy, f"Selected {strategy} for risk-defined exposure"))

    # Confidence note
    if regime in ("squeeze_setup", "breakout", "breakdown"):
        parts.append("Technical setup suggests high-probability directional move")

    return ". ".join(parts) + "."


def _generate_single_example(
    rng: random.Random,
    sample_id: int,
) -> dict[str, str]:
    """Generate one complete training example."""
    # Pick regime and ticker
    regime_name = rng.choice(list(MARKET_REGIMES.keys()))
    regime = MARKET_REGIMES[regime_name]
    symbol = rng.choice(ALL_TICKERS)

    # Generate price
    price_ranges = {
        "SPY": (450, 600), "QQQ": (380, 520), "IWM": (180, 260), "DIA": (340, 440),
    }
    if symbol in price_ranges:
        low, high = price_ranges[symbol]
        spot = round(rng.uniform(low, high), 2)
    else:
        spot = round(rng.uniform(30, 600), 2)

    # Generate market indicators from regime
    rsi = round(_rng_range(rng, *regime["rsi"]), 2)
    iv_rank = round(_rng_range(rng, *regime["iv_rank"]), 4)
    iv_percentile = round(min(1, max(0, iv_rank + rng.uniform(-0.15, 0.15))), 4)
    macd_hist = round(_rng_range(rng, *regime["macd_hist"]), 4)
    trend = regime["trend"]

    # IV and vol
    base_iv = rng.uniform(0.12, 0.65)
    current_iv = round(base_iv * (1 + iv_rank * 0.5), 4)
    hist_vol = round(current_iv * rng.uniform(0.7, 1.1), 4)

    # Earnings
    earnings_soon = regime_name == "pre_earnings" or (rng.random() < 0.15)
    earnings_date = ""
    if earnings_soon:
        days_until = rng.randint(1, 14)
        earnings_date = (datetime.now(timezone.utc) + timedelta(days=days_until)).strftime("%Y-%m-%d")

    # Bollinger position (0 = lower band, 1 = upper band)
    if trend == "bullish":
        bollinger_pos = round(_rng_range(rng, 0.4, 0.9), 2)
    elif trend == "bearish":
        bollinger_pos = round(_rng_range(rng, 0.1, 0.6), 2)
    else:
        bollinger_pos = round(_rng_range(rng, 0.2, 0.8), 2)

    # DTE
    dte = rng.randint(7, 45)

    # Generate option chain
    chain = _generate_option_chain(rng, spot, (dte, dte + 21), current_iv)

    # News
    news_category = "bullish" if trend == "bullish" else "bearish" if trend == "bearish" else "neutral"
    if earnings_soon:
        news_category = rng.choice(["earnings", "iv_event", news_category])
    elif iv_rank > 0.75:
        news_category = rng.choice(["iv_event", news_category])

    news_templates = NEWS_TEMPLATES.get(news_category, NEWS_TEMPLATES["neutral"])
    news_items = []
    for _ in range(rng.randint(1, 3)):
        template = rng.choice(news_templates)
        headline = template.format(
            ticker=symbol,
            partner=rng.choice(PARTNERS),
            bank=rng.choice(BANKS),
            move_pct=round(rng.uniform(-10, 10), 1),
            imp_pct=round(rng.uniform(3, 12), 1),
        )
        news_items.append({"title": headline})

    # Select strategy
    strategy, action = _select_strategy(
        rng, regime_name, rsi, iv_rank, trend, earnings_soon, iv_percentile, bollinger_pos, macd_hist
    )

    # Build legs
    exp_date = chain["calls"][0]["expiration"] if chain["calls"] else (
        datetime.now(timezone.utc) + timedelta(days=dte)
    ).strftime("%Y-%m-%d")
    legs = _build_legs(rng, strategy, spot, chain, exp_date)

    # Build context
    context = {
        "symbol": symbol,
        "underlying": {
            "price": spot,
            "rsi_14": rsi,
            "macd": {
                "macd": round(macd_hist + rng.uniform(-0.3, 0.3), 4),
                "signal": round(macd_hist * 0.7 + rng.uniform(-0.2, 0.2), 4),
                "histogram": macd_hist,
            },
            "bollinger": {
                "upper": round(spot * (1 + 0.02 * (1 - bollinger_pos)), 2),
                "middle": round(spot, 2),
                "lower": round(spot * (1 - 0.02 * bollinger_pos), 2),
            },
            "earnings_date": earnings_date,
            "news": news_items,
        },
        "options_chain": chain,
        "iv_metrics": {
            "current_iv": current_iv,
            "iv_rank": iv_rank,
            "iv_percentile": iv_percentile,
            "historical_vol": hist_vol,
        },
    }

    # Build response
    if action == "HOLD":
        confidence = round(_rng_range(rng, 0.2, 0.55), 2)
        reasoning = _build_reasoning(rng, "none", regime_name, rsi, iv_rank, trend, earnings_soon, symbol)
    else:
        confidence = round(_rng_range(rng, 0.65, 0.98), 2)
        reasoning = _build_reasoning(rng, strategy, regime_name, rsi, iv_rank, trend, earnings_soon, symbol)

    response = json.dumps({
        "action": action,
        "strategy": strategy,
        "underlying": symbol,
        "legs": legs,
        "confidence": confidence,
        "reasoning": reasoning,
    })

    return {
        "instruction": "Given this options chain and market context, what is the best options strategy? Output a JSON trade decision.",
        "input": json.dumps(context, indent=2),
        "output": response,
    }


def _generate_diverse_instructions() -> list[str]:
    """Generate varied instruction phrasings for data augmentation."""
    return [
        "Given this options chain and market context, what is the best options strategy? Output a JSON trade decision.",
        "Analyze the following market data and options chain. Recommend an options trade with rationale. Output JSON.",
        "You are an options trading expert. Based on this market context, what position should be entered? Return JSON.",
        "Review this options chain and technical indicators. What is the optimal options strategy right now? JSON output.",
        "Given current market conditions and this options chain, generate a trade recommendation. Output valid JSON.",
        "Evaluate the Greeks, IV rank, and price action below. What options strategy maximizes risk-adjusted return? JSON.",
        "Based on the provided options chain and market data, construct an options trade. Return structured JSON.",
        "Analyze IV rank, technical indicators, and options pricing. What trade should be placed? Output JSON format.",
        "Given this market snapshot with options Greeks and price data, recommend a position. JSON response required.",
        "What is the highest probability options trade given this market context and chain data? Return JSON.",
    ]


def build_dataset(output_dir: str = "./training_data", num_samples: int = 160_000) -> None:
    """
    Build the full training dataset and save as JSONL.
    Target: 160K examples for ~$40 training cost on Lightning.ai A10G ($0.71/hr).
    """
    os.makedirs(output_dir, exist_ok=True)

    rng = random.Random(42)
    np_rng = np.random.RandomState(42)

    instructions = _generate_diverse_instructions()

    logger.info("Generating %d training examples across %d market regimes...",
                num_samples, len(MARKET_REGIMES))

    all_examples: list[dict[str, str]] = []

    # Ensure balanced distribution across regimes
    samples_per_regime = num_samples // len(MARKET_REGIMES)
    extra_samples = num_samples - samples_per_regime * len(MARKET_REGIMES)

    for regime_idx, regime_name in enumerate(MARKET_REGIMES.keys()):
        regime_count = samples_per_regime + (1 if regime_idx < extra_samples else 0)
        logger.info("Generating %d examples for regime: %s", regime_count, regime_name)

        for i in range(regime_count):
            # Use varied instructions for data augmentation
            example = _generate_single_example(rng, i)
            example["instruction"] = rng.choice(instructions)
            all_examples.append(example)

            if (i + 1) % 10000 == 0:
                logger.info("  Progress: %d / %d", i + 1, regime_count)

    # Shuffle
    logger.info("Shuffling %d examples...", len(all_examples))
    rng.shuffle(all_examples)

    # Split 95/5
    split_idx = int(len(all_examples) * 0.95)
    train_data = all_examples[:split_idx]
    test_data = all_examples[split_idx:]

    # Save as JSONL
    train_path = Path(output_dir) / "train.jsonl"
    test_path = Path(output_dir) / "test.jsonl"

    logger.info("Writing %d train examples to %s", len(train_data), train_path)
    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    logger.info("Writing %d test examples to %s", len(test_data), test_path)
    with open(test_path, "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")

    # Print stats
    strategy_counts = {}
    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    for ex in all_examples:
        resp = json.loads(ex["output"])
        s = resp.get("strategy", "none")
        a = resp.get("action", "HOLD")
        strategy_counts[s] = strategy_counts.get(s, 0) + 1
        action_counts[a] = action_counts.get(a, 0) + 1

    print(f"\n{'='*60}")
    print(f"Dataset Built: {len(train_data):,} train / {len(test_data):,} test")
    print(f"{'='*60}")
    print(f"\nStrategy Distribution:")
    for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        print(f"  {s:25s}: {c:6d} ({c/len(all_examples)*100:.1f}%)")
    print(f"\nAction Distribution:")
    for a, c in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {a:25s}: {c:6d} ({c/len(all_examples)*100:.1f}%)")
    print(f"\nFiles:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    build_dataset()
