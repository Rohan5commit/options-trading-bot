"""
Data pipeline: fetches options chains, Greeks, price action, IV, earnings, and news.
Combines Alpaca API, yfinance, and Polygon.io free tier.
"""
import functools
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# ── Retry decorator ────────────────────────────────────────────────────────────


def _retry(max_retries: int = 3, backoff_factor: float = 1.0):
    """Exponential-backoff retry decorator for network calls."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    wait = backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        func.__name__, attempt, max_retries, exc, wait,
                    )
                    time.sleep(wait)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ── Alpaca helpers ─────────────────────────────────────────────────────────────


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
    }


@_retry()
def _fetch_option_chain_alpaca(underlying: str) -> list[dict[str, Any]]:
    """
    Fetch the option chain snapshot for an underlying via Alpaca data API.
    Returns list of contract snapshots with Greeks.
    """
    url = f"https://data.alpaca.markets/v1beta1/options/snapshots/{underlying}"
    params: dict[str, Any] = {
        "feed": "indicative",
        "limit": 1000,
    }
    resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    snapshots = data.get("snapshots", {})
    results = []
    for symbol, snap in snapshots.items():
        greeks = snap.get("greeks", {})
        quote = snap.get("latest_quote", {})
        trade = snap.get("latest_trade", {})
        results.append({
            "symbol": symbol,
            "bid": quote.get("bp", 0),
            "ask": quote.get("ap", 0),
            "bid_size": quote.get("bs", 0),
            "ask_size": quote.get("as", 0),
            "last_trade": trade.get("p", 0),
            "delta": greeks.get("delta", 0),
            "gamma": greeks.get("gamma", 0),
            "theta": greeks.get("theta", 0),
            "vega": greeks.get("vega", 0),
            "rho": greeks.get("rho", 0),
            "implied_volatility": greeks.get("iv", 0),
        })
    return results


@_retry()
def _fetch_option_contracts_alpaca(underlying: str) -> list[dict[str, Any]]:
    """Fetch active option contracts from Alpaca Trading API."""
    url = "https://paper-api.alpaca.markets/v2/options/contracts"
    params = {
        "underlying_symbols": underlying,
        "status": "active",
        "limit": 100,
    }
    resp = requests.get(url, headers=_alpaca_headers(), params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("option_contracts", [])


@_retry()
def _fetch_account_alpaca() -> dict[str, Any]:
    """Fetch account details from Alpaca."""
    url = "https://paper-api.alpaca.markets/v2/account"
    resp = requests.get(url, headers=_alpaca_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


@_retry()
def _fetch_stock_quote_alpaca(symbol: str) -> dict[str, Any]:
    """Fetch latest stock quote from Alpaca."""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
    resp = requests.get(url, headers=_alpaca_headers(), timeout=30)
    resp.raise_for_status()
    quote = resp.json().get("quote", {})
    return {
        "bid": quote.get("bp", 0),
        "ask": quote.get("ap", 0),
        "mid": (quote.get("bp", 0) + quote.get("ap", 0)) / 2,
    }


# ── Technical indicators (via yfinance) ───────────────────────────────────────


def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI from a price series."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    if pd.isna(val) or not math.isfinite(val):
        return 50.0
    return float(val)


def _compute_macd(prices: pd.Series) -> dict[str, float]:
    """Compute MACD, signal, and histogram."""
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": float(macd_line.iloc[-1]),
        "signal": float(signal_line.iloc[-1]),
        "histogram": float(histogram.iloc[-1]),
    }


def _compute_bollinger(prices: pd.Series, period: int = 20, num_std: float = 2.0) -> dict[str, float]:
    """Compute Bollinger Bands."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return {
        "upper": float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else 0,
        "middle": float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else 0,
        "lower": float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else 0,
    }


# ── IV metrics (via Polygon.io free tier) ──────────────────────────────────────


@_retry()
def _fetch_iv_history_polygon(underlying: str, days: int = 30) -> list[dict[str, Any]]:
    """
    Fetch historical implied volatility from Polygon.io free tier.
    Falls back to an empty list if the API key is missing or the call fails.
    """
    if not config.POLYGON_API_KEY:
        return []

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    url = f"https://api.polygon.io/v3/snapshot/options/{underlying}"
    params = {
        "limit": 50,
        "order": "desc",
        "sort": "implied_volatility",
        "apiKey": config.POLYGON_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        ivs = [
            r.get("implied_volatility", 0)
            for r in results
            if r.get("implied_volatility")
        ]
        return [{"date": end.strftime("%Y-%m-%d"), "iv": v} for v in ivs]
    except Exception as exc:
        logger.warning("Polygon IV fetch failed for %s: %s", underlying, exc)
        return []


def _compute_iv_metrics(chain: list[dict[str, Any]]) -> dict[str, float]:
    """
    Compute current IV from the option chain.
    IV rank and percentile are not computed cross-sectionally (meaningless).
    Instead, report current IV and the IV range across strikes.
    """
    ivs = [c.get("implied_volatility", 0) for c in chain if c.get("implied_volatility", 0) > 0]
    if not ivs:
        return {"current_iv": 0, "iv_rank": 0, "iv_percentile": 0}

    current_iv = float(np.mean(ivs))
    min_iv = float(np.min(ivs))
    max_iv = float(np.max(ivs))

    # Report IV as percentile within the current range (not historical)
    iv_range = max_iv - min_iv
    iv_rank = (current_iv - min_iv) / iv_range if iv_range > 0 else 0.5

    return {
        "current_iv": round(current_iv, 4),
        "iv_rank": round(min(max(iv_rank, 0), 1), 4),
        "iv_percentile": round(min(max(iv_rank, 0), 1), 4),
    }


# ── News and earnings (via yfinance) ───────────────────────────────────────────


def _fetch_news_yfinance(symbol: str) -> list[dict[str, str]]:
    """Fetch recent news headlines for a symbol via yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        return [
            {"title": n.get("title", ""), "link": n.get("link", "")}
            for n in news[:5]
        ]
    except Exception as exc:
        logger.warning("News fetch failed for %s: %s", symbol, exc)
        return []


def _fetch_earnings_yfinance(symbol: str) -> dict[str, str]:
    """Fetch upcoming earnings date for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None and hasattr(cal, 'iloc') and len(cal) > 0:
            earnings_date = str(cal.iloc[0].get("Earnings Date", ""))
            return {"earnings_date": earnings_date}
        elif cal is not None and isinstance(cal, dict) and "Earnings Date" in cal:
            earnings_date = str(cal["Earnings Date"])
            return {"earnings_date": earnings_date}
    except Exception as exc:
        logger.warning("Earnings fetch failed for %s: %s", symbol, exc)
    return {"earnings_date": ""}


# ── Build full market context ──────────────────────────────────────────────────


def _filter_contracts(contracts: list[dict], underlying_price: float) -> list[dict]:
    """Filter to the most relevant contracts: ATM ± 10 strikes, limit to MAX_CONTRACTS."""
    today = datetime.now(timezone.utc).date()
    filtered = []
    for c in contracts:
        exp_str = c.get("expiration_date", "")
        if not exp_str:
            continue
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte < config.MIN_DTE or dte > config.MAX_DTE:
            continue
        strike = float(c.get("strike_price", 0))
        if underlying_price > 0:
            c["_strike_dist"] = abs(strike - underlying_price)
        else:
            c["_strike_dist"] = 0
        c["_dte"] = dte
        filtered.append(c)

    filtered.sort(key=lambda x: (x["_dte"], x["_strike_dist"]))

    max_contracts = config.MAX_CONTRACTS_PER_SYMBOL
    return filtered[:max_contracts]


def _get_contract_details(contracts: list[dict], chain: list[dict]) -> dict[str, list[dict]]:
    """
    Merge contract metadata (OI, expiration, type, strike) with chain snapshots (Greeks, quotes).
    Filters to contracts within DTE range.
    """
    today = datetime.now(timezone.utc).date()
    calls = []
    puts = []

    for contract in contracts:
        exp_str = contract.get("expiration_date", "")
        if not exp_str:
            continue
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        dte = (exp_date - today).days
        if dte < config.MIN_DTE or dte > config.MAX_DTE:
            continue

        sym = contract.get("symbol", "")
        snap = next((s for s in chain if s["symbol"] == sym), None)
        if not snap:
            continue

        bid = snap.get("bid", 0)
        ask = snap.get("ask", 0)
        mid = (bid + ask) / 2 if (bid + ask) > 0 else 0
        spread_pct = ((ask - bid) / mid) if mid > 0 else 999

        entry = {
            "symbol": sym,
            "type": contract.get("type", ""),
            "strike": float(contract.get("strike_price", 0)),
            "expiration": exp_str,
            "dte": dte,
            "open_interest": int(contract.get("open_interest") or 0),
            "bid": bid,
            "ask": ask,
            "mid": round(mid, 2),
            "spread_pct": round(spread_pct, 4),
            "last_trade": snap.get("last_trade", 0),
            "delta": snap.get("delta", 0),
            "gamma": snap.get("gamma", 0),
            "theta": snap.get("theta", 0),
            "vega": snap.get("vega", 0),
            "rho": snap.get("rho", 0),
            "implied_volatility": snap.get("implied_volatility", 0),
        }

        if entry["type"] == "call":
            calls.append(entry)
        else:
            puts.append(entry)

    return {"calls": sorted(calls, key=lambda x: x["strike"]), "puts": sorted(puts, key=lambda x: x["strike"])}


def build_context_for_symbol(symbol: str) -> dict[str, Any] | None:
    """
    Build complete market context for a single symbol.
    Returns None if price data is unavailable (fix #8).
    """
    logger.info("Building context for %s", symbol)

    # Stock price and technical indicators (needed first for ATM filtering)
    latest_price = 0
    avg_volume = 0
    rsi = 50.0
    macd = {"macd": 0, "signal": 0, "histogram": 0}
    bollinger = {"upper": 0, "middle": 0, "lower": 0}
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo", interval="1d")
        if hist.empty or len(hist) < 5:
            raise ValueError("Insufficient price history")
        close = hist["Close"]
        volume = hist["Volume"]
        latest_price = float(close.iloc[-1])
        avg_volume = float(volume.mean())

        rsi = _compute_rsi(close)
        macd = _compute_macd(close)
        bollinger = _compute_bollinger(close)
    except Exception as exc:
        logger.warning("Price data failed for %s: %s", symbol, exc)

    # Skip symbols with no price data (fix #8)
    if latest_price <= 0:
        logger.warning("No valid price for %s, skipping", symbol)
        return None

    # Fetch option chain from Alpaca, filter to ATM ± 10 strikes
    contracts = _fetch_option_contracts_alpaca(symbol)
    contracts = _filter_contracts(contracts, latest_price)
    chain = _fetch_option_chain_alpaca(symbol)
    options_chain = _get_contract_details(contracts, chain)

    # IV metrics
    iv_metrics = _compute_iv_metrics(chain)

    # News and earnings
    news = _fetch_news_yfinance(symbol)
    earnings = _fetch_earnings_yfinance(symbol)

    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "underlying": {
            "price": latest_price,
            "avg_volume_20d": int(avg_volume),
            "rsi_14": round(rsi, 2),
            "macd": macd,
            "bollinger": bollinger,
            "earnings_date": earnings.get("earnings_date", ""),
            "news": news,
        },
        "options_chain": options_chain,
        "iv_metrics": iv_metrics,
    }


def build_full_market_context() -> list[dict[str, Any]]:
    """
    Build market context for every symbol in the watchlist.
    Returns a list of per-symbol context dicts.
    """
    contexts: list[dict[str, Any]] = []
    for symbol in config.WATCHLIST:
        try:
            ctx = build_context_for_symbol(symbol)
            if ctx is None:
                continue
            # Skip symbols with no options data
            total_contracts = len(ctx["options_chain"].get("calls", [])) + len(ctx["options_chain"].get("puts", []))
            if total_contracts == 0:
                logger.info("No options contracts found for %s, skipping", symbol)
                continue
            contexts.append(ctx)
        except Exception as exc:
            logger.error("Failed to build context for %s: %s", symbol, exc)
    logger.info("Built market context for %d symbols", len(contexts))
    return contexts


def get_account_equity() -> float:
    """Fetch current account equity from Alpaca."""
    try:
        account = _fetch_account_alpaca()
        equity = float(account.get("equity", 0))
        logger.info("Account equity: $%.2f", equity)
        return equity
    except Exception as exc:
        logger.error("Failed to fetch account equity: %s", exc)
        return 0.0
