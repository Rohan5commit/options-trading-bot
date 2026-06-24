"""
Dataset builder for LoRA fine-tuning.
Constructs instruction-following training data from options chain data,
financial news, IV history, and synthetic examples.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are an expert options trader. Given market context, output a JSON trade decision."

STRATEGIES = [
    "long_call", "long_put", "bull_call_spread", "bear_put_spread",
    "iron_condor", "straddle", "strangle", "calendar_spread",
]


def _generate_synthetic_examples(num_samples: int = 5000) -> list[dict[str, str]]:
    """
    Generate synthetic instruction-response pairs for training.
    Each example pairs a market context with the optimal strategy.
    """
    import random

    examples = []
    rng = random.Random(42)

    for _ in range(num_samples):
        # Generate random market context
        rsi = rng.uniform(20, 80)
        iv_rank = rng.uniform(0, 1)
        dte = rng.randint(7, 45)
        price = rng.uniform(50, 500)
        earnings_soon = rng.choice([True, False])
        trend = rng.choice(["bullish", "bearish", "neutral"])

        # Determine the appropriate strategy based on context
        if iv_rank > 0.70:
            if trend == "bullish":
                strategy = "bull_call_spread"
                action = "BUY"
            elif trend == "bearish":
                strategy = "bear_put_spread"
                action = "BUY"
            else:
                strategy = "iron_condor"
                action = "BUY"
        elif iv_rank < 0.30 and earnings_soon:
            strategy = "straddle"
            action = "BUY"
        elif trend == "bullish" and rsi < 65:
            strategy = "long_call"
            action = "BUY"
        elif trend == "bearish" and rsi > 35:
            strategy = "long_put"
            action = "BUY"
        else:
            strategy = "none"
            action = "HOLD"

        # Build context JSON
        context = {
            "symbol": rng.choice(["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]),
            "underlying": {
                "price": round(price, 2),
                "rsi_14": round(rsi, 2),
                "macd": {"macd": round(rng.uniform(-2, 2), 2), "signal": round(rng.uniform(-2, 2), 2)},
                "bollinger": {
                    "upper": round(price * 1.02, 2),
                    "middle": round(price, 2),
                    "lower": round(price * 0.98, 2),
                },
                "earnings_date": "2025-02-01" if earnings_soon else "",
                "news": [{"title": f"Market update for {'bullish' if trend == 'bullish' else 'bearish' if trend == 'bearish' else 'neutral'} outlook"}],
            },
            "options_chain": {
                "calls": [
                    {
                        "symbol": f"TEST{dte}C{int(price*1000):08d}",
                        "type": "call",
                        "strike": price,
                        "expiration": "2025-02-21",
                        "dte": dte,
                        "open_interest": rng.randint(100, 5000),
                        "bid": round(rng.uniform(0.5, 5), 2),
                        "ask": round(rng.uniform(5, 10), 2),
                        "delta": round(rng.uniform(0.3, 0.7), 2),
                        "gamma": round(rng.uniform(0.01, 0.05), 3),
                        "theta": round(rng.uniform(-0.1, -0.01), 3),
                        "vega": round(rng.uniform(0.1, 0.3), 3),
                        "implied_volatility": round(rng.uniform(0.15, 0.50), 4),
                    }
                ],
                "puts": [],
            },
            "iv_metrics": {
                "current_iv": round(rng.uniform(0.15, 0.50), 4),
                "iv_rank": round(iv_rank, 4),
                "iv_percentile": round(rng.uniform(0, 1), 4),
                "historical_vol": round(rng.uniform(0.15, 0.40), 4),
            },
        }

        # Build response
        if action == "HOLD":
            response = json.dumps({
                "action": "HOLD",
                "strategy": "none",
                "underlying": context["symbol"],
                "legs": [],
                "confidence": round(rng.uniform(0.3, 0.6), 2),
                "reasoning": f"IV rank is {iv_rank:.0%}, RSI is {rsi:.0f}. No clear edge for {context['symbol']} today.",
            })
        else:
            strike_offset = rng.uniform(0, 0.05) * price
            if strategy in ("long_call", "bull_call_spread"):
                strike = round(price + strike_offset, 1)
                leg2_strike = round(price + strike_offset * 2, 1)
                legs = [{"type": "call", "strike": strike, "expiration": "2025-02-21", "quantity": 1, "side": "buy"}]
                if strategy == "bull_call_spread":
                    legs.append({"type": "call", "strike": leg2_strike, "expiration": "2025-02-21", "quantity": 1, "side": "sell"})
            elif strategy in ("long_put", "bear_put_spread"):
                strike = round(price - strike_offset, 1)
                leg2_strike = round(price - strike_offset * 2, 1)
                legs = [{"type": "put", "strike": strike, "expiration": "2025-02-21", "quantity": 1, "side": "buy"}]
                if strategy == "bear_put_spread":
                    legs.append({"type": "put", "strike": leg2_strike, "expiration": "2025-02-21", "quantity": 1, "side": "sell"})
            elif strategy == "iron_condor":
                put_sell = round(price - strike_offset, 1)
                put_buy = round(price - strike_offset * 2, 1)
                call_sell = round(price + strike_offset, 1)
                call_buy = round(price + strike_offset * 2, 1)
                legs = [
                    {"type": "put", "strike": put_buy, "expiration": "2025-02-21", "quantity": 1, "side": "buy"},
                    {"type": "put", "strike": put_sell, "expiration": "2025-02-21", "quantity": 1, "side": "sell"},
                    {"type": "call", "strike": call_sell, "expiration": "2025-02-21", "quantity": 1, "side": "sell"},
                    {"type": "call", "strike": call_buy, "expiration": "2025-02-21", "quantity": 1, "side": "buy"},
                ]
            elif strategy == "straddle":
                legs = [
                    {"type": "call", "strike": price, "expiration": "2025-02-21", "quantity": 1, "side": "buy"},
                    {"type": "put", "strike": price, "expiration": "2025-02-21", "quantity": 1, "side": "buy"},
                ]
            else:
                legs = [{"type": "call", "strike": price, "expiration": "2025-02-21", "quantity": 1, "side": "buy"}]

            response = json.dumps({
                "action": action,
                "strategy": strategy,
                "underlying": context["symbol"],
                "legs": legs,
                "confidence": round(rng.uniform(0.70, 0.95), 2),
                "reasoning": (
                    f"IV rank at {iv_rank:.0%} {'favors premium selling' if iv_rank > 0.60 else 'supports directional plays'}. "
                    f"RSI {rsi:.0f} suggests {'oversold bounce potential' if rsi < 35 else 'overbought risk' if rsi > 65 else 'neutral momentum'}. "
                    f"{'Earnings soon increases IV' if earnings_soon else 'No upcoming earnings catalyst'}. "
                    f"Recommended {strategy} for risk-defined exposure."
                ),
            })

        examples.append({
            "instruction": "Given this options chain and market context, what is the best options strategy?",
            "input": json.dumps(context, indent=2),
            "output": response,
        })

    return examples


def _fetch_alpaca_historical_options() -> list[dict[str, str]]:
    """
    Fetch historical options data from Alpaca to build training samples.
    Returns instruction-response pairs.
    """
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    if not api_key or not secret_key:
        logger.warning("Alpaca credentials not set — skipping historical options fetch")
        return []

    # This would fetch real historical options data in production
    # For now, return empty — the synthetic data covers this
    return []


def build_dataset(output_dir: str = "./training_data") -> None:
    """
    Build the full training dataset and save as JSONL.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_examples: list[dict[str, str]] = []

    # 1. Synthetic examples (primary training data)
    logger.info("Generating synthetic training examples...")
    synthetic = _generate_synthetic_examples(5000)
    all_examples.extend(synthetic)
    logger.info("Generated %d synthetic examples", len(synthetic))

    # 2. Historical Alpaca data (supplementary)
    logger.info("Fetching historical options data...")
    historical = _fetch_alpaca_historical_options()
    all_examples.extend(historical)
    logger.info("Added %d historical examples", len(historical))

    # Split into train/test (90/10)
    import random
    random.seed(42)
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)
    train_data = all_examples[:split_idx]
    test_data = all_examples[split_idx:]

    # Save as JSONL
    train_path = Path(output_dir) / "train.jsonl"
    test_path = Path(output_dir) / "test.jsonl"

    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(test_path, "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")

    logger.info("Dataset saved: %d train, %d test", len(train_data), len(test_data))
    print(f"Dataset built: {len(train_data)} train / {len(test_data)} test samples")
    print(f"Train: {train_path}")
    print(f"Test: {test_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_dataset()
