"""
LLM Decision Engine. Sends market context to the Modal inference endpoint
and parses structured trade decisions from the LLM response.
"""
import json
import logging
import re
from typing import Any

import config

logger = logging.getLogger(__name__)

# ── System prompt (carefully engineered for structured JSON output) ────────────

SYSTEM_PROMPT = """\
You are an expert US equity options trader with deep knowledge of Greeks, \
implied volatility surfaces, and multi-leg strategies. Your sole objective \
is maximum risk-adjusted profit using paper trading accounts.

RULES:
1. Output ONLY valid JSON — no markdown fences, no explanation before or after.
2. Analyze: price action (RSI, MACD, Bollinger), IV rank/percentile, \
   earnings proximity, news sentiment, open interest, bid-ask spreads.
3. Strategies: long_call, long_put, bull_call_spread, bear_put_spread, \
   iron_condor, straddle, strangle, calendar_spread.
4. Only recommend a trade when you have a clear statistical edge.
   - High IV rank (>0.60) → favor premium selling (spreads, iron condors).
   - Strong trend (RSI >65 or <35, MACD divergence) → favor directional plays.
   - Low IV rank (<0.30) with catalyst → favor long options.
5. For multi-leg strategies, list ALL legs with exact strikes and expirations.
6. Assign a confidence score from 0.00 to 1.00.
7. Keep reasoning under 200 words.

OUTPUT FORMAT (JSON only):
{
  "action": "BUY" | "SELL" | "HOLD",
  "strategy": "<strategy_name>",
  "underlying": "<TICKER>",
  "legs": [
    {"type": "call"|"put", "strike": 190.0, "expiration": "2025-02-21", "quantity": 1, "side": "buy"|"sell"}
  ],
  "confidence": 0.85,
  "reasoning": "Brief rationale..."
}

If no trade is warranted, output:
{"action": "HOLD", "strategy": "none", "underlying": "<TICKER>", "legs": [], "confidence": 0.0, "reasoning": "No clear edge today."}
"""

# ── JSON parsing ───────────────────────────────────────────────────────────────


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Robustly extract a JSON object from LLM output.
    Handles cases where the model wraps JSON in markdown fences or adds preamble.
    """
    # Attempt 1: direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: find the first { ... } block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse JSON from LLM output: %s", text[:300])
    return None


def _validate_decision(decision: dict[str, Any]) -> bool:
    """Validate the structure of an LLM trade decision."""
    required = ["action", "strategy", "underlying", "legs", "confidence", "reasoning"]
    for field in required:
        if field not in decision:
            logger.warning("Missing required field: %s", field)
            return False

    if decision["action"] not in ("BUY", "SELL", "HOLD"):
        logger.warning("Invalid action: %s", decision["action"])
        return False

    if decision["action"] == "HOLD":
        return True

    if decision["strategy"] not in config.STRATEGIES:
        logger.warning("Invalid strategy: %s", decision["strategy"])
        return False

    if not isinstance(decision["legs"], list) or len(decision["legs"]) == 0:
        logger.warning("Legs must be a non-empty list for action %s", decision["action"])
        return False

    for leg in decision["legs"]:
        for field in ["type", "strike", "expiration", "quantity", "side"]:
            if field not in leg:
                logger.warning("Leg missing field: %s", field)
                return False
        if leg["type"] not in ("call", "put"):
            logger.warning("Invalid leg type: %s", leg["type"])
            return False
        if leg["side"] not in ("buy", "sell"):
            logger.warning("Invalid leg side: %s", leg["side"])
            return False

    return True


# ── Public API ─────────────────────────────────────────────────────────────────


def get_trade_decisions(
    market_contexts: list[dict[str, Any]],
    exit_recommendations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    For each symbol's market context, call the LLM and parse the trade decision.
    Returns a list of validated trade decisions (only those passing confidence gate).
    """
    import modal_inference

    decisions: list[dict[str, Any]] = []

    for ctx in market_contexts:
        symbol = ctx.get("symbol", "UNKNOWN")
        try:
            # Build the user prompt with full context
            user_prompt = f"Analyze this options chain and market data:\n{json.dumps(ctx, indent=2)}"

            # Add exit recommendations if available
            if exit_recommendations:
                open_pos = [
                    r for r in exit_recommendations
                    if r.get("underlying") == symbol
                ]
                if open_pos:
                    user_prompt += f"\n\nExisting open positions to evaluate for exit:\n{json.dumps(open_pos, indent=2)}"

            logger.info("Calling LLM for %s", symbol)
            raw_response = modal_inference.call_inference(user_prompt, SYSTEM_PROMPT)

            # Parse the response
            decision = _extract_json(raw_response)
            if decision is None:
                logger.warning("LLM returned unparseable output for %s, skipping", symbol)
                continue

            if not _validate_decision(decision):
                logger.warning("LLM decision failed validation for %s", symbol)
                continue

            # Apply confidence gate
            confidence = float(decision.get("confidence", 0))
            if decision["action"] != "HOLD" and confidence < config.MIN_CONFIDENCE:
                logger.info(
                    "Decision for %s below confidence threshold (%.2f < %.2f), skipping",
                    symbol, confidence, config.MIN_CONFIDENCE,
                )
                continue

            decisions.append(decision)
            logger.info(
                "LLM decision for %s: %s %s (confidence: %.2f)",
                symbol, decision["action"], decision["strategy"], confidence,
            )

        except Exception as exc:
            logger.error("LLM call failed for %s: %s", symbol, exc)
            # Fallback: skip trading for this symbol
            continue

    logger.info("Received %d valid trade decisions from LLM", len(decisions))
    return decisions
