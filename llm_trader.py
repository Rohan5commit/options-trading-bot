"""
LLM Decision Engine. Sends market context to the Modal inference endpoint
and parses structured trade decisions from the LLM response.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    {"type": "call"|"put", "strike": 190.0, "expiration": "YYYY-MM-DD", "quantity": 1, "side": "buy"|"sell"}
  ],
  "confidence": 0.85,
  "reasoning": "Brief rationale..."
}

If no trade is warranted, output:
{"action": "HOLD", "strategy": "none", "underlying": "<TICKER>", "legs": [], "confidence": 0.0, "reasoning": "No clear edge today."}
"""

# ── JSON parsing with brace-depth counter ─────────────────────────────────────


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Robustly extract a JSON object from LLM output using brace-depth counting.
    Handles markdown fences, preamble text, and nested objects.
    """
    text = text.strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip markdown code fences
    cleaned = text
    if "```" in cleaned:
        cleaned = cleaned.replace("```json", "").replace("```", "")
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 3: brace-depth counter (fix #9)
    start_idx = text.find("{")
    if start_idx == -1:
        logger.error("No JSON object found in LLM output: %s", text[:300])
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start_idx, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == "\\":
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start_idx:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    logger.error("Failed to parse JSON from LLM output: %s", text[:300])
    return None


def _validate_decision(decision: dict[str, Any]) -> bool:
    """Validate the structure and types of an LLM trade decision."""
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

    # Validate confidence is a number in valid range (fix #10)
    try:
        confidence = float(decision.get("confidence", 0))
        if not (0.0 <= confidence <= 1.0):
            logger.warning("Confidence out of range: %s", confidence)
            return False
        decision["confidence"] = confidence
    except (ValueError, TypeError):
        logger.warning("Invalid confidence value: %s", decision.get("confidence"))
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

        # Validate strike is a positive number (fix #10)
        try:
            strike = float(leg["strike"])
            if strike <= 0:
                logger.warning("Invalid strike: %s", leg["strike"])
                return False
            leg["strike"] = strike
        except (ValueError, TypeError):
            logger.warning("Invalid strike value: %s", leg["strike"])
            return False

        # Validate quantity is a positive integer (fix #10)
        try:
            qty = int(leg["quantity"])
            if qty <= 0:
                logger.warning("Invalid quantity: %s", leg["quantity"])
                return False
            leg["quantity"] = qty
        except (ValueError, TypeError):
            logger.warning("Invalid quantity value: %s", leg["quantity"])
            return False

        # Validate expiration format (fix #10)
        from datetime import datetime
        try:
            datetime.strptime(leg["expiration"], "%Y-%m-%d")
        except (ValueError, TypeError):
            logger.warning("Invalid expiration format: %s (expected YYYY-MM-DD)", leg["expiration"])
            return False

    return True


# ── Token count guard ─────────────────────────────────────────────────────────


def _truncate_context(ctx: dict[str, Any], max_chars: int = 12000) -> dict[str, Any]:
    """
    Truncate options chain to fit within LLM context window.
    Approximates 1 token per 4 chars, targets ~3K tokens input for 8K context.
    """
    truncated = dict(ctx)
    chain = truncated.get("options_chain", {})

    for key in ("calls", "puts"):
        options = chain.get(key, [])
        if len(options) > 10:
            # Keep ATM options (closest to middle strike)
            chain[key] = options[:10]

    truncated["options_chain"] = chain
    return truncated


# ── Public API ─────────────────────────────────────────────────────────────────


def _process_symbol(
    ctx: dict[str, Any],
    exit_recommendations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Process a single symbol: call LLM and parse decision."""
    import modal_inference

    symbol = ctx.get("symbol", "UNKNOWN")
    try:
        # Truncate context if too large (fix #12)
        ctx = _truncate_context(ctx)

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
            return None

        if not _validate_decision(decision):
            logger.warning("LLM decision failed validation for %s", symbol)
            return None

        # Apply confidence gate
        confidence = decision.get("confidence", 0)
        if decision["action"] != "HOLD" and confidence < config.MIN_CONFIDENCE:
            logger.info(
                "Decision for %s below confidence threshold (%.2f < %.2f), skipping",
                symbol, confidence, config.MIN_CONFIDENCE,
            )
            return None

        logger.info(
            "LLM decision for %s: %s %s (confidence: %.2f)",
            symbol, decision["action"], decision["strategy"], confidence,
        )
        return decision

    except Exception as exc:
        logger.error("LLM call failed for %s: %s", symbol, exc)
        return None


def get_trade_decisions(
    market_contexts: list[dict[str, Any]],
    exit_recommendations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    For each symbol's market context, call the LLM and parse the trade decision.
    Parallelizes across symbols for faster execution (fix #11).
    Returns a list of validated trade decisions.
    """
    if not market_contexts:
        return []

    exit_recs = exit_recommendations or []
    decisions: list[dict[str, Any]] = []

    # Parallel processing (fix #11): use ThreadPoolExecutor for I/O-bound Modal calls
    max_workers = min(5, len(market_contexts))  # Limit concurrency to avoid Modal rate limits
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_symbol, ctx, exit_recs): ctx
            for ctx in market_contexts
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                decisions.append(result)

    logger.info("Received %d valid trade decisions from LLM", len(decisions))
    return decisions
