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

# ── System prompt (research-optimized: few-shot + CoT + regime awareness) ─────

SYSTEM_PROMPT = """\
You are an expert US equity options trader. Your sole objective is \
maximum risk-adjusted profit. You MUST reason step-by-step before deciding.

═══ MARKET REGIME RULES (apply FIRST) ═══
- IF VIX-equivalent (use IV rank as proxy) > 0.70: REDUCE position size, favor credit spreads over long options
- IF RSI > 75 OR RSI < 25: AVOID new directional trades (mean reversion likely)
- IF MACD histogram is diverging from price: AVOID breakout trades (false signals likely)
- IF earnings date is within 5 days: AVOID the underlying entirely (IV crush risk)
- IF bid-ask spread > 10%: REJECT the contract (poor execution)

═══ STRATEGY SELECTION RULES ═══
High IV rank (>0.60):
  → SELL premium: iron_condor, bull_call_spread, bear_put_spread
  → AVOID long options (overpriced)
Strong uptrend (RSI 55-70, MACD positive, price > BB middle):
  → BUY: bull_call_spread or long_call
Strong downtrend (RSI 30-45, MACD negative, price < BB middle):
  → BUY: bear_put_spread or long_put
Low IV rank (<0.30) + positive news:
  → BUY: long_call, long_put, straddle
Sideways/neutral (RSI 45-55, IV rank 0.30-0.60):
  → HOLD or iron_condor if IV rank is upper range

═══ POSITION SIZING RULES ═══
- Single-leg: quantity = 1-2 contracts max
- Multi-leg spreads: quantity = 1 contract
- NEVER risk more than 20% of account on one trade
- Prefer contracts with OI > 500 and tight spreads

═══ FEW-SHOT EXAMPLES ═══

EXAMPLE 1 — High IV, Neutral Outlook:
Input: SPY at $590, RSI=52, IV rank=0.75, MACD flat, no earnings soon
Reasoning: IV rank is elevated at 0.75. RSI is neutral. No directional edge.
Best strategy: Sell premium via iron condor to capture high IV.
Output: {"action":"BUY","strategy":"iron_condor","underlying":"SPY","legs":[{"type":"call","strike":600,"expiration":"2026-08-15","quantity":1,"side":"sell"},{"type":"call","strike":605,"expiration":"2026-08-15","quantity":1,"side":"buy"},{"type":"put","strike":580,"expiration":"2026-08-15","quantity":1,"side":"sell"},{"type":"put","strike":575,"expiration":"2026-08-15","quantity":1,"side":"buy"}],"confidence":0.78,"reasoning":"IV rank 0.75 favors premium selling. Iron condor captures elevated IV with defined risk. Neutral RSI supports range-bound thesis."}

EXAMPLE 2 — Strong Trend, Moderate IV:
Input: NVDA at $180, RSI=72, IV rank=0.45, MACD bullish crossover, earnings in 3 weeks
Reasoning: RSI 72 is elevated but not extreme. MACD crossover confirms uptrend. IV rank moderate. Earnings in 3 weeks is a concern but not immediate.
Best strategy: Bull call spread to define risk before earnings.
Output: {"action":"BUY","strategy":"bull_call_spread","underlying":"NVDA","legs":[{"type":"call","strike":180,"expiration":"2026-08-08","quantity":1,"side":"buy"},{"type":"call","strike":185,"expiration":"2026-08-08","quantity":1,"side":"sell"}],"confidence":0.82,"reasoning":"MACD bullish crossover + RSI confirming uptrend. Bull call spread limits risk before earnings. IV rank moderate so debit spread is appropriate."}

EXAMPLE 3 — No Edge, Hold:
Input: AAPL at $220, RSI=50, IV rank=0.35, MACD flat, no catalyst
Reasoning: All indicators neutral. No directional bias. IV rank low. No catalyst.
Best action: Hold — wait for a clearer setup.
Output: {"action":"HOLD","strategy":"none","underlying":"AAPL","legs":[],"confidence":0.0,"reasoning":"All indicators neutral. No edge identified. Waiting for better setup with clearer direction or higher IV."}

═══ OUTPUT FORMAT ═══
Think step-by-step using the rules above, then output ONLY valid JSON:
{
  "action": "BUY" | "HOLD",
  "strategy": "<strategy_name>",
  "underlying": "<TICKER>",
  "legs": [
    {"type": "call"|"put", "strike": 190.0, "expiration": "YYYY-MM-DD", "quantity": 1, "side": "buy"|"sell"}
  ],
  "confidence": 0.85,
  "reasoning": "Brief rationale referencing specific indicators..."
}

Rules for output:
- HOLD when no edge exists — do NOT force trades
- legs must have valid expiration dates (YYYY-MM-DD format)
- strike prices must be realistic (round numbers near current price)
- confidence must reflect actual conviction (0.0-1.0)
- reasoning must reference specific indicators from the input
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
