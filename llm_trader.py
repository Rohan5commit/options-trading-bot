"""
LLM Decision Engine. Sends market context to the Modal inference endpoint
and parses structured trade decisions from the LLM response.
"""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import config

logger = logging.getLogger(__name__)

# ── System prompt (research-optimized: few-shot + CoT + regime awareness) ─────

SYSTEM_PROMPT = """\
You are an expert US equity options trader. Your sole objective is \
maximum risk-adjusted profit.

RULES:
1. Analyze the data, then output your decision as a JSON object.
2. Do NOT include any text before or after the JSON.
3. Do NOT use markdown, code fences, or explanations.
4. Output ONLY the raw JSON object, nothing else.

═══ MARKET REGIME RULES (apply FIRST) ═══
- IF IV rank > 0.70: favor credit spreads (iron_condor, bull_call_spread, bear_put_spread)
- IF RSI > 75 OR RSI < 25: AVOID new directional trades
- IF earnings within 5 days: AVOID the underlying entirely
- IF bid-ask spread > 10%: skip that contract

═══ STRATEGY SELECTION ═══
High IV rank (>0.60) → SELL premium
Strong uptrend (RSI 55-70, MACD positive) → BUY bull_call_spread
Strong downtrend (RSI 30-45, MACD negative) → BUY bear_put_spread
Low IV (<0.30) + catalyst → BUY long options
Neutral (RSI 45-55) → HOLD

═══ FEW-SHOT EXAMPLES ═══

Input: SPY $590, RSI=52, IV rank=0.75, MACD flat
Output: {"action":"BUY","strategy":"iron_condor","underlying":"SPY","legs":[{"type":"call","strike":600,"expiration":"2026-08-15","quantity":1,"side":"sell"},{"type":"call","strike":605,"expiration":"2026-08-15","quantity":1,"side":"buy"},{"type":"put","strike":580,"expiration":"2026-08-15","quantity":1,"side":"sell"},{"type":"put","strike":575,"expiration":"2026-08-15","quantity":1,"side":"buy"}],"confidence":0.78,"reasoning":"IV rank 0.75 favors premium selling. Neutral RSI supports range-bound thesis."}

Input: NVDA $180, RSI=72, IV rank=0.45, MACD bullish crossover
Output: {"action":"BUY","strategy":"bull_call_spread","underlying":"NVDA","legs":[{"type":"call","strike":180,"expiration":"2026-08-08","quantity":1,"side":"buy"},{"type":"call","strike":185,"expiration":"2026-08-08","quantity":1,"side":"sell"}],"confidence":0.82,"reasoning":"MACD bullish crossover confirms uptrend. Bull call spread limits risk."}

Input: AAPL $220, RSI=50, IV rank=0.35, MACD flat
Output: {"action":"HOLD","strategy":"none","underlying":"AAPL","legs":[],"confidence":0.0,"reasoning":"All indicators neutral. No edge identified."}

═══ OUTPUT FORMAT ═══
Output ONLY this JSON (no other text):
{"action":"BUY" or "HOLD","strategy":"<name>","underlying":"<TICKER>","legs":[{"type":"call" or "put","strike":190.0,"expiration":"YYYY-MM-DD","quantity":1,"side":"buy" or "sell"}],"confidence":0.85,"reasoning":"brief rationale"}
"""

# ── Regex patterns for freeform text fallback ──────────────────────────────────

_ACTION_RE = re.compile(r'\b(action)\s*[:=]\s*["\']?(BUY|SELL|HOLD)["\']?', re.I)
_STRATEGY_RE = re.compile(r'\b(strategy)\s*[:=]\s*["\']?(\w+)["\']?', re.I)
_CONFIDENCE_RE = re.compile(r'\b(confidence)\s*[:=]\s*["\']?(\d+\.?\d*)["\']?', re.I)
_REASONING_RE = re.compile(r'\b(reasoning)\s*[:=]\s*["\']?(.+?)["\']?\s*[,}]', re.I | re.S)
_LEG_RE = re.compile(
    r'\{[^{}]*"type"\s*:\s*"(call|put)"[^{}]*"strike"\s*:\s*(\d+\.?\d*)[^{}]*"expiration"\s*:\s*"(\d{4}-\d{2}-\d{2})"[^{}]*"quantity"\s*:\s*(\d+)[^{}]*"side"\s*:\s*"(buy|sell)"[^{}]*\}',
    re.I
)
# Alternate leg pattern (different key order)
_LEG_RE_ALT = re.compile(
    r'"(call|put)"[^{}]*?(\d+\.?\d*)[^{}]*?"(\d{4}-\d{2}-\d{2})"[^{}]*?(\d+)[^{}]*?"(buy|sell)"',
    re.I
)
_STRATEGIES = {
    "long_call", "long_put", "bull_call_spread", "bear_put_spread",
    "iron_condor", "straddle", "strangle", "calendar_spread", "none"
}


# ── JSON parsing with brace-depth counter + freeform fallback ──────────────────


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Robustly extract a JSON object from LLM output.
    Tries: direct parse → markdown strip → brace-depth → freeform regex fallback.
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

    # Attempt 3: brace-depth counter
    start_idx = text.find("{")
    if start_idx != -1:
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

    # Attempt 4: freeform text extraction (LLM ignored JSON instruction)
    return _parse_freeform_text(text)


def _parse_freeform_text(text: str) -> dict[str, Any] | None:
    """
    Last-resort parser: extract trade decision from free-form LLM text.
    The LLM often outputs good analysis but ignores JSON formatting instructions.
    """
    # Extract action
    m = _ACTION_RE.search(text)
    action = m.group(2).upper() if m else None

    if action is None:
        # Try to infer action from text
        lower = text.lower()
        if "hold" in lower and ("no edge" in lower or "no clear" in lower or "neutral" in lower):
            action = "HOLD"
        elif "buy" in lower or "long" in lower:
            action = "BUY"
        elif "sell" in lower or "short" in lower:
            action = "SELL"
        else:
            logger.warning("Cannot determine action from freeform text")
            return None

    # Extract strategy
    m = _STRATEGY_RE.search(text)
    strategy = m.group(2).lower() if m else None

    if strategy is None or strategy not in _STRATEGIES:
        # Try to infer strategy from text
        lower = text.lower()
        # Check for specific strategy names
        for s in _STRATEGIES:
            if s != "none" and (s in lower or s.replace("_", " ") in lower):
                strategy = s
                break
        # Check for common patterns
        if strategy is None:
            if "iron condor" in lower:
                strategy = "iron_condor"
            elif "bull call" in lower or "bull spread" in lower:
                strategy = "bull_call_spread"
            elif "bear put" in lower or "bear spread" in lower:
                strategy = "bear_put_spread"
            elif "straddle" in lower:
                strategy = "straddle"
            elif "strangle" in lower:
                strategy = "strangle"
            elif "calendar" in lower:
                strategy = "calendar_spread"
            elif "long call" in lower or ("call" in lower and "buy" in lower):
                strategy = "long_call"
            elif "long put" in lower or ("put" in lower and "buy" in lower):
                strategy = "long_put"
            elif "sell" in lower and "call" in lower:
                strategy = "bull_call_spread"  # selling a call = part of spread
            elif "sell" in lower and "put" in lower:
                strategy = "bear_put_spread"  # selling a put = part of spread
            elif action == "HOLD":
                strategy = "none"
            else:
                logger.warning("Cannot determine strategy from freeform text")
                return None

    # Extract confidence
    m = _CONFIDENCE_RE.search(text)
    confidence = float(m.group(2)) if m else 0.7  # default if not found

    # Extract reasoning
    m = _Reasoning_RE.search(text) if hasattr(text, '_Reasoning_RE') else _REASONING_RE.search(text)
    if m is None:
        # Grab first 200 chars after "reasoning" keyword
        idx = text.lower().find("reasoning")
        reasoning = text[idx:idx + 200].strip() if idx != -1 else "Parsed from freeform text"
    else:
        reasoning = m.group(2).strip()

    # Extract legs - try structured JSON patterns first, then simple patterns
    legs = []
    # Pattern 1: Full JSON leg objects
    for m in _LEG_RE.finditer(text):
        legs.append({
            "type": m.group(1).lower(),
            "strike": float(m.group(2)),
            "expiration": m.group(3),
            "quantity": int(m.group(4)),
            "side": m.group(5).lower(),
        })

    if not legs:
        for m in _LEG_RE_ALT.finditer(text):
            legs.append({
                "type": m.group(1).lower(),
                "strike": float(m.group(2)),
                "expiration": m.group(3),
                "quantity": int(m.group(4)),
                "side": m.group(5).lower(),
            })

    # Pattern 2: Simple "Buy/Sell [strike] [call/put] exp [date]"
    if not legs:
        simple_leg_re = re.compile(
            r'(buy|sell)\s+(?:a\s+)?(?:AAPL|SPY|QQQ|NVDA|AMD|TSLA|META|AMZN|GOOGL|MSFT|\w{1,5})\s+'
            r'(call|put)\s+(?:option\s+)?(?:with\s+)?(?:strike\s+(?:price\s+)?[:=]?\s*)?(\d+\.?\d*)'
            r'(?:.*?(?:expir(?:ation)?\s+(?:date\s+)?[:=]?\s*)?["\']?(\d{4}-\d{2}-\d{2})["\']?)?',
            re.I
        )
        for m in simple_leg_re.finditer(text):
            legs.append({
                "type": m.group(2).lower(),
                "strike": float(m.group(3)),
                "expiration": m.group(4) if m.group(4) else "2026-08-15",
                "quantity": 1,
                "side": m.group(1).lower(),
            })

    # Pattern 3: Multi-line "Buy/Sell ... Strike: 225 ... Expiration: 2026-08-08"
    if not legs:
        # Find all "buy/sell" blocks with strike and expiration on separate lines
        block_re = re.compile(
            r'(buy|sell)\s+\w+\s+(call|put)\s+option.*?'
            r'strike\s+price\s*:\s*(\d+\.?\d*).*?'
            r'expir(?:ation)?\s+date\s*:\s*(\d{4}-\d{2}-\d{2})',
            re.I | re.S
        )
        for m in block_re.finditer(text):
            legs.append({
                "type": m.group(2).lower(),
                "strike": float(m.group(3)),
                "expiration": m.group(4),
                "quantity": 1,
                "side": m.group(1).lower(),
            })

    # If HOLD, legs can be empty
    if action == "HOLD":
        legs = []

    # Extract underlying ticker
    underlying_match = re.search(r'\b([A-Z]{1,5})\b', text[:200])
    underlying = underlying_match.group(1) if underlying_match else "UNKNOWN"

    decision = {
        "action": action,
        "strategy": strategy,
        "underlying": underlying,
        "legs": legs,
        "confidence": confidence,
        "reasoning": reasoning[:300],
    }

    logger.info("Parsed freeform text into decision: %s %s", action, strategy)
    return decision


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

    # Validate confidence is a number in valid range
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

        try:
            strike = float(leg["strike"])
            if strike <= 0:
                logger.warning("Invalid strike: %s", leg["strike"])
                return False
            leg["strike"] = strike
        except (ValueError, TypeError):
            logger.warning("Invalid strike value: %s", leg["strike"])
            return False

        try:
            qty = int(leg["quantity"])
            if qty <= 0:
                logger.warning("Invalid quantity: %s", leg["quantity"])
                return False
            leg["quantity"] = qty
        except (ValueError, TypeError):
            logger.warning("Invalid quantity value: %s", leg["quantity"])
            return False

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
        ctx = _truncate_context(ctx)

        user_prompt = f"Analyze this options chain and market data:\n{json.dumps(ctx, indent=2)}"

        if exit_recommendations:
            open_pos = [
                r for r in exit_recommendations
                if r.get("underlying") == symbol
            ]
            if open_pos:
                user_prompt += f"\n\nExisting open positions to evaluate for exit:\n{json.dumps(open_pos, indent=2)}"

        logger.info("Calling LLM for %s", symbol)
        raw_response = modal_inference.call_inference(user_prompt, SYSTEM_PROMPT)

        # Parse the response (with freeform fallback)
        decision = _extract_json(raw_response)
        if decision is None:
            logger.warning("LLM returned unparseable output for %s, skipping", symbol)
            logger.warning("Raw output: %s", raw_response[:500])
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
    Parallelizes across symbols for faster execution.
    """
    if not market_contexts:
        return []

    exit_recs = exit_recommendations or []
    decisions: list[dict[str, Any]] = []

    max_workers = min(5, len(market_contexts))
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
