"""
LLM Decision Engine. Sends market context to the Modal inference endpoint
and parses structured trade decisions from the LLM response.
"""
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import config

logger = logging.getLogger(__name__)

# ── In-memory daily trade counter (fixes MAX_DAILY_TRADES being dead code) ────
_trades_opened_today = 0
_trades_lock = threading.Lock()

# ── System prompt (research-optimized: few-shot + CoT + regime awareness) ─────

SYSTEM_PROMPT = """\
You are an expert US equity options trader. Your sole objective is \
maximum risk-adjusted profit. You MUST find trades to take — defaulting \
to HOLD is unacceptable unless conditions are truly extreme.

RULES:
1. Analyze the data, then output your decision as a JSON object.
2. Do NOT include any text before or after the JSON.
3. Do NOT use markdown, code fences, or explanations.
4. Output ONLY the raw JSON object, nothing else.

═══ MARKET REGIME RULES (apply FIRST) ═══
- IF IV rank > 0.70: favor credit spreads (iron_condor, bull_put_spread)
- IF RSI > 75 OR RSI < 25: AVOID new directional trades
- IF earnings within 5 days: AVOID the underlying entirely
- IF bid-ask spread > 10%: skip that contract

═══ STRATEGY SELECTION ═══
High IV rank (>0.60) → SELL premium (iron_condor or bull_put_spread)
Moderate IV rank (0.30-0.60) with trend → directional spread
Strong uptrend (RSI >55, MACD positive) → BUY bull_call_spread
Downtrend (RSI <45, MACD negative) → BUY bear_put_spread
Low IV (<0.30) + any catalyst → BUY long options
Near support with IV >0.20 → bull_put_spread
Near resistance with IV >0.20 → iron_condor or bear_call_spread

You MUST choose a strategy. The only acceptable HOLD is when:
- RSI is exactly 50 AND IV rank is below 0.15 AND no news/catalyst
- OR the contract liquidity is terrible (bid-ask >15%)

═══ FEW-SHOT EXAMPLES ═══

Input: SPY $739, RSI=41.9, IV=0.22, MACD negative, price near Bollinger lower band
Output: {"action":"BUY","strategy":"bull_put_spread","underlying":"SPY","legs":[{"type":"put","strike":730,"expiration":"2026-07-31","quantity":1,"side":"sell"},{"type":"put","strike":725,"expiration":"2026-07-31","quantity":1,"side":"buy"}],"confidence":0.72,"reasoning":"RSI 41.9 oversold near Bollinger lower band 736. IV 0.22 supports credit selling. Bull put spread captures premium with defined risk."}

Input: NVDA $180, RSI=72, IV=0.45, MACD bullish crossover
Output: {"action":"BUY","strategy":"bull_call_spread","underlying":"NVDA","legs":[{"type":"call","strike":180,"expiration":"2026-08-08","quantity":1,"side":"buy"},{"type":"call","strike":185,"expiration":"2026-08-08","quantity":1,"side":"sell"}],"confidence":0.82,"reasoning":"MACD bullish crossover confirms uptrend. Bull call spread limits risk."}

Input: AAPL $220, RSI=65, IV=0.38, MACD positive, strong uptrend
Output: {"action":"BUY","strategy":"bull_call_spread","underlying":"AAPL","legs":[{"type":"call","strike":220,"expiration":"2026-08-08","quantity":1,"side":"buy"},{"type":"call","strike":225,"expiration":"2026-08-08","quantity":1,"side":"sell"}],"confidence":0.75,"reasoning":"RSI 65 confirms uptrend. IV 0.38 moderate. Bull call spread leverages momentum with defined risk."}

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
    "long_call", "long_put", "bull_call_spread", "bull_put_spread",
    "bear_put_spread", "bear_call_spread", "iron_condor", "straddle",
    "strangle", "calendar_spread", "none"
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
        if "hold" in lower:
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
            elif "bull put" in lower:
                strategy = "bull_put_spread"
            elif "bear put" in lower or "bear spread" in lower:
                strategy = "bear_put_spread"
            elif "bear call" in lower:
                strategy = "bear_call_spread"
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
            elif action == "HOLD":
                strategy = "none"
            else:
                logger.warning("Cannot determine strategy from freeform text")
                return None

    # Extract confidence
    m = _CONFIDENCE_RE.search(text)
    confidence = float(m.group(2)) if m else 0.7  # default if not found

    # Extract reasoning
    m = _REASONING_RE.search(text)
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


# ── Rule-based override (when LLM is too conservative) ────────────────────────


def _leg_passes_liquidity(entry: dict[str, Any]) -> str | None:
    """
    Enforce the contract-level filters from config on an override candidate leg.
    Returns a rejection reason, or None if the leg is tradable.
    The old override ignored spread_pct / open_interest / dte entirely and sold
    illiquid wide-market spreads, giving away the edge on entry fills.
    """
    if entry.get("mid", 0) <= 0:
        return "no mid quote"
    if entry.get("spread_pct", 999) > config.MAX_BID_ASK_SPREAD_PCT:
        return f"bid-ask {entry.get('spread_pct', 0):.1%} > {config.MAX_BID_ASK_SPREAD_PCT:.0%} limit"
    if entry.get("open_interest", 0) < config.MIN_OPEN_INTEREST:
        return f"OI {entry.get('open_interest', 0)} < {config.MIN_OPEN_INTEREST} minimum"
    dte = entry.get("dte", 0)
    if dte < config.MIN_DTE or dte > config.MAX_DTE:
        return f"DTE {dte} outside [{config.MIN_DTE}, {config.MAX_DTE}]"
    return None


def _build_override_leg(chain_entry: dict[str, Any], side: str) -> dict[str, Any]:
    """Build an override leg carrying chain analytics for executor/risk sizing."""
    return {
        "type": chain_entry.get("type", "put"),
        "strike": chain_entry.get("strike", 0),
        "expiration": chain_entry.get("expiration", ""),
        "quantity": 1,
        "side": side,
        "mid": chain_entry.get("mid", 0),
        "bid": chain_entry.get("bid", 0),
        "ask": chain_entry.get("ask", 0),
        "spread_pct": chain_entry.get("spread_pct", 999),
        "open_interest": chain_entry.get("open_interest", 0),
        "dte": chain_entry.get("dte", 0),
        "delta": chain_entry.get("delta", 0),
    }


def _rule_based_override(ctx: dict[str, Any]) -> dict[str, Any] | None:
    """
    Generate a trade when LLM says HOLD but conditions are favorable.
    Returns None if no override is needed.
    Generates ONLY bull_put_spread (credit): short an OTM put below support,
    long a further-OTM put as defined-risk protection. Profits if the stock
    stays ABOVE the short strike; max loss is capped at width minus credit.
    """
    global _trades_opened_today
    import state_manager

    # Gate 1: Position limit
    positions = state_manager.load_positions()
    if len(positions) >= config.MAX_OPEN_POSITIONS:
        return None

    # Gate 2: Daily trade limit (in-memory counter — thread-safe)
    with _trades_lock:
        if _trades_opened_today >= config.MAX_DAILY_TRADES:
            return None

    # Gate 3: Market regime filter.
    # If this IS SPY and RSI < 30, skip (strong downtrend — falling knife).
    underlying_data = ctx.get("underlying", {})
    if ctx.get("symbol") == "SPY" and underlying_data.get("rsi_14", 50) < 30:
        return None

    underlying = ctx.get("symbol", "")
    price = underlying_data.get("price", 0)
    rsi = underlying_data.get("rsi_14", 50)
    iv = ctx.get("iv_metrics", {}).get("current_iv", 0)
    ma_20 = (underlying_data.get("bollinger") or {}).get("middle", 0)
    puts = ctx.get("options_chain", {}).get("puts", [])

    if not puts or price <= 0:
        return None

    # Gate 4: Trend filter — sell puts only into healthy pullbacks.
    # RSI < 40 while price holds above the 20-day MA = dip in an uptrend
    # (high win-rate regime for short puts). RSI < 40 BELOW the MA = falling
    # knife — the old code traded those and bled on max-loss gaps.
    if config.REQUIRE_ABOVE_MA:
        if ma_20 > 0:
            if price <= ma_20:
                logger.info(
                    "Override skip for %s: price $%.2f below 20-day MA $%.2f",
                    underlying, price, ma_20,
                )
                return None
        else:
            logger.warning(
                "Override for %s: no 20-day MA available, proceeding without trend gate",
                underlying,
            )

    # Gate 5: Oversold + elevated IV.
    # Only trade when RSI < 40 (pullback) and IV > 0.25 (premium worth selling).
    if not (rsi < 40 and iv > 0.25):
        return None

    # Strike selection: OTM short put + defined-risk long put.
    # Short strike must sit at least MIN_OTM_PCT below spot so normal noise
    # doesn't test it; long leg is the nearest liquid strike below at the
    # SAME expiration (tight width = small max loss). Both legs must pass
    # liquidity filters. Nearest expiry first.
    max_short_strike = price * (1 - config.MIN_OTM_PCT)
    expirations = sorted({p.get("expiration", "") for p in puts if p.get("expiration")})

    for exp in expirations:
        exp_puts = [p for p in puts if p.get("expiration") == exp]
        shorts = sorted(
            (p for p in exp_puts
             if 0 < p.get("strike", 0) <= max_short_strike),
            key=lambda p: p["strike"],
            reverse=True,
        )
        for short_entry in shorts:
            reason = _leg_passes_liquidity(short_entry)
            if reason is not None:
                logger.debug("Override skip %s %s short leg: %s", underlying, exp, reason)
                continue
            # Delta gate on the short leg (when Greeks are available):
            # target low-delta OTM shorts, not coin-flip ATM ones.
            short_delta = abs(short_entry.get("delta") or 0)
            if short_delta and not (config.SHORT_DELTA_MIN <= short_delta <= config.SHORT_DELTA_MAX):
                logger.debug(
                    "Override skip %s %s: short |delta| %.2f outside [%.2f, %.2f]",
                    underlying, exp, short_delta,
                    config.SHORT_DELTA_MIN, config.SHORT_DELTA_MAX,
                )
                continue
            longs = sorted(
                (p for p in exp_puts if 0 < p.get("strike", 0) < short_entry["strike"]),
                key=lambda p: p["strike"],
                reverse=True,
            )
            long_entry = None
            for candidate in longs:
                if _leg_passes_liquidity(candidate) is None:
                    long_entry = candidate
                    break
            if long_entry is None:
                continue

            sell_strike = short_entry["strike"]
            buy_strike = long_entry["strike"]
            width = round(sell_strike - buy_strike, 2)
            otm_pct = (price - sell_strike) / price * 100
            return {
                "action": "BUY",
                "strategy": "bull_put_spread",
                "underlying": underlying,
                "legs": [
                    _build_override_leg(short_entry, "sell"),
                    _build_override_leg(long_entry, "buy"),
                ],
                "confidence": 0.70,
                "reasoning": (
                    f"Rule override: RSI {rsi:.1f} pullback above 20MA ${ma_20:.2f}, "
                    f"IV {iv:.2f}; short {sell_strike} put {otm_pct:.1f}% OTM "
                    f"(|delta| {short_delta:.2f}) / long {buy_strike} put, "
                    f"width ${width:.2f}, {exp}"
                ),
            }

    logger.info(
        "Override skip for %s: no liquid OTM put spread found (RSI %.1f, IV %.2f)",
        underlying, rsi, iv,
    )
    return None


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

        # Rule-based override: if LLM says HOLD but conditions are favorable, generate a trade
        if decision.get("action") == "HOLD":
            override = _rule_based_override(ctx)
            if override:
                logger.info("Rule-based override for %s: %s %s", symbol, override["action"], override["strategy"])
                decision = override

        # Track ALL BUY decisions (overrides AND LLM-originated)
        # Enforcement happens in risk_manager.validate() — here we only count
        if decision.get("action") == "BUY":
            with _trades_lock:
                global _trades_opened_today
                _trades_opened_today += 1

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
    global _trades_opened_today
    if not market_contexts:
        return []

    # Reset daily counter at start of each pipeline run
    with _trades_lock:
        _trades_opened_today = 0

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
