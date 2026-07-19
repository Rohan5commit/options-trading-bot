"""
Order Executor. Submits validated trades to Alpaca and handles fills.
Supports single-leg and multi-leg (spread) orders.
Detects SELL decisions targeting existing positions and closes them.
"""
import functools
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import requests

import config
import state_manager

logger = logging.getLogger(__name__)

MAX_POLL_ATTEMPTS = 30
POLL_INTERVAL_SEC = 2


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        "Content-Type": "application/json",
    }


def _retry(max_retries: int = 3, backoff_factor: float = 1.0):
    """Exponential-backoff retry for API calls."""
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
            raise last_exc
        return wrapper
    return decorator


# ── Order submission ───────────────────────────────────────────────────────────


def _build_order_payload(decision: dict[str, Any], is_close: bool = False) -> dict[str, Any]:
    """
    Build the Alpaca order JSON payload from a validated trade decision.
    For multi-leg strategies, uses order_class=mleg with OptionLegRequest format.
    For close orders, flips side and uses _to_close intent.
    """
    legs = decision.get("legs", [])

    # Single-leg orders
    if len(legs) == 1:
        leg = legs[0]
        side = leg["side"]
        if is_close:
            side = "sell" if side == "buy" else "buy"
        return {
            "symbol": leg["symbol"],
            "qty": str(leg["quantity"]),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "order_class": "single",
        }

    # Multi-leg orders (spreads, iron condors, etc.)
    alpaca_legs = []
    for leg in legs:
        side = leg["side"]
        if is_close:
            side = "sell" if side == "buy" else "buy"
        position_intent = (
            f"{'buy' if side == 'buy' else 'sell'}_to_close" if is_close
            else f"{'buy' if side == 'buy' else 'sell'}_to_open"
        )
        alpaca_legs.append({
            "symbol": leg["symbol"],
            "ratio_qty": str(leg["quantity"]),
            "side": side,
            "position_intent": position_intent,
        })

    return {
        "order_class": "mleg",
        "qty": "1",
        "type": "market",
        "time_in_force": "day",
        "legs": alpaca_legs,
    }


@_retry()
def _submit_order(payload: dict[str, Any]) -> dict[str, Any]:
    """Submit an order to Alpaca Trading API."""
    url = f"{config.ALPACA_BASE_URL}/v2/orders"
    resp = requests.post(
        url, headers=_alpaca_headers(), json=payload, timeout=30
    )
    resp.raise_for_status()
    order = resp.json()
    logger.info("Order submitted: %s (status: %s)", order.get("id"), order.get("status"))
    return order


def _poll_order(order_id: str) -> dict[str, Any]:
    """Poll order status until filled, canceled, or expired."""
    url = f"{config.ALPACA_BASE_URL}/v2/orders/{order_id}"
    for attempt in range(MAX_POLL_ATTEMPTS):
        try:
            resp = requests.get(url, headers=_alpaca_headers(), timeout=30)
            resp.raise_for_status()
            order = resp.json()
            status = order.get("status", "")
            if status in ("filled", "canceled", "expired", "rejected"):
                logger.info("Order %s final status: %s", order_id, status)
                return order
            logger.debug("Order %s status: %s (attempt %d)", order_id, status, attempt + 1)
        except Exception as exc:
            logger.warning("Poll attempt %d failed for order %s: %s", attempt + 1, order_id, exc)
        if attempt < MAX_POLL_ATTEMPTS - 1:
            time.sleep(POLL_INTERVAL_SEC)

    logger.warning("Order %s polling timed out after %d attempts", order_id, MAX_POLL_ATTEMPTS)
    return {"id": order_id, "status": "unknown"}


# ── Close position ─────────────────────────────────────────────────────────────


@_retry()
def close_position(contract_symbol: str) -> dict[str, Any]:
    """Close an existing option position by contract symbol."""
    url = f"{config.ALPACA_BASE_URL}/v2/positions/{contract_symbol}"
    resp = requests.delete(url, headers=_alpaca_headers(), timeout=30)
    resp.raise_for_status()
    order = resp.json()
    logger.info("Close position order submitted for %s: %s", contract_symbol, order.get("id"))
    return order


# ── Position matching for SELL/close decisions ────────────────────────────────


def _find_existing_position(decision: dict[str, Any]) -> dict[str, Any] | None:
    """
    Find an existing open position that matches a SELL/close decision.
    Matches by underlying symbol and strategy.
    """
    if decision.get("action") != "SELL":
        return None

    underlying = decision.get("underlying", "")
    strategy = decision.get("strategy", "")
    positions = state_manager.load_positions()

    for pos in positions:
        if pos.get("underlying") == underlying and pos.get("strategy") == strategy:
            return pos
    return None


# ── Public API ─────────────────────────────────────────────────────────────────


def execute_trades(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Execute a list of validated trade decisions.
    Returns execution results for each decision.
    """
    results: list[dict[str, Any]] = []

    for decision in decisions:
        # HOLD decisions are no-ops
        if decision.get("action") == "HOLD":
            results.append({
                "decision": decision,
                "status": "skipped",
                "reason": "HOLD — no trade",
            })
            continue

        symbol = decision.get("underlying", "UNKNOWN")
        strategy = decision.get("strategy", "unknown")
        action = decision.get("action", "BUY")
        logger.info("Executing %s %s on %s", action, strategy, symbol)

        try:
            # Check if this is a SELL targeting an existing position (fix #3)
            existing_pos = _find_existing_position(decision)
            is_close = existing_pos is not None

            if is_close:
                logger.info(
                    "Closing existing position %s for %s %s",
                    existing_pos["id"], strategy, symbol,
                )

            payload = _build_order_payload(decision, is_close=is_close)
            order = _submit_order(payload)
            order_id = order.get("id", "")

            # Poll for fill
            final_order = _poll_order(order_id) if order_id else order
            fill_status = final_order.get("status", "unknown")

            if is_close:
                # Close existing position in state
                entry_price = existing_pos.get("entry_price", 0)
                filled_price = float(final_order.get("filled_avg_price", 0))
                quantity = existing_pos.get("quantity", 1)
                realized_pnl = round((filled_price - entry_price) * quantity * 100, 2) if entry_price > 0 else 0

                results.append({
                    "decision": decision,
                    "order": final_order,
                    "existing_position": existing_pos,
                    "status": fill_status,
                    "realized_pnl": realized_pnl,
                })

                if fill_status == "filled":
                    state_manager.remove_position(existing_pos["id"])
                    logger.info("Position %s closed: realized P&L $%.2f", existing_pos["id"], realized_pnl)
                else:
                    logger.warning("Close order for %s not fully filled (status: %s)", symbol, fill_status)
            else:
                # Build new position record
                position_id = f"pos_{uuid.uuid4().hex[:8]}"
                legs_detail = []
                for leg in decision.get("legs", []):
                    legs_detail.append({
                        "symbol": leg.get("symbol", ""),
                        "type": leg.get("type", ""),
                        "strike": leg.get("strike", 0),
                        "expiration": leg.get("expiration", ""),
                        "quantity": leg.get("quantity", 1),
                        "side": leg.get("side", ""),
                    })

                position = {
                    "id": position_id,
                    "order_id": order_id,
                    "underlying": symbol,
                    "strategy": strategy,
                    "legs": legs_detail,
                    "entry_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "entry_price": float(final_order.get("filled_avg_price", 0)),
                    "quantity": int(final_order.get("filled_qty", 0)),
                    "current_price": float(final_order.get("filled_avg_price", 0)),
                    "unrealized_pnl": 0.0,
                    "fill_status": fill_status,
                    "llm_reasoning": decision.get("reasoning", ""),
                    "llm_confidence": decision.get("confidence", 0),
                }

                if fill_status == "filled":
                    state_manager.add_position(position)
                    logger.info("Position %s opened: %s %s", position_id, strategy, symbol)
                else:
                    logger.warning(
                        "Order for %s not fully filled (status: %s)", symbol, fill_status
                    )

                results.append({
                    "decision": decision,
                    "order": final_order,
                    "position": position,
                    "status": fill_status,
                })

        except Exception as exc:
            logger.error("Failed to execute trade for %s: %s", symbol, exc)
            results.append({
                "decision": decision,
                "status": "error",
                "error": str(exc),
            })

    # Log executed trades to daily entry
    _log_executed_trades(results)

    logger.info("Execution complete: %d trades processed", len(results))
    return results


def _log_executed_trades(results: list[dict[str, Any]]) -> None:
    """Append executed trades to today's daily log entry."""
    entry = state_manager.get_today_entry()
    if entry is None:
        entry = state_manager.create_today_entry()

    for result in results:
        decision = result.get("decision", {})
        status = result.get("status", "")

        if status == "filled":
            action = decision.get("action", "BUY")
            if action == "SELL" and "existing_position" in result:
                # Log as closed trade
                existing = result["existing_position"]
                entry["trades_closed"].append({
                    "symbol": decision.get("underlying", ""),
                    "strategy": decision.get("strategy", ""),
                    "entry_price": existing.get("entry_price", 0),
                    "exit_price": result.get("order", {}).get("filled_avg_price", 0),
                    "realized_pnl": result.get("realized_pnl", 0),
                    "reason": f"LLM SELL decision (confidence: {decision.get('confidence', 0):.2f})",
                })
                entry["realized_pnl"] += result.get("realized_pnl", 0)
            else:
                # Log as opened trade
                entry["trades_opened"].append({
                    "symbol": decision.get("underlying", ""),
                    "strategy": decision.get("strategy", ""),
                    "action": action,
                    "confidence": decision.get("confidence", 0),
                    "reasoning": decision.get("reasoning", ""),
                    "entry_price": result.get("position", {}).get("entry_price", 0),
                })
                entry["llm_confidence_scores"].append(decision.get("confidence", 0))
        elif status == "error":
            # Log execution errors separately from risk rejections
            if "execution_errors" not in entry:
                entry["execution_errors"] = []
            entry["execution_errors"].append({
                "symbol": decision.get("underlying", ""),
                "reason": f"Execution error: {result.get('error', 'unknown')}",
            })

    state_manager.save_today_entry(entry)
