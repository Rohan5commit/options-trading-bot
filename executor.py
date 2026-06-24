"""
Order Executor. Submits validated trades to Alpaca and handles fills.
Supports single-leg and multi-leg (spread) orders.
"""
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


def _build_order_payload(decision: dict[str, Any]) -> dict[str, Any]:
    """
    Build the Alpaca order JSON payload from a validated trade decision.
    For multi-leg strategies, uses order_class=mleg with OptionLegRequest format.
    """
    legs = decision.get("legs", [])
    strategy = decision.get("strategy", "")

    # Single-leg orders
    if len(legs) == 1:
        leg = legs[0]
        return {
            "symbol": leg["symbol"],
            "qty": str(leg["quantity"]),
            "side": leg["side"],
            "type": "market",
            "time_in_force": "day",
            "order_class": "single",
        }

    # Multi-leg orders (spreads, iron condors, etc.)
    alpaca_legs = []
    for leg in legs:
        position_intent = (
            f"{'buy' if leg['side'] == 'buy' else 'sell'}_to_open"
        )
        alpaca_legs.append({
            "symbol": leg["symbol"],
            "ratio_qty": str(leg["quantity"]),
            "side": leg["side"],
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
    url = "https://paper-api.alpaca.markets/v2/orders"
    resp = requests.post(
        url, headers=_alpaca_headers(), json=payload, timeout=30
    )
    resp.raise_for_status()
    order = resp.json()
    logger.info("Order submitted: %s (status: %s)", order.get("id"), order.get("status"))
    return order


def _poll_order(order_id: str) -> dict[str, Any]:
    """Poll order status until filled, canceled, or expired."""
    url = f"https://paper-api.alpaca.markets/v2/orders/{order_id}"
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
        time.sleep(POLL_INTERVAL_SEC)

    logger.warning("Order %s polling timed out after %d attempts", order_id, MAX_POLL_ATTEMPTS)
    return {"id": order_id, "status": "unknown"}


# ── Close position ─────────────────────────────────────────────────────────────


@_retry()
def close_position(contract_symbol: str) -> dict[str, Any]:
    """Close an existing option position by contract symbol."""
    url = f"https://paper-api.alpaca.markets/v2/positions/{contract_symbol}"
    resp = requests.delete(url, headers=_alpaca_headers(), timeout=30)
    resp.raise_for_status()
    order = resp.json()
    logger.info("Close position order submitted for %s: %s", contract_symbol, order.get("id"))
    return order


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
        logger.info("Executing %s %s on %s", decision["action"], strategy, symbol)

        try:
            payload = _build_order_payload(decision)
            order = _submit_order(payload)
            order_id = order.get("id", "")

            # Poll for fill
            final_order = _poll_order(order_id) if order_id else order
            fill_status = final_order.get("status", "unknown")

            # Build position record
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
        if result.get("status") == "filled":
            entry["trades_opened"].append({
                "symbol": decision.get("underlying", ""),
                "strategy": decision.get("strategy", ""),
                "action": decision.get("action", ""),
                "confidence": decision.get("confidence", 0),
                "reasoning": decision.get("reasoning", ""),
                "entry_price": result.get("position", {}).get("entry_price", 0),
            })
            entry["llm_confidence_scores"].append(decision.get("confidence", 0))
        elif result.get("status") == "error":
            entry["risk_rejections"].append({
                "symbol": decision.get("underlying", ""),
                "reason": f"Execution error: {result.get('error', 'unknown')}",
            })

    state_manager.save_today_entry(entry)
