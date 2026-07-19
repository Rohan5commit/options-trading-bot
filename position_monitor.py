"""
Position Monitor. Checks all open positions daily and closes based on
hard rules and LLM exit recommendations.
"""
import logging
from datetime import datetime, timezone
from typing import Any

import config
import data_fetcher
import executor
import state_manager

logger = logging.getLogger(__name__)


def _calculate_dte(expiration: str) -> int:
    """Calculate days-to-expiration from a date string."""
    try:
        exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        return (exp_date - today).days
    except (ValueError, TypeError):
        return 999


def _compute_net_position_value(pos: dict[str, Any], leg_prices: dict[str, float]) -> float:
    """
    Compute the net value of a multi-leg position.
    For debit spreads: net = sum(buy_prices) - sum(sell_prices)
    For credit spreads: net = sum(sell_prices) - sum(buy_prices)
    """
    net = 0.0
    for leg in pos.get("legs", []):
        sym = leg.get("symbol", "")
        price = leg_prices.get(sym, 0)
        qty = leg.get("quantity", 1)
        if leg.get("side") == "buy":
            net += price * qty
        else:
            net -= price * qty
    return net


def _check_hard_exit(position: dict[str, Any]) -> tuple[bool, str]:
    """
    Check if a position should be force-closed due to hard rules.
    Returns (should_exit, reason).
    """
    legs = position.get("legs", [])
    if not legs:
        return False, ""

    # DTE exit: close at threshold DTE regardless of P&L
    for leg in legs:
        dte = _calculate_dte(leg.get("expiration", ""))
        if dte <= config.DTE_EXIT_THRESHOLD:
            return True, f"DTE {dte} <= threshold {config.DTE_EXIT_THRESHOLD}"

    # Hard loss exit: close if loss > 100% of debit paid
    entry_price = position.get("entry_price", 0)
    current_price = position.get("current_price", entry_price)
    quantity = position.get("quantity", 1)
    if entry_price > 0:
        debit_paid = entry_price * quantity * 100  # options multiplier
        current_value = current_price * quantity * 100
        loss = debit_paid - current_value
        if loss >= debit_paid * config.HARD_EXIT_LOSS_PCT:
            return True, (
                f"Loss ${loss:.0f} >= {config.HARD_EXIT_LOSS_PCT*100:.0f}% "
                f"of debit paid (${debit_paid:.0f})"
            )

    # Profit target exit: close if profit > PROFIT_TARGET_PCT of debit paid
    if entry_price > 0:
        profit = current_value - debit_paid
        if profit >= debit_paid * config.PROFIT_TARGET_PCT:
            return True, (
                f"Profit ${profit:.0f} >= {config.PROFIT_TARGET_PCT*100:.0f}% "
                f"of debit paid (${debit_paid:.0f})"
            )

    return False, ""


def _get_current_price_for_contract(contract_symbol: str) -> float:
    """Fetch the current mid price for an option contract."""
    try:
        import requests
        headers = {
            "APCA-API-KEY-ID": config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        }
        underlying = "".join(c for c in contract_symbol if c.isalpha())
        url = f"{config.ALPACA_DATA_URL}/v1beta1/options/snapshots/{underlying}"
        resp = requests.get(url, headers=headers, params={"feed": "indicative"}, timeout=30)
        resp.raise_for_status()
        snapshots = resp.json().get("snapshots", {})
        snap = snapshots.get(contract_symbol, {})
        quote = snap.get("latest_quote", {})
        bid = quote.get("bp", 0)
        ask = quote.get("ap", 0)
        return (bid + ask) / 2 if (bid + ask) > 0 else 0
    except Exception as exc:
        logger.warning("Failed to fetch price for %s: %s", contract_symbol, exc)
        return 0


def check_exits() -> list[dict[str, Any]]:
    """
    Check all open positions for exit conditions.
    Returns a list of exit recommendations for the LLM to evaluate.
    """
    positions = state_manager.load_positions()
    if not positions:
        logger.info("No open positions to monitor")
        return []

    logger.info("Checking %d open positions for exits", len(positions))
    exit_recommendations: list[dict[str, Any]] = []
    closed_positions: list[dict[str, Any]] = []

    for pos in positions:
        position_id = pos.get("id", "")
        underlying = pos.get("underlying", "")
        legs = pos.get("legs", [])

        if not legs:
            logger.warning("Position %s has no legs, skipping", position_id)
            continue

        # Fetch current price for each leg
        leg_prices: dict[str, float] = {}
        for leg in legs:
            contract_sym = leg.get("symbol", "")
            if contract_sym:
                price = _get_current_price_for_contract(contract_sym)
                if price > 0:
                    leg_prices[contract_sym] = price

        # Compute net position value from all legs
        net_value = _compute_net_position_value(pos, leg_prices)
        pos["current_price"] = net_value

        # Calculate unrealized P&L
        entry_price = pos.get("entry_price", 0)
        quantity = pos.get("quantity", 1)
        if entry_price > 0:
            pos["unrealized_pnl"] = round(
                (net_value - entry_price) * quantity * 100, 2
            )

        # Check hard exit rules
        should_exit, reason = _check_hard_exit(pos)
        if should_exit:
            logger.info(
                "Hard exit triggered for %s: %s", position_id, reason
            )
            try:
                # Save daily entry BEFORE removing position (fix #14)
                entry = state_manager.get_today_entry()
                if entry is None:
                    entry = state_manager.create_today_entry()

                # Close all legs (not just buy legs) (fix #2)
                for leg in legs:
                    contract_sym = leg.get("symbol", "")
                    if contract_sym:
                        try:
                            executor.close_position(contract_sym)
                        except Exception as exc:
                            logger.error("Failed to close leg %s: %s", contract_sym, exc)

                # Record closed position
                closed_positions.append({
                    **pos,
                    "exit_reason": reason,
                    "exit_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "exit_price": net_value,
                    "realized_pnl": pos.get("unrealized_pnl", 0),
                })

                # Update daily log BEFORE removing position
                entry["trades_closed"].append({
                    "symbol": underlying,
                    "strategy": pos.get("strategy", ""),
                    "entry_price": entry_price,
                    "exit_price": net_value,
                    "realized_pnl": pos.get("unrealized_pnl", 0),
                    "reason": reason,
                })
                entry["realized_pnl"] += pos.get("unrealized_pnl", 0)
                state_manager.save_today_entry(entry)

                # NOW remove from state
                state_manager.remove_position(position_id)

            except Exception as exc:
                logger.error("Failed to close position %s: %s", position_id, exc)
        else:
            # Compute min DTE safely (fix #13)
            dte_values = [_calculate_dte(leg.get("expiration", "")) for leg in legs]
            min_dte = min(dte_values) if dte_values else 999

            # Add to exit recommendations for LLM evaluation
            exit_recommendations.append({
                "id": position_id,
                "underlying": underlying,
                "strategy": pos.get("strategy", ""),
                "entry_price": entry_price,
                "current_price": net_value,
                "unrealized_pnl": pos.get("unrealized_pnl", 0),
                "dte": min_dte,
                "llm_reasoning": pos.get("llm_reasoning", ""),
            })

    # Update positions state with new prices
    remaining = [
        p for p in positions
        if p.get("id") not in {c.get("id") for c in closed_positions}
    ]
    state_manager.save_positions(remaining)

    logger.info(
        "Exit check complete: %d closed, %d remaining, %d sent to LLM",
        len(closed_positions), len(remaining), len(exit_recommendations),
    )
    return exit_recommendations
