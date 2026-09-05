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


def _is_credit_position(pos: dict[str, Any]) -> bool:
    """
    Classify a position as credit vs debit WITHOUT trusting the entry_price sign.
    Alpaca has returned positive filled_avg_price for credit mleg fills before,
    which sent credit spreads down the debit exit branch (and vice versa).
    Priority: stored flag (executor-written) → strategy sets → entry sign fallback.
    """
    if "is_credit" in pos:
        return bool(pos["is_credit"])
    strategy = pos.get("strategy", "")
    if strategy in config.CREDIT_STRATEGIES:
        return True
    if strategy in config.DEBIT_STRATEGIES:
        return False
    return pos.get("entry_price", 0) < 0


def _compute_net_position_value(
    pos: dict[str, Any], leg_prices: dict[str, float]
) -> float | None:
    """
    Compute the signed net value per share of a multi-leg position.
    net = sum(buy_prices) - sum(sell_prices).
    Negative net = spread BuyBack cost (credit position still alive).
    Positive net = spread sale value (debit position still alive).
    Returns None if ANY leg is missing a quote — callers must NOT evaluate
    exits on partial data (a missing expensive leg collapses net toward 0 and
    triggers phantom "profit target" exits).
    """
    net = 0.0
    for leg in pos.get("legs", []):
        sym = leg.get("symbol", "")
        if sym not in leg_prices:
            return None
        price = leg_prices[sym]
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
    Handles both debit spreads (entry_price > 0) and credit spreads (entry_price < 0).
    """
    legs = position.get("legs", [])
    if not legs:
        return False, ""

    # DTE exit: close at threshold DTE regardless of P&L
    for leg in legs:
        dte = _calculate_dte(leg.get("expiration", ""))
        if dte <= config.DTE_EXIT_THRESHOLD:
            return True, f"DTE {dte} <= threshold {config.DTE_EXIT_THRESHOLD}"

    entry_price = position.get("entry_price", 0)
    current_price = position.get("current_price", entry_price)
    quantity = position.get("quantity", 1)
    strategy = position.get("strategy", "")
    is_credit = _is_credit_position(position)

    # Credit spread logic (bull_put_spread, iron_condor, bear_call_spread)
    # Classification comes from leg sides/strategy, NOT the entry_price sign.
    # Profit: buyback cost drops below (1 - PROFIT_TARGET_PCT) of credit.
    # Loss: buyback cost rises above (1 + STOP_LOSS_PCT) of credit.
    if is_credit:
        credit_received = abs(entry_price) * quantity * 100
        current_value = abs(current_price) * quantity * 100
        # Profit target: close if we can buy back for less than 50% of credit
        if current_value <= credit_received * (1 - config.PROFIT_TARGET_PCT):
            profit = credit_received - current_value
            return True, (
                f"Profit target: spread value ${current_value:.0f} < "
                f"{(1-config.PROFIT_TARGET_PCT)*100:.0f}% of credit (${credit_received:.0f})"
            )
        # Stop loss: close if buyback cost exceeds (1 + STOP_LOSS_PCT) of credit
        if current_value >= credit_received * (1 + config.STOP_LOSS_PCT):
            loss = current_value - credit_received
            return True, (
                f"Stop loss: loss ${loss:.0f} > {config.STOP_LOSS_PCT*100:.0f}% "
                f"of credit (${credit_received:.0f})"
            )

    # Debit spread logic (bull_call_spread, bear_put_spread, long_call/put)
    # Entry price is positive (debit paid), current price is positive (spread value)
    # Profit: spread value increases → current_price > entry_price
    # Loss: spread value decreases → current_price < entry_price
    elif entry_price > 0:
        debit_paid = entry_price * quantity * 100
        current_value = current_price * quantity * 100
        # Hard loss exit: close if loss > 100% of debit paid
        loss = debit_paid - current_value
        if loss >= debit_paid * config.HARD_EXIT_LOSS_PCT:
            return True, (
                f"Loss ${loss:.0f} >= {config.HARD_EXIT_LOSS_PCT*100:.0f}% "
                f"of debit paid (${debit_paid:.0f})"
            )
        # Profit target exit: close if profit > 50% of debit paid
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
        import re as _re
        import requests
        headers = {
            "APCA-API-KEY-ID": config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY,
        }
        # Extract root symbol: "GOOGL260812P00335000" → "GOOGL", "SPY260820C00736000" → "SPY"
        match = _re.match(r'^([A-Za-z]+)', contract_symbol)
        underlying = match.group(1) if match else contract_symbol
        url = f"{config.ALPACA_DATA_URL}/v1beta1/options/snapshots/{underlying}"
        resp = requests.get(url, headers=headers, params={"feed": "indicative"}, timeout=30)
        resp.raise_for_status()
        snapshots = resp.json().get("snapshots", {})
        snap = snapshots.get(contract_symbol, {})
        quote = snap.get("latestQuote", {})
        bid = quote.get("bp", 0)
        ask = quote.get("ap", 0)
        return (bid + ask) / 2 if (bid + ask) > 0 else 0
    except Exception as exc:
        logger.warning("Failed to fetch price for %s: %s", contract_symbol, exc)
        return 0


def _close_all_legs_for_fill(
    pos: dict[str, Any],
    leg_prices: dict[str, float] | None,
) -> tuple[float, float, str, float]:
    """
    Close every leg at market and reconcile realized P&L from actual fills.

    Returns (realized_pnl, exit_net_per_share, basis, slippage) where basis is
    "fills" when every leg filled, else "mid_estimate" fallback.

    Universal cashflow math (works for credit AND debit positions):
        realized = Σ leg_close_cf − entry_cf
    A buy-to-close leg is negative cashflow, a sell-to-close leg positive.
    slippage = fill-based realized − mid-based estimate (negative = fills
    were worse than the mids we decided on).
    """
    legs = pos.get("legs", [])
    entry_price = pos.get("entry_price", 0)
    quantity = pos.get("quantity", 1) or 1
    entry_cf = entry_price * quantity * 100

    total_close_cf = 0.0
    all_filled = True
    for leg in legs:
        contract_sym = leg.get("symbol", "")
        if not contract_sym:
            all_filled = False
            continue
        try:
            # Close all legs (not just buy legs) (fix #2)
            fill_price = executor.close_position_and_poll(contract_sym)
            leg_qty = leg.get("quantity", 1) or 1
            if leg.get("side") == "sell":
                total_close_cf -= fill_price * leg_qty * 100  # buy to close
            else:
                total_close_cf += fill_price * leg_qty * 100  # sell to close
        except Exception as exc:
            logger.error("Failed to close leg %s: %s", contract_sym, exc)
            all_filled = False

    # Mid-based estimate for comparison / fallback
    mid_net: float | None = None
    if leg_prices:
        mid_net = _compute_net_position_value(pos, leg_prices)
    if mid_net is None:
        mid_net = pos.get("current_price", entry_price)
    mid_pnl = round((mid_net - entry_price) * quantity * 100, 2)

    if all_filled and legs:
        fill_pnl = round(total_close_cf - entry_cf, 2)
        exit_net = round(total_close_cf / (quantity * 100), 4)
        slippage = round(fill_pnl - mid_pnl, 2)
        logger.info(
            "Position %s closed on fills: realized $%.2f (mid est $%.2f, slippage $%.2f)",
            pos.get("id", ""), fill_pnl, mid_pnl, slippage,
        )
        return fill_pnl, exit_net, "fills", slippage

    logger.warning(
        "Position %s: %s — booking mid-estimate $%.2f instead of fills",
        pos.get("id", ""),
        "no legs to close" if not legs else "some legs did not fill",
        mid_pnl,
    )
    return mid_pnl, round(mid_net, 4), "mid_estimate", 0.0


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

        entry_price = pos.get("entry_price", 0)
        quantity = pos.get("quantity", 1)

        # DTE exit needs no quotes — evaluate it first so near-expiry
        # positions are closed even when market data is unavailable.
        dte_values = [_calculate_dte(leg.get("expiration", "")) for leg in legs]
        min_dte = min(dte_values) if dte_values else 999
        dte_breach = min_dte <= config.DTE_EXIT_THRESHOLD

        # Fetch current price for each leg
        leg_prices: dict[str, float] = {}
        for leg in legs:
            contract_sym = leg.get("symbol", "")
            if contract_sym:
                price = _get_current_price_for_contract(contract_sym)
                if price > 0:
                    leg_prices[contract_sym] = price

        # Net value is None unless EVERY leg returned a quote. Never evaluate
        # P&L exits on partial data — a missing expensive leg collapses the net
        # toward 0 and triggers phantom "profit target" exits.
        net_value = _compute_net_position_value(pos, leg_prices)
        quotes_complete = net_value is not None

        if not quotes_complete and not dte_breach:
            missing = [l.get("symbol", "?") for l in legs if l.get("symbol", "") not in leg_prices]
            logger.warning(
                "Position %s missing quotes for %d/%d legs %s — "
                "skipping P&L exit evaluation (keeping last known values)",
                position_id, len(missing), len(legs), missing,
            )

        if quotes_complete:
            assert net_value is not None
            pos["current_price"] = net_value

            # Unified P&L: profit = (current net - entry net) * qty * 100.
            # Correct for BOTH debit (entry > 0) and credit (entry < 0) spreads.
            # The old credit branch had the operands flipped and booked every
            # winner as a loss of exactly the credit received.
            pos["unrealized_pnl"] = round(
                (net_value - entry_price) * quantity * 100, 2
            )

        # Check hard exit rules
        if dte_breach:
            should_exit, reason = True, (
                f"DTE {min_dte} <= threshold {config.DTE_EXIT_THRESHOLD}"
            )
        elif quotes_complete:
            should_exit, reason = _check_hard_exit(pos)
        else:
            should_exit, reason = False, ""

        if should_exit:
            logger.info(
                "Hard exit triggered for %s: %s", position_id, reason
            )
            try:
                # Save daily entry BEFORE removing position (fix #14)
                entry = state_manager.get_today_entry()
                if entry is None:
                    entry = state_manager.create_today_entry()

                # Close all legs (not just buy legs) (fix #2), polling for
                # actual fills so realized P&L reflects reality, not mid quotes.
                fill_pnl, exit_net, exit_basis, slippage = _close_all_legs_for_fill(
                    pos, leg_prices if quotes_complete else None
                )

                # Record closed position
                closed_positions.append({
                    **pos,
                    "exit_reason": reason,
                    "exit_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    "exit_price": exit_net,
                    "exit_basis": exit_basis,
                    "exit_slippage": slippage,
                    "realized_pnl": fill_pnl,
                })

                # Update daily log BEFORE removing position
                entry["trades_closed"].append({
                    "symbol": underlying,
                    "strategy": pos.get("strategy", ""),
                    "entry_price": entry_price,
                    "exit_price": exit_net,
                    "realized_pnl": fill_pnl,
                    "reason": f"{reason} [{exit_basis}]",
                })
                entry["realized_pnl"] += fill_pnl
                state_manager.save_today_entry(entry)

                # NOW remove from state
                state_manager.remove_position(position_id)

            except Exception as exc:
                logger.error("Failed to close position %s: %s", position_id, exc)
        else:
            # Add to exit recommendations for LLM evaluation.
            # When quotes were incomplete, current_price/unrealized_pnl below
            # are the last known values (NOT fresh marks).
            exit_recommendations.append({
                "id": position_id,
                "underlying": underlying,
                "strategy": pos.get("strategy", ""),
                "entry_price": entry_price,
                "current_price": pos.get("current_price", entry_price),
                "unrealized_pnl": pos.get("unrealized_pnl", 0),
                "dte": min_dte,
                "quotes_stale": not quotes_complete,
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    recommendations = check_exits()
    if recommendations:
        print(f"{len(recommendations)} positions sent to LLM for exit evaluation")
    else:
        print("No exit recommendations")
