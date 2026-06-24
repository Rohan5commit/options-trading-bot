"""
Risk Manager. Validates LLM trade decisions against hard safety rails
before orders are submitted. Logs all rejections with reasons.
"""
import logging
from datetime import datetime, timezone
from typing import Any

import config
import data_fetcher
import state_manager

logger = logging.getLogger(__name__)


class RiskRejection:
    """Represents a rejected trade with reason."""

    def __init__(self, decision: dict[str, Any], reason: str):
        self.decision = decision
        self.reason = reason
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.decision.get("underlying", ""),
            "strategy": self.decision.get("strategy", ""),
            "action": self.decision.get("action", ""),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


def _check_daily_loss_limit() -> bool:
    """
    Check if daily P&L has exceeded the loss limit.
    Returns True if trading should be HALTED.
    """
    equity = data_fetcher.get_account_equity()
    if equity <= 0:
        logger.warning("Cannot determine account equity — halting trades")
        return True

    entry = state_manager.get_today_entry()
    if entry is None:
        return False

    realized = entry.get("realized_pnl", 0)
    unrealized = entry.get("unrealized_pnl", 0)
    total_pnl_pct = (realized + unrealized) / equity if equity > 0 else 0

    if total_pnl_pct < config.DAILY_LOSS_LIMIT_PCT:
        logger.warning(
            "Daily loss limit breached: %.2f%% (limit: %.2f%%)",
            total_pnl_pct * 100, config.DAILY_LOSS_LIMIT_PCT * 100,
        )
        return True

    return False


def _check_position_limit() -> bool:
    """Check if max concurrent positions is reached."""
    positions = state_manager.load_positions()
    return len(positions) >= config.MAX_OPEN_POSITIONS


def _check_position_sizing(decision: dict[str, Any], equity: float) -> str | None:
    """
    Validate that the proposed position size is within limits.
    Returns rejection reason or None if OK.
    """
    if equity <= 0:
        return "Cannot determine account equity"

    # Estimate total debit/credit from legs
    total_cost = 0
    for leg in decision.get("legs", []):
        mid = leg.get("mid", 0)
        qty = leg.get("quantity", 1)
        if leg.get("side") == "buy":
            total_cost += mid * qty * 100  # options multiplier = 100
        else:
            total_cost -= mid * qty * 100

    max_allowed = equity * config.MAX_POSITION_PCT
    if total_cost > max_allowed:
        return (
            f"Position size ${total_cost:.0f} exceeds {config.MAX_POSITION_PCT*100:.0f}% "
            f"limit (${max_allowed:.0f}) of equity ${equity:.0f}"
        )
    return None


def _validate_legs(decision: dict[str, Any]) -> list[str]:
    """
    Validate individual legs against contract-level filters.
    Returns list of rejection reasons (empty if all legs pass).
    """
    rejections: list[str] = []
    for leg in decision.get("legs", []):
        spread_pct = leg.get("spread_pct", 0)
        open_interest = leg.get("open_interest", 0)
        dte = leg.get("dte", 30)
        sym = leg.get("symbol", "unknown")

        if spread_pct > config.MAX_BID_ASK_SPREAD_PCT:
            rejections.append(
                f"Contract {sym}: bid-ask spread {spread_pct*100:.1f}% exceeds "
                f"{config.MAX_BID_ASK_SPREAD_PCT*100:.0f}% limit"
            )

        if open_interest < config.MIN_OPEN_INTEREST:
            rejections.append(
                f"Contract {sym}: open interest {open_interest} below "
                f"minimum {config.MIN_OPEN_INTEREST}"
            )

        if dte < config.MIN_DTE or dte > config.MAX_DTE:
            rejections.append(
                f"Contract {sym}: DTE {dte} outside range [{config.MIN_DTE}, {config.MAX_DTE}]"
            )

    return rejections


def validate(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Validate a list of LLM trade decisions against all risk rules.
    Returns only the decisions that pass all checks.
    """
    rejections: list[RiskRejection] = []
    approved: list[dict[str, Any]] = []

    # Global checks
    daily_loss_halted = _check_daily_loss_limit()
    if daily_loss_halted:
        logger.warning("Daily loss limit reached — rejecting ALL new trades")
        for d in decisions:
            rejections.append(RiskRejection(d, "Daily loss limit breached — all trades halted"))
        _log_rejections(rejections)
        return []

    at_position_limit = _check_position_limit()
    if at_position_limit:
        logger.warning("Max open positions reached — rejecting ALL new trades")
        for d in decisions:
            rejections.append(RiskRejection(d, f"Max {config.MAX_OPEN_POSITIONS} open positions reached"))
        _log_rejections(rejections)
        return []

    equity = data_fetcher.get_account_equity()

    for decision in decisions:
        # HOLD decisions pass through
        if decision.get("action") == "HOLD":
            approved.append(decision)
            continue

        # Position sizing check
        sizing_rejection = _check_position_sizing(decision, equity)
        if sizing_rejection:
            rejections.append(RiskRejection(decision, sizing_rejection))
            continue

        # Per-leg validation
        leg_rejections = _validate_legs(decision)
        if leg_rejections:
            for reason in leg_rejections:
                rejections.append(RiskRejection(decision, reason))
            continue

        # All checks passed
        approved.append(decision)
        logger.info(
            "Approved: %s %s on %s (confidence: %.2f)",
            decision["action"], decision["strategy"],
            decision["underlying"], decision.get("confidence", 0),
        )

    _log_rejections(rejections)

    logger.info(
        "Risk check complete: %d approved, %d rejected out of %d total",
        len(approved), len(rejections), len(decisions),
    )
    return approved


def _log_rejections(rejections: list[RiskRejection]) -> None:
    """Log rejections and append them to today's daily entry."""
    for r in rejections:
        logger.warning("REJECTED: %s — %s", r.decision.get("underlying", ""), r.reason)

    # Persist rejections to daily log
    entry = state_manager.get_today_entry()
    if entry is None:
        entry = state_manager.create_today_entry()
    entry["risk_rejections"].extend([r.to_dict() for r in rejections])
    state_manager.save_today_entry(entry)
