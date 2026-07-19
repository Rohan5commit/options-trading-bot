"""
Main orchestrator. Runs the full daily trading pipeline sequentially.
Entry point for GitHub Actions and local execution.
"""
import json
import logging
import sys
from datetime import datetime, timezone

import config
import data_fetcher
import email_reporter
import executor
import llm_trader
import position_monitor
import risk_manager
import state_manager

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """
    Execute the full daily trading pipeline:
    1. Check exits on existing positions
    2. Fetch market data for watchlist
    3. Get LLM trade decisions via Modal
    4. Validate decisions with risk manager
    5. Execute approved trades
    6. Send email summary
    """
    start_time = datetime.now(timezone.utc)
    logger.info("=" * 60)
    logger.info("Options Trading Bot — Daily Pipeline Start")
    logger.info("Time: %s", start_time.isoformat())
    logger.info("=" * 60)

    # Get or create today's log entry (fix #5: don't wipe on re-run)
    entry = state_manager.get_today_entry()
    if entry is None:
        entry = state_manager.create_today_entry()
        state_manager.save_today_entry(entry)

    # Record starting equity (only if not already set)
    if entry.get("account_equity", 0) <= 0:
        try:
            equity = data_fetcher.get_account_equity()
            entry["account_equity"] = equity
            state_manager.save_today_entry(entry)
        except Exception as exc:
            logger.error("Failed to fetch starting equity: %s", exc)
            equity = 0.0
    else:
        equity = entry["account_equity"]

    # ── Step 1: Check exits on existing positions ──────────────────────────
    logger.info("\n--- Step 1: Checking exits on open positions ---")
    exit_recommendations = []
    try:
        exit_recommendations = position_monitor.check_exits()
        logger.info(
            "Exit check complete: %d recommendations for LLM",
            len(exit_recommendations),
        )
    except Exception as exc:
        logger.error("Position monitor failed: %s", exc)

    # ── Step 2: Fetch market data ──────────────────────────────────────────
    logger.info("\n--- Step 2: Fetching market data ---")
    market_contexts = []
    try:
        market_contexts = data_fetcher.build_full_market_context()
        logger.info("Market context built for %d symbols", len(market_contexts))
    except Exception as exc:
        logger.error("Data fetcher failed: %s", exc)

    # ── Step 3: Get LLM trade decisions ────────────────────────────────────
    logger.info("\n--- Step 3: Getting LLM trade decisions ---")
    decisions = []
    try:
        decisions = llm_trader.get_trade_decisions(
            market_contexts, exit_recommendations
        )
        logger.info("Received %d trade decisions from LLM", len(decisions))
    except Exception as exc:
        logger.error("LLM trader failed: %s", exc)
        # Fallback: if LLM fails, skip trading for the day
        logger.warning("LLM failure — skipping trades for today")

    # ── Step 4: Risk validation ────────────────────────────────────────────
    logger.info("\n--- Step 4: Validating with risk manager ---")
    approved_decisions = []
    try:
        approved_decisions = risk_manager.validate(decisions)
        logger.info("Risk manager approved %d / %d decisions", len(approved_decisions), len(decisions))
    except Exception as exc:
        logger.error("Risk manager failed: %s", exc)

    # ── Step 5: Execute orders ─────────────────────────────────────────────
    logger.info("\n--- Step 5: Executing orders ---")
    execution_results = []
    try:
        execution_results = executor.execute_trades(approved_decisions)
        logger.info("Executed %d trades", len(execution_results))
    except Exception as exc:
        logger.error("Executor failed: %s", exc)

    # ── Step 6: Update final equity ────────────────────────────────────────
    logger.info("\n--- Step 6: Updating final equity ---")
    try:
        final_equity = data_fetcher.get_account_equity()
        entry = state_manager.get_today_entry()
        if entry:
            entry["account_equity"] = final_equity
            # Calculate unrealized P&L from open positions
            positions = state_manager.load_positions()
            unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)
            entry["unrealized_pnl"] = unrealized
            entry["total_pnl"] = entry.get("realized_pnl", 0) + unrealized
            state_manager.save_today_entry(entry)
    except Exception as exc:
        logger.error("Failed to update final equity: %s", exc)

    # ── Step 7: Send email report ──────────────────────────────────────────
    logger.info("\n--- Step 7: Sending email report ---")
    try:
        email_reporter.send_daily_summary()
    except Exception as exc:
        logger.error("Email reporter failed: %s", exc)

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete — Summary")
    logger.info("=" * 60)
    logger.info("Elapsed: %.1f seconds", elapsed)
    logger.info("Decisions from LLM: %d", len(decisions))
    logger.info("Approved by risk: %d", len(approved_decisions))
    logger.info("Trades executed: %d", len(execution_results))
    logger.info("Open positions: %d", len(state_manager.load_positions()))

    entry = state_manager.get_today_entry()
    if entry:
        logger.info("Realized P&L: $%.2f", entry.get("realized_pnl", 0))
        logger.info("Unrealized P&L: $%.2f", entry.get("unrealized_pnl", 0))
        logger.info("Total P&L: $%.2f", entry.get("total_pnl", 0))
        logger.info("Risk rejections: %d", len(entry.get("risk_rejections", [])))

    logger.info("=" * 60)


if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as exc:
        logger.critical("Pipeline failed with unhandled exception: %s", exc, exc_info=True)
        sys.exit(1)
