"""
Email Reporter. Sends a daily summary email after each trading run.
Uses Gmail SMTP with app password authentication.
"""
import logging
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import config
import state_manager

logger = logging.getLogger(__name__)


def _format_currency(value: float) -> str:
    """Format a float as a currency string."""
    if value >= 0:
        return f"${value:,.2f}"
    return f"-${abs(value):,.2f}"


def _build_html_table(rows: list[list[str]], headers: list[str]) -> str:
    """Build an HTML table from rows and headers."""
    html = '<table style="border-collapse:collapse;width:100%;margin:10px 0">'
    html += "<tr>"
    for h in headers:
        html += f'<th style="border:1px solid #ddd;padding:8px;background:#f4f4f4;text-align:left">{h}</th>'
    html += "</tr>"
    for row in rows:
        html += "<tr>"
        for cell in row:
            html += f'<td style="border:1px solid #ddd;padding:8px">{cell}</td>'
        html += "</tr>"
    html += "</table>"
    return html


def _build_email_body() -> str:
    """Build the full HTML email body from today's daily log entry."""
    entry = state_manager.get_today_entry()
    if entry is None:
        entry = state_manager.create_today_entry()

    date_str = entry.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    realized = entry.get("realized_pnl", 0)
    unrealized = entry.get("unrealized_pnl", 0)
    total = realized + unrealized

    # Determine P&L color
    pnl_color = "#2e7d32" if total >= 0 else "#c62828"

    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; color: #333; }}
            h1 {{ color: #1a237e; border-bottom: 2px solid #1a237e; padding-bottom: 10px; }}
            h2 {{ color: #283593; margin-top: 20px; }}
            .summary-box {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0; }}
            .pnl-positive {{ color: #2e7d32; font-weight: bold; }}
            .pnl-negative {{ color: #c62828; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Options Trading Bot — Daily Report</h1>
        <p>Date: <strong>{date_str}</strong></p>

        <div class="summary-box">
            <h2>Daily P&L Summary</h2>
            <p>Realized P&L: <span class="{'pnl-positive' if realized >= 0 else 'pnl-negative'}">{_format_currency(realized)}</span></p>
            <p>Unrealized P&L: <span class="{'pnl-positive' if unrealized >= 0 else 'pnl-negative'}">{_format_currency(unrealized)}</span></p>
            <p><strong>Total P&L: <span style="color:{pnl_color}">{_format_currency(total)}</span></strong></p>
            <p>Account Equity: <strong>{_format_currency(entry.get('account_equity', 0))}</strong></p>
        </div>
    """

    # Trades opened today
    opened = entry.get("trades_opened", [])
    if opened:
        rows = []
        for t in opened:
            rows.append([
                t.get("symbol", ""),
                t.get("strategy", ""),
                t.get("action", ""),
                _format_currency(t.get("entry_price", 0)),
                f"{t.get('confidence', 0):.2f}",
                t.get("reasoning", "")[:100],
            ])
        html += "<h2>Trades Opened Today</h2>"
        html += _build_html_table(
            rows,
            ["Symbol", "Strategy", "Action", "Entry Price", "Confidence", "Reasoning"],
        )
    else:
        html += "<h2>Trades Opened Today</h2><p>No trades opened.</p>"

    # Trades closed today
    closed = entry.get("trades_closed", [])
    if closed:
        rows = []
        for t in closed:
            pnl = t.get("realized_pnl", 0)
            pnl_class = "pnl-positive" if pnl >= 0 else "pnl-negative"
            rows.append([
                t.get("symbol", ""),
                t.get("strategy", ""),
                _format_currency(t.get("entry_price", 0)),
                _format_currency(t.get("exit_price", 0)),
                f'<span class="{pnl_class}">{_format_currency(pnl)}</span>',
                t.get("reason", ""),
            ])
        html += "<h2>Trades Closed Today</h2>"
        html += _build_html_table(
            rows,
            ["Symbol", "Strategy", "Entry", "Exit", "P&L", "Exit Reason"],
        )
    else:
        html += "<h2>Trades Closed Today</h2><p>No trades closed.</p>"

    # Risk rejections
    rejections = entry.get("risk_rejections", [])
    if rejections:
        rows = []
        for r in rejections:
            rows.append([
                r.get("symbol", ""),
                r.get("strategy", ""),
                r.get("reason", ""),
            ])
        html += "<h2>Risk Manager Rejections</h2>"
        html += _build_html_table(rows, ["Symbol", "Strategy", "Reason"])
    else:
        html += "<h2>Risk Manager Rejections</h2><p>No rejections.</p>"

    # LLM confidence scores
    scores = entry.get("llm_confidence_scores", [])
    if scores:
        avg_conf = sum(scores) / len(scores)
        html += f"""
        <h2>LLM Confidence Scores</h2>
        <p>Scores: {', '.join(f'{s:.2f}' for s in scores)}</p>
        <p>Average confidence: <strong>{avg_conf:.2f}</strong></p>
        """

    html += """
        <hr style="margin-top:30px">
        <p style="color:#666;font-size:12px">
            Generated by Options Trading Bot — Alpaca Paper Trading<br>
            This is a paper trading report. No real money is at risk.
        </p>
    </body>
    </html>
    """

    return html


def send_daily_summary() -> bool:
    """
    Build and send the daily summary email.
    Returns True if sent successfully, False otherwise.
    """
    if not all([config.EMAIL_USER, config.EMAIL_PASS, config.EMAIL_RECIPIENT]):
        logger.warning("Email credentials not configured — skipping email report")
        return False

    try:
        html_body = _build_email_body()
        entry = state_manager.get_today_entry()
        date_str = entry.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")) if entry else "unknown"

        # Build MIME message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Options Bot Daily Report — {date_str}"
        msg["From"] = config.EMAIL_USER
        msg["To"] = config.EMAIL_RECIPIENT
        msg.attach(MIMEText(html_body, "html"))

        # Send via Gmail SMTP
        with smtplib.SMTP(config.EMAIL_SMTP_HOST, config.EMAIL_SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(config.EMAIL_USER, config.EMAIL_PASS)
            server.sendmail(
                config.EMAIL_USER,
                config.EMAIL_RECIPIENT,
                msg.as_string(),
            )

        logger.info("Daily summary email sent to %s", config.EMAIL_RECIPIENT)
        return True

    except smtplib.SMTPAuthenticationError as exc:
        logger.error("SMTP authentication failed: %s", exc)
        return False
    except smtplib.SMTPException as exc:
        logger.error("SMTP error: %s", exc)
        return False
    except Exception as exc:
        logger.error("Failed to send email: %s", exc)
        return False
