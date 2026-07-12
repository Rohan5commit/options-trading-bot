# Options Trading Bot

LLM-driven autonomous US options trading bot using Alpaca Broker API (paper trading), Modal serverless GPU inference, and Lightning.ai for model training.

## Architecture

```
GitHub Actions (Daily 9:35 ET)
    → position_monitor.py   (check exits)
    → data_fetcher.py       (market context)
    → llm_trader.py         (LLM decisions via Modal)
    → risk_manager.py       (validate)
    → executor.py           (place orders)
    → email_reporter.py     (send summary)

Modal GPU (L4)              → Llama-3-8B + LoRA inference
Lightning.ai (A10G)         → LoRA fine-tuning ($0.71/hr)
Alpaca Paper Trading API    → Options chain, orders, positions
yfinance + Polygon.io       → Price data, IV, news, earnings
```

## Setup

### 1. Alpaca Paper Trading
1. Create an account at [Alpaca Markets](https://alpaca.markets/)
2. Enable paper trading and options trading (Level 3 for multi-leg)
3. Generate API keys

### 2. Modal (Inference)
1. Sign up at [Modal](https://modal.com/)
2. `pip install modal && modal setup`
3. Deploy the inference endpoint:
   ```bash
   modal deploy modal_inference.py
   ```

### 3. Lightning.ai (Training)
1. Create a free [Lightning.ai](https://lightning.ai) account
2. Open a Studio with **A10G GPU** ($0.71/hr)
3. In the Studio terminal:
   ```bash
   git clone https://github.com/Rohan5commit/options-trading-bot.git
   cd options-trading-bot
   export HF_TOKEN=your_hf_token
   export HF_CHECKPOINT_REPO=Rohan5commit/options-llm-checkpoints
   bash finetune/run_lightning.sh
   ```
4. Training auto-checkpoints to HuggingFace Hub every 1000 steps
5. When budget runs out, open new Lightning.ai account and resume (checkpoints persist on Hub)

### 4. GitHub Secrets
Add these secrets to your GitHub repository:

| Secret | Description |
|--------|-------------|
| `ALPACA_API_KEY` | Alpaca paper trading API key |
| `ALPACA_SECRET_KEY` | Alpaca paper trading secret key |
| `MODAL_TOKEN_ID` | Modal auth token |
| `MODAL_TOKEN_SECRET` | Modal auth secret |
| `HF_TOKEN` | HuggingFace token |
| `POLYGON_API_KEY` | Polygon.io API key (free tier) |
| `EMAIL_USER` | Gmail for reports |
| `EMAIL_PASS` | Gmail app password |
| `EMAIL_RECIPIENT` | Report recipient |

### 5. Local Development
```bash
pip install -r requirements.txt
python main.py
```

## File Structure

```
├── .github/workflows/
│   ├── trade.yml            # Daily trading cron
│   └── retrain.yml          # Retrain workflow (Lightning.ai)
├── finetune/
│   ├── build_dataset.py     # Training data construction (100K examples)
│   ├── train.py             # LoRA fine-tuning (runs on Lightning.ai)
│   ├── run_lightning.sh     # Lightning.ai training script
│   ├── lightning_helper.py  # Account switching & budget tracking
│   └── Dockerfile           # Lightning.ai container image
├── state/
│   ├── positions.json       # Open positions
│   └── daily_log.json       # Trade history
├── config.py                # All configuration
├── data_fetcher.py          # Market data pipeline
├── llm_trader.py            # LLM decision engine
├── risk_manager.py          # Risk validation
├── executor.py              # Order execution
├── position_monitor.py      # Exit logic
├── state_manager.py         # JSON state persistence
├── email_reporter.py        # Daily email summary
├── modal_inference.py       # Modal GPU endpoint
├── main.py                  # Pipeline orchestrator
└── requirements.txt
```

## Risk Rules

| Rule | Threshold |
|------|-----------|
| Max single position | 20% of account equity |
| Max concurrent positions | 5 |
| Daily loss limit | -15% of starting equity |
| Bid-ask spread filter | > 10% of mid price |
| Minimum open interest | 100 |
| Hard exit (loss) | > 100% of debit paid |
| DTE exit | Close at 1 DTE |

## Strategies

- Long calls / puts
- Bull call spreads
- Bear put spreads
- Iron condors
- Straddles / strangles
- Calendar spreads

## Fine-Tuning

The bot uses a LoRA-adapted Llama-3-8B model trained on:
- 160K synthetic options trading scenarios
- 12 market regimes (bull/bear/sideways × high/low/normal IV × earnings/no-earnings)
- Black-Scholes Greeks, IV term structure
- 10 varied instruction templates

**Training cost:** ~$40 on Lightning.ai A10G ($0.71/hr × 56hrs)
**Inference cost:** ~$0.30/month on Modal L4

### Budget & Account Switching

Total budget: $45 ($40 training + $5 buffer).

- Checkpoints save to HuggingFace Hub every 1000 steps
- When one account's budget runs out, open a new account
- Training resumes from latest Hub checkpoint automatically

```bash
# Check training status and budget
python finetune/lightning_helper.py status

# Reset for new account
python finetune/lightning_helper.py reset

# View checkpoints on Hub
python finetune/lightning_helper.py checkpoints
```

## Email Reports

Daily summary includes:
- Trades opened with LLM reasoning
- Trades closed with P&L
- Open positions with unrealized P&L
- Risk rejections
- LLM confidence scores
- Account equity

## Paper Trading Only

This bot operates exclusively on Alpaca's paper trading environment. No real money is used. The `ALPACA_PAPER=True` flag is hardcoded in `config.py`.
