# Options Trading Bot

LLM-driven autonomous US options trading bot using Alpaca Broker API (paper trading), Modal serverless GPU inference, and two-phase training on Lightning.ai + Modal.

## Architecture

```
GitHub Actions (Daily 9:35 ET)
    → position_monitor.py   (check exits)
    → data_fetcher.py       (market context)
    → llm_trader.py         (LLM decisions via Modal)
    → risk_manager.py       (validate)
    → executor.py           (place orders)
    → email_reporter.py     (send summary)

Modal GPU (A10G)             → Llama-3-8B + LoRA inference + Phase 2 training
Lightning.ai (L4)            → Phase 1 training ($0.48/hr)
Alpaca Paper Trading API     → Options chain, orders, positions
yfinance + Polygon.io        → Price data, IV, news, earnings
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

### 3. Training (Two-Phase)

**Total budget: $45** ($15 Lightning + $30 Modal)

#### Phase 1: Lightning.ai (L4 @ $0.48/hr)
1. Create a free [Lightning.ai](https://lightning.ai) account
2. Open a Studio with **L4 GPU** ($0.48/hr)
3. In the Studio terminal:
   ```bash
   git clone https://github.com/Rohan5commit/options-trading-bot.git
   cd options-trading-bot
   export HF_TOKEN=your_hf_token
   bash finetune/run_lightning.sh
   ```
4. Phase 1 uses $15 = 31.25 hours of training
5. Checkpoints save to HuggingFace Hub every 1000 steps

#### Phase 2: Modal (A10G @ $1.10/hr)
1. After Phase 1 budget exhausted, sign up at [Modal](https://modal.com/)
2. Install and authenticate:
   ```bash
   pip install modal && modal setup
   ```
3. Create secret:
   ```bash
   modal secret create options-training \
     HF_TOKEN=your_hf_token \
     HF_CHECKPOINT_REPO=Rohan556/options-llm-checkpoints
   ```
4. Deploy training:
   ```bash
   modal deploy finetune/modal_train.py
   ```
5. Phase 2 uses $30 = 27.3 hours of training

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
│   └── retrain.yml          # Retrain workflow
├── finetune/
│   ├── build_dataset.py     # Training data (160K examples)
│   ├── train.py             # LoRA fine-tuning
│   ├── run_lightning.sh     # Phase 1: Lightning.ai script
│   ├── modal_train.py       # Phase 2: Modal training script
│   ├── lightning_helper.py  # Status tracking & instructions
│   └── Dockerfile           # Container image
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

**Training plan:**
| Phase | Platform | GPU | Budget | Hours |
|-------|----------|-----|--------|-------|
| Phase 1 | Lightning.ai | L4 @ $0.48/hr | $15 | 31.25 |
| Phase 2 | Modal | A10G @ $1.10/hr | $30 | 27.3 |
| **Total** | | | **$45** | **58.5** |

**Inference cost:** ~$0.30/month on Modal A10G

### Budget & Checkpoints

- Checkpoints save to HuggingFace Hub every 1000 steps
- Training resumes automatically across platforms
- No data is lost when switching from Lightning to Modal

```bash
# Check training status
python finetune/lightning_helper.py status

# Get Phase 2 instructions
python finetune/lightning_helper.py phase2

# Reset for platform switch
python finetune/lightning_helper.py reset
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
