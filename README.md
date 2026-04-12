---
title: Forex Trading OpenEnv
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# ForexTrading OpenEnv Environment

This is a real-world task environment simulating Forex Trading for AI agents. It complies strictly with the OpenEnv specification and is packaged for simple deployment on Hugging Face Spaces.

## Motivation & Real-World Utility
Forex trading involves constant monitoring of technical indicators, managing drawdowns, and executing trades (Buy, Sell, Hold, Close) to maximize capital. This environment tests an agent's ability to act as a profitable proprietary trader, combining pattern recognition with strict risk management.

## Observation Space
The environment returns a structured `Observation` model:
- `current_equity` (float): The account balance.
- `unrealized_pnl` (float): PnL of an open position.
- `open_trade` (str): "NONE", "BUY", or "SELL".
- `features` (Dict): Real-time calculated technical indicators (e.g., `rsi_14`, `ma_spread`, etc.).

## Action Space
Agents submit a discrete action via the `Action` model:
- `0`: **HOLD**
- `1`: **BUY**
- `2`: **SELL**
- `3`: **CLOSE**

## Tasks & Difficulties
This environment contains 3 specific tasks evaluated by deterministic graders:
1. **`first_blood` (Easy)**: The agent must complete a single trade with a net positive profit.
2. **`consistent_gainer` (Medium)**: The agent must end the episode with an overall account growth of at least 0.5%.
3. **`risk_manager` (Hard)**: The agent must achieve > 1.0% overall profit margin while rigorously keeping maximum drawdown below 1.0%.

## Local Setup & Usage

To build and run the Dockerized HuggingFace Space environment:

```bash
docker build -t forex-env .
docker run -p 7860:7860 forex-env
```

If the external market CSV is not present, the environment falls back to a
deterministic synthetic EUR/USD-like dataset so the repository can run
out-of-the-box for local validation.

### Running the Inference Script

Ensure the environment requires API access variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hf_or_openai_token_here"
```

Execute the validation script:

```bash
python inference.py
```
