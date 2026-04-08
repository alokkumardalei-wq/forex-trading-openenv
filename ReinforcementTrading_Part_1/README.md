

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

## OpenEnv API

The Space exposes the required OpenEnv endpoints:

- `POST /reset` resets the episode and returns the initial observation.
- `POST /step` applies one action and returns the next observation, reward, done flag, and info payload.
- `GET /state` returns the current episode state, including step count and cumulative reward.

The interactive API docs are available at `/docs` once the Space is running.

## Tasks & Difficulties
This environment contains 3 specific tasks evaluated by deterministic graders:
1. **`first_blood` (Easy)**: The agent must complete a single trade with a net positive profit.
2. **`consistent_gainer` (Medium)**: The agent must end the episode with an overall account growth of at least 0.5%.
3. **`risk_manager` (Hard)**: The agent must achieve > 1.0% overall profit margin while rigorously keeping maximum drawdown below 1.0%.

The task definitions are also stored in `openenv.yaml` at the project root.
Task scores are intentionally fractional and stay strictly inside `(0, 1)` so the grader can measure partial progress without hitting endpoint values.

## Local Setup & Usage

To build and run the Dockerized HuggingFace Space environment:

```bash
docker build -t forex-env .
docker run -p 7860:7860 forex-env
```

If the external market CSV is not present, the environment falls back to a
deterministic synthetic EUR/USD-like dataset so the repository can run
out-of-the-box for local validation.

## Hugging Face Space Setup

This project is packaged as a Docker Space and listens on port `7860`.

Add these Space settings:

- Secret `HF_TOKEN`
- Variable `API_BASE_URL=https://router.huggingface.co/v1`
- Variable `MODEL_NAME=Qwen/Qwen2.5-72B-Instruct`

`LOCAL_IMAGE_NAME` is optional and only needed if you use a local Docker image workflow.

### Running the Inference Script

Ensure the environment requires API access variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_hugging_face_token_here"
```

Execute the inference script:

```bash
python inference.py
```

By default, `inference.py` evaluates the `first_blood` task.
To run a different task locally, set `TASK_NAME` before launching:

```bash
export TASK_NAME="consistent_gainer"
python inference.py
```

To run all three tasks in one local session, set `RUN_ALL_TASKS=1`.

## Baseline Inference Output

`inference.py` is the baseline agent used for validation. It uses the OpenAI client and emits structured stdout in the required format:

- `[START] task=<task_name> env=<benchmark> model=<model_name>`
- `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
- `[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>`

Scores and rewards are normalized to the `0.0` to `1.0` range and formatted to 2 decimal places.

## Submission Validation

Run the local validator before submitting:

```bash
bash validate-submission.sh https://alok245-forex-trading-openenv.hf.space .
```

The validator checks:

- the live Space `/reset` endpoint
- the Docker build
- `openenv validate`
