import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

logging.basicConfig(level=logging.DEBUG, format="[DEBUG] %(message)s")
logger = logging.getLogger(__name__)

# Required submission variables.
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional helpers for local or alternate launch flows.
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK", "forex_trading")
TASK_NAME = os.getenv("TASK_NAME", "first_blood")
RUN_ALL_TASKS = os.getenv("RUN_ALL_TASKS", "0").lower() in {"1", "true", "yes"}
BASE_URL = os.getenv("BASE_URL", "http://localhost:7860")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

DEFAULT_ACTION = "0"
TASK_SEQUENCE = ["first_blood", "consistent_gainer", "risk_manager"]


def compact_text(value: Optional[str]) -> str:
    if not value:
        return "null"
    return str(value).replace("\n", " ").replace("\r", " ")


def format_system_prompt() -> str:
    return """You are a highly skilled Forex trading agent operating in a strictly controlled environment.
Your objective is to maximize profit acting in a simulated market based on historical data points.
You must output ONLY an integer corresponding to your chosen action space:
0: HOLD (Do nothing)
1: BUY (Go Long)
2: SELL (Go Short)
3: CLOSE (Close open positions)

Return JUST the integer. Nothing else."""


def format_user_prompt(step: int, obs: Dict[str, Any], last_reward: float, history: List[Dict[str, Any]]) -> str:
    msg = f"Step {step}:\n"
    msg += f"Observation: {json.dumps(obs, indent=2)}\n"
    msg += f"Last Reward: {last_reward}\n"
    if history:
        msg += "Previous steps:\n"
        for item in history[-4:]:
            msg += f"- step={item['step']} action={item['action']} reward={item['reward']:.2f}\n"
    msg += "\nWhat is your action (0, 1, 2, or 3)?"
    return msg


def extract_action(text: str) -> str:
    text = (text or "").strip()
    if text in {"0", "1", "2", "3"}:
        return text

    match = re.search(r"\b([0-3])\b", text)
    if match:
        return match.group(1)

    logger.debug("Warning: Model gave non-standard output %r. Defaulting to 0.", text)
    return DEFAULT_ACTION


async def get_model_action(
    client: AsyncOpenAI,
    step: int,
    obs: Dict[str, Any],
    last_reward: float,
    history: List[Dict[str, Any]],
) -> str:
    if not HF_TOKEN:
        logger.debug("No API key found in environment variables (HF_TOKEN).")
        return DEFAULT_ACTION

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": format_system_prompt()},
                {"role": "user", "content": format_user_prompt(step, obs, last_reward, history)},
            ],
            temperature=0.1,
            max_tokens=10,
        )
        content = response.choices[0].message.content or ""
        return extract_action(content)
    except Exception as exc:
        logger.debug("Model request failed: %s", exc)
        return DEFAULT_ACTION


async def evaluate_task(client: AsyncOpenAI, task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    steps_taken = 0
    rewards_history: List[float] = []
    total_score = 0.0
    success = False
    history: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30.0) as http:
        obs: Dict[str, Any] = {}
        last_reward = 0.0
        done = False

        try:
            reset_payload = {"task": task, "episode_id": "eval_1"}
            resp = await http.post(f"{BASE_URL}/reset", json=reset_payload)
            if resp.status_code != 200:
                logger.debug("Reset failed: %s", resp.text)
                print("[END] success=false steps=0 score=0.00 rewards=", flush=True)
                return

            obs = resp.json()

            for step in range(1, MAX_STEPS + 1):
                action_str = await get_model_action(client, step, obs, last_reward, history)
                action_int = int(action_str)

                step_error: Optional[str] = None
                try:
                    s_resp = await http.post(f"{BASE_URL}/step", json={"action": action_int})
                    s_resp.raise_for_status()
                    data = s_resp.json()
                except Exception as step_exc:
                    step_error = str(step_exc)
                    print(
                        f"[STEP] step={step} action={action_str} reward=0.00 done=false error={compact_text(step_error)}",
                        flush=True,
                    )
                    steps_taken = step
                    break

                obs = data.get("observation", obs)
                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))

                info = data.get("info") or {}
                step_error = info.get("last_action_error") or data.get("error")

                rewards_history.append(reward)
                total_score += reward
                steps_taken = step
                last_reward = reward
                history.append({"step": step, "action": action_int, "reward": reward})

                print(
                    f"[STEP] step={step} action={action_str} reward={reward:.2f} done={str(done).lower()} "
                    f"error={compact_text(step_error)}",
                    flush=True,
                )

                if done:
                    break

            success = done and total_score > 0.0
            final_score = min(max(total_score, 0.01), 0.99)
            rewards_str = ",".join(f"{reward:.2f}" for reward in rewards_history)
            print(
                f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.2f} "
                f"rewards={rewards_str}",
                flush=True,
            )

        except Exception as exc:
            logger.debug("Episode failed: %s", exc)
            final_score = min(max(total_score, 0.01), 0.99)
            rewards_str = ",".join(f"{reward:.2f}" for reward in rewards_history)
            print(
                f"[END] success=false steps={steps_taken} score={final_score:.2f} rewards={rewards_str}",
                flush=True,
            )


async def main() -> None:
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy_string",
    )

    tasks = TASK_SEQUENCE if RUN_ALL_TASKS or TASK_NAME.lower() == "all" else [TASK_NAME]
    for task in tasks:
        await evaluate_task(client, task)


if __name__ == "__main__":
    asyncio.run(main())
