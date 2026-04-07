import os
import sys
import json
import httpx
import logging
import asyncio
from openai import AsyncOpenAI
import traceback

logging.basicConfig(level=logging.DEBUG, format='[DEBUG] %(message)s')
logger = logging.getLogger(__name__)

# Target API config
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
DEBUG_FALLBACK = "0"

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    logger.debug("No API Key found in environment variables (HF_TOKEN or OPENAI_API_KEY).")

MAX_STEPS = 50
PORT = 7860
BASE_URL = f"http://localhost:{PORT}"

def format_system_prompt() -> str:
    return """You are a highly skilled Forex trading agent operating in a strictly controlled environment. 
Your objective is to maximize profit acting in a simulated market based on historical data points.
You must output ONLY an integer corresponding to your chosen action space:
0: HOLD (Do nothing)
1: BUY (Go Long)
2: SELL (Go Short)
3: CLOSE (Close open positions)

Return JUST the integer. Nothing else."""

def format_user_prompt(step: int, obs: dict, last_reward: float, history: list) -> str:
    msg = f"Step {step}:\n"
    msg += f"Observation: {json.dumps(obs, indent=2)}\n"
    msg += f"Last Reward: {last_reward}\n\n"
    msg += "What is your action (0, 1, 2, or 3)?"
    return msg

async def get_model_action(client: AsyncOpenAI, step: int, obs: dict, last_reward: float, history: list) -> str:
    try:
        if not API_KEY:
            raise ValueError("No API Key provided")
            
        sys_prompt = format_system_prompt()
        user_prompt = format_user_prompt(step, obs, last_reward, history)
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=10,
        )
        content = response.choices[0].message.content.strip()
        # Fallback parsing
        if content in ["0", "1", "2", "3"]:
            return content
        else:
            logger.debug(f"Warning: Model gave non-standard output '{content}'. Defaulting to '0'.")
            return "0"
            
    except Exception as e:
        logger.debug(f"Model request failed: {e}")
        return DEBUG_FALLBACK

async def evaluate_task(client: AsyncOpenAI, task: str):
    print(f"[START] task={task} env=forex_trading model={MODEL_NAME}", flush=True)
    
    steps_taken = 0
    total_score = 0.0
    rewards_history = []
    
    async with httpx.AsyncClient(timeout=30.0) as http:
        try:
            # 1. Reset
            reset_payload = {"task": task, "episode_id": "eval_1"}
            resp = await http.post(f"{BASE_URL}/reset", json=reset_payload)
            if resp.status_code != 200:
                logger.debug(f"Reset failed: {resp.text}")
                print(f"[END] success=false steps=0 score=0.000 rewards=", flush=True)
                return
                
            obs = resp.json()
            
            last_reward = 0.0
            done = False
            reward = 0.0
            history = []
            
            for step in range(1, MAX_STEPS + 1):
                # 2. Get action
                action_str = await get_model_action(client, step, obs, last_reward, history)
                action_int = int(action_str)
                
                # 3. Step environment
                step_payload = {"action": action_int}
                try:
                    s_resp = await http.post(f"{BASE_URL}/step", json=step_payload)
                    s_resp.raise_for_status()
                    data = s_resp.json()
                except Exception as step_e:
                    logger.debug(f"Step request failed: {step_e}")
                    raise step_e
                
                obs = data.get("observation", {})
                reward = float(data.get("reward", 0.0))
                done = bool(data.get("done", False))
                
                last_reward = reward
                rewards_history.append(f"{reward:.2f}")
                total_score += reward
                steps_taken += 1
                
                # Update history
                history.append({"step": step, "action": action_int, "reward": reward})
                
                # Emit step log
                print(f"[STEP] step={step} action={action_int} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)
                
                if done:
                    break
                    
            # Completed loop
            # Calculate final normalized score bounding between 0, 1
            if total_score < 0:
                final_score = 0.0
            elif total_score > 1.0:
                final_score = 1.0
            else:
                final_score = total_score
                
            success = done and (final_score > 0.0)
            rewards_str = ",".join(rewards_history)
            print(f"[END] success={str(success).lower()} steps={steps_taken} score={final_score:.3f} rewards={rewards_str}", flush=True)
            
        except Exception as full_e:
            rewards_str = ",".join(rewards_history)
            final_score = total_score
            success = False
            print(f"[END] success=false steps={steps_taken} score={final_score:.3f} rewards={rewards_str}", flush=True)

async def main():
    # If no API key is set, the wrapper will fail but the script bounds fallback to 0
    dummy_key = API_KEY if API_KEY else "dummy_string"
    
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=dummy_key,
    )
    
    tasks = ["first_blood", "consistent_gainer", "risk_manager"]
    
    for t in tasks:
        await evaluate_task(client, t)

if __name__ == "__main__":
    asyncio.run(main())
