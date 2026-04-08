import random
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_env import ForexTradingEnv
from indicators import load_and_preprocess_data

app = FastAPI(title="ForexTrading OpenEnv", version="1.0.0")

# --- Pydantic Models ---

class Action(BaseModel):
    # Action space: 0: HOLD, 1: BUY, 2: SELL, 3: CLOSE
    action: int = Field(..., description="0: HOLD, 1: BUY, 2: SELL, 3: CLOSE")

class Observation(BaseModel):
    # A dictionary reflecting the technical features from the underlying env
    features: Dict[str, float] = Field(..., description="Technical indicators (e.g., rsi_14, ma_spread)")
    current_equity: float = Field(..., description="Current account equity in USD")
    open_trade: str = Field("NONE", description="NONE, BUY, or SELL")
    unrealized_pnl: float = Field(0.0, description="Current unrealized PnL")

class RewardInfo(BaseModel):
    # Any extra debugging/tracking metrics
    drawdown: float = Field(..., description="Current drawdown percentage")
    total_profit_pct: float = Field(..., description="Total profit percentage from start")
    trade_closed: bool = Field(False, description="Whether a trade was closed in this step")
    base_reward: float = Field(0.0, description="Raw reward from the environment")
    internal_action: int = Field(0, description="Action passed to the internal environment")

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: RewardInfo

class StateResult(BaseModel):
    step_count: int
    episode_id: str
    total_reward: float
    task_name: str

# --- Global Environment State ---

class EnvManager:
    def __init__(self):
        # We load a small snippet for speed during validation, or the main file
        # Fallback if the full file isn't found
        self.file_path = "data/test_EURUSD_Candlestick_1_Hour_BID_20.02.2023-22.02.2025.csv"
        try:
            self.df, self.feature_cols = load_and_preprocess_data(self.file_path)
            # Use smaller dataset to fit within memory/time limits for the competition
            self.df = self.df.iloc[:2000].copy()
        except Exception as e:
            # Deterministic fallback so the service still boots if the source
            # market file cannot be loaded.
            self.df, self.feature_cols = load_and_preprocess_data(
                "__synthetic_fallback__.csv"
            )
            self.df = self.df.iloc[:2000].copy()
            
        self.sl_opts = [10, 20, 30]
        self.tp_opts = [10, 20, 30]
        self.win_size = 30
        
        self.env: Optional[ForexTradingEnv] = None
        
        self.episode_id = ""
        self.step_count = 0
        self.total_reward = 0.0
        self.task_name = "first_blood" # Default task
        
        self.max_equity = 10000.0 # Standard start

    def initialize_env(self):
        self.env = ForexTradingEnv(
            df=self.df,
            window_size=self.win_size,
            sl_options=self.sl_opts,
            tp_options=self.tp_opts,
            spread_pips=1.0,
            commission_pips=0.0,
            random_start=False,
            feature_columns=self.feature_cols,
        )

env_manager = EnvManager()


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False, response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>ForexTrading OpenEnv</title>
        <style>
          :root {
            color-scheme: dark;
            --bg: #0b1220;
            --card: #121a2b;
            --text: #e6edf7;
            --muted: #99a7bd;
            --accent: #7ee0a8;
            --line: rgba(255, 255, 255, 0.08);
          }
          html, body {
            margin: 0;
            min-height: 100%;
            background:
              radial-gradient(circle at top left, rgba(126, 224, 168, 0.16), transparent 28%),
              radial-gradient(circle at top right, rgba(104, 171, 255, 0.10), transparent 22%),
              var(--bg);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }
          .wrap {
            max-width: 880px;
            margin: 0 auto;
            padding: 72px 24px;
          }
          .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 8px 14px;
            font-size: 14px;
            color: var(--muted);
            background: rgba(255, 255, 255, 0.03);
          }
          .dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent);
            box-shadow: 0 0 0 6px rgba(126, 224, 168, 0.12);
          }
          h1 {
            margin: 22px 0 12px;
            font-size: clamp(38px, 6vw, 72px);
            line-height: 0.98;
            letter-spacing: -0.05em;
          }
          p {
            max-width: 760px;
            font-size: 18px;
            line-height: 1.7;
            color: var(--muted);
            margin: 0 0 28px;
          }
          .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 28px;
          }
          .card {
            background: rgba(18, 26, 43, 0.88);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 18px 16px;
            backdrop-filter: blur(10px);
          }
          .card h2 {
            margin: 0 0 10px;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #b7c4d9;
          }
          .card code {
            display: block;
            margin: 0 0 8px;
            color: #f2f7ff;
            font-size: 14px;
            word-break: break-word;
          }
          .card span {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.55;
          }
          a {
            color: var(--accent);
            text-decoration: none;
          }
          a:hover {
            text-decoration: underline;
          }
        </style>
      </head>
      <body>
        <main class="wrap">
          <div class="badge"><span class="dot"></span> OpenEnv service running</div>
          <h1>ForexTrading OpenEnv</h1>
          <p>
            A real-world forex trading benchmark with three graded tasks,
            deterministic fallbacks for local validation, and a FastAPI/OpenEnv
            interface ready for evaluation.
          </p>
          <div class="grid">
            <section class="card">
              <h2>Start an episode</h2>
              <code>POST /reset</code>
              <span>Resets the environment and returns the initial observation.</span>
            </section>
            <section class="card">
              <h2>Take a step</h2>
              <code>POST /step</code>
              <span>Submit one of the four discrete actions: hold, buy, sell, or close.</span>
            </section>
            <section class="card">
              <h2>Inspect state</h2>
              <code>GET /state</code>
              <span>View the current episode id, step count, and cumulative reward.</span>
            </section>
            <section class="card">
              <h2>Docs</h2>
              <code><a href="/docs">/docs</a></code>
              <span>Open the auto-generated API docs to explore the request and response models.</span>
            </section>
          </div>
        </main>
      </body>
    </html>
    """

def convert_obs(raw_obs, env: ForexTradingEnv) -> Observation:
    # `raw_obs` is a flatten array. The last step in raw_obs is the current step's features.
    # To keep it simple, we extract properties directly from the `env` state.
    
    features_dict = {}
    if env.feature_columns:
        # Get the latest row of features
        latest_features = env.df.iloc[env.current_step][env.feature_columns]
        features_dict = latest_features.to_dict()
    
    trade_type = "NONE"
    if env.position != 0:
        trade_type = "BUY" if env.position == 1 else "SELL"
        
    return Observation(
        features=features_dict,
        current_equity=float(env.equity_usd),
        open_trade=trade_type,
        unrealized_pnl=float(env._compute_unrealized_pips())
    )

def calculate_task_reward(env: ForexTradingEnv, base_reward: float, last_trade_info: dict, done: bool, task: str) -> float:
    """ Applies task-specific routing for rewards and grading """
    profit_pct = (env.equity_usd - env.initial_equity_usd) / env.initial_equity_usd
    max_eq = max(env.equity_curve) if env.equity_curve else env.initial_equity_usd
    drawdown = (max_eq - env.equity_usd) / max_eq
    
    reward = 0.0
    
    if task == "first_blood":
        # Task 1: Complete one profitable trade
        if last_trade_info and last_trade_info.get("event") == "CLOSE":
            if last_trade_info.get("net_pips", 0.0) > 0:
                reward = 1.0
                env.terminated = True # Finish immediately once achieved
    
    elif task == "consistent_gainer":
        # Task 2: Achieve > 0.5% profit
        if done:
            # Grade up to 1.0
            if profit_pct > 0.005:
                reward = 1.0
            elif profit_pct > 0:
                reward = profit_pct / 0.005
            else:
                reward = 0.0
        else:
            # Minor shaping reward for closed profitable trades
            if last_trade_info and last_trade_info.get("net_pips", 0.0) > 0:
                reward = 0.1
                
    elif task == "risk_manager":
        # Task 3: Value profit, heavily penalize drawdown
        if done:
            if drawdown > 0.01:
                reward = 0.0 # Failed drawdown limit
            else:
                # Target > 1.0% profit
                if profit_pct > 0.01:
                    reward = 1.0
                elif profit_pct > 0:
                    reward = profit_pct / 0.01
                else:
                    reward = 0.0
        else:
            if drawdown > 0.01:
                # Instantly end episode if drawdown exceeds 1%
                env.terminated = True
                reward = 0.0 # keep score in [0, 1]

    return float(max(0.0, min(1.0, reward)))

# --- API Endpoints ---

class ResetRequest(BaseModel):
    task: str = "first_blood"
    episode_id: str = "default_ep"

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None):
    # By default, openenv validation might send an empty dictionary or no body.
    if req is None:
        req = ResetRequest()
        
    env_manager.task_name = req.task
    env_manager.episode_id = req.episode_id
    env_manager.step_count = 0
    env_manager.total_reward = 0.0
    env_manager.initialize_env()
    
    raw_obs = env_manager.env.reset()
    env_manager.max_equity = env_manager.env.equity_usd
    
    return convert_obs(raw_obs, env_manager.env)

@app.post("/step", response_model=StepResult)
def step(action: Action):
    if env_manager.env is None or (env_manager.env.terminated or env_manager.env.truncated):
        raise HTTPException(status_code=400, detail="Environment needs reset.")
        
    # Translate Action (0: HOLD, 1: BUY, 2: SELL, 3: CLOSE) -> internal env action
    # Internal: 0=HOLD, 1=CLOSE, 2..10=OPEN Short, 11..19=OPEN Long
    internal_action = 0
    if action.action == 0:
        internal_action = 0
    elif action.action == 1:
        # BUY (Long) with sl=10, tp=10
        internal_action = 11
    elif action.action == 2:
        # SELL (Short) with sl=10, tp=10
        internal_action = 2
    elif action.action == 3:
        # CLOSE
            internal_action = 1
        
    raw_obs, base_reward, terminated, truncated, info = env_manager.env.step(internal_action)
    done = terminated or truncated
    
    env_manager.step_count += 1
    
    # Calculate specialized task reward
    last_trade = info.get("last_trade_info") or {}
    task_reward = calculate_task_reward(env_manager.env, base_reward, last_trade, done, env_manager.task_name)
    
    env_manager.total_reward += task_reward
    
    # Check if the task forced a done state (e.g. drawdown limits hit)
    if (env_manager.env.terminated or env_manager.env.truncated):
        done = True
        
    max_eq = max(env_manager.env.equity_curve) if env_manager.env.equity_curve else env_manager.env.initial_equity_usd
    dd = (max_eq - env_manager.env.equity_usd) / max_eq
    prof = (env_manager.env.equity_usd - env_manager.env.initial_equity_usd) / env_manager.env.initial_equity_usd

    reward_info = RewardInfo(
        drawdown=float(dd),
        total_profit_pct=float(prof),
        trade_closed=(last_trade.get("event") == "CLOSE"),
        base_reward=float(base_reward),
        internal_action=internal_action
    )
    
    return StepResult(
        observation=convert_obs(raw_obs, env_manager.env),
        reward=task_reward,
        done=done,
        info=reward_info
    )

@app.get("/state", response_model=StateResult)
def state():
    return StateResult(
        step_count=env_manager.step_count,
        episode_id=env_manager.episode_id,
        total_reward=env_manager.total_reward,
        task_name=env_manager.task_name
    )

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
