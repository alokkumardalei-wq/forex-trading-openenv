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
        self.previous_grade = 0.0

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

def get_current_grade(env: ForexTradingEnv, task: str) -> float:
    profit_pct = (env.equity_usd - env.initial_equity_usd) / env.initial_equity_usd
    max_eq = max(env.equity_curve) if env.equity_curve else env.initial_equity_usd
    drawdown = (max_eq - env.equity_usd) / max_eq

    if task == "first_blood":
        # Any closed profitable trade terminates and succeeds
        if profit_pct > 0:
            return 1.0
        return 0.0

    elif task == "consistent_gainer":
        # Partial progress up to 0.5% profit
        if profit_pct > 0:
            return min(1.0, profit_pct / 0.005)
        return 0.0

    elif task == "risk_manager":
        # Requires drawdown < 1.0%, partial progress up to 1.0% profit
        if drawdown > 0.01:
            env.terminated = True
            return 0.0
        if profit_pct > 0:
            return min(1.0, profit_pct / 0.01)
        return 0.0
    return 0.0

def calculate_step_reward(env: ForexTradingEnv, base_reward: float, last_trade_info: dict, done: bool, task: str) -> float:
    """ Computes step reward as the differential of the clamped current grade. """
    # first_blood termination logic
    if task == "first_blood" and last_trade_info and last_trade_info.get("event") == "CLOSE":
        if last_trade_info.get("net_pips", 0.0) > 0:
            env.terminated = True
            
    # Risk manager termination logic
    max_eq = max(env.equity_curve) if env.equity_curve else env.initial_equity_usd
    drawdown = (max_eq - env.equity_usd) / max_eq
    if task == "risk_manager" and drawdown > 0.01:
        env.terminated = True

    grade = get_current_grade(env, task)
    
    # Strictly inside (0, 1), e.g., [0.01, 0.99]
    clamped_grade = max(0.01, min(0.99, grade))
    
    step_reward = clamped_grade - env_manager.previous_grade
    env_manager.previous_grade = clamped_grade
    return float(step_reward)

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
    env_manager.previous_grade = 0.0
    
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
    task_reward = calculate_step_reward(env_manager.env, base_reward, last_trade, done, env_manager.task_name)
    
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
