import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from envs.limo_soccer_env_static import LimoSoccerEnvStaticRobot

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

LOG_DIR = os.path.join(PROJECT_ROOT, "models/models_static")
MODEL_PATH = os.path.join(LOG_DIR, "ppo_limo_checkpoint") 
VEC_PATH = os.path.join(LOG_DIR, "vecnormalize_checkpoint.pkl")

def make_env(render_mode="human", seed=0):
    def _init():
        env = LimoSoccerEnvStaticRobot(render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    # Build test env with rendering
    venv = DummyVecEnv([make_env(render_mode="human", seed=42)])

    # Load normalization if present
    if os.path.exists(VEC_PATH):
        venv = VecNormalize.load(VEC_PATH, venv)
        venv.training = False
        venv.norm_reward = False
        print("Loaded VecNormalize")
    else:
        print("No VecNormalize found, running without obs/reward normalization")

    # Load model 
    model = PPO.load(MODEL_PATH, env=venv)
    print("Model loaded.")

    n_episodes = 10
    for ep in range(n_episodes):
        obs = venv.reset()
        done = False
        total_reward = 0.0
        step = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            total_reward += float(reward)
            venv.render()
            step += 1
            if done:
                print(f"Episode {ep+1} ended. Steps={step}, Reward={total_reward:.2f}")
                break
        time.sleep(1.0)
    venv.close()
