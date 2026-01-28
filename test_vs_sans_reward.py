"""
Évaluation d'un agent PPO sur l'environnement Limo Soccer Duel
contre un robot statique ou un autre agent pré-entraîné.
Affiche le rendu en temps réel et calcule la récompense totale par épisode.
"""
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from limo_soccer_env_duel_sans_reward import LimoSoccerEnvDuel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(CURRENT_DIR, "models_duel_sans_reward_2")
MODEL_PATH = os.path.join(LOG_DIR, "ppo_limo_checkpoint") 
VEC_PATH = os.path.join(LOG_DIR, "vecnormalize_checkpoint.pkl")

# Le modèle 7 correspond au modèle qui à été entrainé contre le deuxième robot static
opponent_model_path = "models_7/ppo_limo_checkpoint.zip"

def make_env(render_mode="human", seed=0):
    def _init():
        env = LimoSoccerEnvDuel(opponent_model_path, render_mode=render_mode)
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
