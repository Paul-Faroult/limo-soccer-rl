import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from limo_soccer_env_duel_sans_reward import LimoSoccerEnvDuel

# ======================================================
# CONFIGURATION
# ======================================================

N_EPISODES = 500   # nombre d'épisodes pour la comparaison
MAX_STEPS = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CSV_PATH = os.path.join(EVAL_DIR, "results.csv")

# --------- MODELS A ET B ----------
MODEL_A = {
    "name": "base",
    "model_path": "models_duel_sans_reward_2/ppo_limo_checkpoint",
    "vec_path": "models_duel_sans_reward_2/vecnormalize_checkpoint.pkl"
}

MODEL_B = {
    "name": "finetune",
    "model_path": "models_duel_sans_reward_2_finetune/ppo_limo_finetune",
    "vec_path": "models_duel_sans_reward_2_finetune/vecnormalize_finetune.pkl"
}

# ======================================================
# FONCTION POUR CRÉER L'ENV
# ======================================================

def make_env(opponent_path):
    def _init():
        env = LimoSoccerEnvDuel(opponent_path, render_mode=None)
        env = Monitor(env)
        return env
    return _init

# ======================================================
# LOAD MODEL
# ======================================================

def load_model(cfg, opponent_zip):
    env = DummyVecEnv([make_env(opponent_zip)])
    env = VecNormalize.load(cfg["vec_path"], env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(cfg["model_path"], env=env)
    return model, env

model_A, env_A = load_model(MODEL_A, MODEL_B["model_path"] + ".zip")
model_B, env_B = load_model(MODEL_B, MODEL_A["model_path"] + ".zip")

# ======================================================
# BOUCLE D'ÉVALUATION
# ======================================================

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode", "model",
        "goals", "goals_conceded",
        "collisions", "reward",
        "steps", "win"
    ])

    for ep in range(N_EPISODES):
        for model, env, label in [
            (model_A, env_A, "A"),
            (model_B, env_B, "B")
        ]:
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            goals = 0
            goals_conceded = 0
            collisions = 0

            while not done and steps < MAX_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, infos = env.step(action)

                info = infos[0]
                total_reward += reward[0]
                steps += 1

                goals += info.get("goals", 0)
                goals_conceded += info.get("goals_conceded", 0)
                collisions += info.get("static_collisions", 0)

            win = 1 if goals > goals_conceded else 0

            writer.writerow([
                ep, label,
                goals, goals_conceded,
                collisions, total_reward,
                steps, win
            ])

        if ep % 50 == 0:
            print(f"{ep}/{N_EPISODES} épisodes évalués")

print("Évaluation terminée :", CSV_PATH)
