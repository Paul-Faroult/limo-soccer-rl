import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from limo_soccer_env_duel_sans_reward import LimoSoccerEnvDuel

# ======================================================
# CONFIG
# ======================================================

N_EPISODES = 10_000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CSV_PATH = os.path.join(EVAL_DIR, "results_duel.csv")

MODEL_A = {
    "name": "base",
    "model": "models_duel_sans_reward_2/ppo_limo_checkpoint.zip",
    "vec": "models_duel_sans_reward_2/vecnormalize_checkpoint.pkl"
}

MODEL_B = {
    "name": "no_limit",
    "model": "models_duel_no_limit/ppo_limo_finetune.zip"
}

# ======================================================
# ENV
# ======================================================

def make_env(opponent_path):
    def _init():
        env = LimoSoccerEnvDuel(opponent_path, render_mode=None)
        return Monitor(env)
    return _init

# ======================================================
# LOAD ENV + MODELS
# ======================================================

env = DummyVecEnv([make_env(MODEL_B["model"])])
env = VecNormalize.load(MODEL_A["vec"], env)
env.training = False
env.norm_reward = False

model_A = PPO.load(MODEL_A["model"], env=env)
model_B = PPO.load(MODEL_B["model"])

# ======================================================
# EVALUATION LOOP
# ======================================================

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "agent",
        "goals_scored",
        "goals_conceded",
        "result"  # win / lose / draw
    ])

    for ep in range(N_EPISODES):
        obs = env.reset()
        done = False
        final_info = None

        while not done:
            # Base joue
            action_A, _ = model_A.predict(obs, deterministic=True)
            obs, _, done, infos = env.step(action_A)
            final_info = infos[0]

        # Stats des deux agents
        goals_base = final_info.get("goals_agent", 0)
        goals_finetune = final_info.get("goals_opponent", 0)

        # Déterminer le résultat
        if goals_base > goals_finetune:
            res_base = "win"
            res_finetune = "lose"
        elif goals_base < goals_finetune:
            res_base = "lose"
            res_finetune = "win"
        else:
            res_base = res_finetune = "draw"

        # Écriture CSV
        writer.writerow([ep, "base", goals_base, goals_finetune, res_base])
        writer.writerow([ep, "no_limit", goals_finetune, goals_base, res_finetune])

        if ep % 50 == 0:
            print(f"[{ep}/{N_EPISODES}] {goals_base}-{goals_finetune} → {res_base}/{res_finetune}")

print("Évaluation terminée")
print(f"Résultats sauvegardés dans : {CSV_PATH}")
