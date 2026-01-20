import os
import csv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from limo_soccer_env_duel_no_limit import LimoSoccerEnvDuel

# ======================================================
# CONFIG
# ======================================================

N_EPISODES = 5000

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

CSV_PATH = os.path.join(EVAL_DIR, "results_duel.csv")

MODELS = {
    "base": {
        "model": "models_duel_sans_reward_2/ppo_limo_checkpoint.zip",
        "vec":   "models_duel_sans_reward_2/vecnormalize_checkpoint.pkl",
    },
    "no_limit": {
        "model": "models_duel_no_limit/ppo_limo_finetune.zip",
        "vec":   "models_duel_no_limit/vecnormalize_finetune.pkl",
    }
}

# ======================================================
# ENV FACTORY
# ======================================================

def make_env(opponent_model_path):
    def _init():
        env = LimoSoccerEnvDuel(opponent_model_path, render_mode=None)
        return Monitor(env)
    return _init

# ======================================================
# EVALUATION
# ======================================================

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "episode",
        "agent",
        "opponent",
        "goals_scored",
        "goals_conceded",
        "result"
    ])

    for ep in range(N_EPISODES):

        # === MATCH ALLER + RETOUR ===
        for agent_name, opp_name in [("base", "no_limit"), ("no_limit", "base")]:

            agent = MODELS[agent_name]
            opponent = MODELS[opp_name]

            # env avec adversaire figé
            env = DummyVecEnv([make_env(opponent["model"])])
            env = VecNormalize.load(agent["vec"], env)
            env.training = False
            env.norm_reward = False

            model = PPO.load(agent["model"], env=env)

            obs = env.reset()
            done = False
            final_info = None

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, infos = env.step(action)
                final_info = infos[0]

            goals_for = final_info.get("goals_agent", 0)
            goals_against = final_info.get("goals_opponent", 0)

            if goals_for > goals_against:
                result = "win"
            elif goals_for < goals_against:
                result = "lose"
            else:
                result = "draw"

            writer.writerow([
                ep,
                agent_name,
                opp_name,
                goals_for,
                goals_against,
                result
            ])

        if ep % 100 == 0:
            print(f"[{ep}/{N_EPISODES}] épisodes")

print("Évaluation terminée")
print(f"Résultats : {CSV_PATH}")
