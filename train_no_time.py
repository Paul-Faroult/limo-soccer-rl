"""
Fine-tuning : affrontement entre 2 robots
- Chargement d'un modèle déjà entraîné
- Reprise de l'entraînement
- Sauvegarde sous un NOUVEAU nom et dans un NOUVEAU dossier
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure

from limo_soccer_env_duel_no_limit import LimoSoccerEnvDuel

# ======================================================
# CONFIGURATION GLOBALE
# ======================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------- ANCIEN MODÈLE (SOURCE) ----------
BASE_MODEL_DIR = os.path.join(CURRENT_DIR, "models_duel_sans_reward_2")
BASE_MODEL_NAME = "ppo_limo_checkpoint"
BASE_VEC_NAME = "vecnormalize_checkpoint.pkl"

# --------- NOUVEAU MODÈLE (FINE-TUNING) ----------
FINE_TUNE_DIR = os.path.join(CURRENT_DIR, "models_duel_no_limit")
os.makedirs(FINE_TUNE_DIR, exist_ok=True)

MODEL_NAME = "ppo_limo_finetune"
VEC_NAME = "vecnormalize_finetune.pkl"

# --------- PARAMÈTRES ----------
N_ENVS = 4
CHECKPOINT_FREQ = 1_000_000
TIMESTEPS = 20_000_000
DEVICE = "cuda"

# modèle adverse (inchangé)
opponent_model_path = os.path.join(
    BASE_MODEL_DIR,
    BASE_MODEL_NAME + ".zip"
)

# ======================================================
# CALLBACKS
# ======================================================

class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path,
                f"{MODEL_NAME}_{self.num_timesteps}"
            )
            self.model.save(model_path)

            if hasattr(self.training_env, "save"):
                self.training_env.save(
                    os.path.join(
                        self.save_path,
                        f"{VEC_NAME}_{self.num_timesteps}"
                    )
                )

            if self.verbose:
                print(f"Checkpoint sauvegardé à {self.num_timesteps} steps")

        return True


class GoalTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.goal_buffer = []
        self.collision_buffer = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "goals" in info:
                self.goal_buffer.append(info["goals"])
            if "static_collisions" in info:
                self.collision_buffer.append(info["static_collisions"])

        if self.n_calls % 100 == 0:
            if self.goal_buffer:
                self.logger.record(
                    "soccer/goals_per_100_steps",
                    np.sum(self.goal_buffer)
                )
                self.goal_buffer.clear()

            if self.collision_buffer:
                self.logger.record(
                    "soccer/collisions_per_100_steps",
                    np.sum(self.collision_buffer)
                )
                self.collision_buffer.clear()

        return True


# ======================================================
# ENVIRONNEMENT
# ======================================================

def make_env_fn(seed=0, render_mode=None):
    def _init():
        env = LimoSoccerEnvDuel(
            opponent_model_path=opponent_model_path,
            render_mode=render_mode
        )
        env = Monitor(env)
        return env
    return _init


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    # --------- ENV PARALLÈLE ----------
    venv = SubprocVecEnv(
        [make_env_fn(i) for i in range(N_ENVS)]
    )

    # --------- CHARGEMENT VecNormalize ----------
    old_vec_path = os.path.join(BASE_MODEL_DIR, BASE_VEC_NAME)

    if os.path.exists(old_vec_path):
        venv = VecNormalize.load(old_vec_path, venv)
        venv.training = True
        venv.norm_reward = True
        print("VecNormalize chargé depuis l'ancien modèle")
    else:
        raise FileNotFoundError(
            "VecNormalize introuvable : impossible de fine-tuner proprement"
        )

    # --------- CHARGEMENT DU MODÈLE ----------
    old_model_path = os.path.join(
        BASE_MODEL_DIR,
        BASE_MODEL_NAME + ".zip"
    )

    if not os.path.exists(old_model_path):
        raise FileNotFoundError("Modèle source introuvable")

    model = PPO.load(
        old_model_path,
        env=venv,
        device=DEVICE
    )

    print("Ancien modèle chargé → reprise de l'entraînement")

    # --------- LOGGER TENSORBOARD ----------
    tb_log_path = os.path.join(FINE_TUNE_DIR, "tb")

    new_logger = configure(
        tb_log_path,
        ["stdout", "tensorboard"]
    )
    model.set_logger(new_logger)

    # Pour afficher Tensorboard, aller dans un terminal,
    # tensorboard --logdir "C:\Users\fpaul\OneDrive\Documents\Github\limo-soccer-rl\models_duel_no_limit\tb"
    # puis ouvir le local host

    # --------- CALLBACKS ----------
    callbacks = CallbackList([
        SaveOnStepCallback(CHECKPOINT_FREQ, FINE_TUNE_DIR),
        GoalTensorboardCallback()
    ])

    # --------- ENTRAÎNEMENT ----------
    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=False  # TRÈS IMPORTANT
    )

    # --------- SAUVEGARDE FINALE ----------
    model.save(os.path.join(FINE_TUNE_DIR, MODEL_NAME))
    venv.save(os.path.join(FINE_TUNE_DIR, VEC_NAME))

    print("Fine-tuning terminé. Nouveau modèle sauvegardé.")