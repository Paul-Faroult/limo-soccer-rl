"""
Entrainement affrontement entre 2 robot sans mis en place de reward pour les collisions.
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
import numpy as np

from limo_soccer_env_duel_sans_reward import LimoSoccerEnvDuel

# dossier où se trouve train.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(CURRENT_DIR, "models_duel_sans_reward_2")
os.makedirs(LOG_DIR, exist_ok=True)

CHECKPOINT_FREQ = 1_000_000  # sauvegarde tous les X timesteps

MODEL_NAME = "ppo_limo_checkpoint"
VEC_NAME = "vecnormalize_checkpoint.pkl"

N_ENVS = 4

opponent_model_path = "models_7/ppo_limo_checkpoint.zip"

# --------- callback pour sauvegarde automatique ----------
class SaveOnStepCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            # sauvegarde modèle et VecNormalize
            model_path = os.path.join(self.save_path, f"{MODEL_NAME}_{self.num_timesteps}")
            self.model.save(model_path)
            if hasattr(self.training_env, 'save'):
                self.training_env.save(os.path.join(self.save_path, f"{VEC_NAME}_{self.num_timesteps}"))
            if self.verbose > 0:
                print(f"Checkpoint sauvegardé à {self.num_timesteps} timesteps")
        return True
    
# --------- callback pour suivi des goals et collisions ----------
class GoalTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.goal_buffer = []
        self.collison_buffer = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "goals" in info:
                self.goal_buffer.append(info["goals"])
            if "static_collisions" in info:
                self.collison_buffer.append(info["static_collisions"])

        # log toutes les 100 steps
        if self.n_calls % 100 == 0:
            if len(self.goal_buffer) > 0:
                self.logger.record(
                    "soccer/goals_per_step",
                    np.sum(self.goal_buffer)
                )
                self.goal_buffer.clear()

            if len(self.collison_buffer) > 0:
                self.logger.record(
                    "soccer/collisions_per_step",
                    np.sum(self.collison_buffer)
                )
                self.collison_buffer.clear()
        return True
    
# --------- création de l'environnement ----------
def make_env_fn(seed=0, render_mode=None):
    def _init():
        env = LimoSoccerEnvDuel(opponent_model_path, render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":

    # créer l'environnement parallélisé
    venv = SubprocVecEnv([make_env_fn(i) for i in range(N_ENVS)])
    
    # tenter de charger un VecNormalize existant
    vec_path = os.path.join(LOG_DIR, VEC_NAME)
    if os.path.exists(vec_path):
        venv = VecNormalize.load(vec_path, venv)
        venv.training = True
        print("VecNormalize chargé depuis le checkpoint")
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)

    # charger le modèle si existant
    model_path = os.path.join(LOG_DIR, MODEL_NAME)
    if os.path.exists(model_path + ".zip"):
        model = PPO.load(model_path, env=venv)
        print("Modèle chargé depuis le checkpoint")

        from stable_baselines3.common.logger import configure

        new_tb_log = os.path.join(LOG_DIR, "tb")

        new_logger = configure(
            new_tb_log,
            ["stdout", "tensorboard"]
        )

        model.set_logger(new_logger)

    else:
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            learning_rate=3e-4,
            device = "cuda", # activation du GPU
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99, 
            gae_lambda=0.95, 
            ent_coef=1e-3,    
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(LOG_DIR, "tb")
        )
        # Pour afficher Tensorboard, aller dans un terminal,
        # tensorboard --logdir "C:\Users\fpaul\OneDrive\Documents\Github\limo-soccer-rl\models_duel_sans_reward_2\tb"
        # puis ouvir le local host


    # Callback combiné
    callbacks = CallbackList([
        SaveOnStepCallback(CHECKPOINT_FREQ, LOG_DIR),
        GoalTensorboardCallback()
    ])

    # démarrer l'entraînement 
    TIMESTEPS = 20_000_000
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

    # sauvegarde finale
    model.save(os.path.join(LOG_DIR, MODEL_NAME))
    venv.save(os.path.join(LOG_DIR, VEC_NAME))
    print("Entraînement terminé, modèles sauvegardés.")
