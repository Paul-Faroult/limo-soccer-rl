"""
Script pour entraîner un modèle PPO sur l'environnement LimoSoccerEnv.
Le script gère les checkpoints automatiques, la normalisation VecNormalize,
le suivi du nombre de buts et l'arrêt anticipé si un objectif de buts est atteint.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from limo_soccer_env import LimoSoccerEnv
from stable_baselines3.common.callbacks import CallbackList

# dossier où se trouve train.py
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)

CHECKPOINT_FREQ = 500_000  # sauvegarde tous les X timesteps

MODEL_NAME = "ppo_limo_checkpoint"
VEC_NAME = "vecnormalize_checkpoint.pkl"

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
    
# --------- callback pour suivi des goals ----------
class GoalLoggerCallback(BaseCallback):
    """
    Callback pour afficher le nombre de buts marqués pendant l'entraînement.
    """
    def __init__(self, verbose=1, log_freq=10_000):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.goals = 0
        self.last_timesteps = 0

    def _on_step(self) -> bool:
        # On récupère l'environnement vectorisé
        env = self.training_env
        if hasattr(env, 'envs'):
            # pour DummyVecEnv, on a envs = [LimoSoccerEnv]
            for e in env.envs:
                if hasattr(e, 'goals_scored'):
                    self.goals += e.goals_scored
                    e.goals_scored = 0  # reset compteur dans l'env

        # Affichage toutes les log_freq timesteps
        if (self.num_timesteps - self.last_timesteps) >= self.log_freq:
            print(f"[{self.num_timesteps} timesteps] Total goals scored: {self.goals}")
            self.last_timesteps = self.num_timesteps
        return True
    
class GoalEarlyStopCallback(BaseCallback):
    """
    Stop learning when the total number of goals across envs >= target_goals.
    """
    def __init__(self, target_goals: int = 1, verbose=1):
        super().__init__(verbose)
        self.target_goals = target_goals
        self.accum_goals = 0
        self.last_report = 0
        self.stopped_early = False

    def _on_step(self) -> bool:
        # training_env can be a VecEnv with attribute envs (DummyVecEnv)
        env = self.training_env
        if hasattr(env, "envs"):
            for e in env.envs:
                if hasattr(e, "goals_scored") and e.goals_scored > 0:
                    self.accum_goals += e.goals_scored
                    e.goals_scored = 0  # reset in env after accounting
        # report periodically
        if self.num_timesteps - self.last_report >= 50_000 and self.verbose:
            print(f"[GoalEarlyStop] timesteps={self.num_timesteps} total_goals={self.accum_goals}")
            self.last_report = self.num_timesteps

        if self.accum_goals >= self.target_goals:
            if self.verbose:
                print(f"[GoalEarlyStop] Reached target_goals={self.target_goals} at timesteps={self.num_timesteps}. Stopping training.")
            self.stopped_early = True
            return False  # stop training
        return True
# --------- création de l'environnement ----------
def make_env_fn(seed=0, render_mode=None):
    def _init():
        env = LimoSoccerEnv(render_mode=render_mode)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    # créer l'environnement vectorisé
    venv = DummyVecEnv([make_env_fn()])
    
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
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            verbose=1,
            learning_rate=5e-5,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=1e-4,    
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=os.path.join(LOG_DIR, "tb")
        )
    # Callback combiné
    callbacks = CallbackList([
        SaveOnStepCallback(CHECKPOINT_FREQ, LOG_DIR),
        GoalEarlyStopCallback(target_goals=1, verbose=1),
        GoalLoggerCallback(log_freq=50_000)  # tous les 50k timesteps
    ])

    # démarrer l'entraînement 
    TIMESTEPS = 20_000_000
    model.learn(total_timesteps=TIMESTEPS, callback=callbacks)
    #tensorboard --logdir "C:\Users\fpaul\OneDrive\Documents\Cours_5a\Projet\Presentation_28_11\models\tb
    # si arrêté par goalstop, lancer une évaluation rapide
    if GoalEarlyStopCallback(target_goals=1, verbose=1).stopped_early:
        print("Training stopped early due to goal. Running quick evaluation (10 episodes).")
        # Prépare env de test (non normalisé si tu veux voir le comportement exact)
        test_env = DummyVecEnv([make_env_fn(render_mode="human")])
        # charger normalization si utilisé -> si tu utilises VecNormalize pour training, tu dois charger vec
        try:
            vec_path = os.path.join(LOG_DIR, VEC_NAME)
            if os.path.exists(vec_path):
                test_env = VecNormalize.load(vec_path, test_env)
                test_env.training = False
                test_env.norm_reward = False
        except Exception as e:
            print("Could not load VecNormalize:", e)

        model.save(os.path.join(LOG_DIR, MODEL_NAME + "_on_goal"))
        for ep in range(10):
            obs, _ = test_env.reset()
            done = False
            total_r = 0.0
            steps = 0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, terminated, truncated, info = test_env.step(action)
                total_r += float(r)
                steps += 1
                test_env.render()
                if terminated or truncated:
                    print(f"Eval ep {ep}: steps={steps} reward={total_r:.2f}")
                    break
        test_env.close()

    # sauvegarde finale
    model.save(os.path.join(LOG_DIR, MODEL_NAME))
    venv.save(os.path.join(LOG_DIR, VEC_NAME))
    print("Entraînement terminé, modèles sauvegardés.")
