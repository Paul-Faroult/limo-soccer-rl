"""
Environnement de duel Limo Soccer sans récompense directe sur l'adversaire ni limite de temps.

L'agent principal affronte un adversaire contrôlé par un modèle PPO figé.
Les observations de l'adversaire sont normalisées via VecNormalize chargé
depuis un checkpoint externe.

Modifier OPP_PATH pour changer le modèle adverse.
"""

from limo_soccer_env import (
    LimoSoccerEnv,
    FIELD_LEFT, FIELD_RIGHT, FIELD_TOP, FIELD_BOTTOM,
    FIELD_W, FIELD_H,
    CAR_W, CAR_H,
    BALL_R,
    WIDTH, HEIGHT,
    FPS,
    COLOR_BG,
    COLOR_FIELD,
    COLOR_BORDER,
    COLOR_GOAL,
    COLOR_BAD_GOAL,
    COLOR_BALL,
    COLOR_CAR,
    COLOR_HEADLIGHT,
    MAX_ANGULAR_SPEED,
    MAX_LINEAR_SPEED,
    ACCELERATION,
    DT,
    LINEAR_DAMP
)

import numpy as np
import pygame
import math
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium import spaces
import os

from limo_soccer_env_static_opponent import LimoSoccerEnvStaticRobot

OPP_PATH = "models_duel_sans_reward_2"

def clamp(x, a, b):
    return max(a, min(b, x))

def angle_normalize(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

class LimoSoccerEnvGhost(LimoSoccerEnvStaticRobot):
    """
    ENV UTILISÉ UNIQUEMENT POUR :
    - charger VecNormalize
    - normaliser les observations du robot adverse

     NE JAMAIS L’UTILISER POUR STEP / RENDER
    """

    def __init__(self):
        super().__init__(render_mode=None)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

# ============================================================
# ENV DUEL
# ============================================================
class LimoSoccerEnvDuel(LimoSoccerEnv):

    def __init__(self, opponent_model_path, render_mode="human"):
        super().__init__(render_mode)

        # -------------------
        # état adversaire
        # -------------------
        self.opp_pos = np.zeros(2, dtype=float)
        self.opp_angle = 0.0
        self.opp_speed = 0.0

        self.opponent = PPO.load(opponent_model_path, device="cpu")

        # observation = obs parent + (x,y) robot statique
        low = np.array([-1, -1, -math.pi, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, math.pi, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Dummy env juste pour VecNormalize
        dummy_env = DummyVecEnv([lambda: LimoSoccerEnvGhost()])

        self.opp_vecnorm = VecNormalize.load(
            os.path.join(OPP_PATH,"vecnormalize_checkpoint.pkl"),
            dummy_env
        )

        # Pour tensorboard
        self.static_collisions = 0

        # Constante pour eviter le deuxième robot
        self.SAFE_DIST_STATIC = CAR_W * 1.5
        self.COLLISION_DIST_STATIC = CAR_W

        # variables analyse
        self.goals_agent = 0
        self.goals_opponent = 0

        self.max_goals = 5 # nombre total de buts pour arrêter l'épisode

        self.steps_since_last_goal = 0
        self.nb_goals = 0
        self.max_steps_no_goal = 30 * FPS  # 30 secondes × FPS

    # ========================================================
    # RESET
    # ========================================================
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # ------------------ Robot apparait aléatoirement sur un côté------------------
        side = np.random.choice(["top", "bottom", "left", "right"])

        if side == "top":
            rx = np.random.uniform(FIELD_LEFT, FIELD_RIGHT)
            ry = np.random.uniform (FIELD_TOP + CAR_H, FIELD_H / 2)
        elif side == "bottom":
            rx = np.random.uniform(FIELD_LEFT, FIELD_RIGHT)
            ry = np.random.uniform(FIELD_BOTTOM - CAR_H, FIELD_H / 2)
        elif side == "left":
            rx = np.random.uniform(FIELD_LEFT + CAR_W, FIELD_W / 2)
            ry = np.random.uniform(FIELD_TOP, FIELD_BOTTOM)
        else: # right
            rx = np.random.uniform(FIELD_RIGHT - CAR_W, FIELD_W / 2)
            ry = np.random.uniform(FIELD_TOP, FIELD_BOTTOM)
        
        self.opp_pos[:] = np.array([rx, ry])
        self.opp_angle = np.random.uniform(-math.pi, math.pi)
        self.opp_speed = 0.0

        self.static_collisions = 0

        # variables analyse
        self.goals_agent = 0
        self.goals_opponent = 0

        self.steps_since_last_goal = 0

        self.nb_goals = 0

        return self._observe(), info
    
    def _apply_action_only(self, action):
        action = np.asarray(action, dtype=np.float32)
        accel = float(clamp(action[0], -1.0, 1.0))
        steer = float(clamp(action[1], -1.0, 1.0))

        # rotation
        angular_speed = steer * MAX_ANGULAR_SPEED
        self.car_angle = angle_normalize(self.car_angle + angular_speed * DT)

        # vitesse linéaire
        target_speed = accel * MAX_LINEAR_SPEED
        speed_diff = target_speed - self.car_speed
        max_accel_step = ACCELERATION * DT
        speed_diff = clamp(speed_diff, -max_accel_step, max_accel_step)
        self.car_speed += speed_diff

        if abs(accel) < 1e-3:
            self.car_speed *= (1.0 - LINEAR_DAMP * DT)

    def _compute_reward(self):
        goal_left = self._is_ball_in_left_goal()
        goal_right = self._is_ball_in_right_goal()

        reward = super()._compute_reward()

        if goal_left:
            self.goals_agent += 1

        if goal_right:
            self.goals_opponent += 1

        return float(reward)

    
    # ========================================================
    # STEP (PHYSIQUE UNIQUE)
    # ========================================================
    def step(self, action):

        # ==================================================
        # ACTION ADVERSAIRE (modèle figé)
        # ==================================================
        obs_opp = self._observe_opponent()[None, :]
        obs_opp = self.opp_vecnorm.normalize_obs(obs_opp)

        action_opp, _ = self.opponent.predict(obs_opp, deterministic=True)
        action_opp = action_opp[0].copy()
        action_opp[1] *= -1.0  # miroir steering

        # ==================================================
        # SAUVEGARDE ÉTAT AGENT
        # ==================================================
        save_pos = self.car_pos.copy()
        save_angle = self.car_angle
        save_speed = self.car_speed

        # ==================================================
        # SIMULATION ADVERSAIRE (SANS REWARD)
        # ==================================================
        self.car_pos = self.opp_pos
        self.car_angle = self.opp_angle
        self.car_speed = self.opp_speed

        self._apply_action_only(action_opp)
        self._simulate_physics()

        self.opp_pos = self.car_pos.copy()
        self.opp_angle = self.car_angle
        self.opp_speed = self.car_speed

        # ==================================================
        # RESTAURATION AGENT
        # ==================================================
        self.car_pos = save_pos
        self.car_angle = save_angle
        self.car_speed = save_speed

        # collision voiture ↔ voiture
        self._handle_car_collision()

        # ==================================================
        # STEP NORMAL DE L’AGENT PRINCIPAL
        # ==================================================
        obs, reward, terminated, truncated, info = super().step(action)

        # ajouter la métrique
        info["static_collisions"] = self.static_collisions

        # nouvel arrêt basé sur le nombre de buts
        total_goals = self.goals_agent + self.goals_opponent
        terminated = total_goals >= self.max_goals
        truncated = False

        # Si un but a été marqué, reset du compteur
        if self.goals_agent + self.goals_opponent > self.nb_goals:
            self.steps_since_last_goal = 0
            self.nb_goals += 1

        self.steps_since_last_goal += 1

        # Vérifier si 30 secondes sans but
        if self.steps_since_last_goal >= self.max_steps_no_goal:
            self._reset_ball_center()
            self.steps_since_last_goal = 0

        if terminated:
            if self.goals_agent > self.goals_opponent:
                info["result"] = "win"
            elif self.goals_agent < self.goals_opponent:
                info["result"] = "lose"
            else:
                info["result"] = "draw"

            info["goals_agent"] = self.goals_agent
            info["goals_opponent"] = self.goals_opponent
            self.static_collisions = 0

        return obs, reward, terminated, truncated, info

    # ========================================================
    # COLLISIONS
    # ========================================================
    def _handle_car_collision(self):
        diff = self.car_pos - self.opp_pos
        dist = np.linalg.norm(diff)
        min_dist = CAR_W

        if dist < min_dist and dist > 1e-6:
            n = diff / dist
            overlap = min_dist - dist
            self.car_pos += n * overlap * 0.5
            self.opp_pos -= n * overlap * 0.5
            self.car_speed *= -0.3
            self.opp_speed *= -0.3
            self.static_collisions += 1
    
    # -------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------
    def _observe(self):
        base_obs = super()._observe()

        cx = FIELD_LEFT + FIELD_W / 2
        cy = FIELD_TOP + FIELD_H / 2

        sx = (self.opp_pos[0] - cx) / (FIELD_W / 2)
        sy = (self.opp_pos[1] - cy) / (FIELD_H / 2)

        return np.concatenate([base_obs, [sx, sy]]).astype(np.float32)
    
    # ========================================================
    # OBS OPPOSANT
    # ========================================================
    def _observe_opponent(self):
        """
        Observation miroir pour que l'adversaire
        pense attaquer la cage de gauche
        """

        cx = FIELD_LEFT + FIELD_W / 2
        cy = FIELD_TOP + FIELD_H / 2

        # --- position adversaire (miroir X) ---
        ox = -(self.opp_pos[0] - cx) / (FIELD_W / 2)
        oy =  (self.opp_pos[1] - cy) / (FIELD_H / 2)

        # angle miroir
        oa = math.pi - self.opp_angle
        oa = (oa + math.pi) % (2 * math.pi) - math.pi

        # --- balle (miroir X) ---
        bx = -(self.ball_pos[0] - cx) / (FIELD_W / 2)
        by =  (self.ball_pos[1] - cy) / (FIELD_H / 2)

        # --- agent vu comme obstacle (miroir X) ---
        ax = -(self.car_pos[0] - cx) / (FIELD_W / 2)
        ay =  (self.car_pos[1] - cy) / (FIELD_H / 2)

        return np.array([ox, oy, oa, bx, by, ax, ay], dtype=np.float32)
    
    # ========================================================
    # RENDER
    # ========================================================
    def render(self):
        if self.render_mode is None:
            return None

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        surf = self.screen

        # ===============================
        # TERRAIN (copié du parent)
        # ===============================
        surf.fill(COLOR_BG)
        pygame.draw.rect(surf, COLOR_FIELD, (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H))
        pygame.draw.rect(surf, COLOR_BORDER, (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H), 3)

        # goals
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        goal_w = int(0.08 * WIDTH / 800.0 * 100)

        pygame.draw.rect(
            surf,
            COLOR_GOAL,
            pygame.Rect(FIELD_LEFT, gy0, goal_w, gy1 - gy0),
        )
        pygame.draw.rect(
            surf,
            COLOR_BAD_GOAL,
            pygame.Rect(FIELD_RIGHT - goal_w, gy0, goal_w, gy1 - gy0),
        )

        # ===============================
        # BALLE
        # ===============================
        pygame.draw.circle(
            surf,
            COLOR_BALL,
            (int(self.ball_pos[0]), int(self.ball_pos[1])),
            BALL_R,
        )

        # ===============================
        # VOITURE PRINCIPALE
        # ===============================
        car_surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, COLOR_CAR, (0, 0, CAR_W, CAR_H), border_radius=6)

        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.car_angle))
        rect = rotated.get_rect(center=self.car_pos.astype(int))
        surf.blit(rotated, rect.topleft)

        # phare avant
        fx = self.car_pos[0] + math.cos(self.car_angle) * (CAR_H / 2)
        fy = self.car_pos[1] + math.sin(self.car_angle) * (CAR_H / 2)
        pygame.draw.circle(surf, COLOR_HEADLIGHT, (int(fx), int(fy)), 5)

        # ===============================
        # DEUXIÈME VOITURE 
        # ===============================
        static_surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(static_surf, (160, 160, 160), (0, 0, CAR_W, CAR_H), border_radius=6)

        # phare
        pygame.draw.circle(
            static_surf,
            (255, 220, 140),
            (CAR_W - 6, CAR_H // 2),
            4,
        )

        rotated_static = pygame.transform.rotate(
            static_surf,
            -math.degrees(self.opp_angle),
        )

        static_rect = rotated_static.get_rect(
            center=self.opp_pos.astype(int)
        )
        surf.blit(rotated_static, static_rect.topleft)

        # ===============================
        # FINAL
        # ===============================
        obs = self._observe()
        font = pygame.font.SysFont("Consolas", 14)
        txt = f"step:{self.step_count} obs:{np.round(obs,2)}"
        surf.blit(font.render(txt, True, (0, 0, 0)), (4, 4))
        
        pygame.display.flip()
        self.clock.tick(FPS)

# ============================================================
# TEST MANUEL
# ============================================================
if __name__ == "__main__":

    env = LimoSoccerEnvDuel(
        opponent_model_path=os.path.join(OPP_PATH,"ppo_limo_checkpoint.zip"),
        render_mode="human"
    )

    obs, _ = env.reset()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        accel = (keys[pygame.K_UP] - keys[pygame.K_DOWN])
        steer = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT])

        action = np.array([accel, steer], dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)

        env.render()

        if terminated or truncated:
            time.sleep(0.7)
            obs, _ = env.reset()

    env.close()
