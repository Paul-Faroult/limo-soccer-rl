"""
LimoSoccerEnvStaticRobot : environnement avec un robot statique comme obstacle.

Description :
-------------
- Version du jeu LimoSoccer où un robot statique est placé aléatoirement sur le terrain.
- Ne génère pas de reward pour les collisions avec le robot statique.
- Permet à l'agent principal de s'entraîner à éviter un obstacle fixe tout en jouant au football.
- Observation :
    * Position, angle et vitesse du robot principal et de la balle
    * Position normalisée du robot statique
- Physique :
    * Gestion collisions voiture ↔ balle
    * Gestion collisions voiture ↔ robot statique
- Rendu :
    * Affichage du terrain, de la balle, de la voiture principale et du robot statique avec Pygame
    * Visualisation de l'état pour debug ou analyse
- Tensorboard :
    * Compte le nombre de collisions avec le robot statique dans info["static_collisions"]

Usage :
------
- Peut être utilisé pour l'entraînement ou l'évaluation de modèles RL.
- Test manuel possible via clavier (flèches directionnelles) en exécutant le fichier directement.
"""
# Hérite du fichier principal
from limo_soccer_env import (
    LimoSoccerEnv,
    FIELD_LEFT, FIELD_RIGHT, FIELD_TOP, FIELD_BOTTOM,
    FIELD_W, FIELD_H,
    CAR_W, CAR_H,
    BALL_R,
    FPS,
    BALL_RESTITUTION,
    BALL_PUSH_FACTOR,
    EPS,
    COLOR_BG,
    COLOR_BALL,
    COLOR_BAD_GOAL,
    COLOR_BORDER,
    COLOR_CAR,
    COLOR_FIELD,
    COLOR_GOAL,
    COLOR_HEADLIGHT,
    WIDTH, 
    HEIGHT
)

import numpy as np
import pygame
from gymnasium import spaces
import math
import time

class LimoSoccerEnvStaticRobot(LimoSoccerEnv):
    def __init__(self, render_mode="human"):
        super().__init__(render_mode)

        # --- robot statique ---
        self.static_robot_pos = np.zeros(2, dtype=float)
        self.static_robot_angle = 0.0

        # observation = obs parent + (x,y) robot statique
        low = np.array([-1, -1, -math.pi, -1, -1, -1, -1], dtype=np.float32)
        high = np.array([1, 1, math.pi, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        # Constante pour eviter le deuxième robot
        self.SAFE_DIST_STATIC = CAR_W * 1.5
        self.COLLISION_DIST_STATIC = CAR_W

        # Pour tensorboard
        self.static_collisions = 0

    # -------------------------------------------------
    # RESET
    # -------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # --- choix du mode obstacle ---
        mode = np.random.choice(["car_ball", "ball_goal"])

        if mode == "car_ball":
            start = self.car_pos
            end = self.ball_pos
        else:
            start = self.ball_pos
            end = np.array([
                FIELD_LEFT,
                FIELD_TOP + FIELD_H / 2.0
            ])

        v = end - start
        d = np.linalg.norm(v)
        if d < 1e-6:
            v = np.array([1.0, 0.0])
            d = 1.0
        v /= d

        # --- position sur le trajet ---
        alpha = np.random.uniform(0.25, 0.75)
        base_pos = start + alpha * d * v

        # --- écart latéral LARGE ---
        perp = np.array([-v[1], v[0]])
        
        hard_case = np.random.rand() < 0.3  # 30% des épisodes

        if hard_case:
            lateral_offset = np.random.uniform(-30, 30)  # quasi sur le chemin
        else:
            lateral_offset = np.random.uniform(-100, 100)

        pos = base_pos + perp * lateral_offset

        # --- clamp terrain ---
        pos[0] = np.clip(pos[0], FIELD_LEFT + CAR_W, FIELD_RIGHT - CAR_W)
        pos[1] = np.clip(pos[1], FIELD_TOP + CAR_H, FIELD_BOTTOM - CAR_H)

        self.static_robot_pos[:] = pos
        self.static_robot_angle = np.random.uniform(-math.pi, math.pi)

        self.prev_static_dist = None
        self.static_collisions = 0

        return self._observe(), info

    # -------------------------------------------------
    # OBSERVATION
    # -------------------------------------------------
    def _observe(self):
        base_obs = super()._observe()

        cx = FIELD_LEFT + FIELD_W / 2
        cy = FIELD_TOP + FIELD_H / 2

        sx = (self.static_robot_pos[0] - cx) / (FIELD_W / 2)
        sy = (self.static_robot_pos[1] - cy) / (FIELD_H / 2)

        return np.concatenate([base_obs, [sx, sy]]).astype(np.float32)

    # -------------------------------------------------
    # PHYSIQUE (SURCHARGE PROPRE)
    # -------------------------------------------------
    def _simulate_physics(self):
        # 1) physique normale
        super()._simulate_physics()

        # -------------------------------
        # COLLISION BALLE ↔ ROBOT STATIQUE
        # -------------------------------
        diff = self.ball_pos - self.static_robot_pos
        dist = np.linalg.norm(diff)
        contact = BALL_R + CAR_W / 2

        if dist < contact and dist > 1e-6:
            n = diff / (dist + EPS)
            overlap = contact - dist

            self.ball_pos += n * overlap

            rel_vel = np.dot(self.ball_vel, n)
            if rel_vel < 0:
                impulse = -(1 + BALL_RESTITUTION) * rel_vel
                self.ball_vel += n * impulse * BALL_PUSH_FACTOR

        # --------------------------------
        # COLLISION VOITURE ↔ ROBOT STATIQUE
        # --------------------------------
        diff = self.car_pos - self.static_robot_pos
        dist = np.linalg.norm(diff)
        contact = CAR_W

        if dist < contact and dist > 1e-6:
            n = diff / (dist + EPS)
            overlap = contact - dist

            self.car_pos += n * overlap
            self.car_speed *= -0.3

    # -------------------------------------------------
    # RENDER
    # Nécéssaire de la réecrire entièrement car une fois le .flip() effectué on ne peut rien rajouter donc,
    # si on appelle la fonction depuis la classe mère on ne peut pas rajouter d'élément proprement
    # -> ça fait des éléments clignotants.
    # -------------------------------------------------
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
        # DEUXIÈME VOITURE (STATIQUE)
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
            -math.degrees(self.static_robot_angle),
        )

        static_rect = rotated_static.get_rect(
            center=self.static_robot_pos.astype(int)
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
    
    # Surcharge de Step pour afficher les collisions dans Tensorboard
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # ajouter la métrique
        info["static_collisions"] = self.static_collisions

        # reset pour l’épisode suivant
        if terminated or truncated:
            self.static_collisions = 0

        return obs, reward, terminated, truncated, info


# =====================================================
# TEST MANUEL CLAVIER
# =====================================================
if __name__ == "__main__":
    print("=== Test LimoSoccerEnvStaticRobot ===")

    env = LimoSoccerEnvStaticRobot(render_mode="human")
    obs, _ = env.reset()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        accel = 0.0
        steer = 0.0

        if keys[pygame.K_UP]:
            accel += 1.0
        if keys[pygame.K_DOWN]:
            accel -= 1.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0

        action = np.array([accel, steer], dtype=np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)

        env.render()

        if terminated or truncated:
            time.sleep(0.5)
            obs, _ = env.reset()

    env.close()
    print("Test terminé.")
