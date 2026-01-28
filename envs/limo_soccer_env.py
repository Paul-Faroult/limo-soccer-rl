"""
Limo Soccer environment (2D) .
Observation : [car_x, car_y, car_angle, ball_x, ball_y]
Action      : [accel_norm, steer_norm] in [-1,1]^2
Le robot et la balle apparaissent de manière aléatoire
"""
from typing import Optional, Dict, Any, Tuple
import math
import time
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# ---------------- CONFIG ----------------
WIDTH = 800
HEIGHT = 800

FIELD_MARGIN = 80
FIELD_LEFT = FIELD_MARGIN
FIELD_TOP = FIELD_MARGIN
FIELD_RIGHT = WIDTH - FIELD_MARGIN
FIELD_BOTTOM = HEIGHT - FIELD_MARGIN
FIELD_W = FIELD_RIGHT - FIELD_LEFT
FIELD_H = FIELD_BOTTOM - FIELD_TOP

CAR_W = 70
CAR_H = 48

BALL_R = 12

PIXELS_PER_METER = FIELD_W / 3.0
CAR_LENGTH_M = 0.35
CAR_LENGTH_PX = CAR_LENGTH_M * PIXELS_PER_METER

MAX_LINEAR_SPEED = 0.5 * PIXELS_PER_METER
MAX_ANGULAR_SPEED = 2 * math.pi / 3.2
ACCELERATION = MAX_LINEAR_SPEED / 0.5
LINEAR_DAMP = 2.0

BALL_RESTITUTION = 0.5
BALL_LINEAR_DAMP = 0.95
BALL_PUSH_FACTOR = 2.2
BALL_MAX_SPEED = 2 * MAX_LINEAR_SPEED

FPS = 10
DT = 1.0 / FPS
MAX_EPISODE_STEPS = FPS * 20

COLOR_BG = (18, 120, 40)
COLOR_FIELD = (34, 139, 34)
COLOR_BORDER = (255, 255, 255)
COLOR_CAR = (20, 120, 220)
COLOR_BALL = (220, 30, 30)
COLOR_GOAL = (50, 200, 50)
COLOR_BAD_GOAL = (200, 50, 50)
COLOR_HEADLIGHT = (255, 200, 70)

EPS = 1e-8

# ---------------- UTIL ----------------
def clamp(x, a, b):
    return max(a, min(b, x))

def angle_normalize(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi

# ---------------- ENV ----------------
class LimoSoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = "human"):
        super().__init__()
        self.render_mode = render_mode

        # espace action / observation 
        self.action_space = spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                                       high=np.array([1., 1.], dtype=np.float32),
                                       dtype=np.float32)
        low = np.array([-1.0, -1.0, -3.14, -1.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 3.14, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # dynamique de la voiture et de la balle
        self.car_pos = np.array([FIELD_LEFT + 20.0, FIELD_TOP + FIELD_H / 2.0], dtype=float)
        self.car_angle = 0.0
        self.car_speed = 0.0

        self.ball_pos = np.array([FIELD_LEFT + FIELD_W / 2.0, FIELD_TOP + FIELD_H / 2.0], dtype=float)
        self.ball_vel = np.array([0.0, 0.0], dtype=float)

        # Nombre de step 
        self.step_count = 0

        # rendu pygame 
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Limo Soccer Env")
            self.clock = pygame.time.Clock()

        # variable pour timer replacement balle par arbitre
        self._referee_zone_start = None
        self._referee_zone_name = None

        #  nombre de but
        self.goals_scored = 0

    # ---- observation / reset ----
    """
    Normalisation des observations entre -1 et 1
    """
    def _observe(self):
        # centre du terrain
        cx = FIELD_LEFT + FIELD_W / 2
        cy = FIELD_TOP + FIELD_H / 2

        # normalisation entre -1 et 1
        rx = (self.car_pos[0] - cx) / (FIELD_W / 2)
        ry = (self.car_pos[1] - cy) / (FIELD_H / 2)
        bx = (self.ball_pos[0] - cx) / (FIELD_W / 2)
        by = (self.ball_pos[1] - cy) / (FIELD_H / 2)

        #orientation de la voiture
        ro = self.car_angle

        return np.array([rx, ry, ro, bx, by], dtype=np.float32)


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self.seed(seed)

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
        
        self.car_pos[:] = np.array([rx, ry])
        self.car_angle = np.random.uniform(-math.pi, math.pi)
        self.car_speed = 0.0

        self._reset_ball_center()

        self.step_count = 0
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        
        self.init_robot_to_ball = self._robot_to_ball_dist()
        self.init_ball_to_goal = self._ball_dist_to_left_goal()

        return self._observe(), {}

    # ---- fonctions d'aides pour les distances ----

    # devant de la voiture
    def _robot_front(self) -> np.ndarray:
        return self.car_pos + np.array([math.cos(self.car_angle), math.sin(self.car_angle)]) * (CAR_LENGTH_PX / 2.0)

    # distance balle, voiture
    def _robot_to_ball_dist(self) -> float:
        front = self._robot_front()
        return float(np.linalg.norm(self.ball_pos - front) + EPS)

    # distance balle, cage
    def _ball_dist_to_left_goal(self) -> float:
        gx = FIELD_LEFT
        gy = FIELD_TOP + FIELD_H / 2.0
        return float(math.hypot(self.ball_pos[0] - gx, self.ball_pos[1] - gy))

    # zone des cage
    def _is_in_goal_zone(self, x, y) -> bool:
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        goal_w = int(0.08 * (WIDTH / 800.0) * 100)
        in_left = (FIELD_LEFT <= x <= FIELD_LEFT + goal_w) and (gy0 <= y <= gy1)
        in_right = (FIELD_RIGHT - goal_w <= x <= FIELD_RIGHT) and (gy0 <= y <= gy1)
        return in_left or in_right

    # balle dans la cage de gauche (la bonne)
    def _is_ball_in_left_goal(self) -> bool:
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        fully_past_line = (self.ball_pos[0] <= (FIELD_LEFT - BALL_R))
        vertically_in_goal = (gy0 <= self.ball_pos[1] <= gy1)
        return bool(fully_past_line and vertically_in_goal)

    # balle dans la cage de droite (la mauvaise)
    def _is_ball_in_right_goal(self) -> bool:
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        fully_past_line = (self.ball_pos[0] >= (FIELD_RIGHT + BALL_R))
        vertically_in_goal = (gy0 <= self.ball_pos[1] <= gy1)
        return bool(fully_past_line and vertically_in_goal)

    # voiture en dehors du terrain
    def _is_out_of_arena(self) -> bool:
        x, y = self.car_pos
        return (x < FIELD_LEFT - 30) or (x > FIELD_RIGHT + 30) or (y < FIELD_TOP - 30) or (y > FIELD_BOTTOM + 30)

    # ---- placement de la balle ----
    def _reset_ball_center(self):
        square_ratio = 0.4
        square_w = FIELD_W * square_ratio
        square_h = FIELD_H * square_ratio
        square_left = FIELD_LEFT + (FIELD_W - square_w) / 2
        square_top = FIELD_TOP + (FIELD_H - square_h) / 2
        bx = np.random.uniform(square_left + BALL_R, square_left + square_w - BALL_R)
        by = np.random.uniform(square_top + BALL_R, square_top + square_h - BALL_R)
        self.ball_pos[:] = np.array([bx, by], dtype=float)
        self.ball_vel[:] = 0.0

        # reset shaping records
        self.best_robot_to_ball = None
        self.best_ball_to_goal = None

    # fonction pour empecher de "rentrer" dans le mur
    def _clamp_vec_to_bounds(self, pos, r):
        x = clamp(pos[0], FIELD_LEFT + r, FIELD_RIGHT - r)
        y = clamp(pos[1], FIELD_TOP + r, FIELD_BOTTOM - r)
        return np.array([x, y], dtype=float)

    # ---- Fonction pour simuler action de l'arbitre qui replace la balle ----
    def _referee_nudge_ball(self):
        REQUIRED_STAY = 1.0
        now = time.time()
        bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])
        thresh = CAR_W / 2.0
        in_top_zone = ((by - BALL_R) < (FIELD_TOP + thresh)) and (not self._is_in_goal_zone(bx, by))
        in_bottom_zone = ((FIELD_BOTTOM - (by + BALL_R)) < thresh) and (not self._is_in_goal_zone(bx, by))
        current_zone = "top" if in_top_zone and not in_bottom_zone else ("bottom" if in_bottom_zone and not in_top_zone else None)
        if current_zone is None:
            self._referee_zone_start = None
            self._referee_zone_name = None
            return
        if self._referee_zone_name != current_zone:
            self._referee_zone_name = current_zone
            self._referee_zone_start = now
            return
        start = self._referee_zone_start
        if start is None:
            self._referee_zone_start = now
            return
        if (now - start) < REQUIRED_STAY:
            return
        if current_zone == "top":
            self.ball_pos[1] = FIELD_TOP + thresh + BALL_R + 1.0
        else:
            self.ball_pos[1] = FIELD_BOTTOM - (thresh + BALL_R + 1.0)
        self.ball_vel[:] = 0.0
        dist_to_car = np.linalg.norm(self.ball_pos - self.car_pos)
        min_safe = (CAR_W + BALL_R) * 0.9
        if dist_to_car < min_safe:
            v = self.ball_pos - self.car_pos
            vn = np.linalg.norm(v)
            if vn < 1e-6:
                perp = np.array([1.0, 0.0])
            else:
                perp = np.array([-v[1], v[0]]) / (vn + EPS)
            lateral_shift = CAR_W
            cand1 = self.ball_pos + perp * lateral_shift
            cand2 = self.ball_pos - perp * lateral_shift
            cand1[0] = clamp(cand1[0], FIELD_LEFT + BALL_R, FIELD_RIGHT - BALL_R)
            cand2[0] = clamp(cand2[0], FIELD_LEFT + BALL_R, FIELD_RIGHT - BALL_R)
            self.ball_pos[0] = cand1[0] if np.linalg.norm(cand1 - self.car_pos) > np.linalg.norm(cand2 - self.car_pos) else cand2[0]
        self.ball_pos = self._clamp_vec_to_bounds(self.ball_pos, BALL_R)
        self._referee_zone_start = None
        self._referee_zone_name = None

    # Fonction qui gère toute la physique de l'environnement
    def _simulate_physics(self):
        
        # deplacement de la voiture 
        dx = math.cos(self.car_angle) * self.car_speed * DT
        dy = math.sin(self.car_angle) * self.car_speed * DT
        self.car_pos += np.array([dx, dy], dtype=float)

        # collision voiture-balle
        diff = self.ball_pos - self.car_pos
        dist = np.linalg.norm(diff)
        contact_dist = (CAR_LENGTH_PX / 2.0) + BALL_R
        if dist < contact_dist and dist > 1e-6:
            n = diff / (dist + EPS)
            overlap = contact_dist - dist
            self.ball_pos += n * (overlap * 0.6)
            self.car_pos -= n * (overlap * 0.4)
            car_vel_vec = np.array([math.cos(self.car_angle) * self.car_speed,
                                    math.sin(self.car_angle) * self.car_speed], dtype=float)
            rel_vel = np.dot(self.ball_vel - car_vel_vec, n)
            if rel_vel < 0:
                impulse = -(1 + BALL_RESTITUTION) * rel_vel
                self.ball_vel += n * impulse * BALL_PUSH_FACTOR
                self.car_speed *= 0.9
            # clamp vitesse de la balle 
            vnorm = np.linalg.norm(self.ball_vel)
            if vnorm > BALL_MAX_SPEED:
                self.ball_vel *= BALL_MAX_SPEED / (vnorm + EPS)

        # vitesse et position de la balle
        self.ball_pos += self.ball_vel * DT
        self.ball_vel *= BALL_LINEAR_DAMP * DT

        goal_top = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        goal_bottom = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0

        # intervention de l'arbitre
        self._referee_nudge_ball()

        # collisions voiture-mur(sauf ouverture de but)
        cx, cy = self.car_pos
        half_w, half_h = CAR_W / 2.0, CAR_H / 2.0
        if cx - half_w < FIELD_LEFT and not (goal_top <= cy <= goal_bottom):
            self.car_pos[0] = FIELD_LEFT + half_w
            self.car_speed *= -0.3
        if cx + half_w > FIELD_RIGHT and not (goal_top <= cy <= goal_bottom):
            self.car_pos[0] = FIELD_RIGHT - half_w
            self.car_speed *= -0.3
        if cy - half_h < FIELD_TOP:
            self.car_pos[1] = FIELD_TOP + half_h
            self.car_speed *= -0.3
        if cy + half_h > FIELD_BOTTOM:
            self.car_pos[1] = FIELD_BOTTOM - half_h
            self.car_speed *= -0.3

        # collisions ball-mur (sauf ouverture de but)
        bx, by = self.ball_pos
        if bx - BALL_R < FIELD_LEFT and not (goal_top <= by <= goal_bottom):
            self.ball_pos[0] = FIELD_LEFT + BALL_R
            self.ball_vel *= -0.3
        if bx + BALL_R > FIELD_RIGHT and not (goal_top <= by <= goal_bottom):
            self.ball_pos[0] = FIELD_RIGHT - BALL_R
            self.ball_vel *= -0.3
        if by - BALL_R < FIELD_TOP:
            self.ball_pos[1] = FIELD_TOP + BALL_R
            self.ball_vel *= -0.3
        if by + BALL_R > FIELD_BOTTOM:
            self.ball_pos[1] = FIELD_BOTTOM - BALL_R
            self.ball_vel *= -0.3

        return
    
    # Fonction qui gère les rewards
    def _compute_reward(self):
        # distance balle voiture et balle bonne cage(gauche)
        dist_rb = self._robot_to_ball_dist()
        dist_bg = self._ball_dist_to_left_goal()

        reward = 0.0

        # ---------- INIT ----------
        if self.best_robot_to_ball is None:
            self.best_robot_to_ball = dist_rb
        if self.best_ball_to_goal is None:
            self.best_ball_to_goal = dist_bg

        # ---------- PARAMETRE ----------
        MIN_DELTA = 1.0        # px (anti jitter)
        W_RB_DENSE = 0.003     # robot -> balle
        W_BG_DENSE = 0.008     # balle -> cage

        W_RB_RECORD = 0.05
        W_BG_RECORD = 0.12

        # ---------- DENSE PROGRESS ----------
        delta_rb = self.best_robot_to_ball - dist_rb
        delta_bg = self.best_ball_to_goal - dist_bg

        if delta_rb > MIN_DELTA:
            reward += W_RB_DENSE * delta_rb

        if delta_bg > MIN_DELTA:
            reward += W_BG_DENSE * delta_bg

        # ---------- RECORD BONUS ----------
        if dist_rb < self.best_robot_to_ball - 1.5:
            reward += W_RB_RECORD
            self.best_robot_to_ball = dist_rb

        if dist_bg < self.best_ball_to_goal - 1.5:
            reward += W_BG_RECORD
            self.best_ball_to_goal = dist_bg

        # ---------- GOALS ----------
        if self._is_ball_in_left_goal():
            reward += 50.0
            self.goals_scored += 1
            self._reset_ball_center()
            self.best_robot_to_ball = None
            self.best_ball_to_goal = None

        if self._is_ball_in_right_goal():
            reward -= 5.0
            self._reset_ball_center()
            self.best_robot_to_ball = None
            self.best_ball_to_goal = None

        # ---------- OUT ----------
        if self._is_out_of_arena():
            reward -= 3.0

        # ---------- UPDATE ----------
        self.prev_robot_to_ball = dist_rb
        self.prev_ball_to_goal = dist_bg

        return float(reward)

    # ---- step ----
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:

        # appliquer l’action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError("Action must be shape (2,)")
        accel = float(clamp(action[0], -1.0, 1.0))
        steer = float(clamp(action[1], -1.0, 1.0))

        # rotation
        angular_speed = steer * MAX_ANGULAR_SPEED
        self.car_angle = angle_normalize(self.car_angle + angular_speed * DT)

        # linear speed contrôle
        target_speed = accel * MAX_LINEAR_SPEED
        speed_diff = target_speed - self.car_speed
        max_accel_step = ACCELERATION * DT
        speed_diff = clamp(speed_diff, -max_accel_step, max_accel_step)
        self.car_speed += speed_diff
        if abs(accel) < 1e-3:
            self.car_speed *= (1.0 - LINEAR_DAMP * DT)

        terminated = False
        truncated = False

        # simulation de la physique
        self._simulate_physics()
    
        # calcul de la reward
        reward = self._compute_reward()

        # observation
        obs = self._observe()

        self.step_count += 1

        # Sortie du terrain
        if self._is_out_of_arena():
            terminated = True

        # Bon but
        if self._is_ball_in_left_goal():
            terminated = True

        # Détection fin d'épisodes (durée)
        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True

        # nombre de but pour affichage tensorboard
        info = {
            "goals": self.goals_scored,
        }

        # reset pour ne pas compter plusieurs fois
        self.goals_scored = 0

        return obs, float(reward), bool(terminated), bool(truncated), info


    # ---- Fonction pour convertir nos action au format csv----
    # PAS UTILISEE pour l'instant
    def step_from_cmd(self, cmd: Dict[str, Any]):
        if cmd.get("emergency_stop", False):
            action = np.array([0.0, 0.0], dtype=np.float32)
        else:
            lin = float(cmd.get("linear_x", 0.0))
            ang = float(cmd.get("angular_z", 0.0))
            accel = clamp(lin / (MAX_LINEAR_SPEED + EPS), -1.0, 1.0)
            steer = clamp(ang / (MAX_ANGULAR_SPEED + EPS), -1.0, 1.0)
            action = np.array([accel, steer], dtype=np.float32)
        return self.step(action)

    # ---- render / close ----
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
        surf.fill(COLOR_BG)
        pygame.draw.rect(surf, COLOR_FIELD, (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H))
        pygame.draw.rect(surf, COLOR_BORDER, (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H), 3)

        gx0 = FIELD_LEFT
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gx1 = FIELD_LEFT + int(0.08 * WIDTH / 800.0 * 100)
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        left_rect = pygame.Rect(int(gx0), int(gy0), int(gx1 - gx0), int(gy1 - gy0))
        pygame.draw.rect(surf, COLOR_GOAL, left_rect)

        rx0 = FIELD_RIGHT - int(0.08 * WIDTH / 800.0 * 100)
        ry0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        rx1 = FIELD_RIGHT
        ry1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        right_rect = pygame.Rect(int(rx0), int(ry0), int(rx1 - rx0), int(ry1 - ry0))
        pygame.draw.rect(surf, COLOR_BAD_GOAL, right_rect)

        pygame.draw.circle(surf, COLOR_BALL, (int(self.ball_pos[0]), int(self.ball_pos[1])), BALL_R)

        car_surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, COLOR_CAR, (0, 0, CAR_W, CAR_H), border_radius=6)
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.car_angle))
        rect = rotated.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
        surf.blit(rotated, rect.topleft)

        fx = self.car_pos[0] + math.cos(self.car_angle) * (CAR_H / 2.0)
        fy = self.car_pos[1] + math.sin(self.car_angle) * (CAR_H / 2.0)
        pygame.draw.circle(surf, COLOR_HEADLIGHT, (int(fx), int(fy)), 5)

        obs = self._observe()
        font = pygame.font.SysFont("Consolas", 14)
        txt = f"step:{self.step_count} obs:{np.round(obs,2)}"
        surf.blit(font.render(txt, True, (0, 0, 0)), (4, 4))

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(FPS)
        else:
            return np.transpose(np.array(pygame.surfarray.array3d(surf)), (1, 0, 2))

    def close(self):
        try:
            if self.screen is not None:
                pygame.quit()
                self.screen = None
        except Exception:
            pass

# ---------------- test manuel ----------------
if __name__ == "__main__":
    """
    Test manuel de l'environnement Limo Soccer.
    Permet de contrôler le robot avec les flèches ou de simuler aléatoirement.
    """
    
    print("=== Limo Soccer Env (manual test) ===")
    env = LimoSoccerEnv(render_mode="human")
    obs, _ = env.reset()
    running = True
    use_keyboard = True
    reward_total = 0.0

    start = time.time()
    count = 0
    while running:
        if use_keyboard:
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
        else:
            action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Reward = {reward}")
        reward_total += reward
        count += 1
        env.render()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        if term or trunc:
            time.sleep(0.8)
            obs, _ = env.reset()
    env.close()
    end= time.time()
    duration = end - start
    nb_reward = count/duration
    print("closing")
    print(f"Reward totale = {reward_total}")
    print(f"Il y a {nb_reward} reward par seconde")
