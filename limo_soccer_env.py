# limo_soccer_env.py
#
# Environnement Limo Soccer (2D, non-Box2D)
# - Voiture rectangulaire, balle, buts visibles.
# - Observation par défaut : LIDAR (3 secteurs : left, front, right).
# - Action : soit vecteur [accel, steer] continu, soit commande Twist-like via step_from_cmd.
# - Conserve la physique (vitesse, impulsions, rebonds).

import math
import time

# Le module "typing" permet d’indiquer le type des variables, arguments et retours de fonctions
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# ---------------- CONFIGURATION GLOBALE ----------------
# Dimensions fenêtre
WIDTH = 800
HEIGHT = 800

# Zone de jeu : on dessine un rectangle à l'intérieur de l'écran

FIELD_MARGIN = 80
FIELD_LEFT = FIELD_MARGIN
FIELD_TOP = FIELD_MARGIN
FIELD_RIGHT = WIDTH - FIELD_MARGIN
FIELD_BOTTOM = HEIGHT - FIELD_MARGIN
FIELD_W = FIELD_RIGHT - FIELD_LEFT
FIELD_H = FIELD_BOTTOM - FIELD_TOP

# Taille voiture (en pixels) 
CAR_W = 70   # largeur latérale (axe x local)
CAR_H = 48   # longueur (axe avant/arrière)

# Balle
BALL_R = 12  # rayon en pixels

# Conversion / constantes de dynamique (réglées pour simuler ~3 m d'arène)
PIXELS_PER_METER = FIELD_W / 3.0      # conversion pixel <-> m (utile pour réalisme)

CAR_LENGTH_M = 0.35
CAR_LENGTH_PX = CAR_LENGTH_M * PIXELS_PER_METER

# Vitesse et accélération
MAX_LINEAR_SPEED = 0.5 * PIXELS_PER_METER     # vitesse max (px/s)
MAX_ANGULAR_SPEED = 2 * math.pi / 3.2         # tourne sur lui-même en ~3.2s
ACCELERATION = MAX_LINEAR_SPEED / 0.5         # pour atteindre vmax en ~0.5s
LINEAR_DAMP = 2.0                          # frottement quand on n'accélère pas

# Physique balle
BALL_RESTITUTION = 0.5    # coefficient de rebond (0..1)
BALL_LINEAR_DAMP = 0.95   # réduit la vitesse chaque frame (5% perdu)
BALL_PUSH_FACTOR = 2.2    # multiplicateur d'impulsion quand la voiture frappe la balle
""" 
j'ai mis 2 fois la vitesse de la voiture qu'on ai toujours l'impression de pousser la balle sinon elle reste collé à la voiture
"""
BALL_MAX_SPEED = 2*MAX_LINEAR_SPEED   # limite vitesse balle px/s

# Profondeur virtuelle du but (pour laisser la balle "entrer")
GOAL_DEPTH = 25

# Fréquence / pas de simulation
FPS = 60
DT = 1.0 / FPS

# Sectors LIDAR-like (angles relatifs à l'avant de la voiture)
"""Ba Thong 14/11 : rectification de l'ordre des angles et  lidar : 90deg -> 180deg"""
SECTOR_CONFIG = {
    "left":  (math.radians(-180), math.radians(-60)),
    "front": (math.radians(-60), math.radians(60)),
    "right": (math.radians(60), math.radians(180)),
}
LIDAR_MAX_RANGE = max(FIELD_W, FIELD_H)

# Episodes / récompenses
MAX_EPISODE_STEPS = FPS * 30

# Couleurs pour rendu (R,G,B)
COLOR_BG = (18, 120, 40)
COLOR_FIELD = (34, 139, 34)
COLOR_BORDER = (255, 255, 255)
COLOR_CAR = (20, 120, 220)
COLOR_BALL = (220, 30, 30)
COLOR_GOAL = (50, 200, 50)
COLOR_BAD_GOAL = (200, 50, 50)
COLOR_HEADLIGHT = (255, 200, 70)

# ---------------- UTILITAIRES ----------------
# Ces fonctions servent à effectuer des calculs géométriques et physiques simples :
# - clamp : limite une valeur dans un intervalle.
# - angle_normalize : remet un angle dans la plage [-π, π].
# - ray_rect_distance : calcule à quelle distance un rayon touche un rectangle.
# - ray_circle_distance : calcule à quelle distance un rayon touche un cercle.
# Ces deux dernières fonctions servent à simuler le capteur LIDAR de la voiture :
#   on tire plusieurs rayons autour d’elle et on mesure la distance jusqu’aux murs ou à la balle.
#   Cela donne une "perception" de l’environnement, comme un sonar ou un laser.

def clamp(x, a, b):
    """
    Cette fonction "coupe" la valeur x pour qu’elle reste entre a et b.

    ➜ Exemple :
       clamp(5, 0, 3) renvoie 3  (car 5 est au-dessus du maximum)
       clamp(-2, 0, 3) renvoie 0 (car -2 est en dessous du minimum)
       clamp(2, 0, 3) renvoie 2  (car 2 est dans l’intervalle)

    Utilité :
       - On s’en sert très souvent pour limiter des valeurs physiques :
         par exemple, empêcher que la vitesse dépasse une valeur maximale,
         ou que la direction d’un capteur soit trop grande.
    """
    return max(a, min(b, x))

def angle_normalize(a):
    """
    angle_normalize(a)
    Ramène un angle (en radians) dans la plage [-π, π].

    ➜ Pourquoi ?
       En physique ou robotique, les angles tournent en rond :
       0°, 360°, 720°... représentent en fait la même direction.
       Pour éviter des erreurs de calcul, on normalise les angles.

    ➜ Exemple :
       angle_normalize(3π) ≈ -π
       angle_normalize(-4π/3) ≈ 2π/3

    Principe :
       On ajoute π, on fait un modulo 2π, puis on enlève π.
       Ça permet de "boucler" l’angle proprement.
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


def ray_rect_distance(px, py, ang,
                      rect_min_x, rect_min_y, rect_max_x, rect_max_y,
                      max_range):
    """
    ray_rect_distance(...)
    Calcule la distance entre un rayon (émis depuis un point (px, py)
    dans la direction 'ang') et un rectangle aligné sur les axes.

    Ce qu’on fait ici :
      - On imagine un rayon partant de la voiture (position (px, py))
        dans une direction donnée (ang).
      - On cherche à savoir à quelle distance ce rayon touche un mur.
      - On regarde les 4 bords du rectangle (gauche, droite, haut, bas),
        et on calcule où le rayon les coupe.

    Mathématiquement :
      Le rayon est défini par : (x, y) = (px, py) + t*(cos(ang), sin(ang))
      avec t ≥ 0.
      Chaque bord du rectangle est défini par une équation simple :
        - x = rect_min_x  (bord gauche)
        - x = rect_max_x  (bord droit)
        - y = rect_min_y  (bord haut)
        - y = rect_max_y  (bord bas)
      On résout pour t quand le rayon touche ces lignes.
      Puis on vérifie si le point d’intersection est bien sur le bord du rectangle.

    Utilité dans le projet :
      Cette fonction sert à simuler un LIDAR :
      la voiture tire des "rayons" pour mesurer la distance
      jusqu’au bord du terrain (les murs).
      On répète cela dans plusieurs directions (devant, gauche, droite)
      pour que la voiture "voit" les obstacles.

    max_range :
      C’est la distance maximale de détection.
      Si le rayon ne touche rien avant cette distance, on renvoie max_range.
    """

    # Direction du rayon (vecteur unitaire)
    dx = math.cos(ang)
    dy = math.sin(ang)
    tx_min = float('inf')  # distance minimale trouvée (on cherche le plus petit t)

    # ---- Bord vertical gauche (x = rect_min_x) et droit (x = rect_max_x)
    if abs(dx) > 1e-8:  # éviter une division par zéro (si le rayon est presque vertical)
        t1 = (rect_min_x - px) / dx # Bord gauche - position x de la voiture / direction du rayon (x)
        t2 = (rect_max_x - px) / dx
        for t in (t1, t2):
            # On ne garde que les intersections devant le rayon (t >= 0)
            if 0 <= t <= max_range:
                # y correspondant
                y = py + t * dy
                # vérifier si ce y est bien sur le bord vertical
                if rect_min_y - 1e-6 <= y <= rect_max_y + 1e-6:
                    tx_min = min(tx_min, t)

    # ---- Bord horizontal haut (y = rect_min_y) et bas (y = rect_max_y)
    if abs(dy) > 1e-8:
        t3 = (rect_min_y - py) / dy # Bord haut - position y de la voiture / direction du rayon (y)
        t4 = (rect_max_y - py) / dy
        for t in (t3, t4):
            if 0 <= t <= max_range:
                x = px + t * dx
                if rect_min_x - 1e-6 <= x <= rect_max_x + 1e-6:
                    tx_min = min(tx_min, t)

    # Si aucune intersection trouvée (t infini), on retourne max_range
    if tx_min == float('inf'):
        return max_range

    # On renvoie la distance trouvée (limitée à max_range)
    return clamp(tx_min, 0.0, max_range)

def ray_circle_distance(px, py, ang, cx, cy, r, max_range):
    """
    ray_circle_distance(...)
    Calcule la distance entre un rayon (partant de (px, py) à l’angle 'ang')
    et un cercle de centre (cx, cy) et de rayon r.

    Ce qu’on cherche :
       On veut savoir à quelle distance le rayon touche le cercle,
       c’est-à-dire à quel moment il "entre en contact" avec la balle.

    Équation du rayon :
        (x, y) = (px, py) + t*(cos(ang), sin(ang))

       Équation du cercle :
        (x - cx)² + (y - cy)² = r²

    En remplaçant x et y par ceux du rayon, on obtient une équation en t :
        ((px + t*dx - cx)² + (py + t*dy - cy)² = r²)
        qu’on développe, et on obtient :
        a*t² + b*t + c = 0
      où :
        a = dx² + dy²
        b = 2 * ((px - cx)*dx + (py - cy)*dy)
        c = (px - cx)² + (py - cy)² - r²

    ➜ On résout cette équation du second degré (quadratique)
      pour trouver les valeurs possibles de t.

      Si le rayon touche le cercle, il y a :
        - 2 solutions (entrée et sortie)
        - 1 solution si tangent (pile au bord)
        - aucune si le rayon passe à côté

    Utilité dans le projet :
      Cette fonction sert à détecter la balle avec le LIDAR :
      quand un rayon part de la voiture, cette fonction indique
      à quelle distance la balle se trouve dans cette direction.
      On garde la distance minimale entre la balle et le mur.

    max_range :
      Si aucune intersection n’est trouvée ou si elle est trop loin,
      on renvoie max_range.
    """

    # Direction du rayon
    dx = math.cos(ang)
    dy = math.sin(ang)

    # Décalage entre le centre du rayon et le centre du cercle
    ox = px - cx # position x de la voiture - position x du centre du cercle
    oy = py - cy

    # Coefficients de l’équation du second degré
    a = dx * dx + dy * dy
    b = 2 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - r * r

    # Discriminant Δ = b² - 4ac
    disc = b * b - 4 * a * c
    if disc < 0:
        # Pas d’intersection
        return max_range

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # On cherche la plus petite intersection positive (devant le rayon)
    t_candidate = float('inf')
    for t in (t1, t2):
        if 0 <= t <= max_range:
            t_candidate = min(t_candidate, t)

    if t_candidate == float('inf'):
        return max_range

    return clamp(t_candidate, 0.0, max_range)


# ---------------- ENVIRONNEMENT ----------------

class LimoSoccerEnv(gym.Env):
    """
    Environnement complet.
    - On garde aussi la méthode step_from_cmd pour accepter des commandes style Twist.
    - reset() -> (obs, {}) : initialise et renvoie l'observation initiale
    - step(action) -> (obs, reward, terminated, truncated, info)
    - render() : affiche la scène (pygame)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = "human"):
        """
        Constructeur :
        - render_mode : "human" pour afficher la fenêtre (pygame), "rgb_array" pour retourner des images.
        """
        super().__init__()
        self.render_mode = render_mode

        # Action : on garde un espace continu (accélération normalisée, rotation normalisée)
        self.action_space = spaces.Box(low=np.array([-1., -1.], dtype=np.float32),
                                       high=np.array([1., 1.], dtype=np.float32),
                                       dtype=np.float32)
        
        """
        23/11 Modification ici pour prendre en compte LIDAR et position
        """
        # Observation : dépend du flag lidar et position
        low = np.array([0.0]*3 + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([LIDAR_MAX_RANGE]*3 + [WIDTH, HEIGHT, WIDTH, HEIGHT], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # États dynamiques (positions, vitesses)
        # voiture : position centre, angle, vitesse linéaire (px/s)
        self.car_pos = np.array([FIELD_LEFT + 20.0, FIELD_TOP + FIELD_H / 2.0])
        self.car_angle = 0.0
        self.car_speed = 0.0

        # balle : position centre, vitesse (px/s)
        self.ball_pos = np.array([FIELD_LEFT + FIELD_W / 2.0, FIELD_TOP + FIELD_H / 2.0])
        self.ball_vel = np.array([0.0, 0.0])

        # compteurs / historiques pour shaping des rewards
        self.step_count = 0
        self.prev_ball_to_goal = None
        self.prev_robot_to_ball = None
        self.goals_scored = 0 # ajout de cette variable pour afficher les buts marqués dans le training

        """"
        Training part
        """
        self.goals_scored = 0
        self.prev_ball_to_goal = None
        self.prev_robot_to_ball = None
        self.prev_ball_vel = np.array([0.0, 0.0])
        self.ball_stagnant_steps = 0
        self.prev_ball_pos = self.ball_pos.copy()


        # rendu pygame (initialisé si render_mode == "human")
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Limo Soccer Env")
            self.clock = pygame.time.Clock()

    # ---------- LIDAR : échantillonnage de rayons ----------
    def compute_sectors(self, samples_per_sector: int = 42) -> Tuple[float, float, float]:
        """
        Fonction : compute_sectors
        --------------------------------
        Objectif :
            Simuler un capteur de type LIDAR pour détecter les obstacles autour de la voiture.
            On découpe le champ de vision en 3 zones :
                - left  (à gauche)
                - front (devant)
                - right (à droite)

        Fonctionnement :
            - Pour chaque zone (secteur), on tire plusieurs rayons (lignes droites)
            dans l’angle correspondant (ex: de -45° à 0° pour la gauche).
            - Pour chaque rayon, on calcule la distance :
                → jusqu’à la balle  (si le rayon la touche)
                → jusqu’aux murs du terrain
            - On garde la plus petite distance trouvée (c’est l’objet le plus proche détecté).
            - À la fin, on retourne trois valeurs (en pixels) : (left, front, right)
            qui représentent les distances minimales détectées dans chaque direction.

        Paramètres :
            samples_per_sector : int
                Nombre de rayons simulés par secteur (plus il y en a, plus la détection est précise,
                mais aussi plus c’est long à calculer).

                #RAYONS PAR SECTEUR REELS : 
                #angle_increment : 0.05 , donc un rayon tous les 0.05 degrés rad
                #donc on est entre -pi, pi donc entre -3.14 et 3.14
                # un rayon tous les 0.05  degrés rad correspond à  6,28/0.05 -> total de 125,6 rayons
                # donc 125,6/3 = 41,86  rayons par secteur dans la vraie vie


        Retour :
            (left, front, right) : Tuple[float, float, float]
                Distances mesurées dans chaque direction (en pixels).
        """

        # --- Position et orientation de la voiture ---
        """
        px, py = self.car_pos       # Coordonnées (x, y) actuelles de la voiture
        """
        # Décalage du LIDAR vers l’avant de la voiture
        # Nouvelle origine du LIDAR (avant de la voiture)
        # Exactement la même formule que pour le placement du phare
        px = self.car_pos[0] + math.cos(self.car_angle) * (CAR_H / 2.0)
        py = self.car_pos[1] + math.sin(self.car_angle) * (CAR_H / 2.0)

        car_ang = self.car_angle    # Angle d’orientation de la voiture (en radians)

        results = []  # Liste qui contiendra la distance minimale pour chaque secteur

        # Parcours des trois secteurs définis globalement dans SECTOR_CONFIG :
        # Exemple de SECTOR_CONFIG :
        # {
        #   "left":  (-np.pi/3, -np.pi/12),
        #   "front": (-np.pi/12,  np.pi/12),
        #   "right": ( np.pi/12,  np.pi/3)
        # }
        # Chaque paire d’angles définit un cône (secteur) devant la voiture.
        #dico.items renvoie une liste de tuples [(clé1, valeur1),...,(cléN, valeurN)]

        
        for name, (a_min, a_max) in SECTOR_CONFIG.items():

            best = LIDAR_MAX_RANGE  # Distance initiale : on part du maximum possible (rien vu encore)

            # On tire plusieurs rayons dans le secteur pour rendre la mesure plus robuste.
            for k in range(samples_per_sector):
                # On répartit uniformément les rayons entre a_min et a_max
                frac = (k + 0.5) / samples_per_sector     # fraction de progression dans le secteur
                theta = a_min + (a_max - a_min) * frac    # angle du rayon (relatif à la voiture)
                ray_ang = car_ang + theta                 # angle absolu du rayon (dans le monde global)

                # ---------- Détection de la balle ----------
                # On calcule la distance entre la voiture et le point où le rayon
                # toucherait la balle (si c’est le cas).
                # - px, py : position de départ du rayon (voiture)
                # - ray_ang : direction du rayon
                # - self.ball_pos : position de la balle
                # - BALL_R : rayon de la balle
                # - LIDAR_MAX_RANGE : distance maximale que le capteur peut "voir"
                
                d_ball = ray_circle_distance(
                    px, py, ray_ang,
                    self.ball_pos[0], self.ball_pos[1],
                    BALL_R, LIDAR_MAX_RANGE
                )

                # ---------- Détection des murs du terrain ----------
                # Même principe, mais on vérifie où le rayon intersecte
                # les bords du terrain rectangulaire.
                d_wall = ray_rect_distance(
                    px, py, ray_ang,
                    FIELD_LEFT, FIELD_TOP, FIELD_RIGHT, FIELD_BOTTOM,
                    LIDAR_MAX_RANGE
                )

                # On garde la plus petite distance (le premier obstacle rencontré)
                best = min(best, d_ball, d_wall)

            # On ajoute la distance la plus courte trouvée pour ce secteur
            results.append(float(best))

        # On retourne les trois distances (gauche, devant, droite)
        return results[0], results[1], results[2]

    # ---------- observation ----------
    """
    Modification le 23/11 pour retourner la position de la balle et du robot en plus du lidar"""
    def _observe(self) -> np.ndarray:
        left, front, right = self.compute_sectors()
        return np.array([
            left, front, right,
            self.car_pos[0], self.car_pos[1],
            self.ball_pos[0], self.ball_pos[1]
        ], dtype=np.float32)

    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Remet l'environnement à zéro.
        Retourne (observation, {}) pour la compatibilité Gymnasium.
        """
        self.seed(seed)

        # position initiale voiture : légèrement à gauche, centre vertical
        self.car_pos = np.array([FIELD_LEFT + 20.0, FIELD_TOP + FIELD_H / 2.0])
        self.car_angle = 0.0
        self.car_speed = 0.0

        # balle au centre
        self.ball_pos = np.array([FIELD_LEFT + FIELD_W  / 2.0, FIELD_TOP + FIELD_H  / 2.0])
        self.ball_vel = np.array([0.0, 0.0])

        self.step_count = 0
        self.prev_ball_to_goal = self._ball_dist_to_left_goal()
        self.prev_robot_to_ball = self._robot_to_ball_dist()

        # init rendu si nécessaire
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        return self._observe(), {}


    # ---------- helpers distances ----------
    """
    Modification pour utiliser le centre de voiture"""
    def _robot_to_ball_dist(self) -> float:
        car_front = self.car_pos + np.array([
            math.cos(self.car_angle),
            math.sin(self.car_angle)
        ]) * (CAR_LENGTH_PX / 2)
        return float(np.linalg.norm(self.ball_pos - car_front))


    def _ball_dist_to_left_goal(self) -> float:
        """Distance entre centre de la balle et centre du but gauche (utile pour shaping)."""
        gx = FIELD_LEFT
        gy = FIELD_TOP + FIELD_H / 2.0
        return float(np.hypot(self.ball_pos[0] - gx, self.ball_pos[1] - gy))
    
    # ---------- détection de but ----------
    def _is_ball_in_left_goal(self) -> bool:
        """
        BUT marqué à gauche :
        - On considère le but marqué si la balle est ENTIEREMENT passée *derrière* la ligne FIELD_LEFT.
          Concrètement : center_x <= FIELD_LEFT - BALL_R.
        - Et la balle doit être dans la zone verticale du but.
        Cette méthode évite de compter un but si la balle touche seulement l'ouverture.
        """
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        fully_past_line = (self.ball_pos[0] <= (FIELD_LEFT - BALL_R))
        vertically_in_goal = (gy0 <= self.ball_pos[1] <= gy1)
        return bool(fully_past_line and vertically_in_goal)

    def _is_ball_in_right_goal(self) -> bool:
        """
        BUT marqué à droite (symétrique) :
        - center_x >= FIELD_RIGHT + BALL_R
        """
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        fully_past_line = (self.ball_pos[0] >= (FIELD_RIGHT + BALL_R))
        vertically_in_goal = (gy0 <= self.ball_pos[1] <= gy1)
        return bool(fully_past_line and vertically_in_goal)

    def _is_out_of_arena(self) -> bool:
        """
        Vérifie si la voiture est sortie trop loin (tolérance de 30 px).
        Utilisé pour terminer l'épisode si la voiture s'est égarée.
        """
        x, y = self.car_pos
        return (x < FIELD_LEFT - 30) or (x > FIELD_RIGHT + 30) or (y < FIELD_TOP - 30) or (y > FIELD_BOTTOM + 30)

    def _is_in_goal_zone(self, x, y) -> bool:
        """
        Renvoie True si le point (x,y) est dans la zone d'ouverture du but (l'ouverture colorée).
        Utilisé pour autoriser la balle à traverser l'ouverture (pas de rebond).
        Attention : ceci n'est pas la détection de but pleine (qui exige la balle entièrement passée).
        """
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        goal_w = int(0.08 * (WIDTH / 800.0) * 100)
        in_left = (FIELD_LEFT <= x <= FIELD_LEFT + goal_w) and (gy0 <= y <= gy1)
        in_right = (FIELD_RIGHT - goal_w <= x <= FIELD_RIGHT) and (gy0 <= y <= gy1)
        return in_left or in_right
    
    def _referee_nudge_ball(self):
        """
        Intervention "arbitre" avec temporisation :

        - Si la balle reste collée contre un mur non-cage (haut ou bas) pendant
        au moins 1.0 seconde (sans quitter la zone), alors l'arbitre la
        replace une fois : posée à l'intérieur (pas d'impulsion, vitesse = 0).
        - Si la balle sort de la zone avant 1s, le timer est réinitialisé.
        - On évite de reposer la balle sur la voiture en la décalant latéralement
        si nécessaire. Le repositionnement tient compte uniquement des murs
        haut/bas (les cages gauche/droite sont ignorées ici).
        """
        # paramètres temporels
        REQUIRED_STAY = 1.0  # secondes à rester dans la zone avant d'intervenir
        now = time.time()

        # positions actuelles de la balle
        bx, by = float(self.ball_pos[0]), float(self.ball_pos[1])

        # seuil : si la distance entre le bord et le bord de la balle est < thresh => zone problématique
        thresh = CAR_W / 2.0      # seuil exprimé en pixels (demi-largeur voiture)
        lateral_shift = CAR_W     # décalage latéral si la balle est trop proche de la voiture

        # helper : test si la balle est dans la zone "collée" au mur haut (hors ouverture)
        in_top_zone = ((by - BALL_R) < (FIELD_TOP + thresh)) and (not self._is_in_goal_zone(bx, by))
        # helper : test si la balle est dans la zone "collée" au mur bas (hors ouverture)
        in_bottom_zone = ((FIELD_BOTTOM - (by + BALL_R)) < thresh) and (not self._is_in_goal_zone(bx, by))

        # initialisation des variables de suivi si elles n'existent pas encore
        # _referee_zone_start : timestamp du moment où la balle est entrée dans la zone
        # _referee_zone_name  : "top" / "bottom" pour se souvenir de quelle zone
        if not hasattr(self, "_referee_zone_start"):
            self._referee_zone_start = None
        if not hasattr(self, "_referee_zone_name"):
            self._referee_zone_name = None

        # Détecter entrée / sortie de zone
        if in_top_zone and not in_bottom_zone:
            current_zone = "top"
        elif in_bottom_zone and not in_top_zone:
            current_zone = "bottom"
        else:
            current_zone = None

        # Si la balle n'est pas dans une zone problématique, on réinitialise le timer
        if current_zone is None:
            self._referee_zone_start = None
            self._referee_zone_name = None
            return  # rien à faire

        # Si on vient juste d'entrer dans une zone *nouvelle*, on démarre le timer
        if self._referee_zone_name != current_zone:
            self._referee_zone_name = current_zone
            self._referee_zone_start = now
            # on attend maintenant REQUIRED_STAY secondes de présence continue
            return

        # Si on est déjà dans la même zone : vérifier la durée
        start = self._referee_zone_start
        if start is None:
            # sécurité : si pour une raison start est None, on remet la montre
            self._referee_zone_start = now
            return

        elapsed = now - start
        if elapsed < REQUIRED_STAY:
            # pas encore le temps requis : on attend
            return

        # Si on arrive ici : la balle est restée dans la même zone pendant >= REQUIRED_STAY
        # => on intervient une seule fois, puis on réinitialise le timer.
        # On place la balle à l'intérieur, sans vitesse (l'arbitre la pose).
        if current_zone == "top":
            new_by = FIELD_TOP + thresh + BALL_R + 1.0  # petit décalage de sécurité
            self.ball_pos[1] = new_by
        else:  # "bottom"
            new_by = FIELD_BOTTOM - (thresh + BALL_R + 1.0)
            self.ball_pos[1] = new_by

        # coupe toute vitesse (arbitre pose la balle)
        self.ball_vel[:] = 0.0

        # éviter de reposer la balle *sur* la voiture : s'assurer d'une distance minimale
        dist_to_car = np.linalg.norm(self.ball_pos - self.car_pos)
        min_safe = (CAR_W + BALL_R) * 0.9
        if dist_to_car < min_safe:
            # vecteur voiture -> balle
            v = self.ball_pos - self.car_pos
            vn = np.linalg.norm(v)
            if vn < 1e-6:
                perp = np.array([1.0, 0.0])  # cas pathologique : on pousse horizontalement
            else:
                perp = np.array([-v[1], v[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-8)

            cand1 = self.ball_pos + perp * lateral_shift
            cand2 = self.ball_pos - perp * lateral_shift

            # clamp horizontaux pour rester dans le terrain
            def clamp_x(x):
                return clamp(x, FIELD_LEFT + BALL_R, FIELD_RIGHT - BALL_R)

            cand1[0] = clamp_x(cand1[0])
            cand2[0] = clamp_x(cand2[0])

            # choisir le candidat le plus éloigné de la voiture
            if np.linalg.norm(cand1 - self.car_pos) > np.linalg.norm(cand2 - self.car_pos):
                self.ball_pos[0] = cand1[0]
                # on garde Y tel quel (déjà fixé)
            else:
                self.ball_pos[0] = cand2[0]

        # s'assurer que la balle reste dans la zone jouable
        self.ball_pos[0] = clamp(self.ball_pos[0], FIELD_LEFT + BALL_R, FIELD_RIGHT - BALL_R)
        self.ball_pos[1] = clamp(self.ball_pos[1], FIELD_TOP + BALL_R, FIELD_BOTTOM - BALL_R)

        # Reset du timer / marqueur pour éviter ré-interventions immédiates
        self._referee_zone_start = None
        self._referee_zone_name = None


    # ---------- STEP (application d'une action) ----------
    def step(self, action: np.ndarray):
        """
        Applique une action et simule une frame.
        - action : array shape (2,) => [accel, steer], chaque composante dans [-1,1]
        Retourne (obs, reward, terminated, truncated, info)
        """

         # ---------- validation / lecture action ----------
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError("Action must be shape (2,)")

        accel = float(clamp(action[0], -1.0, 1.0))   # -1..1
        steer = float(clamp(action[1], -1.0, 1.0))   # -1..1

        # ---------- rotation (steer) ----------
        # La voiture peut tourner sur place proportionnellement à steer.
        angular_speed = steer * MAX_ANGULAR_SPEED
        self.car_angle = angle_normalize(self.car_angle + angular_speed * DT)

        # ---------- vitesse linéaire ----------
        # target_speed est la vitesse souhaitée (accel * vmax)
        target_speed = accel * MAX_LINEAR_SPEED
        speed_diff = target_speed - self.car_speed
        max_accel = ACCELERATION * DT
        speed_diff = clamp(speed_diff, -max_accel, max_accel)
        self.car_speed += speed_diff

        # si l'utilisateur n'accélère pas, on applique un fort freinage pour éviter le 'glissement'
        if abs(accel) < 1e-3:
            self.car_speed *= (1.0 - LINEAR_DAMP * DT)

        # ---------- déplacement de la voiture ----------
        dx = math.cos(self.car_angle) * self.car_speed * DT
        dy = math.sin(self.car_angle) * self.car_speed * DT
        self.car_pos += np.array([dx, dy])

        # ---------- collision voiture-balle ----------
        # On traite la collision centre-centre (approximation suffisante).
        diff = self.ball_pos - self.car_pos
        dist = np.linalg.norm(diff)
        contact_dist = (CAR_LENGTH_PX / 2.0) + BALL_R  # distance où les deux 'touchent' approximativement

        if dist < contact_dist and dist > 1e-6:
            # normale dirigée de la voiture vers la balle
            n = diff / dist

            # 1) Correction de pénétration : on sépare les deux objets
            overlap = contact_dist - dist
            # on déplace surtout la balle (on suppose la voiture plus lourde)
            self.ball_pos += n * (overlap * 0.6)
            # on ajuste aussi la voiture un peu pour éviter qu'elle traverse
            """
            Peut être à supprimer dans la vrai vie la voiture recule par lorsqu'elle tape le ballon
            """
            self.car_pos -= n * (overlap * 0.4) 

            # 2) Calcul de la vitesse relative le long de la normale
            car_vel_vec = np.array([math.cos(self.car_angle) * self.car_speed,
                                    math.sin(self.car_angle) * self.car_speed])
            
            #np.dot() effectue une multiplication matricielle entre les deux tableaux d'entrée
            rel_vel = np.dot(self.ball_vel - car_vel_vec, n)

            # 3) Si les objets se rapprochent (rel_vel < 0) -> appliquer une impulsion (rebond + transfert d'énergie)
            if rel_vel < 0:
                impulse = -(1 + BALL_RESTITUTION) * rel_vel
                # on applique plus d'impulsion si la voiture arrive vite (BALL_PUSH_FACTOR)
                self.ball_vel += n * impulse * BALL_PUSH_FACTOR
                # la voiture perd un peu d'énergie lors de l'impact
                self.car_speed *= 0.9

            """
            Ajout limiteur de vitesse de la balle
            """
            # limiter la vitesse de la balle
            vnorm = np.linalg.norm(self.ball_vel)
            if vnorm > BALL_MAX_SPEED:
                self.ball_vel *= BALL_MAX_SPEED / (vnorm + 1e-8)

        # ---------- intégration de la balle ----------
        # on met à jour la position de la balle par intégration simple Euler
        self.ball_pos += self.ball_vel * DT
        # on applique un amortissement pour simuler frottements (air/sol)
        self.ball_vel *= BALL_LINEAR_DAMP

        # ---------- collisions balle avec murs, en autorisant l'entrée dans l'ouverture du but ----------
        """
        Modification le 25/11 pour éviter les rebonds dans les coins qui provoquait des erreurs
        """
        bx, by = self.ball_pos 
        vx, vy = self.ball_vel

        # limites verticales du but
        goal_top = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        goal_bottom = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0

        # zone où on supprime les rebonds dans les coins (plus large que l'ouverture)
        no_bounce_top = FIELD_TOP + FIELD_H * 0.15
        no_bounce_bottom = FIELD_BOTTOM - FIELD_H * 0.15

        # Rebonds larges (zone étendue)
        if bx - BALL_R < FIELD_LEFT - GOAL_DEPTH:
            bx = FIELD_LEFT - GOAL_DEPTH + BALL_R
            vx = abs(vx) * BALL_RESTITUTION

        elif bx + BALL_R > FIELD_RIGHT + GOAL_DEPTH:
            bx = FIELD_RIGHT + GOAL_DEPTH - BALL_R
            vx = -abs(vx) * BALL_RESTITUTION

        # Rebonds haut/bas
        if by - BALL_R < FIELD_TOP:
            by = FIELD_TOP + BALL_R
            vy = abs(vy) * BALL_RESTITUTION

        elif by + BALL_R > FIELD_BOTTOM:
            by = FIELD_BOTTOM - BALL_R
            vy = -abs(vy) * BALL_RESTITUTION

        # Rebonds gauche hors ouverture MAIS PAS dans les coins
        if (
            bx - BALL_R < FIELD_LEFT
            and not (goal_top <= by <= goal_bottom)
            and not (by < no_bounce_top or by > no_bounce_bottom)
        ):
            bx = FIELD_LEFT + BALL_R
            vx = abs(vx) * BALL_RESTITUTION

        # Rebonds droite hors ouverture MAIS PAS dans les coins
        if (
            bx + BALL_R > FIELD_RIGHT
            and not (goal_top <= by <= goal_bottom)
            and not (by < no_bounce_top or by > no_bounce_bottom)
        ):
            bx = FIELD_RIGHT - BALL_R
            vx = -abs(vx) * BALL_RESTITUTION

        self.ball_pos = np.array([bx, by])
        self.ball_vel = np.array([vx, vy])


        # ----------- ARBITRE : décolle et replace la balle si coincée (mur ou coin) -----------
        self._referee_nudge_ball()

       # ---------- empêcher la voiture de traverser les murs (sauf via l'ouverture du but) ----------
        cx, cy = self.car_pos
        half_w, half_h = CAR_W / 2, CAR_H / 2

        # Si la voiture essaie de traverser à gauche (hors ouverture), on la bloque
        # si la voiture rentre dans le mur on la decal devant et on reduit sa vitesse pour simuler le fait qu'elle se cogne
        if cx - half_w < FIELD_LEFT:
            if not (goal_top <= cy <= goal_bottom):
                self.car_pos[0] = FIELD_LEFT + half_w
                # rebond / perte d'énergie pour voiture
                self.car_speed *= -0.3

        # à droite
        if cx + half_w > FIELD_RIGHT:
            if not (goal_top <= cy <= goal_bottom):
                self.car_pos[0] = FIELD_RIGHT - half_w
                self.car_speed *= -0.3

        # haut / bas
        if cy - half_h < FIELD_TOP:
            self.car_pos[1] = FIELD_TOP + half_h
            self.car_speed *= -0.3
        if cy + half_h > FIELD_BOTTOM:
            self.car_pos[1] = FIELD_BOTTOM - half_h
            self.car_speed *= -0.3
        
        """
        Modification le 23/11 pour reward simple proposé

        # -----------------------
        # Reward Shaping Football RL - Correctif Robot Attiré vers Balle
        # -----------------------
        # -------------------- Constantes --------------------
        REWARD_GOAL = 50.0           # gros bonus quand on marque
        REWARD_BAD_GOAL = -5.0       # mal marquer dans le mauvais but
        REWARD_OUT = -10.0            # sortir du terrain

        # ---------- recompense (reward simple, normalisée) ----------
        obs = self._observe()

        # distances (en pixels)
        car_front = self.car_pos + np.array([math.cos(self.car_angle), math.sin(self.car_angle)]) * (CAR_LENGTH_PX / 2.0)
        robot_to_ball_px = float(np.linalg.norm(self.ball_pos - car_front) + 1e-8)
        ball_to_goal_px = float(self._ball_dist_to_left_goal())

        # conversion pixels -> mètres (valeurs en m)
        robot_to_ball_m = robot_to_ball_px / PIXELS_PER_METER
        ball_to_goal_m = ball_to_goal_px / PIXELS_PER_METER

        # seuil "près" : 20 cm
        NEAR_THRESH_M = 0.2
        if robot_to_ball_m < NEAR_THRESH_M:
            robot_to_ball_m = 0.0

        if ball_to_goal_m < 0.005:  # si la balle est quasiment au but, la distance est 0
            ball_to_goal_m = 0.0

        # reward de base : on punit les distances (plus petit = mieux)
        # on normalise pour garder des amplitudes raisonnables
        # R = -d_robot_ball - d_ball_goal
        REWARD_SCALE = 0.1  # on peut ajuster si besoin
        reward = -REWARD_SCALE * (robot_to_ball_m + ball_to_goal_m)

        # bonus / malus
        if self._is_ball_in_left_goal():
            reward += REWARD_GOAL          # gros bonus pour avoir mis dans le bon but
            self.goals_scored = getattr(self, "goals_scored", 0) + 1
            # replacer la balle au centre (sans terminer l'épisode)
            self.ball_pos = np.array([FIELD_LEFT + FIELD_W / 2.0, FIELD_TOP + FIELD_H / 2.0])
            self.ball_vel = np.array([0.0, 0.0])

        if self._is_ball_in_right_goal():
            reward += REWARD_BAD_GOAL      # malus pour mauvais but
            # replacer la balle
            self.ball_pos = np.array([FIELD_LEFT + FIELD_W / 2.0, FIELD_TOP + FIELD_H / 2.0])
            self.ball_vel = np.array([0.0, 0.0])

        # si la voiture sort du terrain -> pénalité forte et forcer fin d'épisode
        if self._is_out_of_arena():
            reward += REWARD_OUT
            # forcer la fin par timeout
            self.step_count = MAX_EPISODE_STEPS

        # gestion fin d'épisode par time-limit
        terminated = False
        truncated = (self.step_count >= MAX_EPISODE_STEPS)

        # mise à jour historiques (si utilisés ailleurs)
        self.prev_ball_to_goal = ball_to_goal_px
        self.prev_robot_to_ball = robot_to_ball_px

        self.step_count += 1

        info = {}
        return obs, float(reward), terminated, truncated, info
        """
        # -------------------- REWARDS --------------------
        REWARD_GOAL = 10.0           # gros bonus quand on marque
        REWARD_BAD_GOAL = -5.0       # mal marquer dans le mauvais but
        REWARD_OUT = -2.0            # sortir du terrain
        # ---------- recompense (reward shaping) ----------
        obs = self._observe()

        # --- PARAMETERS (coefficients calibrés pour PPO) ---
        STEP_PENALTY = -0.01          # petit coût pour inciter à finir vite
        APPROACH_COEF = 0.8           # encourager aller vers la balle
        BALL_GOAL_COEF = 3.0          # encourager la balle vers le but (fort)
        ALIGNMENT_COEF = 0.5          # bonus d'alignement derrière la balle
        CONTACT_BONUS = 5.0           # bonus quand on "touche" efficacement la balle
        BALL_VEL_GOAL_COEF = 2.0      # récompense pour vitesse de la balle vers le but
        WALL_PENALTY = -0.5           # pénalité si la balle heurte le mur (non-goal)
        STAGNATION_PENALTY = -0.2     # pénalité si la balle reste immobile trop longtemps
        MIN_CONTACT_DIST = 30.0       # seuil distance (px) pour considérer contact/near-hit
        STAGNANT_VEL_THRESH = 1e-2    # seuil pour considérer la balle immobile

        # --- compute reference points ---
        # front of car (approx LIDAR position)
        car_front = self.car_pos + np.array([math.cos(self.car_angle), math.sin(self.car_angle)]) * (CAR_LENGTH_PX / 2.0)

        # distances
        robot_to_ball_vec = self.ball_pos - car_front
        robot_ball_dist = float(np.linalg.norm(robot_to_ball_vec) + 1e-8)

        ball_goal_dist = self._ball_dist_to_left_goal()  # existe déjà

        # deltas (based on previous step)
        delta_rb = (self.prev_robot_to_ball - robot_ball_dist) if self.prev_robot_to_ball is not None else 0.0
        delta_goal = (self.prev_ball_to_goal - ball_goal_dist) if self.prev_ball_to_goal is not None else 0.0

        # base reward
        reward = STEP_PENALTY

        # 1) Reward pour se rapprocher de la balle (signal dense)
        reward += APPROACH_COEF * delta_rb

        # 2) Reward pour déplacer la balle vers le but (shaping fort)
        reward += BALL_GOAL_COEF * delta_goal

        # 3) Bonus d'alignement : position (cosine) entre robot->ball et ball->goal
        b = self.ball_pos
        ball_to_goal = np.array([FIELD_LEFT - b[0], (FIELD_TOP + FIELD_H / 2.0) - b[1]])
        norm_rb = np.linalg.norm(robot_to_ball_vec) + 1e-6
        norm_bg = np.linalg.norm(ball_to_goal) + 1e-6
        cos_align = (robot_to_ball_vec @ ball_to_goal) / (norm_rb * norm_bg)
        # on donne un bonus si on est derrière la balle (orientation utile)
        if cos_align > 0.3:
            reward += ALIGNMENT_COEF * cos_align

        # 4) Reward basé sur la vitesse de la balle vers le but
        # projection de la vitesse sur la direction goal
        ball_vel = self.ball_vel
        if np.linalg.norm(ball_vel) > 1e-8:
            dir_to_goal = ball_to_goal / (np.linalg.norm(ball_to_goal) + 1e-8)
            vel_toward_goal = float(ball_vel @ dir_to_goal)   # positive si vers le but gauche
            # on récompense la vitesse positive vers le but, pénalise si on la pousse à l'opposé
            reward += BALL_VEL_GOAL_COEF * vel_toward_goal * (1.0 / (PIXELS_PER_METER + 1e-8))

        # 5) Contact/impact bonus : détecte si la voiture vient de réduire fortement la distance
        # OU si la balle a subi une accélération (ball vel change) — signe d'un coup réussi
        ball_vel_change = np.linalg.norm(ball_vel - (self.prev_ball_vel if hasattr(self, "prev_ball_vel") else np.array([0.0,0.0])))
        ball_hit = False
        # Condition 1 : grosse réduction de distance cette frame (robot approche very close)
        if (self.prev_robot_to_ball is not None) and (self.prev_robot_to_ball - robot_ball_dist > 5.0) and (robot_ball_dist < MIN_CONTACT_DIST):
            ball_hit = True
        # Condition 2 : gros changement de vitesse de la balle (impact)
        if ball_vel_change > 20.0:  # seuil empirique : ajuster si nécessaire
            ball_hit = True

        if ball_hit:
            reward += CONTACT_BONUS

        # 6) Penalty si la balle heurte les murs sans avancer vers le but (anti-exploit)
        # On détecte heurtoir en comparant position précédente et actuelle : si la balle a rebondi
        if (self.prev_ball_pos is not None):
            # heuristique : si la balle a changé de signe de vx ou vy fortement -> prob rebond
            prev_v = self.prev_ball_pos - (getattr(self, "_prev_prev_ball_pos", self.prev_ball_pos))
            cur_v = self.ball_pos - self.prev_ball_pos
            # si la composante x ou y a changé de direction fortement -> probable rebond
            if np.dot(prev_v, cur_v) < 0 and (abs(prev_v).max() > 1e-6):
                reward += WALL_PENALTY

        # 7) Anti-loop / stagnation: si la balle ne bouge presque pas pendant plusieurs pas
        ball_moved = np.linalg.norm(self.ball_pos - self.prev_ball_pos) if self.prev_ball_pos is not None else 1.0
        if ball_moved < STAGNANT_VEL_THRESH:
            self.ball_stagnant_steps = getattr(self, "ball_stagnant_steps", 0) + 1
        else:
            self.ball_stagnant_steps = 0

        # si stagnation longue, on applique une pénalité croissante
        if self.ball_stagnant_steps > FPS * 2:  # 2 secondes immobile
            reward += STAGNATION_PENALTY * min(5.0, (self.ball_stagnant_steps / FPS))

        # ---------- terminal / goal ---------- 
        terminated = False
        truncated = False

        if self._is_ball_in_left_goal():
            reward += REWARD_GOAL
            self.goals_scored = getattr(self, "goals_scored", 0) + 1
            terminated = True
        elif self._is_ball_in_right_goal():
            reward += REWARD_BAD_GOAL
            terminated = True

        if self._is_out_of_arena():
            reward += REWARD_OUT
            terminated = True

        if self.step_count >= MAX_EPISODE_STEPS:
            truncated = True

        # pénalité forte si la balle reste immobile trop longtemps (fallback)
        if np.linalg.norm(self.ball_vel) < 0.001 and self.step_count > FPS * 10:
            reward -= 50.0
            truncated = True

        # --- update history values for next step ---
        self.prev_ball_to_goal = ball_goal_dist
        self.prev_robot_to_ball = robot_ball_dist
        self.prev_ball_vel = ball_vel.copy()
        # store ball previous positions for wall/rebound heuristic
        self._prev_prev_ball_pos = getattr(self, "prev_ball_pos", self.ball_pos.copy())
        self.prev_ball_pos = self.ball_pos.copy()

        self.step_count += 1


        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ---------- API auxiliaire : recevoir des commandes style Twist ----------
    """
    Remise de la fonction pour convertir mais surement à modifier car pour vitesse lineaire on a que selon x (après peut être normal / à vérifier)
    """
    def step_from_cmd(self, cmd: Dict[str, Any]):
        """
        Convertit un dictionnaire ressemblant à un Twist ROS en action pour step().
        Exemple d'entrée :
          { "linear_x": 0.2, "angular_z": 0.5, "emergency_stop": False }

        - Si emergency_stop True => on met action [0,0].
        - Sinon on normalise linear_x par MAX_LINEAR_SPEED pour obtenir accel dans [-1,1],
          et angular_z par MAX_ANGULAR_SPEED pour steer dans [-1,1].
        Retourne le tuple renvoyé par step(action).
        """

        # Si l'appelant veut stopper d'urgence, on force action 0
        if cmd.get("emergency_stop", False):
            action = np.array([0.0, 0.0], dtype=np.float32)
        else:
            # récupère valeurs (met 0 par défaut si absent)
            lin = float(cmd.get("linear_x", 0.0))
            ang = float(cmd.get("angular_z", 0.0))

            # on mappe lin (px/s) -> accel normalisé, pareil pour ang -> steer normalisé
            # clamp pour rester dans [-1,1]
            accel = clamp(lin / MAX_LINEAR_SPEED, -1.0, 1.0)
            steer = clamp(ang / MAX_ANGULAR_SPEED, -1.0, 1.0)
            action = np.array([accel, steer], dtype=np.float32)

        # appelle la méthode principale step()
        return self.step(action)

    # ---------- rendu ----------
    def render(self):
        """
        Affichage Pygame : dessine le terrain, les buts, la balle et la voiture.
        Rien d'exécutif ici (ne change pas la physique), juste visualisation.
        """
        if self.render_mode is None:
            return None
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        # gérer événements (nécessaire pour que la fenêtre reste responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        surf = self.screen
        surf.fill(COLOR_BG)

        # terrain vert et contour blanc
        pygame.draw.rect(surf, COLOR_FIELD, (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H))
        pygame.draw.rect(surf, COLOR_BORDER, (FIELD_LEFT, FIELD_TOP, FIELD_W, FIELD_H), 3)

        # but gauche (zone colorée)
        gx0 = FIELD_LEFT
        gy0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        gx1 = FIELD_LEFT + int(0.08 * WIDTH / 800.0 * 100)
        gy1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        left_rect = pygame.Rect(int(gx0), int(gy0), int(gx1 - gx0), int(gy1 - gy0))
        pygame.draw.rect(surf, COLOR_GOAL, left_rect)

        # but droit
        rx0 = FIELD_RIGHT - int(0.08 * WIDTH / 800.0 * 100)
        ry0 = FIELD_TOP + FIELD_H / 2.0 - 0.3 * FIELD_H / 2.0
        rx1 = FIELD_RIGHT
        ry1 = FIELD_TOP + FIELD_H / 2.0 + 0.3 * FIELD_H / 2.0
        right_rect = pygame.Rect(int(rx0), int(ry0), int(rx1 - rx0), int(ry1 - ry0))
        pygame.draw.rect(surf, COLOR_BAD_GOAL, right_rect)

        # balle (cercle)
        pygame.draw.circle(surf, COLOR_BALL, (int(self.ball_pos[0]), int(self.ball_pos[1])), BALL_R)

        # voiture (rectangle tourné)
        car_surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, COLOR_CAR, (0, 0, CAR_W, CAR_H), border_radius=6)
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.car_angle))
        rect = rotated.get_rect(center=(int(self.car_pos[0]), int(self.car_pos[1])))
        surf.blit(rotated, rect.topleft)

        # phare (petit point devant la voiture)
        fx = self.car_pos[0] + math.cos(self.car_angle) * (CAR_H / 2.0)
        fy = self.car_pos[1] + math.sin(self.car_angle) * (CAR_H / 2.0)
        pygame.draw.circle(surf, COLOR_HEADLIGHT, (int(fx), int(fy)), 5)

        # affichage texte utile (observations)
        obs = self._observe()
        font = pygame.font.SysFont("Consolas", 14)
        txt = f"step:{self.step_count} obs:{np.round(obs,2)}"
        surf.blit(font.render(txt, True, (0, 0, 0)), (6, 6))

        # rafraîchir fenêtre
        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(FPS)
        else:
            # si on veut récupérer une image numpy
            return np.transpose(np.array(pygame.surfarray.array3d(surf)), (1, 0, 2))

    def close(self):
        """Ferme pygame proprement. Appeler à la fin si on a ouvert la fenêtre."""
        try:
            if self.screen is not None:
                pygame.quit()
                self.screen = None
        except Exception:
            pass

    def seed(self, seed=None):
        """Initialise le RNG interne (Gym-compatible)."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]


# ---------------- TEST MANUEL ----------------
if __name__ == "__main__":
    print("=== Limo Soccer Env (manual test) ===")
    
    env = LimoSoccerEnv(render_mode="human")
    obs, _ = env.reset()
    running = True
    use_keyboard = True  
    reward_total = 0
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
        #print de la reward
        print(f"Reward = {reward}")
        reward_total += reward
        env.render()

        # gestion fermeture fenêtr
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # si épisode fini on attend 0.8s pour regarder puis reset
        if term or trunc:
            time.sleep(0.8)
            obs, _ = env.reset()
        

    env.close()
    print("closing")
    print(f"Reward totale = {reward_total}")