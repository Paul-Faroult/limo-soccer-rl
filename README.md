# 🤖 Limo Soccer RL

Projet de recherche et d’expérimentation en **apprentissage par renforcement (Reinforcement Learning)** appliqué à des robots mobiles de type LIMO évoluant dans un environnement de football robotique.

L’objectif est de construire une pipeline **progressive** allant d’un agent seul jusqu’à un **duel compétitif**, en étudiant la stabilité de l’apprentissage, la généralisation et les performances des politiques PPO.

---

## 🎯 Objectifs du projet

- Concevoir un **environnement Gymnasium** personnalisé pour le football robotique
- Entraîner un agent via **PPO (Stable-Baselines3)**
- Apprendre la navigation et le contrôle du ballon de manière robuste  
- Gérer des comportements adverses  
- Comparer les approches classiques et hiérarchiques d'apprentissage par renforcement
- Comparer les performances des modèles (winrate, buts marqués/encaissés)

Ce projet s’inscrit dans un cadre académique (Projet 5A) et vise une qualité reproductible et analysable.

---

## 🧩 Pipeline expérimentale

Le projet est entièrement versionné via des branches Git :

### Étapes principales (validées)

- **Étape 1 — But cage (basé sur Lidar)**  
  - Position initiale du robot fixe  
  - Ballon au centre  
  - Observations basées uniquement sur le Lidar  

- **Étape 2 — RL classique**  
  - Positions aléatoires du robot et du ballon  
  - PPO avec γ = 0,99  

- **Étape 3 — Adversaire statique**  
  - Adversaire figé  
  - Récompense explicite pour éviter les collisions  

- **Étape 4 — Duel**  
  - Deux agents, buts indépendants  
  - Apprentissage par renforcement compétitif

### Expérimentations (abandonnées ou exploratoires)

- Apprentissage par renforcement hiérarchique  
- PPO avec γ = 0,995  
- Adversaire statique sans pénalité de collision

---

## 📁 Structure du dépôt

- `main` → Implémentation finale du duel
```
limo-soccer-rl/
│
├── envs/
│ ├── limo_soccer_env.py # Environnement solo
│ ├── limo_soccer_env_static.py # Adversaire statique
│ └── limo_soccer_env_duel.py # Duel 1v1
│
├── train/
│ ├── train_solo.py
│ ├── train_static.py
│ └── train_duel.py
│
├── eval/
│ ├── eval_model.py
│ ├── test_vs_models_duel.py
│ └── analyze_results.py
│
├── models/ # Modèles entraînés (ignoré par git)
├── logs/ # TensorBoard logs (ignoré par git)
│
├── requirements.txt
├── README.md
└── .gitignore
```
- `dev` → Branche d'intégration  
- `stage/*` → Jalons de développement validés  
- `experiment/*` → Approches exploratoires ou abandonnées

---

## Outils & bibliothèques

- Python  
- Stable-Baselines3 (PPO)  
- Gymnasium  
- NumPy  
- PyGame (affichage)  
- TensorBoard (analyse de l'entraînement)

---

## ⚙️Installation

### 1. Cloner le dépôt

```
git clone https://github.com/Paul-Faroult/limo-soccer-rl.git
cd limo-soccer-rl
```

### Créer un environnement virtuel

```
python -m venv venv
source venv/bin/activate # Linux / Mac
venv\Scripts\activate # Windows
```

### Installer les dépendances

```
pip install -r requirements.txt
```

---

## 🚀 Entraînement

### Agent solo

## Résultats

L'agent final montre :  

- Interception du ballon robuste  
- Évitement adaptatif de l’adversaire  
- Comportement de marquage de but compétitif

---

## Auteurs

- Paul Faroult  
- Ba Thong Nguyen  

Projet académique développé dans le cadre d'expérimentations en apprentissage par renforcement.
