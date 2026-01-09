# Limo Soccer RL

## Football robotique autonome avec apprentissage par renforcement

Ce projet explore l'entraînement d'un robot autonome sur roues pour jouer à un jeu de football simplifié, en utilisant des techniques d'apprentissage par renforcement (PPO).

Le travail suit une **pipeline expérimentale progressive**, allant d'une tâche simple de marquage de but jusqu'à un duel multi-agents compétitif.

---

## 🎯 Objectifs

- Apprendre la navigation et le contrôle du ballon de manière robuste  
- Éviter le sur-apprentissage sur des trajectoires fixes  
- Gérer des comportements adverses  
- Comparer les approches classiques et hiérarchiques d'apprentissage par renforcement

---

## 🧪 Pipeline expérimentale

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

## 📂 Structure du dépôt

- `main` → Implémentation finale du duel  
- `dev` → Branche d'intégration  
- `stage/*` → Jalons de développement validés  
- `experiment/*` → Approches exploratoires ou abandonnées

---

## 🛠️ Outils & bibliothèques

- Python  
- Stable-Baselines3 (PPO)  
- Gymnasium  
- NumPy  
- PyGame (affichage)  
- TensorBoard (analyse de l'entraînement)

---

## 📈 Résultats

L'agent final montre :  

- Interception du ballon robuste  
- Évitement adaptatif de l’adversaire  
- Comportement de marquage de but compétitif

---

## ✍️ Auteurs

- Paul Faroult  
- Ba Thong Nguyen  

Projet académique développé dans le cadre de recherches et expérimentations en apprentissage par renforcement.
