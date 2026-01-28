"""
Analyse des résultats duel entre agents.

- Lit results_duel.csv
- Nettoie et convertit les résultats en valeurs numériques
- Calcule win rate moyen, buts marqués et concédés
- Sauvegarde un CSV résumé et deux graphiques (win rate et goals)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# ------------------- CONFIG -------------------
INPUT_CSV = os.path.join("evaluation", "results_duel.csv")
OUTPUT_DIR = "evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------- LIRE LES DONNEES -------------------
df = pd.read_csv(INPUT_CSV)

# Assurer que chaque ligne contient un seul résultat correct
df["result"] = df["result"].str.strip().str.lower()
valid_results = ["win", "draw", "lose"]
df = df[df["result"].isin(valid_results)]

# ------------------- CONVERTIR RESULTATS EN NUMERIQUE -------------------
# win=1, draw=0.5, lose=0
result_map = {"win": 1, "draw": 0.5, "lose": 0}
df["result_num"] = df["result"].map(result_map)

# ------------------- RENOMMER LES AGENTS -------------------
# Si tu as des colonnes 'agent' avec "base" ou "finetune"
df["agent_name"] = df["agent"].replace({"base": "Base", "finetune": "Finetune"})

# ------------------- CALCUL DES STATISTIQUES -------------------
summary = df.groupby("agent_name").agg({
    "result_num": "mean",          # Win rate moyen
    "goals_scored": "mean",
    "goals_conceded": "mean"
}).reset_index()

summary.rename(columns={
    "result_num": "win_rate_%",
    "goals_scored": "avg_goals_scored",
    "goals_conceded": "avg_goals_conceded"
}, inplace=True)

summary["win_rate_%"] *= 100

# Sauvegarde CSV
summary_csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary.to_csv(summary_csv_path, index=False)
print(f"Résumé statistique sauvegardé : {summary_csv_path}")

# ------------------- VISUALISATION -------------------
# Bar plot Win rate
plt.figure(figsize=(6,4))
plt.bar(summary["agent_name"], summary["win_rate_%"], color=["skyblue","salmon"])
plt.ylabel("Win rate (%)")
plt.title("Win rate moyen par agent")
plt.ylim(0,100)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, "win_rate.png"))
plt.close()

# Bar plot Avg Goals Scored / Conceded
plt.figure(figsize=(6,4))
width = 0.35
x = range(len(summary))
plt.bar(x, summary["avg_goals_scored"], width, label="Goals Scored", color="lightgreen")
plt.bar([i+width for i in x], summary["avg_goals_conceded"], width, label="Goals Conceded", color="lightcoral")
plt.xticks([i+width/2 for i in x], summary["agent_name"])
plt.ylabel("Goals")
plt.title("Buts marqués et concédés moyens")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.savefig(os.path.join(OUTPUT_DIR, "goals_avg.png"))
plt.close()

print("Plots sauvegardés dans le dossier evaluation")
