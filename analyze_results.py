import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
CSV_PATH = os.path.join(EVAL_DIR, "results.csv")
PLOT_DIR = os.path.join(EVAL_DIR, "plots")
SUMMARY_PATH = os.path.join(EVAL_DIR, "summary.csv")

os.makedirs(PLOT_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(CSV_PATH)

# ======================================================
# SUMMARY STATISTICS
# ======================================================

summary = df.groupby("model").agg({
    "win": "mean",
    "goals": "mean",
    "goals_conceded": "mean",
    "collisions": "mean",
    "reward": "mean",
    "steps": "mean"
}).reset_index()

summary["win"] *= 100
summary.rename(columns={"win": "win_rate_%"}, inplace=True)
summary.to_csv(SUMMARY_PATH, index=False)

print("Résumé statistique sauvegardé :", SUMMARY_PATH)

# ======================================================
# PLOTS
# ======================================================

def bar(metric, ylabel):
    plt.figure()
    df.groupby("model")[metric].mean().plot(kind="bar")
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric}.png"))
    plt.close()

# Barplots simples
bar("win", "Win rate (%)")
bar("goals", "Goals")
bar("collisions", "Collisions")
bar("reward", "Reward")

# Boxplots
plt.figure()
df.boxplot(column=["goals", "collisions", "reward"], by="model")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "boxplots.png"))
plt.close()

print("Graphiques générés dans :", PLOT_DIR)
