import os
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# PATHS
# ======================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

INPUT_CSV = os.path.join(EVAL_DIR, "results_duel.csv")
PLOT_DIR = os.path.join(EVAL_DIR, "plots")
SUMMARY_PATH = os.path.join(EVAL_DIR, "summary.csv")

os.makedirs(PLOT_DIR, exist_ok=True)

# ======================================================
# LOAD DATA
# ======================================================

df = pd.read_csv(INPUT_CSV)

# ======================================================
# METRICS
# ======================================================

df["win"] = (df["result"] == "win").astype(int)
df["loss"] = (df["result"] == "lose").astype(int)
df["draw"] = (df["result"] == "draw").astype(int)

df["goal_diff"] = df["goals_scored"] - df["goals_conceded"]

# ======================================================
# SUMMARY TABLE
# ======================================================

summary = df.groupby("agent").agg(
    matches=("episode", "count"),
    win_rate=("win", "mean"),
    draw_rate=("draw", "mean"),
    loss_rate=("loss", "mean"),
    goals_scored=("goals_scored", "mean"),
    goals_conceded=("goals_conceded", "mean"),
    goal_diff=("goal_diff", "mean"),
).reset_index()

summary[["win_rate", "draw_rate", "loss_rate"]] *= 100
summary.to_csv(SUMMARY_PATH, index=False)

print("Résumé statistique sauvegardé :", SUMMARY_PATH)
print(summary)

# ======================================================
# PLOTS
# ======================================================

def bar(metric, ylabel, title=None):
    plt.figure()
    summary.set_index("agent")[metric].plot(kind="bar")
    plt.ylabel(ylabel)
    plt.title(title if title else metric)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{metric}.png"))
    plt.close()

# --- Barplots principaux ---
bar("win_rate", "Win rate (%)", "Win rate par agent")
bar("goals_scored", "Buts marqués", "Buts marqués par match")
bar("goals_conceded", "Buts concédés", "Buts concédés par match")
bar("goal_diff", "Différence de buts", "Différence moyenne de buts")

# --- Boxplots (distribution) ---
plt.figure()
df.boxplot(column="goals_scored", by="agent")
plt.title("Distribution des buts marqués")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "box_goals_scored.png"))
plt.close()

plt.figure()
df.boxplot(column="goals_conceded", by="agent")
plt.title("Distribution des buts concédés")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "box_goals_conceded.png"))
plt.close()

plt.figure()
df.boxplot(column="goal_diff", by="agent")
plt.title("Distribution de la différence de buts")
plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "box_goal_diff.png"))
plt.close()

print("Graphiques générés dans :", PLOT_DIR)
