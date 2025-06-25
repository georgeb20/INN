import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import joblib

# --- Load Optuna study object ---
study = joblib.load("./saved_network/optuna_study.pkl")
# Place L_inv first, then the rest
obj_names = ["L_inv", "L_pred", "L_latent", "L_pad", "L_boundary"]

# --- Build Pareto DataFrame directly from study.best_trials ---
pareto = study.best_trials
pareto_data = []
for t in pareto:
    row = {
        "Trial": t.number,
        **{name: val for name, val in zip(["L_pred", "L_inv", "L_latent", "L_pad", "L_boundary"], t.values)},
        **t.params
    }
    pareto_data.append(row)
pareto_df = pd.DataFrame(pareto_data)

# --- Select top 5 by L_inv ---
top5 = pareto_df.nsmallest(5, "L_inv").reset_index(drop=True)
trials = top5["Trial"].astype(str)

# --- Bar plot for all losses in top 5 ---
bar_width = 0.15
x = np.arange(5)

best_idx =2

# --- Highlight best overall trial in bar plot ---
plt.figure(figsize=(10, 6))
for i, loss in enumerate(obj_names):
    plt.bar(x + i * bar_width, top5[loss], width=bar_width, label=loss)
# Draw best trial bars again, with black edge and transparent fill (so only outline shows)
plt.bar(
    x[best_idx] + np.arange(len(obj_names)) * bar_width,
    top5.loc[best_idx, obj_names],
    width=bar_width,
    fill=False,                # Only show the edge
    edgecolor="black",
    linewidth=2
)
plt.xticks(x + 2 * bar_width, trials)
plt.xlabel("Trial #")
plt.ylabel("Loss Value")
plt.title("Top 5 Trials (Lowest L_inv)")
plt.legend()
plt.tight_layout()
plt.savefig("./saved_network/top5_Linv_barplot_best.png", dpi=300)
plt.show()
print("Wrote saved_network/top5_Linv_barplot_best.png")


# --- (Optional) Save the top5 table as CSV for supplement ---
top5.to_csv("./saved_network/top5_Linv_losses.csv", index=False)
print("Wrote saved_network/top5_Linv_losses.csv")
