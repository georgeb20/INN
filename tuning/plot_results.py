# tuning/plot_results.py
import joblib, matplotlib.pyplot as plt, optuna, os
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)

study = joblib.load("./saved_network/optuna_study.pkl")

# 1) Optuna’s built-in interactive HTML → PNG/HTML
for viz_fn, name in [
    (plot_optimization_history, "history"),
    (plot_param_importances,   "importance"),
    (plot_parallel_coordinate, "parallel"),
    (plot_slice,               "slice")
]:
    fig = viz_fn(study)
    fig.write_html(f"./saved_network/{name}.html")
    print(f"Wrote saved_network/{name}.html")

# 2) Custom Matplotlib: loss vs λ_forward
import pandas as pd, numpy as np
df = study.trials_dataframe()
plt.figure(figsize=(6,4))
plt.scatter(df["params_λ_forward"], df["value"], alpha=0.7)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("λ_forward"); plt.ylabel("Validation total loss")
plt.title("Effect of λ_forward")
plt.tight_layout()
plt.savefig("./saved_network/loss_vs_lfwd.png", dpi=300)
print("Wrote saved_network/loss_vs_lfwd.png")
