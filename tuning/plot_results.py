# tuning/plot_results.py
import joblib, os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_slice,
    plot_pareto_front
)
import plotly.io as pio

study = joblib.load("./saved_network/optuna_study.pkl")
os.makedirs("./saved_network", exist_ok=True)

df = study.trials_dataframe()

# ------------------------
# 1. Optuna HTML Visualizations
# ------------------------

def save_plotly(fig, name):
    fig.write_html(f"./saved_network/{name}.html")
    print(f"Wrote saved_network/{name}.html")

# Optimization history (for each objective)
for i, name in enumerate(["L_pred", "L_inv", "L_latent", "L_pad", "L_boundary"]):
    fig = plot_optimization_history(study, target=lambda t: t.values[i], target_name=name)
    save_plotly(fig, f"history_{name}")

# Pareto front projections (pairwise, 2D and 3D)
fig2d = plot_pareto_front(
    study,
    targets=lambda t: (t.values[0], t.values[1]),
    target_names=["L_pred","L_inv"]
)
save_plotly(fig2d, "pareto_pred_inv")

fig3d = plot_pareto_front(
    study,
    targets=lambda t: (t.values[0], t.values[1], t.values[2]),
    target_names=["L_pred","L_inv","L_latent"]
)
save_plotly(fig3d, "pareto_pred_inv_latent")


# Parallel coordinate plot for all objectives
for i, name in enumerate(["L_pred", "L_inv", "L_latent", "L_pad", "L_boundary"]):
    fig = plot_parallel_coordinate(
        study,
        target=lambda t: t.values[i],
        target_name=name
    )
    save_plotly(fig, f"parallel_coord_{name}")


# Slice plots for each objective
for i, name in enumerate(["L_pred", "L_inv", "L_latent", "L_pad", "L_boundary"]):
    fig = plot_slice(study, target=lambda t: t.values[i], target_name=name)
    save_plotly(fig, f"slice_{name}")

# ------------------------
# 2. Custom Pareto Analysis: Matplotlib + Seaborn
# ------------------------

# Helper: get Pareto-optimal trials (the front)
pareto = study.best_trials
pareto_idx = [t.number for t in pareto]
df["pareto"] = df["number"].isin(pareto_idx)

# Custom scatter matrix for all 5 objectives
sns.pairplot(df, vars=[f"values_{i}" for i in range(5)],
             hue="pareto", palette={True: "red", False: "gray"}, plot_kws={"alpha":0.5})
plt.suptitle("Pairplot: All objectives, Pareto front in red", y=1.03)
plt.tight_layout()
plt.savefig("./saved_network/pairplot_objectives.png", dpi=300)
plt.close()
print("Wrote saved_network/pairplot_objectives.png")

# Scatter plot: objectives vs each λ (Pareto only)
lambdas = ["λ_forward", "λ_inverse", "λ_latent", "λ_pad", "λ_boundary"]
obj_names = ["L_pred", "L_inv", "L_latent", "L_pad", "L_boundary"]
for lam in lambdas:
    plt.figure(figsize=(12,8))
    for i, obj in enumerate(obj_names):
        plt.subplot(3,2,i+1)
        plt.scatter(df[f"params_{lam}"], df[f"values_{i}"], c=df["pareto"].map({True:"red", False:"gray"}), alpha=0.6)
        plt.xscale("log")
        plt.xlabel(lam)
        plt.ylabel(obj)
        plt.title(f"{obj} vs {lam}")
    plt.tight_layout()
    plt.savefig(f"./saved_network/loss_vs_{lam}.png", dpi=200)
    plt.close()
    print(f"Wrote saved_network/loss_vs_{lam}.png")

# 2D projections of Pareto front (objectives)
from itertools import combinations
plt.figure(figsize=(16,10))
for i, (ix, iy) in enumerate(combinations(range(5), 2)):
    plt.subplot(3, 4, i+1)
    plt.scatter(df[f"values_{ix}"], df[f"values_{iy}"],
                c=df["pareto"].map({True:"red", False:"gray"}), alpha=0.5)
    plt.xlabel(obj_names[ix])
    plt.ylabel(obj_names[iy])
    plt.title(f"{obj_names[ix]} vs {obj_names[iy]}")
plt.tight_layout()
plt.savefig("./saved_network/pareto_2dprojections.png", dpi=300)
plt.close()
print("Wrote saved_network/pareto_2dprojections.png")

# Parallel coordinates plot for Pareto front only
from pandas.plotting import parallel_coordinates
pareto_df = df[df["pareto"]]
# Rename columns for clarity
rename_dict = {f"values_{i}": obj_names[i] for i in range(5)}
pcoord = pareto_df.rename(columns=rename_dict)
pcoord = pcoord[["L_pred", "L_inv", "L_latent", "L_pad", "L_boundary"]]
pcoord["Trial"] = pareto_df["number"].astype(str)
plt.figure(figsize=(10,6))
parallel_coordinates(pcoord, "Trial", color=plt.cm.viridis(np.linspace(0,1,len(pcoord))))
plt.gca().legend_.remove()
plt.title("Parallel coordinates: Pareto front solutions")
plt.ylabel("Objective value")
plt.tight_layout()
plt.savefig("./saved_network/parallel_pareto.png", dpi=300)
plt.close()
print("Wrote saved_network/parallel_pareto.png")

# Objective correlation heatmap
plt.figure(figsize=(7,6))
sns.heatmap(df[[f"values_{i}" for i in range(5)]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation heatmap between objectives")
plt.tight_layout()
plt.savefig("./saved_network/objective_correlation.png", dpi=300)
plt.close()
print("Wrote saved_network/objective_correlation.png")

print("ALL PLOTS COMPLETE")
