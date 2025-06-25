import optuna, joblib, os
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from objective import objective

os.makedirs("./saved_network", exist_ok=True)

study = optuna.create_study(
    directions=["minimize"]*5,          # five objectives
    study_name="inn_pareto",
    sampler=TPESampler(seed=2024, multivariate=True),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=3)
)
study.optimize(objective, n_trials=60)

joblib.dump(study, "./saved_network/optuna_study.pkl")
study.trials_dataframe().to_csv("./saved_network/hparam_search_log.csv", index=False)

# Print all Pareto-optimal parameter sets
print("Pareto-optimal solutions:")
for t in study.best_trials:
    print(f"Trial {t.number}: objectives={t.values}, params={t.params}")
