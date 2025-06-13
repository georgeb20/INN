
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, random, numpy as np, os, joblib
import optuna
from datetime import datetime
import network as net              # import your modules
from utilities import (                 # ← adjust if utilities is elsewhere
    get_dims, preprocess_train, split_data, mmd, boundary_loss
)

DEVICE = torch.device("cuda:0")

# ---------- helper to load and preprocess data once ----------
def _prepare_data(h5_path: str, batch_size: int = 128):
    import h5py
    with h5py.File(h5_path, "r") as h5f:
        rho_raw   = np.log10(np.asarray(h5f["rho"]))
        curve_raw = np.asarray(h5f["curve"])

    curve_raw, scaler = preprocess_train(curve_raw)
    n_trace, x_dim, y_dim, e_dim, z_dim, tot_dim = get_dims(rho_raw, curve_raw)

    zero_pad = np.zeros((n_trace, y_dim))
    z        = np.random.multivariate_normal([0.] * z_dim, np.eye(z_dim), n_trace)

    n_train = int(n_trace * .8)
    n_test  = n_trace - n_train

    rho_tr,  rho_te  = split_data(rho_raw,  n_train, n_test)
    cv_tr,   cv_te   = split_data(curve_raw, n_train, n_test)
    pad_tr,  pad_te  = split_data(zero_pad, n_train, n_test)
    z_tr,    z_te    = split_data(z,        n_train, n_test)

    in_tr  = torch.from_numpy(np.concatenate([rho_tr, pad_tr], axis=-1).astype("float32"))
    out_tr = torch.from_numpy(np.concatenate([z_tr,  cv_tr],  axis=-1).astype("float32"))

    in_te  = torch.from_numpy(np.concatenate([rho_te, pad_te], axis=-1).astype("float32"))
    out_te = torch.from_numpy(np.concatenate([z_te,  cv_te],  axis=-1).astype("float32"))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(in_tr, out_tr),
        batch_size=batch_size, shuffle=True, num_workers=0
    )

    dim_dict = {"model": x_dim, "data": y_dim, "latent": z_dim, "pad": e_dim}

    return in_tr, out_tr, in_te, out_te, loader, dim_dict, tot_dim

# Load once so every trial re-uses tensors in pinned CPU RAM
DATA = _prepare_data("../training_data/2_5_layers_100000.h5")

# ---------- objective ----------
def objective(trial: optuna.Trial) -> float:
    # ––– hyper-params to sample –––
    weights = {
        "forward" : trial.suggest_loguniform("λ_forward",  1e-3, 10.0),
        "inverse" : trial.suggest_loguniform("λ_inverse",  1e-3, 10.0),
        "latent"  : trial.suggest_loguniform("λ_latent",   1e-3, 10.0),
        "pad"     : trial.suggest_loguniform("λ_pad",      1e-3, 10.0),
        "boundary": trial.suggest_loguniform("λ_boundary", 1e-3, 10.0),
    }
    n_layer      = trial.suggest_int("n_layer", 3, 6)
    lr           = trial.suggest_loguniform("lr", 1e-4, 3e-3)
    wd           = trial.suggest_loguniform("wd", 1e-6, 1e-3)
    gamma        = trial.suggest_uniform("gamma", 0.95, 0.999)

    # ––– deterministic for reproducibility –––
    torch.manual_seed(trial.number); np.random.seed(trial.number); random.seed(trial.number)

    # ––– build fresh INN –––
    in_tr, out_tr, in_te, out_te, loader, dim_dict, tot_dim = DATA
    inn = net.define_inn(io_dim=tot_dim, n_layer=n_layer).to(DEVICE)

    # ––– quick train (8 epochs) –––
    train_curve, test_curve = net.inn_forward(
        inn, DEVICE, loader, epoch=8,
        in_train_ts=in_tr,  out_train_ts=out_tr,
        in_test_ts=in_te,   out_test_ts=out_te,
        dim_dict=dim_dict,  weight_dict=weights,
        lr=lr, weight_decay=wd, gamma=gamma,
        save_folder="./saved_network/tmp_"
    )

    # keep curves for later visualisation
    trial.set_user_attr("train_curve", train_curve)
    trial.set_user_attr("test_curve",  test_curve)

    return test_curve[-1, 5]    # final total loss
