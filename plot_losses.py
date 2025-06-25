import numpy as np
import matplotlib.pyplot as plt

# ── diagnostics file ───────────────────────────────────────────────────────────
file_path = "saved_network/diagnostics_06-14-2025_23-26-35.npz"
# ───────────────────────────────────────────────────────────────────────────────

data  = np.load(file_path)
train = data["train"]                # shape (n_epochs, 6)
test  = data["test"]

loss_names = [
    "L_pred", "L_inv", "L_latent",
    "L_pad",  "L_boundary", "Total"
]

# two stacked axes
fig, (ax_train, ax_test) = plt.subplots(2, 1, figsize=(4, 8), sharex=True, constrained_layout=True)

# plot loops ------------------------------------------------------------
for idx, name in enumerate(loss_names):
    ax_train.plot(train[:, idx], label=name)
    ax_test .plot(test[:,  idx], label=name)

# cosmetics -------------------------------------------------------------
a=0
for ax, title in zip((ax_train, ax_test), ("(a)\nTraining Losses", "(b)\nTesting Losses")):
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if a==0:
        a=1
        ax.legend()

ax_test.set_xlabel("Epoch")

plt.tight_layout()
plt.show()
