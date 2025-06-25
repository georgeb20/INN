import h5py
import matplotlib.pyplot as plt

h5_file_path = "./saved_network/loss_curve_06-14-2025_15-20-11.h5"  # Replace <timestamp> with the actual timestamp

with h5py.File(h5_file_path, 'r') as h5f:
    train_L_pred = h5f['train_L_pred'][:]
    train_L_inv = h5f['train_L_inv'][:]
    train_L_latent = h5f['train_L_latent'][:]
    train_L_pad = h5f['train_L_pad'][:]
    train_L_boundary = h5f['train_L_boundary'][:]
    train_total_loss = h5f['train_total_loss'][:]

    test_L_pred = h5f['test_L_pred'][:]
    test_L_inv = h5f['test_L_inv'][:]
    test_L_latent_t = h5f['train_L_latent'][:]
    train_L_pad_t = h5f['train_L_pad'][:]
    train_L_boundary_t = h5f['train_L_boundary'][:]
    test_total_loss = h5f['test_total_loss'][:]

# Define colors and styles
colors = ['blue', 'orange', 'green', 'red', 'purple','black']  # Different colors for each line
styles = ['o-', 's-', '^-', 'd-', 'x-','*-']  # Different markers for each line

# Plot training losses
fig_train, axs_train = plt.subplots(6, 1, figsize=(8, 15), sharex=True)
fig_train.suptitle("Training Loss Components", fontsize=16)

loss_components_train = [
    (train_L_pred, "L_pred"),
    (train_L_inv, "L_inv"),
    (train_L_latent, "L_latent"),
    (train_L_pad, "L_pad"),
    (train_L_boundary, "L_boundary"),
    (train_total_loss, "Total Train Loss")
]

for i, (loss_data, title) in enumerate(loss_components_train):
    axs_train[i].plot(loss_data, styles[i], color=colors[i], label=title, linewidth=1.5, markersize=4)
    axs_train[i].set_title(title)
    axs_train[i].set_ylabel("Loss")
    axs_train[i].legend()
    axs_train[i].grid(True, linestyle='--', alpha=0.7)

axs_train[-1].set_xlabel("Epoch")
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout for the main title
plt.show()

# Plot testing losses
fig_test, axs_test = plt.subplots(6, 1, figsize=(8, 15), sharex=True)
fig_test.suptitle("Testing Loss Components", fontsize=16)

loss_components_test = [
    (train_L_pred, "L_pred_t"),
    (train_L_inv, "L_inv_t"),
    (train_L_latent, "L_latent_t"),
    (train_L_pad, "L_pad_t"),
    (train_L_boundary, "L_boundary_t"),
    (train_total_loss, "Total Test Loss")
]

for i, (loss_data, title) in enumerate(loss_components_test):
    axs_test[i].plot(loss_data, styles[i], color=colors[i], label=title, linewidth=1.5, markersize=4)
    axs_test[i].set_title(title)
    axs_test[i].set_ylabel("Loss")
    axs_test[i].legend()
    axs_test[i].grid(True, linestyle='--', alpha=0.7)

axs_test[-1].set_xlabel("Epoch")
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
