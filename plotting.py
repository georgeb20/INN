import numpy as np
import matplotlib.pyplot as plt
import utilities as utl
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def plot_1d_slice(
    trace: int,
    inverted_rho_log: np.ndarray,
    true_rho_log: np.ndarray,
    tvd_pixel: np.ndarray,
    tvd_edge: np.ndarray,
    letter: str = ''
):
    # ── pick the right depth axis ────────────────────────────────────────────────
    depth_axis = (
        tvd_pixel[:, 0]            # 2-D case: one column per trace
        if tvd_pixel.ndim == 2
        else np.asarray(tvd_pixel)     # already 1-D
    ).ravel()                           # guarantee 1-D

    # ── basic statistics ────────────────────────────────────────────────────────
    mean_inv        = np.mean(inverted_rho_log, axis=1)
    percentile_low  = np.percentile(inverted_rho_log, 10, axis=1)
    percentile_high = np.percentile(inverted_rho_log, 90, axis=1)

    # ── figure setup ────────────────────────────────────────────────────────────
    # 1. choose tight x-limits around your data
    x_min, x_max = inverted_rho_log.min(), inverted_rho_log.max()

    # 2. make it wider & high-DPI
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)

    # 3. … now draw spaghetti as before …
    ax.set_xlim(x_min - 0.05, x_max + 0.05)

    for col in inverted_rho_log.T:
        ax.plot(
            col, depth_axis,
            color='0.25',   # grey
            alpha=0.35,     # ⬆︎ from 0.12 → less transparent
            linewidth=1.2
        )


    # overlays
    ax.plot(true_rho_log[trace], depth_axis,
            color='blue', lw=2.2, label='True Model')
    ax.plot(mean_inv, depth_axis,
            color='red',  lw=2.2, label='Mean Inversion')
    ax.fill_betweenx(depth_axis, percentile_low, percentile_high,
                     color='orange', alpha=0.75, label='10th–90th Percentile')

    # cosmetics
    ax.set_xlabel(r'$\log_{10}(\rho)\,(\Omega\!\cdot\!\text{ft})$'
                  f'\n\n({letter})', fontsize=11)
    ax.set_ylabel('Depth (ft)', fontsize=11)
    ax.invert_yaxis()
    ax.set_title(f'1-D Resistivity Slice at X = {trace*10}', fontsize=12)
    ax.grid(ls=':', lw=0.5, color='gray')
    ax.legend(facecolor='white', edgecolor='black')
    plt.tight_layout()
    plt.show()


def plot_uncertainty_distribution(predicted_samples_list,x_lim_range,true_val,letter='z'):
    """
    Plots the distribution of predicted samples to visualize uncertainty and displays the 80% confidence interval.
    
    Parameters:
    - predicted_samples_list: A list of 100 predicted samples (doubles).
    """
    # Convert the list to a numpy array
    predicted_samples_array = np.array(predicted_samples_list)
    
    # Calculate the 80% confidence interval (10th and 90th percentiles)
    ci_80 = np.percentile(predicted_samples_array, [10, 90])
    avg_uncertainty = np.mean(ci_80)
    print(avg_uncertainty)

    mean_value = np.mean(predicted_samples_array)
    median_value = np.median(predicted_samples_array)
    std_dev = np.std(predicted_samples_array)
    # Plot the distribution
    plt.figure(figsize=(6,6))
    
    sns.histplot(predicted_samples_array, bins=20, kde=True, color='green', edgecolor='white')
    
    # Plot vertical lines for the 80% confidence interval
    plt.axvline(ci_80[0], color='black', linestyle='-',linewidth=3, label=f'80% CI lower: {ci_80[0]:.2f}')
    plt.axvline(ci_80[1], color='black', linestyle='-', linewidth=3,label=f'80% CI upper: {ci_80[1]:.2f}')
    plt.axvline(mean_value, color='red', linestyle='--', linewidth=3,label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='yellow', linestyle='--', linewidth=3,label=f'Median: {median_value:.2f}')
    plt.axvline(true_val, color='blue', linestyle='--', linewidth=3,label=f'True Value: {true_val:.2f}')

    plt.xlim(x_lim_range)

    plt.title('Distribution of Predicted Samples with 80% Confidence Interval')
    plt.xlabel(f"Predicted Value \n\n ({letter})")
    plt.ylabel('Density')
    plt.legend(loc='upper right', title=f'Std Dev: {std_dev:.2f}')

    plt.tight_layout()
    plt.show()


def f1(x):
    return 0.9 * (np.sin(x / 8 + 1.2 * np.pi) * 7 + 10 - (np.cos(x / 3) * 2.8)) - 40

def f2(x):
    return 2 * (np.sin(x / 10 + 3.2 * np.pi) * 7 + 5 - (2 * np.sin(-x / 3) * 0.8)) + 27
import numpy as np

import numpy as np

import numpy as np

def f33(x):
    x = np.asarray(x)
    return (
        31
        + 8 * np.sin((x + 200) / 27 + np.pi / 12) * np.cos((x + 200) / 12)
        + 15 * np.cos((x + 200) / 10) ** 9
        - 15 * np.tanh(((x + 200) - 500) / 25)
    )

def f22(x):
    x = np.asarray(x)
    return (
        -16
        + 12 * (
            np.cos((x + 200) / 29 - np.pi / 3) * np.sin((x + 200) / 42 + np.pi / 5)
            - 0.7 * np.sin((x + 200) / 76) * np.cos((x + 200) / 34 + np.pi / 4)
        )
        + 4 * np.sin((x + 200) / 20) * np.cos((x + 200) / 6)
        + 6 * np.tanh(((x + 200) - 450) / 100)
        + 2 * np.cos((x + 200) / 30) ** 2
        - 0.005 * x
        + 35
    )

def f11(x):
    x = np.asarray(x)
    return (
        -57
        + 11 * (
            np.cos((x + 200) / 15 - np.pi / 12) * np.sin((x + 200) / 13 + np.pi / 17)
            - 0.6 * np.sin((x + 200) / 16) * np.cos((x + 200) / 50 + np.pi / 3)
        )
        + 5 * np.sin((x + 200) / 12 + np.pi / 5) * np.cos((x + 200) / 15)
        - 12 * np.tanh(((x + 200) - 600) / 20)
        + 5 * np.exp(-((x + 200) - 400) ** 2 / 120)
        + 7 * np.tanh(((x + 200) - 200) / 30)
        + 3 * np.sin((x + 200) / 5 + 0.3 * (x / 100))
    )


def _add_twin_axis(ax, x_data, y1_data, y2_data,y3_data,invert_range=(75, -75)):
    ax_twin = ax.twinx()
    ax_twin.plot(x_data, y1_data, color='white', linewidth=2, label='f1(x)')
    ax_twin.plot(x_data, y2_data, color='white', linewidth=2, label='f2(x)')
    ax_twin.plot(x_data, y3_data, color='white', linewidth=2, label='f2(x)')

    ax_twin.set_ylim(invert_range)
    ax_twin.set_yticks([])
    ax_twin.set_yticklabels([])
    return ax_twin

def plot_results(true_2d_model, inv_2d_model, predicted_samples):
    """
    Plots the true and inverted resistivity models, the prediction uncertainty,
    and the relative model misfit.

    Parameters
    ----------
    true_2d_model : np.ndarray
        The true 2D resistivity model.
    inv_2d_model : np.ndarray
        The inverted 2D resistivity model.
    predicted_samples : np.ndarray
        Prediction samples for uncertainty estimation.

    Returns
    -------
    max_avg_uncertainty_index : int
        X-index at which the average uncertainty is maximum.
    min_avg_uncertainty_index : int
        X-index at which the average uncertainty is minimum.
    max_vals : np.ndarray
        Samples at the position of maximum uncertainty.
    min_vals : np.ndarray
        Samples at the position of minimum uncertainty.
    """

    # Compute model misfit
    model_misfit = utl.model_misfit(true_2d_model, inv_2d_model, type="r2norm")

    # Create base figure and axes
    fig, axs = plt.subplots(4, 1, figsize=(18, 12), constrained_layout=True)

    # 1) Prepare data for the "true" model with f1, f2
    x_grid = np.linspace(0, 1000, true_2d_model.shape[0])  
    x_fine = np.linspace(0, true_2d_model.shape[1], 500)
    
    y1 = f1(x_fine)
    y2 = f2(x_fine)

    X, Y = np.meshgrid(x_fine, np.linspace(-75, 75, 1000))
    Z = np.zeros_like(X)

    # Assign different log10(rho) values in different regions
    Z[y2 < Y] = np.log10(10)     # below
    Z[y1 > Y] = np.log10(50)     # above
    Z[(Y < y2) & (Y > y1)] = np.log10(100)  # between y1 and y2

    # 1. True Model Plot
    p0 = axs[0].pcolormesh(X, Y, Z, cmap='jet', vmin=0, vmax=2, shading='auto')
    axs[0].plot(x_fine, y1, color='white', linewidth=2, label='f1(x)')
    axs[0].plot(x_fine, y2, color='white', linewidth=2, label='f2(x)')
    axs[0].invert_yaxis()
    axs[0].hlines(0, x_grid[0], x_grid[-1], "black", linestyle="dashed", linewidth=2)

    # 2) Predicted Samples and Confidence Intervals
    predicted_samples = np.flip(np.transpose(predicted_samples, (1, 0, 2)), axis=0)
    lower_percentile = np.percentile(predicted_samples, 10, axis=2)
    upper_percentile = np.percentile(predicted_samples, 90, axis=2)
    _CI = upper_percentile - lower_percentile  # 80% confidence interval width

    # 2. Inverted Model Plot
    x = np.arange(true_2d_model.shape[1] + 1)
    y = np.arange(true_2d_model.shape[0] + 1)
    x_mesh, y_mesh = np.meshgrid(x, y)

    p1 = axs[1].pcolormesh(x_mesh, y_mesh, inv_2d_model, cmap="jet", vmin=0, vmax=2, shading='auto')
    _add_twin_axis(axs[1], x_fine, y1, y2)
    axs[1].hlines(15, x_grid[0], x_grid[-1], "black", linestyle="dashed", linewidth=2)

    # Print min/max misfit values
    print(f'Minimum Model Misfit: {min(model_misfit)}')
    print(f'Maximum Model Misfit: {max(model_misfit)}')

    # 3. Confidence Interval Width Plot
    p2 = axs[2].pcolormesh(x_mesh, y_mesh, _CI, cmap='magma', shading='flat')
    _add_twin_axis(axs[2], x_fine, y1, y2)
    axs[2].hlines(15, x_grid[0], x_grid[-1], "black", linestyle="dashed", linewidth=2)
    axs[2].set_title('Width of 80% Confidence Interval Indicating Prediction Uncertainty '
                     '($\\log_{10}(\\rho)(\\Omega \\cdot ft)$)')

    # 4. Relative Model Misfit Plot
    axs[3].plot(
        np.linspace(1, true_2d_model.shape[1], true_2d_model.shape[1]),
        model_misfit, "bP--", label="Relative Model Misfit"
    )
    axs[3].set_xlabel("(d) X (ft)")
    axs[3].set_ylabel("Relative Model Misfit (%)")
    axs[3].set_title("Relative L2-Norm Model Misfit (%)")

    # Add colorbars
    cbar0 = fig.colorbar(p0, ax=axs[0], aspect=17)
    cbar0.set_label("$\\log_{10}(\\rho)(\\Omega \\cdot ft)$", labelpad=15)
    cbar1 = fig.colorbar(p1, ax=axs[1], aspect=17)
    cbar1.set_label("$\\log_{10}(\\rho)(\\Omega \\cdot ft)$", labelpad=15)
    cbar2 = fig.colorbar(p2, ax=axs[2], aspect=17)
    cbar2.set_label("$\\log_{10}(\\rho)(\\Omega \\cdot ft)$", labelpad=15)

    # Set axis ranges and labels
    axs[0].set_xticks(x_grid, minor=True)
    axs[0].set_xlim(0, true_2d_model.shape[1])
    axs[0].set_xlabel("(a) X (ft)")
    axs[0].set_ylabel("Depth (ft)")
    axs[0].set_title("True Resistivity Model ($\\log_{10}(\\rho)(\\Omega \\cdot ft)$)")
    axs[0].set_yticks([-50, 0, 50])
    axs[0].set_yticklabels([4950, 5000, 5050])

    axs[1].set_ylim(0, true_2d_model.shape[0])
    axs[1].set_xlim(0, true_2d_model.shape[1])
    axs[1].set_xlabel("(b) X (ft)")
    axs[1].set_ylabel("Depth (ft)")
    axs[1].set_title("Inverted Resistivity Model ($\\log_{10}(\\rho)(\\Omega \\cdot ft)$)")
    axs[1].set_yticks([25, 15, 5])
    axs[1].set_yticklabels([4950, 5000, 5050])

    axs[2].set_ylim(0, true_2d_model.shape[0])
    axs[2].set_xlim(0, true_2d_model.shape[1])
    axs[2].set_xlabel("(c) X (ft)")
    axs[2].set_ylabel("Depth (ft)")
    axs[2].set_yticks([25, 15, 5])
    axs[2].set_yticklabels([4950, 5000, 5050])

    axs[3].set_xlim(0, true_2d_model.shape[1])

    # Customize X-tick labels on all subplots
    custom_ticks =  [0,100,200,300,400,500,600,700,800]
    for i in range(4):
        axs[i].set_xticklabels(custom_ticks)
        labels = axs[i].get_xticklabels()
        new_labels = [f"X{label.get_text()}" for label in labels]
        axs[i].set_xticklabels(new_labels)

    # Draw dividing lines across figure
    fig.add_artist(plt.Line2D([0, 1], [0.25, 0.25], color='black', transform=fig.transFigure, linewidth=1))
    fig.add_artist(plt.Line2D([0, 1], [0.5, 0.5], color='black', transform=fig.transFigure, linewidth=1))
    fig.add_artist(plt.Line2D([0, 1], [0.75, 0.75], color='black', transform=fig.transFigure, linewidth=1))

    # Calculate uncertainty metrics
    avg_uncertainty = np.mean(_CI, axis=0)
    max_avg_uncertainty_index = np.argmax(avg_uncertainty)
    min_avg_uncertainty_index = np.argmin(avg_uncertainty)
    max_uncertainty_index = np.unravel_index(np.argmax(_CI), _CI.shape)
    min_uncertainty_index = np.unravel_index(np.argmin(_CI), _CI.shape)

    for ax in axs:
        ax.axvline(x=max_avg_uncertainty_index, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=min_avg_uncertainty_index, color='black', linestyle='-', linewidth=2)

    print(_CI[max_uncertainty_index[0]][max_uncertainty_index[1]])
    print(_CI[min_uncertainty_index[0]][min_uncertainty_index[1]])
    true_max_val = true_2d_model[max_uncertainty_index]
    true_min_val = true_2d_model[min_uncertainty_index]
    max_vals = predicted_samples[max_uncertainty_index]
    min_vals = predicted_samples[min_uncertainty_index]

    print(max_uncertainty_index)
    print(min_uncertainty_index)

    plt.show()

    return max_avg_uncertainty_index, min_avg_uncertainty_index, max_vals, min_vals,true_max_val,true_min_val
def plot_results_realistic(true_2d_model, inv_2d_model, predicted_samples):
    """
    Plots the true and inverted resistivity models, the prediction uncertainty,
    and the relative model misfit.

    Parameters
    ----------
    true_2d_model : np.ndarray
        The true 2D resistivity model.
    inv_2d_model : np.ndarray
        The inverted 2D resistivity model.
    predicted_samples : np.ndarray
        Prediction samples for uncertainty estimation.

    Returns
    -------
    max_avg_uncertainty_index : int
        X-index at which the average uncertainty is maximum.
    min_avg_uncertainty_index : int
        X-index at which the average uncertainty is minimum.
    max_vals : np.ndarray
        Samples at the position of maximum uncertainty.
    min_vals : np.ndarray
        Samples at the position of minimum uncertainty.
    """

    # Compute model misfit
    model_misfit = utl.model_misfit(true_2d_model, inv_2d_model, type="r2norm")

    # Create base figure and axes
    fig, axs = plt.subplots(4, 1, figsize=(18, 12), constrained_layout=True)

    # 1) Prepare data for the "true" model with f1, f2
    x_grid = np.linspace(0, 1000, true_2d_model.shape[0])  
    x_fine = np.linspace(0, true_2d_model.shape[1], 500)
    
    y1 = f11(x_fine)
    y2 = f22(x_fine)
    y3 = f33(x_fine)

    X, Y = np.meshgrid(x_fine, np.linspace(-75, 75, 1000))
    Z = np.zeros_like(X)

    # Turn the 1-D boundaries into 2-D rows so broadcasting works cleanly
    y1_grid = y1[None, :]   # shape (1, nx)
    y2_grid = y2[None, :]
    y3_grid = y3[None, :]

    # ── ➎  region-by-region assignment (log10 scale)  ─────────────────────────────
    # Feel free to change these resistivity values to whatever you need.
    Z[Y > y2_grid]                     = np.log10(3)    # *below* second boundary
    Z[Y < y1_grid]                     = np.log10(15)    # *above* first boundary
    Z[(Y <= y2_grid) & (Y >= y1_grid)] = np.log10(3)   # between y1 and y2
    Z[(Y <= y3_grid) & (Y > y2_grid)] = np.log10(15)

    # 1. True Model Plot
    p0 = axs[0].pcolormesh(X, Y, Z, cmap='jet', vmin=0, vmax=2, shading='auto')
    axs[0].plot(x_fine, y1, color='white', linewidth=2, label='f1(x)')
    axs[0].plot(x_fine, y2, color='white', linewidth=2, label='f2(x)')
    axs[0].plot(x_fine, y3, color='white', linewidth=2, label='f3(x)')
    axs[0].invert_yaxis()
    axs[0].hlines(0, x_grid[0], x_grid[-1], "black", linestyle="dashed", linewidth=2)

    # 2) Predicted Samples and Confidence Intervals
    predicted_samples = np.flip(np.transpose(predicted_samples, (1, 0, 2)), axis=0)
    lower_percentile = np.percentile(predicted_samples, 10, axis=2)
    upper_percentile = np.percentile(predicted_samples, 90, axis=2)
    _CI = upper_percentile - lower_percentile  # 80% confidence interval width

    # 2. Inverted Model Plot
    x = np.arange(true_2d_model.shape[1] + 1)
    y = np.arange(true_2d_model.shape[0] + 1)
    x_mesh, y_mesh = np.meshgrid(x, y)

    p1 = axs[1].pcolormesh(x_mesh, y_mesh, inv_2d_model, cmap="jet", vmin=0, vmax=2, shading='auto')
    _add_twin_axis(axs[1], x_fine, y1, y2,y3)
    axs[1].hlines(15, x_grid[0], x_grid[-1], "black", linestyle="dashed", linewidth=2)

    # Print min/max misfit values
    print(f'Minimum Model Misfit: {min(model_misfit)}')
    print(f'Maximum Model Misfit: {max(model_misfit)}')

    # 3. Confidence Interval Width Plot
    p2 = axs[2].pcolormesh(x_mesh, y_mesh, _CI, cmap='magma', shading='flat')
    _add_twin_axis(axs[2], x_fine, y1, y2,y3)
    axs[2].hlines(15, x_grid[0], x_grid[-1], "black", linestyle="dashed", linewidth=2)
    axs[2].set_title('Width of 80% Confidence Interval Indicating Prediction Uncertainty '
                     '($\\log_{10}(\\rho)(\\Omega \\cdot ft)$)')

    # 4. Relative Model Misfit Plot
    axs[3].plot(
        np.linspace(1, true_2d_model.shape[1], true_2d_model.shape[1]),
        model_misfit, "bP--", label="Relative Model Misfit"
    )
    axs[3].set_xlabel("(d) X (ft)")
    axs[3].set_ylabel("Relative Model Misfit (%)")
    axs[3].set_title("Relative L2-Norm Model Misfit (%)")

    # Add colorbars
    cbar0 = fig.colorbar(p0, ax=axs[0], aspect=17)
    cbar0.set_label("$\\log_{10}(\\rho)(\\Omega \\cdot ft)$", labelpad=15)
    cbar1 = fig.colorbar(p1, ax=axs[1], aspect=17)
    cbar1.set_label("$\\log_{10}(\\rho)(\\Omega \\cdot ft)$", labelpad=15)
    cbar2 = fig.colorbar(p2, ax=axs[2], aspect=17)
    cbar2.set_label("$\\log_{10}(\\rho)(\\Omega \\cdot ft)$", labelpad=15)

    # Set axis ranges and labels
    axs[0].set_xticks(x_grid, minor=True)
    axs[0].set_xlim(0, true_2d_model.shape[1])
    axs[0].set_xlabel("(a) X (ft)")
    axs[0].set_ylabel("Depth (ft)")
    axs[0].set_title("True Resistivity Model ($\\log_{10}(\\rho)(\\Omega \\cdot ft)$)")
    axs[0].set_yticks([-50, 0, 50])
    axs[0].set_yticklabels([4950, 5000, 5050])

    axs[1].set_ylim(0, true_2d_model.shape[0])
    axs[1].set_xlim(0, true_2d_model.shape[1])
    axs[1].set_xlabel("(b) X (ft)")
    axs[1].set_ylabel("Depth (ft)")
    axs[1].set_title("Inverted Resistivity Model ($\\log_{10}(\\rho)(\\Omega \\cdot ft)$)")
    axs[1].set_yticks([25, 15, 5])
    axs[1].set_yticklabels([4950, 5000, 5050])

    axs[2].set_ylim(0, true_2d_model.shape[0])
    axs[2].set_xlim(0, true_2d_model.shape[1])
    axs[2].set_xlabel("(c) X (ft)")
    axs[2].set_ylabel("Depth (ft)")
    axs[2].set_yticks([25, 15, 5])
    axs[2].set_yticklabels([4950, 5000, 5050])

    axs[3].set_xlim(0, true_2d_model.shape[1])

    # Customize X-tick labels on all subplots
    custom_ticks =  [0,100,200,300,400,500,600,700,800]
    for i in range(4):
        axs[i].set_xticklabels(custom_ticks)
        labels = axs[i].get_xticklabels()
        new_labels = [f"X{label.get_text()}" for label in labels]
        axs[i].set_xticklabels(new_labels)

    # Draw dividing lines across figure
    fig.add_artist(plt.Line2D([0, 1], [0.25, 0.25], color='black', transform=fig.transFigure, linewidth=1))
    fig.add_artist(plt.Line2D([0, 1], [0.5, 0.5], color='black', transform=fig.transFigure, linewidth=1))
    fig.add_artist(plt.Line2D([0, 1], [0.75, 0.75], color='black', transform=fig.transFigure, linewidth=1))

    # Calculate uncertainty metrics
    avg_uncertainty = np.mean(_CI, axis=0)
    max_avg_uncertainty_index = np.argmax(avg_uncertainty)
    min_avg_uncertainty_index = np.argmin(avg_uncertainty)
    max_uncertainty_index = np.unravel_index(np.argmax(_CI), _CI.shape)
    min_uncertainty_index = np.unravel_index(np.argmin(_CI), _CI.shape)

    # for ax in axs:
    #     ax.axvline(x=max_avg_uncertainty_index, color='black', linestyle='-', linewidth=2)
    #     ax.axvline(x=min_avg_uncertainty_index, color='black', linestyle='-', linewidth=2)

    print(_CI[max_uncertainty_index[0]][max_uncertainty_index[1]])
    print(_CI[min_uncertainty_index[0]][min_uncertainty_index[1]])
    true_max_val = true_2d_model[max_uncertainty_index]
    true_min_val = true_2d_model[min_uncertainty_index]
    max_vals = predicted_samples[max_uncertainty_index]
    min_vals = predicted_samples[min_uncertainty_index]

    print(max_uncertainty_index)
    print(min_uncertainty_index)

    plt.show()

    return max_avg_uncertainty_index, min_avg_uncertainty_index, max_vals, min_vals,true_max_val,true_min_val
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_with_misfit_only(true_2d_model, inv_2d_models, inv_2d_models_titles):
    fig, axs = plt.subplots(len(inv_2d_models), 1, figsize=(6, 9), constrained_layout=True)
    letters = ['b', 'c', 'd']

    for i in range(len(inv_2d_models)):
        # compute misfit
        model_misfit = utl.model_misfit(true_2d_model, inv_2d_models[i], type="r2norm")

        # print min and max
        min_m = np.min(model_misfit)
        max_m = np.max(model_misfit)
        print(f"{inv_2d_models_titles[i]} – Min misfit: {min_m:.3f}%, Max misfit: {max_m:.3f}%")

        # dynamic padding
        pad = 0.05 * (max_m - min_m) if max_m != min_m else 1
        y_lo = max(0, min_m - pad)
        y_hi = max_m + pad

        ax = axs[i]
        ax.set_ylim(0, true_2d_model.shape[0])
        ax.set_xlim(0, true_2d_model.shape[1])

        # plot (keeping your original x-range approach)
        x = np.linspace(1, model_misfit.size, model_misfit.size)
        ax.plot(x, model_misfit, "bP--", label="Relative Model Misfit")

        # apply dynamic y-limits
        ax.set_ylim(y_lo, y_hi)

        ax.set_xlabel(f"({letters[i]}) X (ft)")
        ax.set_ylabel("Relative Model Misfit (%)")
        ax.set_title(f"{inv_2d_models_titles[i]} Relative L²-Norm Misfit (%)")

        # your original tick-label logic
        arr =  [0,100,200,300,400,500,600,700,800]
        ax.set_xticklabels(arr)  # Change the tick labels
        labels = ax.get_xticklabels()
        new_labels = [f"X{lbl.get_text()}" for lbl in labels]
        ax.set_xticklabels(new_labels)

    plt.show()

def plot_2d_with_inversion_only(true_2d_model, inv_2d_models,inv_2d_models_titles):
    ''''''

    x_grid = np.linspace(0, 1000, true_2d_model.shape[0])  

    fig, axs = plt.subplots(1,len(inv_2d_models), figsize=(12,6), constrained_layout=True)
    fig.suptitle(
        r"Inverted Resistivity Models — 10% ($\log_{10}(\rho)\;(\Omega\!\cdot\!\text{ft})$)",
        fontsize=12, weight="bold"
    )

    x = np.arange(true_2d_model.shape[1] + 1)  
    y = np.arange(true_2d_model.shape[0] + 1)  

    x_mesh, y_mesh = np.meshgrid(x, y)

    letters = ['b','c','d']
    for i in range(len(inv_2d_models)):
        p = axs[i].pcolormesh(x_mesh, y_mesh, inv_2d_models[i], cmap="jet", vmin=0, vmax=2,shading ='flat')
        axs[i].set_ylim(0, true_2d_model.shape[0])
        axs[i].set_xlim(0, true_2d_model.shape[1])
        axs[i].set_xlabel(f"({letters[i]}) X (ft)")
        axs[i].set_ylabel("Depth (ft)")
        #axs[i].set_title(f"{inv_2d_models_titles[i]} Inverted Resistivity Model, ($\\log_{{10}}(\\rho) (\\Omega \\cdot ft)$)")
        #axs[i].set_title(f"{inv_2d_models_titles[i]} 10% Noise, ($\\log_{{10}}(\\rho) (\\Omega \\cdot ft)$)")
        axs[i].set_title(inv_2d_models_titles[i], pad=4)

        axs[i].set_yticks([25, 15, 5])  # Set the original ticks
        axs[i].set_yticklabels([4950, 5000, 5050])  # Change the tick labels

        if i > 0:
            axs[i].set_ylabel("")
            axs[i].set_yticklabels([])


        arr =  [0,100,200,300,400,500,600,700,800]
        axs[i].set_xticklabels(arr)  # Change the tick labels
        labels = axs[i].get_xticklabels()
        new_labels = [f"X{label.get_text()}" for label in labels]
        axs[i].set_xticklabels(new_labels)
        axs[i].hlines(15, x_grid[0], x_grid[-1],"black", linestyle="dashed",linewidth=2)
        x = np.linspace(0, true_2d_model.shape[1], 500)
        x_copy = np.linspace(0, true_2d_model.shape[1], 500)

        y1 = f1(x)
        y2 = f2(x)
        X, Y = np.meshgrid(x, np.linspace(-75, 75, 1000))
        ax_twin = axs[i].twinx()
        ax_twin.plot(x_copy, y1, color='white', linewidth=2, linestyle="dashed",label='f1(x)')
        ax_twin.plot(x_copy, y2, color='white', linewidth=2, linestyle="dashed",label='f2(x)')
        ax_twin.set_ylim([75, -75])  # Set the y-axis limits to -70 to 70
        ax_twin.set_yticks([])        # Remove ticks
        ax_twin.set_yticklabels([])   # Remove tick labels
    cbar = fig.colorbar(p, ax=axs, location='right', shrink=0.82, aspect=25)
    cbar.set_label(r"$\log_{10}(\rho)\;(\Omega\!\cdot\!\text{ft})$")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_all_noise_inversions(true_2d_model, inversion_by_noise, solver_titles, noise_levels):
    """
    Plot a grid of inverted resistivity models for multiple noise levels,
    with consistent ticks: X ticks only labeled on bottom row, Y ticks only
    labeled on first column.
    """
    n_rows = len(noise_levels)
    n_cols = len(solver_titles)

    # common mesh
    x = np.arange(true_2d_model.shape[1] + 1)
    y = np.arange(true_2d_model.shape[0] + 1)
    X, Y = np.meshgrid(x, y)


    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(4*n_cols, 3.5*n_rows),
                            sharex=True, sharey=False,
                            constrained_layout=False)
    # overall suptitle
    fig.suptitle(
        r"Inverted Resistivity Models Across Noise Levels "
        r"($\log_{10}(\rho)\;(\Omega\!\cdot\!\text{ft})$)",
        fontsize=15, weight="bold"
    )
    fig.supxlabel("X (ft)",     fontsize=15, y=0.01)
    fig.supylabel("Depth (ft)", fontsize=15, x=0.02)
    letters = ['(a)', '(b)', '(c)']
    for i, noise in enumerate(noise_levels):
        results = inversion_by_noise[noise]
        for j, model in enumerate(results):
            ax = axs[i, j] if n_rows > 1 else axs[j]
            pcm = ax.pcolormesh(X, Y, model, cmap="jet", vmin=0, vmax=2, shading="flat")

            # column titles on first row
            if i == 0:
                ax.set_title(solver_titles[j], pad=6, fontsize=17, weight="medium",fontweight='bold')

            # y ticks on all, but labels only on first column
            if j == 0:
                ax.set_yticks([25, 15, 5])  # Set the original ticks
                ax.set_yticklabels([4950, 5000, 5050])  # Change the tick labels
                ax.set_ylabel(f"{int(noise*100)}% noise", fontsize=15,fontweight='bold')
            else:
                ax.set_yticklabels([])


            # x ticks on all, but labels only on bottom row
            #ax.set_xticks(xt)
            if i == n_rows - 1:
                labels = ax.get_xticklabels()
                n_ticks = 9
                xt_pos = np.linspace(0, true_2d_model.shape[1], n_ticks)
                # 3) build your “fake” labels by scaling up by 10
                xt_lab = [f"X{int(pos * (800/true_2d_model.shape[1]))}" for pos in xt_pos]
                ax.set_xticks(xt_pos)
                ax.set_xticklabels(xt_lab, fontsize=10)
                ax.set_xlabel("\n"+letters[j], fontsize=15,fontweight='bold')

            else:
                #ax.set_xticklabels([])
                a=1

            # dashed horizon line
            ax.hlines(15, x[0], x[-1], "k", linestyle="dashed", linewidth=1)

    # single colorbar on the right
    cbar = fig.colorbar(pcm, ax=axs, location="right", shrink=0.82, aspect=20, pad=0.03)
    cbar.set_label(r"$\log_{10}(\rho)\;(\Omega\!\cdot\!\text{ft})$", fontsize=15)


    mng = plt.get_current_fig_manager()
    try:
        # Qt backends
        mng.window.showMaximized()
    except AttributeError:
        try:
            # TkAgg backend
            mng.window.state('zoomed')
        except AttributeError:
            # fallback to fullscreen toggle
            mng.full_screen_toggle()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_all_noise_misfits(true_2d_model, inversion_by_noise, solver_titles, noise_levels):
    """
    Plot a grid of misfit curves for multiple noise levels and solvers,
    computing misfit internally just like plot_2d_with_misfit_only.
    
    Parameters
    ----------
    true_2d_model : 2D array
        The reference resistivity model (for misfit computation and trace count).
    inversion_by_noise : dict
        Mapping noise_level -> list of inverted_result arrays in same order as solver_titles.
    solver_titles : list of str
        Column titles (e.g. ["INN","LMA","Occam"]).
    noise_levels : list of float
        Ordered list of noise levels to plot (e.g. [0.0, 0.1, 0.2, 0.3]).
    """
    n_rows = len(noise_levels)
    n_cols = len(solver_titles)
    n_traces = true_2d_model.shape[1]
    letters = ['(a)', '(b)', '(c)']

    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(4*n_cols, 3.5*n_rows),
                            sharex=True, sharey=False,
                            constrained_layout=False)

    fig.suptitle("Relative Model Misfit Across Noise Levels", fontsize=15, weight="bold")
    fig.supxlabel("X (ft)",     fontsize=15, y=0.01)
    fig.supylabel("Relative Model Misfit Misfit (%)", fontsize=15, rotation="vertical")

    # define common tick positions
    n_ticks = 9
    xt = np.linspace(1, n_traces, n_ticks, dtype=int)

    for i, noise in enumerate(noise_levels):
        inverted_list = inversion_by_noise[noise]
        for j, inv_model in enumerate(inverted_list):
            ax = axs[i, j] if n_rows > 1 else axs[j]

            # compute misfit as in your original
            mf = utl.model_misfit(true_2d_model, inv_model, type="r2norm")
            avg_misfit = np.mean(mf)  # average misfit for this trace
            # dynamic y-range
            mn, mx = mf.min(), mf.max()
            pad = 0.05*(mx - mn) if mx != mn else 1.0
            y_lo = max(0, mn - pad)
            y_hi = mx + pad

            # plot
            x = np.arange(1, n_traces+1)
            ax.plot(x, mf, "bP--", label="Relative Model Misfit")
            ax.set_xlim(1, n_traces)
            ax.set_ylim(y_lo, y_hi)

            # column titles on top row
            if i == 0:
                ax.set_title(solver_titles[j], pad=6, fontsize=17, weight="medium",fontweight='bold')

            # row labels on first column
            if j == 0:
                pct = int(noise * 100)
                ax.set_ylabel(f"{pct}% noise", fontsize=15, fontweight='bold')
            else:
                ax.set_ylabel("")

            # X ticks everywhere, but label only bottom row
            if i == n_rows - 1:
                labels = ax.get_xticklabels()
                n_ticks = 9
                xt_pos = np.linspace(0, true_2d_model.shape[1], n_ticks)
                # 3) build your “fake” labels by scaling up by 10
                xt_lab = [f"X{int(pos * (800/true_2d_model.shape[1]))}" for pos in xt_pos]
                ax.set_xticks(xt_pos)
                ax.set_xticklabels(xt_lab, fontsize=10)
                ax.set_xlabel("\n"+letters[j], fontsize=15,fontweight='bold')

            # Y ticks everywhere, but label only first column
            yt = np.linspace(0, 50, 5)
            ax.set_yticks(yt)
            if j == 0:
                ax.set_yticklabels([f"{v:.1f}" for v in yt], fontsize=8)
            else:
                ax.set_yticklabels([])

            # print min/max
            #print(f"{solver_titles[j]} @ {pct}% – Min: {mn:.3f}%, Max: {mx:.3f}%, Mean: {avg_misfit:.3f}%")

    plt.show()
