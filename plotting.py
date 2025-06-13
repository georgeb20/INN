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
    trace,
    inverted_rho_log,
    true_rho_log,
    tvd_pixel,
    tvd_edge,
    letter=''
):
    # Basic stats for plotting lines
    n_pixel = inverted_rho_log.shape[0]
    mean_inv = np.mean(inverted_rho_log, axis=1)
    percentile_low = np.percentile(inverted_rho_log, 10, axis=1)
    percentile_high = np.percentile(inverted_rho_log, 90, axis=1)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(5, 6))
    min_inverted = np.min(inverted_rho_log)
    max_inverted = np.max(inverted_rho_log)

    cmap = plt.cm.viridis.copy()  # Copy to avoid modifying the original colormap
    cmap.set_under("grey")  # Set the color for values below vmin to white

    # 2D histogram of the inverted data
    h = ax.hist2d(
        inverted_rho_log.flatten(),
        tvd_pixel.flatten(),
        bins=[150, n_pixel],
        range=[[0.8,2.2], [tvd_edge[0], tvd_edge[-1]]],
        cmap=cmap,
        vmin=0.001
    )

    # Add colorbar
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label("Density", fontsize=11, color="black")

    # Superimpose horizontal lines:
    ax.stairs(
        true_rho_log[trace].flatten(),
        tvd_edge,
        orientation="horizontal",
        linewidth=2.5,
        color="blue",
        label="True Model"
    )
    ax.stairs(
        mean_inv,
        tvd_edge,
        orientation="horizontal",
        linewidth=2.5,
        color="red",
        label="Mean Inversion"
    )
    ax.stairs(
        percentile_low,
        tvd_edge,
        orientation="horizontal",
        linewidth=1.5,
        color="white",
        linestyle="--",
        label="10th/90th Percentile"
    )
    ax.stairs(
        percentile_high,
        tvd_edge,
        orientation="horizontal",
        linewidth=1.5,
        color="white",
        linestyle="--"
    )

    # Axis labels and inversion of y-axis 
    ax.set_xlabel(f"$\\mathrm{{log}}_{{10}}(\\rho)\\, (\\Omega \\cdot \\text{{ft}})$\n\n ({letter})", color='black', fontsize=11)
    ax.set_ylabel("Depth (ft)", color='black', fontsize=11)
    ax.invert_yaxis()

    ax.set_yticks([-50, 0, 50])
    ax.set_yticklabels([4950, 5000, 5050])

    # Plot title
    ax.set_title("1D Resistivity Slice at X = {}".format(trace * 10), color='black', fontsize=12)

    ax.text(
        0.05,
        0.95,
        f'x={trace * 10}',
        color='black',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=14,
        bbox=dict(facecolor='black', alpha=0.3, boxstyle='round,pad=0.3')
    )

    # Add a faint grid
    ax.grid(color='gray', linestyle=':', linewidth=0.5)

    # Add legend
    ax.legend(facecolor='white', edgecolor='black')

    # Tight layout
    plt.tight_layout()

    # Show the plot
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

def _add_twin_axis(ax, x_data, y1_data, y2_data, invert_range=(75, -75)):
    ax_twin = ax.twinx()
    ax_twin.plot(x_data, y1_data, color='white', linewidth=2, label='f1(x)')
    ax_twin.plot(x_data, y2_data, color='white', linewidth=2, label='f2(x)')
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

def plot_2d_with_misfit_only(true_2d_model,inv_2d_models,inv_2d_models_titles):
    fig, axs = plt.subplots(len(inv_2d_models), 1, figsize=(6,9),constrained_layout=True)
    letters = ['b','c','d']

    for i in range(len(inv_2d_models)):
        axs[i].set_ylim(0, true_2d_model.shape[0])
        axs[i].set_xlim(0, true_2d_model.shape[1])
        model_misfit = utl.model_misfit(true_2d_model, inv_2d_models[i], type="r2norm")
        p3 = axs[i].plot(np.linspace(1, 80, 80), model_misfit, "bP--", label="Relative Model Misfit")
        axs[i].set_xlabel(f"({letters[i]}) X (ft)")
        axs[i].set_ylabel("Relative Model Misfit (%)")
        axs[i].set_ylim(0,10)
        axs[i].set_title(f"{inv_2d_models_titles[i]} Relative L2-Norm Misfit (%)")
        arr =  [0,100,200,300,400,500,600,700,800]
        axs[i].set_xticklabels(arr)  # Change the tick labels
        labels = axs[i].get_xticklabels()
        new_labels = [f"X{label.get_text()}" for label in labels]
        axs[i].set_xticklabels(new_labels)
    plt.show()
def plot_2d_with_inversion_only(true_2d_model, inv_2d_models,inv_2d_models_titles):
    ''''''

    x_grid = np.linspace(0, 1000, true_2d_model.shape[0])  

    n_trace = true_2d_model.shape[1]
    fig, axs = plt.subplots(len(inv_2d_models), 1, figsize=(6,9),constrained_layout=True)

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
        axs[i].set_title(f"{inv_2d_models_titles[i]} Inverted Resistivity Model, ($\\log_{{10}}(\\rho) (\\Omega \\cdot ft)$)")
        axs[i].set_yticks([25, 15, 5])  # Set the original ticks
        axs[i].set_yticklabels([4950, 5000, 5050])  # Change the tick labels
        arr =  [0,100,200,300,400,500,600,700,800]
        axs[i].set_xticklabels(arr)  # Change the tick labels
        labels = axs[i].get_xticklabels()
        new_labels = [f"X{label.get_text()}" for label in labels]
        axs[i].set_xticklabels(new_labels)
        cbar = fig.colorbar(p, ax=axs[i], aspect=17)
        cbar.set_label("$\log_{10}(ρ)(\Omega \cdot ft)$", labelpad=15)
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
    plt.show()