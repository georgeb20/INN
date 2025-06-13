import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

train_data_folder = "./training_data/"
train_data_h5name = "2_5_layers_100000.h5"
with h5py.File(train_data_folder + train_data_h5name, 'r') as h5f:

    rho_raw = np.log10(np.array(h5f['rho']))  # shape = (n_trace, n_pixel)


    print(rho_raw)
    curve_raw = np.array(h5f['curve'])  # shape = (n_trace, n_exp)

    rho_mean = np.mean(rho_raw)
    rho_std = np.std(rho_raw)
    rho_skewness = np.mean((rho_raw - rho_mean) ** 3) / rho_std ** 3
    rho_kurtosis = np.mean((rho_raw - rho_mean) ** 4) / rho_std ** 4 - 3

    curve_mean = np.mean(curve_raw)
    curve_std = np.std(curve_raw)
    curve_skewness = np.mean((curve_raw - curve_mean) ** 3) / curve_std ** 3
    curve_kurtosis = np.mean((curve_raw - curve_mean) ** 4) / curve_std ** 4 - 3

    # Print the calculated metrics
    rho_min = np.min(rho_raw)
    rho_max = np.max(rho_raw)

    curve_min = np.min(curve_raw)
    curve_max = np.max(curve_raw)

    # Print the calculated min and max values

    print("rho_raw:")
    print("Minimum:", rho_min)
    print("Maximum:", rho_max)
    print("Mean:", rho_mean)
    print("Standard Deviation:", rho_std)
    print("Skewness:", rho_skewness)
    print("Kurtosis:", rho_kurtosis)

    print("\ncurve_raw:")
    print("Minimum:", curve_min)
    print("Maximum:", curve_max)
    print("Mean:", curve_mean)
    print("Standard Deviation:", curve_std)
    print("Skewness:", curve_skewness)
    print("Kurtosis:", curve_kurtosis)
    plt.figure(figsize=(6, 4))

    # Plot for rho_raw
    
    plt.hist(rho_raw.flatten(), bins=500, color='blue', alpha=0.7)
    plt.title('Distribution of 100,000 Resitivity Samples')
    plt.xlabel("$\log_{10}(ρ)(\Omega \cdot m)$")
    plt.ylabel('Frequency')    
    plt.show()   
    
    # Plot for curve_raw
    
    plt.hist(curve_raw.flatten(), bins=50, color='green', alpha=0.7)
    plt.title('Distribution of 100,000 EM Response')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    
    plt.show()   
    
    
