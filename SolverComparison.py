import h5py
import numpy as np
import torch
import time
import utilities as utl
import network as net
import plotting as pt

class InversionSolver:
    def __init__(self, data_file):
        self.data_file = data_file
        self.rho_p_raw = None
        self.curve_p_raw = None
        self.inverted_result = None
        self.inversion_time = None

    def load_data(self):
        test_data_folder = "./test_data/"
        with h5py.File(test_data_folder + self.data_file, 'r') as h5f:
            self.rho_p_raw = (np.array(h5f['rho']))
            self.curve_p_raw = np.array(h5f['curve'])
            # Add noise based on the standard deviation of each row
            # noise_curve = np.zeros_like(self.curve_p_raw)
            # for i in range(self.curve_p_raw.shape[0]):
            #     noise_level = 0.2 * np.std(self.curve_p_raw[i, :])
            #     noise_curve[i, :] = np.random.normal(0, noise_level, self.curve_p_raw.shape[1])
            # self.curve_p_raw += noise_curve

    def run_solver(self):
        raise NotImplementedError("Subclasses must implement run_solver method")

    def compare_results(self):
        inv_scaled = np.power(10,self.inverted_result)
        mse = np.mean((self.rho_p_raw - inv_scaled) ** 2)  
        mae = np.mean(np.abs(self.rho_p_raw - inv_scaled))  
        rmse = np.sqrt(mse)  
        r_squared = 1 - np.sum((self.rho_p_raw - inv_scaled) ** 2) / np.sum((self.rho_p_raw - np.mean(self.rho_p_raw)) ** 2) 
        
        print('MSE:', mse)
        print('MAE:', mae)
        print('RMSE:', rmse)
        print('R-squared:', r_squared)

class INNInversionSolver(InversionSolver):
    def __init__(self, data_file):
        super().__init__(data_file)
        self.all_z_samples = None
        self.true_rho_log=None
        self.tvd_pixel=None
        self.tvd_edge =None
        
    def run_solver(self):
        train_data_folder = "./training_data/"
        load_network_path = "./saved_network/"
        train_data_h5name = "2_5_layers_100000.h5"
        load_network_name = "inn_best.pth"
        with h5py.File(train_data_folder + train_data_h5name, 'r') as h5f:
            curve_raw = np.array(h5f['curve'])

        _ , scaler_att = utl.preprocess_train(curve_raw)
        curve_test_preprocessed = utl.preprocess_test(self.curve_p_raw, scaler_att)

        start_time = time.time()

        inv_rho_log_mean = np.zeros_like(self.rho_p_raw.T)
        inv_rho_log_std = np.zeros_like(self.rho_p_raw.T)

        n_trace = self.curve_p_raw.shape[0]

        n_sample = 50 # Number of sampling point for latent parameter z vector

        rho_log_pr_all = np.zeros((n_trace, 30, n_sample))
        tvd_pixel = np.linspace(-70, 70, self.rho_p_raw.shape[1])
        n_trace, x_dim, y_dim, e_dim, z_dim, tot_dim = utl.get_dims(self.rho_p_raw, curve_test_preprocessed)
        # Define INN and load trained network parameters
        inn = net.define_inn(tot_dim)
        state_dict = torch.load(load_network_path+load_network_name, weights_only=True)
        inn.load_state_dict(state_dict)
        inn.eval()
        for i_trace in range(n_trace):
            print("Inverting trace {0} ...".format(i_trace))
            curve_p_std = [curve_test_preprocessed[i_trace]]
 
            rho_log_pr_list = np.zeros((x_dim, n_sample))  # in shape (n_pixel, n_inv_sample)
            tvd_pr_list = np.tile(tvd_pixel, (n_sample,1)).T  # in shape (n_pixel, n_inv_sample)
            # Generate inverted parameter distribution (probability density q(m|d)) through 
            # repeat sampling on every latent parameter z following the independent standard normal distribution
            for i_sample in range(n_sample):
                z_pr = np.random.multivariate_normal([0.]*z_dim, np.eye(z_dim), 1)
                out_pr = np.concatenate([z_pr, curve_p_std], axis=-1).astype('float32')
                out_pr_ts = torch.from_numpy(out_pr)

                all_pr_inv,_ = inn(out_pr_ts,rev = True)
                rho_log_pr = all_pr_inv[0, 0:x_dim].detach().cpu().numpy()
                rho_log_pr_list[:, i_sample] = rho_log_pr
            
            # Calculate the mean and std for the inverted parameter distribution

            rho_log_pr_all[i_trace] = (rho_log_pr_list)
            mean_inv = np.median(rho_log_pr_list, axis = 1)
            std_inv = np.std(rho_log_pr_list, axis = 1)

            # Combine the results from all the 1D traces
            inv_rho_log_mean[:,i_trace] = (mean_inv)
            inv_rho_log_std[:,i_trace] = std_inv

        end_time = time.time()
        self.all_z_samples = rho_log_pr_all
        tvd_pixel2 = np.linspace(-70, 70, self.rho_p_raw.shape[1]+1)

        self.true_rho_log = np.log10(self.rho_p_raw)
        self.tvd_pixel = tvd_pr_list
        self.tvd_edge = tvd_pixel2
        
        self.inverted_result = inv_rho_log_mean

        self.inversion_time = end_time - start_time
        print(f"Time to invert model was: {self.inversion_time}")
        self.rho_p_raw = self.rho_p_raw.T
    def plot_results(self):
            true_2d_model = np.flip(np.log10((self.rho_p_raw)),axis=0)
            inv_2d_model = np.flip((self.inverted_result),axis=0)
            L2Norm = utl.relative_l2_norm(true_2d_model,inv_2d_model)
            print('L2 Norm:', L2Norm)
            highest_trace,lowest_trace,highest_samples,lowest_samples = pt.plot_results(true_2d_model, inv_2d_model,self.all_z_samples)
            combined_min = min(np.min(highest_samples), np.min(lowest_samples)) -.1
            combined_max = max(np.max(highest_samples), np.max(lowest_samples)) +.1
            x_lim_range = (combined_min,combined_max)
            pt.plot_uncertainty_distribution(highest_samples,x_lim_range,letter='a')
            pt.plot_uncertainty_distribution(lowest_samples,x_lim_range,letter='b')
            pt.plot_1d_slice(highest_trace,self.all_z_samples[highest_trace], self.true_rho_log, self.tvd_pixel, self.tvd_edge,letter= 'a')
            pt.plot_1d_slice(lowest_trace,self.all_z_samples[lowest_trace], self.true_rho_log, self.tvd_pixel, self.tvd_edge, letter='b')
class LmaInversionSolver(InversionSolver):
    def run_solver(self):     
        with h5py.File("lma_results.h5", 'r') as h5f:
            self.inverted_result = np.log10(np.array(h5f['rho'])).T

class OccumsInversionSolver(InversionSolver):
    def run_solver(self):
        with h5py.File("occams_result.h5", 'r') as h5f:
            self.inverted_result = np.log10(np.array(h5f['rho'])).T


inn_solver = INNInversionSolver('INNvsLMA.h5')
inn_solver.load_data()
INN_Results = inn_solver.run_solver()
comparison_metrics = inn_solver.compare_results()
inn_solver.plot_results()

# lma_solver = LmaInversionSolver('INNvsLMA.h5')
# lma_solver.load_data()
# lma_solver.run_solver()



# occum_solver = OccumsInversionSolver('INNvsLMA.h5')
# occum_solver.load_data()
# occum_solver.run_solver()


# inn_solver.inverted_result=np.flip(inn_solver.inverted_result,axis=0)
# lma_solver.inverted_result=np.flip(lma_solver.inverted_result,axis=0)
# occum_solver.inverted_result=np.flip(occum_solver.inverted_result,axis=0)



# inversion_list = [inn_solver.inverted_result,lma_solver.inverted_result,occum_solver.inverted_result]

# titles = ["INN","LMA","Occam"]
# pt.plot_2d_with_inversion_only(np.flip(np.log10(inn_solver.rho_p_raw),axis=0),inversion_list,titles)
# pt.plot_2d_with_misfit_only(np.flip(np.log10(inn_solver.rho_p_raw),axis=0),inversion_list,titles)