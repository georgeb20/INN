import h5py
import numpy as np
import torch
import time
import utilities as utl
import network as net
import plotting as pt
import matplotlib.pyplot as plt
import FrEIA.modules as Fm            # already in your code
import os
from matplotlib.patches import FancyArrowPatch

class InversionSolver:
    def __init__(self, data_file):
        self.data_file = data_file
        self.rho_p_raw = None
        self.curve_p_raw = None
        self.inverted_result = None
        self.inversion_time = None

    def load_data(self, noise_level=0):
        test_data_folder = "./test_data/"
        with h5py.File(test_data_folder + self.data_file, 'r') as h5f:
            self.rho_p_raw = (np.array(h5f['rho']))
            try:
                self.curve_p_raw = np.array(h5f['curve'])
            except Exception as e:
                self.curve_p_raw = np.zeros((self.rho_p_raw.shape[0], 1))

            noise_curve = np.zeros_like(self.curve_p_raw)#80x48
            for i in range(self.curve_p_raw.shape[1]):
                curve = self.curve_p_raw[:,i]
                std_curve = np.std(curve)
                noise_std = noise_level * std_curve
                added_noise = noise_std * np.random.randn(self.curve_p_raw.shape[0])
                noise_curve[:,i] = added_noise

            self.curve_p_raw += noise_curve

    def run_solver(self,data_file=None):
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

    def invert_with_trace(inn: torch.nn.Module,x):
        """
        Run one reverse pass through `inn`, capturing the vector after every
        coupling block.

        Parameters
        ----------
        inn   : the trained SequenceINN (in eval() mode)
        z     : (1, z_dim)  standard-normal sample
        cond  : (1, d_cond) pre-processed curve (already on the same device)
        z_dim : dimension of z
        x_dim : number of pixels in the resistivity model (30 in your case)

        Returns
        -------
        states : list[Tensor]  length = n_layers + 1
                states[0] is the raw z,
                states[k] is the vector after the k-th coupling block,
                states[-1][:, :x_dim] is  log10(ρ)  predicted by the INN.
        """

        states = [x.clone()]

        for node in reversed(list(inn.children())):          
            x, _ = node(x, rev=True)                       
            states.append(x.clone())                   

        return states

    def run_solver(self,data_file = None):
        train_data_folder = "./training_data/"
        load_network_path = "./saved_network/"
        train_data_h5name = "2_5_layers_100000.h5"
        load_network_name = "inn_best_paper.pth"
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
        inn = net.define_inn(tot_dim, n_layer=5)
        state_dict = torch.load(load_network_path+load_network_name, weights_only=True)
        inn.load_state_dict(state_dict)
        inn.eval()
        with torch.no_grad():
            for i_trace in range(n_trace):
                print("Inverting trace {0} ...".format(i_trace))
                curve_p_std = [curve_test_preprocessed[i_trace]]
    
                rho_log_pr_list = np.zeros((x_dim, n_sample))  # in shape (n_pixel, n_inv_sample)
                tvd_pr_list = np.tile(tvd_pixel, (n_sample,1)).T  # in shape (n_pixel, n_inv_sample)
                # Generate inverted parameter distribution (probability density q(m|d)) through 
                # repeat sampling on every latent parameter z following the independent standard normal distribution
                for i_sample in range(n_sample):
                    trace_needed = (i_trace == 0 and i_sample == 0)   # e.g. only save 1 trace
                    z_pr = np.random.multivariate_normal([0.]*z_dim, np.eye(z_dim), 1)
                    out_pr = np.concatenate([z_pr, curve_p_std], axis=-1).astype('float32')
                    out_pr_ts = torch.from_numpy(out_pr)
                    if trace_needed:
                        out_pr2 = np.concatenate([z_pr, curve_p_std], axis=-1).astype('float32')      # (1, tot_dim)
                        x       = torch.from_numpy(out_pr2)                        # torch.FloatTensor
                        x = x.unsqueeze(0)                 # (1, tot)

                        blocks = [m for m in inn.modules()
                                if isinstance(m, Fm.GLOWCouplingBlock)]

                        states = [z_pr[0].copy()]
                        for blk in reversed(blocks):
                            x, _ = blk(x, rev=True)
                            rho_slice  = x[0][0, :x_dim]
                            states.append(rho_slice.cpu().numpy())

                        self.layer_snapshots = states

                    all_pr_inv,_ = inn(out_pr_ts,rev = True)

                    rho_log_pr = all_pr_inv[0, 0:x_dim].detach().cpu().numpy()
                    rho_log_pr_list[:, i_sample] = rho_log_pr
                
                # Calculate the mean and std for the inverted parameter distribution

                rho_log_pr_all[i_trace] = (rho_log_pr_list)
                mean_inv = np.mean(rho_log_pr_list, axis = 1)
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
            one = np.log10((self.rho_p_raw[:,0]))
            inv_2d_model = np.flip((self.inverted_result),axis=0)
            L2Norm = utl.l2_norm(true_2d_model,inv_2d_model)
            print('Relative L2 Norm INN % :', L2Norm)
            highest_trace,lowest_trace,highest_samples,lowest_samples, tmaxv, tminval= pt.plot_results_realistic(true_2d_model, inv_2d_model,self.all_z_samples)
            combined_min = min(np.min(highest_samples), np.min(lowest_samples)) -.1
            combined_max = max(np.max(highest_samples), np.max(lowest_samples)) +.1
            x_lim_range = (combined_min,combined_max)
            pt.plot_uncertainty_distribution(highest_samples,x_lim_range,tmaxv,letter='a')
            pt.plot_uncertainty_distribution(lowest_samples,x_lim_range,tminval,letter='b')
            pt.plot_1d_slice(highest_trace,self.all_z_samples[highest_trace], self.true_rho_log, self.tvd_pixel, self.tvd_edge,letter= 'a')
            pt.plot_1d_slice(lowest_trace,self.all_z_samples[lowest_trace], self.true_rho_log, self.tvd_pixel, self.tvd_edge, letter='b')
            #self.plot_layer_snapshots(self.layer_snapshots,one,self.tvd_pixel)

    from matplotlib.patches import FancyArrowPatch

    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D

    def plot_layer_snapshots(self,
            snapshots: list[np.ndarray],
            true_model=None,
            pixel_depths: np.ndarray = None
        ):
        """
        Plot the inverse evolution of a 1-D resistivity vector through network blocks,
        from final output back to the latent noise z.
        """
        if len(snapshots) != 6:
            raise ValueError(f"Need 6 layers (z + 5 blocks). Got {len(snapshots)}.")

        # Depth axis
        n_pixel = snapshots[0].shape[0]
        depth_axis = np.asarray(pixel_depths) if pixel_depths is not None else np.arange(n_pixel)

        # Shared x‐limits
        stack = np.vstack(snapshots)
        x_min, x_max = stack.min(), stack.max()

        # Reverse titles & data for inverse flow
        titles = ['Block 1', 'Block 2', 'Block 3', 'Block 4', 'Block 5', 'z']
        rev_snapshots = snapshots[::-1]

        # Figure setup
        fig, axes = plt.subplots(
            1, 6,
            figsize=(12, 4),
            dpi=200,
            sharey=True,
        )
        fig.subplots_adjust(top=0.80)  # leave room for legend
        fig.suptitle('Inverse Resistivity Generation Through GLOW Blocks', fontsize=14)
        fig.supxlabel(
            r'$\log_{10}(\rho)\;(\Omega\!\cdot\!\mathrm{ft})$',
            fontsize=12,
            y=0.03
        )

        # Plot each reversed snapshot
        for k, (ax, vec) in enumerate(zip(axes, rev_snapshots)):
            ax.plot(vec, depth_axis,
                    marker='o', linewidth=1.2, alpha=0.6,
                    color='red')
            if true_model is not None:
                ax.plot(true_model, depth_axis,
                        linewidth=1.2, alpha=0.8, color='blue')
            ax.set_xlim(x_min - 0.1, x_max + 0.05)
            ax.invert_yaxis()
            ax.set_xlabel(titles[k], fontsize=10)

            if k == 0:
                ax.set_yticks([40, 0, -40])
                ax.set_yticklabels([4950, 5000, 5050])
                ax.set_ylabel(
                    'Depth (ft)' if pixel_depths is not None else 'Pixel Index',
                    fontsize=11
                )
            ax.grid(ls=':', lw=0.5, color='gray')

        # Create proxy artists for legend
        from matplotlib.lines import Line2D

        proxy_inn = Line2D([0], [0], marker='o', color='red', linestyle='-', lw=1.2, alpha=0.6)
        proxy_true = Line2D([0], [0], color='blue', linestyle='-', lw=1.2, alpha=0.8)
        fig.legend(
            handles=[proxy_inn, proxy_true],
            labels=['Generated Model', 'True Model'],
            loc='upper center',
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.95)
        )

        # Optional arrow annotation
        arrow = FancyArrowPatch(
            (0.92, 0.12), (0.08, 0.12),
            transform=fig.transFigure,
            arrowstyle='-|>',
            mutation_scale=20,
            color='red',
            linewidth=2
        )
        fig.add_artist(arrow)

        plt.tight_layout()
        plt.show()




class LmaInversionSolver(InversionSolver):
    def run_solver(self,data_file=None):     
        with h5py.File(data_file, 'r') as h5f:
            self.inverted_result = np.log10(np.array(h5f['rho'])).T
        self.rho_p_raw = self.rho_p_raw.T
        true_2d_model = np.flip(np.log10((self.rho_p_raw)),axis=0)
        inv_2d_model = np.flip((self.inverted_result),axis=0)
        L2Norm = utl.l2_norm(true_2d_model,inv_2d_model)
        print('Relative L2 Norm % LMA:', L2Norm)


class OccumsInversionSolver(InversionSolver):
    def run_solver(self,data_file=None):
        with h5py.File(data_file, 'r') as h5f:
            self.inverted_result = np.log10(np.array(h5f['rho'])).T
        self.rho_p_raw = self.rho_p_raw.T
        true_2d_model = np.flip(np.log10((self.rho_p_raw)),axis=0)
        inv_2d_model = np.flip((self.inverted_result),axis=0)
        L2Norm = utl.l2_norm(true_2d_model,inv_2d_model)
        print('Relative L2 Norm % Occam', L2Norm)




inn_solver = INNInversionSolver('RealisticSyntheticCase.h5')
inn_solver.load_data(0)
INN_Results = inn_solver.run_solver()

# inn_solver10 = INNInversionSolver('00_noise_Synthetic_Case.h5')
# inn_solver10.load_data(0.1)
# INN_Results10 = inn_solver10.run_solver()

# inn_solver20 = INNInversionSolver('00_noise_Synthetic_Case.h5')
# inn_solver20.load_data(0.2)
# INN_Results20 = inn_solver20.run_solver()

# inn_solver30 = INNInversionSolver('00_noise_Synthetic_Case.h5')
# inn_solver30.load_data(0.3)
# INN_Results30 = inn_solver30.run_solver()

inn_solver.plot_results()

# inn_solver10.plot_results()

# inn_solver20.plot_results()

# inn_solver30.plot_results()

# #############################################################
# lma_solver = LmaInversionSolver('00_noise_Synthetic_Case.h5')
# lma_solver.load_data()
# lma_solver.run_solver("lma_results/00_noise_lma_results.h5")

# lma_solver10 = LmaInversionSolver('00_noise_Synthetic_Case.h5')
# lma_solver10.load_data()
# lma_solver10.run_solver("lma_results/10_noise_lma_results.h5")

# lma_solver20 = LmaInversionSolver('00_noise_Synthetic_Case.h5')
# lma_solver20.load_data()
# lma_solver20.run_solver("lma_results/20_noise_lma_results.h5")

# lma_solver30 = LmaInversionSolver('00_noise_Synthetic_Case.h5')
# lma_solver30.load_data()
# lma_solver30.run_solver("lma_results/30_noise_lma_results.h5")

# #############################################################
# occum_solver = OccumsInversionSolver('00_noise_Synthetic_Case.h5')
# occum_solver.load_data()
# occum_solver.run_solver("occams_results/00_noise_occam_results.h5")

# occum_solver10 = OccumsInversionSolver('00_noise_Synthetic_Case.h5')
# occum_solver10.load_data()
# occum_solver10.run_solver("occams_results/10_noise_occam_results.h5")

# occum_solver20 = OccumsInversionSolver('00_noise_Synthetic_Case.h5')
# occum_solver20.load_data()
# occum_solver20.run_solver("occams_results/20_noise_occam_results.h5")

# occum_solver30 = OccumsInversionSolver('00_noise_Synthetic_Case.h5')
# occum_solver30.load_data()
# occum_solver30.run_solver("occams_results/30_noise_occam_results.h5")

# # collect them with their noise levels
# solvers = {
#     0.0:   (inn_solver,   lma_solver,   occum_solver),
#     0.1:   (inn_solver10, lma_solver10, occum_solver10),
#     0.2:   (inn_solver20, lma_solver20, occum_solver20),
#     0.3:   (inn_solver30, lma_solver30, occum_solver30),
# }

# # 1) flip every solver’s result
# for lvl, (inn, lma, occ) in solvers.items():
#     inn.inverted_result  = np.flip(inn.inverted_result,  axis=0)
#     lma.inverted_result  = np.flip(lma.inverted_result,  axis=0)
#     occ.inverted_result  = np.flip(occ.inverted_result,  axis=0)

# # 2) build your inversion lists
# #    each entry is [INN_result, LMA_result, Occam_result]
# inversion_by_noise = {
#     lvl: [inn.inverted_result, lma.inverted_result, occ.inverted_result]
#     for lvl, (inn, lma, occ) in solvers.items()
# }


# solver_titles = ["INN", "LMA", "Occam"]
# noise_levels = [0.0, 0.1, 0.2, 0.3]

# raw = np.flip(np.log10(inn_solver.rho_p_raw), axis=0)
# pt.plot_all_noise_inversions(raw, inversion_by_noise, solver_titles, noise_levels)
# pt.plot_all_noise_misfits(raw,inversion_by_noise, solver_titles, noise_levels)
