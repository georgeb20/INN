clc
clear

n_pixel = 30;
data_size = 80;
input_file = "../test_data/INNvsLMA.h5"
rho_all = LMA_SOLVER(input_file);

filename = '../xlma_results';
disp('Start saving rho');
h5create([filename, '.h5'], '/rho', [n_pixel data_size]);
h5write([filename, '.h5'], '/rho', collect_all_rho);
disp('Start saving curvedata');
h5create([filename, '.h5'], '/curve', [8 * n_exp data_size]);
h5write([filename, '.h5'], '/curve', collect_all_data);