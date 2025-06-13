clc
clear

n_pixel = 30;
data_size = 80;
input_file = "../test_data/INNvsLMA.h5";
rho_test = h5read(input_file, '/rho');
curve_test = h5read(input_file,'/curve');
rho_pred = LMA_SOLVER(rho_test,curve_test);
rho_all = fliplr(rho_all);
filename = '../xlma_results';
disp('Start saving rho');
h5create([filename, '.h5'], '/rho', [n_pixel data_size]);
h5write([filename, '.h5'], '/rho', rho_all);
