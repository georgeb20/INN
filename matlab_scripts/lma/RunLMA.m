clc
clear

forwardModelFolder = fullfile(pwd,'../forward_model');
addpath(forwardModelFolder);

n_pixel = 30;
data_size = 80;

input_file = "../../test_data/00_noise_Synthetic_Case.h5";
rho_test = h5read(input_file, '/rho');
curve_test = h5read(input_file,'/curve');

noise_curve = zeros(size(curve_test));
for i = 1:size(curve_test,1) 
    curve = curve_test(i,:);   
    std_curve = std(curve);   
    noise_std = 0 * std_curve;
    added_noise = noise_std * randn(1,size(curve_test,2));
    noise_curve(i,:) = added_noise;
end
curve_test = curve_test + noise_curve;

tStart = tic;
rho_pred = LMA_SOLVER(rho_test,curve_test);
elapsedTime = toc(tStart);
fprintf('Total elapsed time: %.2f seconds\n', elapsedTime);

filename = '../../lma_results/00_noise_lma_results';
disp('Start saving rho');
h5create([filename, '.h5'], '/rho', [n_pixel data_size]);
h5write([filename, '.h5'], '/rho', rho_pred);
