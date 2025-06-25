clc;
clear;

% Parameters
lower_bound = 0;    % Lower bound
upper_bound = 3;    % Upper bound
mu = (lower_bound+upper_bound)/2;   % Mean
sigma = 1;  % Standard deviation

% Generate truncated normal distribution
x = linspace(lower_bound, upper_bound, 1000); % Generate values between lower and upper bounds
pdf_values = normpdf(x, mu, sigma); % Calculate PDF values

% Plot the distribution
plot(x, pdf_values);
xlabel('Value');
ylabel('Probability Density');
title('Truncated Normal Distribution');
%%
function idx = findClosestIndex(array, value)
    [~, idx] = min(abs(array - value));
end

rng(42,'twister');


tic;


n_pixel = 30;

freq_input = [2000 2000 12000 12000 48000 48000]';
tool_sp_input = [408 954 408 954 408 954]';

n_exp = numel(freq_input);

interface = linspace(-70,70,n_pixel-1)';
epsilonr = ones(n_pixel,1);
TVD = 0; 
Dip = 90;

data_size = 100000;
collect_all_data = zeros(8*n_exp, data_size);
collect_all_rho = zeros(n_pixel, data_size);

lower_bound = min(interface)+10;
upper_bound = max(interface)-10;
DataGrid = repmat(struct('layers', [], 'split', [], 'rho', []), data_size, 1);

% Parallel loop to fill DataGrid
parfor ii = 1:data_size
    n_layer = randi([2, 5], 1, 1);
    rho_temp = 10.^x(randsample(length(x), n_layer, true, pdf_values));
    
    interface_temp = zeros(n_layer, 1);
    
    if n_layer == 1
        interface_temp = n_pixel;
    else
        intersects = zeros(n_layer - 1, 1);
        % Define bounds for intersections based on the number of layers
        bounds = linspace(upper_bound, lower_bound, n_layer);
        % Calculate intersections and indices
        for i = 1:(n_layer - 1)
            intersects(i) = (bounds(i) - bounds(i + 1)) * rand() + bounds(i + 1);
        end
        indices = arrayfun(@(y) findClosestIndex(interface, y), intersects);
        indices = [n_pixel; indices(:); 0];  % Add endpoints for indexing
        % Calculate layer thicknesses
        for i = 1:n_layer
            interface_temp(i) = indices(i) - indices(i + 1);
        end
        if n_layer==5
            disp(interface_temp);
        end
    end
        DataGrid(ii) = struct('layers', n_layer, 'split', interface_temp, 'rho', rho_temp);
end

% Parallel loop to compute data
parfor i_data = 1:data_size
    rho_temp = DataGrid(i_data).rho;
    interface_layer = DataGrid(i_data).split;
    rho = repelem(rho_temp, interface_layer);
    data_curve_all = zeros(8*n_exp, 1);
    for i_exp = 1:n_exp
        Resp_full_raw_truth = mexDipole(n_pixel, rho, rho, epsilonr, epsilonr, interface, TVD, Dip, 0, ...
            freq_input(i_exp), tool_sp_input(i_exp)/2, -tool_sp_input(i_exp)/2);  % Forward model 
        H_field_tensor_truth = reshape(Resp_full_raw_truth, 3, 3);  % Reshape H tensor 
        truth_curve = fromFieldtoCurves(H_field_tensor_truth);  % Calculate 8 curves base on H tensor 
        data_curve_all(1 + (i_exp - 1) * 8:i_exp * 8, 1) = truth_curve;  % record the 8 curves in each experiment to a list for all experiments
    end

    collect_all_data(:, i_data) = data_curve_all;
    collect_all_rho(:, i_data) = rho;
end

elapsed_time = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);

filename = '../training_data/2_5_layers_100000';
disp('Start saving rho');
h5create([filename, '.h5'], '/rho', [n_pixel data_size]);
h5write([filename, '.h5'], '/rho', collect_all_rho);
disp('Start saving curvedata');
h5create([filename, '.h5'], '/curve', [8 * n_exp data_size]);
h5write([filename, '.h5'], '/curve', collect_all_data);