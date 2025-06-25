clc
clear

f1 = @(x) .9*(sin(x/8+1.2*pi)*7+10-(cos(x/3)*2.8)) - 40;  % Original
f2 = @(x) 2*(sin(x/10+3.2*pi)*7+5-(2*sin(-x/3)*0.8)) + 27;


figure(1)
hold on
fplot(f1)
fplot(f2)
hold off

x = 1:1:130;
xlim([min(x),max(x)])
ylim([-75 75])
h = gca;
set(h, 'YDir', 'reverse');
%%
tic;

freq_input = [2000 2000 12000 12000 48000 48000]';
tool_sp_input = [408 954 408 954 408 954]';
n_exp = numel(freq_input);

n_pixel = 30;
n_layer = 3;

interface = linspace(-70,70,n_pixel-1)';  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
epsilonr = [linspace(1,1,n_pixel)]';  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TVD = 0;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dip = 90;


data_size = length(x);
collect_all_data = zeros(8*n_exp, data_size);
collect_all_rho = zeros(n_pixel, data_size);


rho_temp = [1
15
1];


y = linspace(-70,70,n_pixel-1)';
y2 = f1(x);
y1 = f2(x);
DataGrid = repmat(struct('layers', [], 'split', [], 'rho', []), data_size, 1);

for ii = 1:data_size
    y_intersect_1 = interp1(x, y1, x(ii));
    y_intersect_2 = interp1(x, y2, x(ii));
    
    [~, index_intersect_1] = min(abs(y - y_intersect_1));
    [~, index_intersect_2] = min(abs(y - y_intersect_2));

    layer1 = n_pixel-index_intersect_1;
    layer2 = index_intersect_1-index_intersect_2;
    layer3 = n_pixel - (layer1+layer2);
    
    interface_temp = [layer3;layer2;layer1];

    DataGrid(ii) = struct('layers',n_layer,'split',interface_temp,'rho',rho_temp);
end

for i_data = 1:data_size
    rho_temp = DataGrid(i_data).rho;
    interface_layer = DataGrid(i_data).split;
    rho = repelem(rho_temp, interface_layer);
    for i_exp = 1:n_exp

        Resp_full_raw_truth = mexDipole(n_pixel, rho, rho, epsilonr, epsilonr, interface, TVD, Dip , 0 ,...
            freq_input(i_exp), tool_sp_input(i_exp)/2 , -tool_sp_input(i_exp)/2);  % Forward solver 
        H_field_tensor_truth = reshape(Resp_full_raw_truth,3,3);  % Reshape H tensor 
        truth_curve = fromFieldtoCurves(H_field_tensor_truth);  % Calculate 8 curves base on H tensor 
        truth_curve_all(1+(i_exp-1)*8:i_exp*8, 1) = truth_curve;  % record the 8 curves in each experiment to a list for all experiments
    end

    collect_all_data(:, i_data) = truth_curve_all;
    collect_all_rho(:, i_data) = rho;
end

elapsed_time = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsed_time);

filename = '../test_data/RealisticSyntheticCase';
fullfile = [filename, '.h5'];

if exist(fullfile, 'file')
    delete(fullfile);
    disp(['Deleted existing file: ', fullfile]);
end
disp('Start saving rho');
h5create([filename,'.h5'], '/rho', [n_pixel data_size]);
h5write([filename,'.h5'], '/rho', collect_all_rho);
disp('Start saving curvedata');
h5create([filename,'.h5'], '/curve', [8*n_exp data_size]);
h5write([filename,'.h5'], '/curve', collect_all_data);