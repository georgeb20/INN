clc
clear

forwardModelFolder = fullfile(pwd,'../forward_model');
addpath(forwardModelFolder);



f3 = @(x) (31 ...                             
     + 8 * sin((x + 200)/27 + pi/12) .* cos((x + 200)/12) ...
     + 15  * cos((x + 200)/10).^9 ...
     -15 * tanh(((x + 200) - 500)/25));


f2 = @(x) ( ...
   -16 ...                                            
  + 12 * ( cos((x+200)/29 - pi/3) .* sin((x+200)/42 + pi/5) ...
          - 0.7 * sin((x+200)/76) .* cos((x+200)/34 + pi/4) ) ... 
  + 4  * sin((x+200)/20) .* cos((x+200)/6) ...         
  + 6  * tanh(((x+200)-450)/100) ...                   
  + 2  * (cos((x+200)/30)).^2 ...                    
  - 0.005*(x) +35);                                      


f1 = @(x) ( ...
   -57 ...                                             
  + 11 * ( cos((x+200)/15 - pi/12) .* sin((x+200)/13 + pi/17) ...
          - 0.6 * sin((x+200)/16) .* cos((x+200)/50 + pi/3) ) ... 
  + 5  * sin((x+200)/12 + pi/5) .* cos((x+200)/15) ... 
  -12 * tanh(((x+200)-600)/20) ...                   
  + 5  * exp(-((x+200)-400).^2/120) ...                
  + 7  * tanh(((x+200)-200)/30) ...                  
  + 3  * sin((x+200)/5 + 0.3*(x/100)) );              


n_pixel = 30;

x = 1:1:100;
y = linspace(-70,70,n_pixel-1)';
y1 = f3(x);
y2 = f2(x);
y3 = f1(x);


figure(1)
fplot(f1) 
hold on
fplot(f2)
fplot(f3)
%fplot(f4) 
hold off
xlim([min(x),max(x)])
ylim([-75 75])
h = gca;
set(h, 'YDir', 'reverse');
%%
tic;

inch = 0.0254;
feet = 12 * inch;

DataGrid = struct('layers','','split','','rho','');



freq_input = [2000 2000 12000 12000 48000 48000]';
tool_sp_input = [408 954 408 954 408 954]';
n_exp = length(freq_input);

interface = linspace(-70,70,n_pixel-1)';  
epsilonr = [linspace(1,1,n_pixel)]';  
TVD = 0;                              
Dip = 90;

data_size = length(x);
collect_all_data = zeros(8*n_exp, data_size);
collect_all_rho  = zeros(n_pixel, data_size);

rho_temp = [15
            3
            15
            3];


n_layer = 4;

for ii = 1:data_size
    y_intersect_1 = interp1(x, y1, x(ii));
    y_intersect_2 = interp1(x, y2, x(ii));
    y_intersect_3 = interp1(x, y3, x(ii));

    [~, index_intersect_1] = min(abs(y - y_intersect_1));
    [~, index_intersect_2] = min(abs(y - y_intersect_2));
    [~, index_intersect_3] = min(abs(y - y_intersect_3));

    layer1 = n_pixel - index_intersect_1;
    layer2 = index_intersect_1 - index_intersect_2;
    layer3 = index_intersect_2 - index_intersect_3;
    layer4 = n_pixel - (layer1 + layer2 + layer3);

    interface_temp = [layer4; layer3; layer2; layer1];

    DataGrid(ii) = struct('layers', n_layer, ...
                          'split',  interface_temp, ...
                          'rho',    rho_temp);
end

% Forward modeling loop
for i_data = 1:data_size
    rho_temp        = DataGrid(i_data).rho;
    interface_layer = DataGrid(i_data).split;
    rho = repelem(rho_temp, interface_layer);
    
    truth_curve_all = zeros(8*n_exp, 1);
    
    for i_exp = 1:n_exp
        Resp_full_raw_truth = mexDipole( ...
            n_pixel, rho, rho, ...
            epsilonr, epsilonr, ...
            interface, TVD, Dip, 0, ...
            freq_input(i_exp), ...
            tool_sp_input(i_exp)/2, ...
            -tool_sp_input(i_exp)/2);
        
        H_field_tensor_truth = reshape(Resp_full_raw_truth,3,3);
        truth_curve = fromFieldtoCurves(H_field_tensor_truth);
        
        truth_curve_all(1+(i_exp-1)*8 : i_exp*8, 1) = truth_curve;
    end

    collect_all_data(:, i_data) = truth_curve_all;
    collect_all_rho(:, i_data)  = rho;
end

filename = '../../test_data/RealisticSyntheticCase';
disp('Start saving rho');
fullfile = [filename, '.h5'];

if exist(fullfile, 'file')
    delete(fullfile);
    disp(['Deleted existing file: ', fullfile]);
end

h5create([filename,'.h5'], '/rho', [n_pixel data_size]);
h5write([filename,'.h5'], '/rho', collect_all_rho);
disp('Start saving curvedata');
h5create([filename,'.h5'], '/curve', [8*n_exp data_size]);
h5write([filename,'.h5'], '/curve', collect_all_data);

