clc
clear

parentFolder = fullfile(pwd, '../..');
addpath(parentFolder);
utilsFOlder = fullfile(pwd, '../utilities/');
addpath(utilsFOlder);
forwardModelFolder = fullfile(pwd,'../forward_model');
addpath(forwardModelFolder);

runConfigFile = 'inversion1D.yaml';
result = runMARE1DEM(runConfigFile);



result_record_YS = zeros(30, 80);
for ii = 1:80
    result_record_YS(:,ii) = result{ii};
end

save("result_to_George_01sm_0718.mat", "result_record_YS")


filename = '../../occams_results/00_noise_lma_results';
disp('Start saving rho');
h5create([filename,'.h5'], '/rho', [30 80]);
h5write([filename,'.h5'], '/rho', result_record_YS);