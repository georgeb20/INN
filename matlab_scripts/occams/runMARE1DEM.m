% runMARE1DEM - run 1D inversion for MARE2DEM
function result = runMARE1DEM(varargin)
    % create a diary and naming it with curretn time stamp
    diary(['diary_' datestr(now,'yyyy-mm-dd-HH-MM-SS') '.txt'])
    
    CONSTS = constants();
    inch = CONSTS.inch;
    feet = CONSTS.feet;
    
    % warning('off', 'MATLAB:singularMatrix');
    lFwdOnly             = false;   % use command line argument -F to set this to true.
    lSaveJacobian        = false;   % use command line argument -FJ to set this to true
    lSaveSensitivity     = false;   % use command line argument -FS to set this to true
    lCompJ = false;
    
    yamlfile = "test_batch.yaml";
    if ~isempty(varargin)
        yamlfile = varargin{1};
    end
    runConfig = yaml.loadFile(yamlfile); % load configuration from YAML file
    disp(yaml.dump(runConfig, "block")); % show yaml file
    sInv_method = runConfig.inversion_method; % 'occam' or 'staticmu'
    
    if isunix
        ParallelComp.parpool_init(runConfig.partition,runConfig.numOfWorkers)
    end

    time0 = tic;

    InversionIO.displayBanner()
    lFwdOnly = runConfig.FwdOnly;
    lSaveJacobian = runConfig.SaveJacobian;
    lSaveSensitivity = runConfig.SaveSensitivity;

    if lFwdOnly
        forward_fdm();
    else % inversion call
        if ~isempty(runConfig.FwdData)
            load(runConfig.FwdData)

            measurement = Curve_mat;
            noise_curve = zeros(size(measurement));
            for i = 1:size(measurement,1) 
                curve = measurement(i,:);   
                std_curve = std(curve);   
                noise_std = 0 * std_curve;
                added_noise = noise_std * randn(1,80);
                noise_curve(i,:) = added_noise;
            end
            measurement = measurement + noise_curve;

            TVD = TVD/feet;
            numPoint = size(measurement,2);

            % Constraint parameter
            model.r_lb = runConfig.lowerBoundGlobal;  %low bound of resistivity
            model.r_ub = runConfig.upperBoundGlobal;  %upper bound
            model.anis_lb = runConfig.lowerBoundAnisGlobal;  % Lower bound for Rv/Rh ratio
            model.anis_ub = runConfig.upperBoundAnisGlobal;  % Upper bound for Rv/Rh ratio
            model.anisotropy = runConfig.anisotropy; % >0: anisotropy is fixed; 0: anisotropy will be inverted
            model.nParamsAnis = runConfig.nParamsAnis; % number of anisotropy values to be inverted
            if model.anisotropy>0 % overwrite if fixed anisotropy
                model.nParamsAnis = 0;
                runConfig.nParamsAnis = 0;
            end

            model.Dip = tool.Dip;                    % Record the dip angle
            model.freq = tool.freq;
            model.spac = tool.ToolMulti;
            model.solver = runConfig.inversion_method;  % 'occam' or 'staticmu'

            % Configuration of inversion
            config.scope = str2num(runConfig.tool_investigation_scope{1});          % the tool investigation range
            config.maxItr = runConfig.maxItr;                 % max iteration
            config.fdstep = runConfig.fdstep;               % step factor for finite difference
            config.tolFunc = 1e-6;              % tolerance of function value
            config.tolchange = 1e-4;            % relative change of function value
            config.tolStep = 1e-6;               % tolerance of movement step

            % The configuration for the initialization function
            config.resolution = runConfig.resolution_of_pixel;              % the resolution of pixel
            config.meshType = runConfig.mesh_type;        % initial mesh type: equal or unequal
            config.comp = runConfig.computation_type;             % specify computation type (based on last iteration result or not)
                                                % 'iter' or 'noiter'
            config.initType = runConfig.initial_type;           % initial model type
            config.homo_Rh = runConfig.homo_Rh;

            % Initialize test model
            model.ref = []; % no reference model
            prior = 1;  %1: use previous station as initial model; 
                        %0: not use, initial model is homogeneous

            if runConfig.initial_is_homo
                rhoh(:) = runConfig.homo_Rh;
            end
            if runConfig.cBoundsTransform~='linear'
                Rh_init = log10(rhoh);
            else
                Rh_init = rhoh;
            end
            ratio = 2;
            Para_init = [Rh_init; ratio];

            model.mref = Para_init;
            model.ind_ref = ones(length(Para_init),1)*0.2;
            model.ZbedInput = model.Zbed; % record Zbed from input for model-based simulation
            
            freq = tool.freq/1000; %kHz
            spac = tool.ToolMulti; %inch
            n_freq = length(freq);
            n_spacing = length(spac);
            if runConfig.weightSpaceFreq=='uniform'
                wspac = ones(size(spac));
                wfreq = ones(size(freq));
            elseif runConfig.weightSpaceFreq=='nonuniform' %#ok<*BDSCA>
                wspac = 1./spac./sum(1./spac);
                wfreq = 1./freq./sum(1./freq);
            end
            w = ones(size(Curve_mat,1),1);
            n_freq_spac = n_freq * n_spacing;
            ind = 1;
            for ii = 1 : n_freq
                for jj = 1 : n_spacing
                    w(ind:n_freq_spac:end) = 1*wfreq(ii)*wspac(jj);
                    % w(ind:n_freq_spac:end) = [1 1 3 3 2 2 2 2]*wfreq(ii)*wspac(jj);
                    ind = ind + 1;
                end
            end
            
            weight = diag(w);

            fprintf('loaded data\n')
            
        else
            error('No forward data is provided. Abort!')
        end
        
        run_time = zeros(numPoint,1);
        misfit = zeros(numPoint,1);
        anis = zeros(numPoint,1);

        InvGroupPts = runConfig.InvGroupPts;
        inv_station_start = runConfig.inv_station_start;
        inv_station_end = runConfig.inv_station_end;
        
        if runConfig.inversionSteps>0
            totalStep = runConfig.inversionSteps;
        else
            totalStep = 0;
            runConfig.inversionSteps = 1;
        end
        result = cell(max(totalStep,1),length(TVD));

        for step=1:max(totalStep,1)
            fprintf("Running step %d....\n",step);
            if totalStep
                switch step
                    case 1 % short spacing
                        meas = 'f1s2 f2s2 f3s2';
%                         meas = 'f2s1 f3s1 f4s1 f5s1 f2s2 f3s2 f4s2 f5s2';
%                         meas = 'f4s1 f5s1 f6s1';
                    case 2 % long spacing
                        meas = 'f1s2 f2s2 f3s2 f4s2';
                    case 3 % short and ling spacing
                        meas = 'f2s1 f3s1 f4s1 f5s1 f2s2 f3s2 f4s2 f5s2';
                end
                [measIndices, fIndices, sIndices] = getMeasIndex(meas,size(measurement,1)/8);
                model.freq = unique(CONSTS.frequencies(fIndices));
                model.spac = unique(CONSTS.spacings(sIndices));
            else
                measIndices = 1:size(measurement,1);
            end
            
            if strcmp(config.comp, 'iter')
                for i = inv_station_start:InvGroupPts:inv_station_end
                    fprintf("There are %d inversion station left.\n",inv_station_end-i);
                    if i+InvGroupPts-1<=inv_station_end
                        data.measurement = measurement(measIndices,i:i+InvGroupPts-1);
                        data.stderr = runConfig.stderr;
                        data.weight = repmat(w(measIndices),1,InvGroupPts);
                        model.TVD = TVD(i:i+InvGroupPts-1);             % fetch the current tvd value
                    else
                        data.measurement = measurement(measIndices,i:inv_station_end);
                        data.stderr = runConfig.stderr;
                        data.weight = repmat(w(measIndices),1,inv_station_end-i+1);
                        model.TVD = TVD(i:inv_station_end);             % fetch the current tvd value
                    end

                    if i == inv_station_start
                        [init, mesh] = initialize(TVD(i),config,model);
                    else
                        [~, mesh] = initialize(TVD(i),config,model);
                        nL = length(mesh)+1;
                        init_linear = result{step,i-InvGroupPts}; % without logarithm
                        init = init_linear;
    %                     init_rh = 0.5*(log(init_linear(1:nL)-model.r_lb)-log(model.r_ub-init_linear(1:nL)));
    %                     init_anis = 0.5*(log(init_linear(nL+1)-model.anis_lb)-log(model.anis_ub-init_linear(nL+1)));
    %                     init = [init_rh; init_anis];
                    end 

                    model.Zbed = mesh;              % record Zbed (for forward calculation)
                   
                    % run the inversion control
                    tic;[result{step,i}, info_out] = inversion1D(init, data, model, runConfig, tool);
                    run_time(i) = toc;              % record the runtime
                   
                    misfit(i) = info_out(3);
                    if model.nParamsAnis>0
                        anis(i) = result{step,i}(end);
                    else
                        anis(i) = model.anisotropy;
                    end
                    fprintf("The anisotropy is %f \n", anis(i));
                end %InvGroupPts
            else %noiter, so using parallel computing
                numElements = numel(inv_station_start:InvGroupPts:inv_station_end);
                TVD_group = cell(numElements);
                measurement_par = measurement(measIndices,:);
                data.stderr = runConfig.stderr;
                w_par = w(measIndices);
                tool_para = tool;

                result_para = cell(runConfig.inversionSteps,numElements);
                misfit_para = zeros(numElements,1);
                anis_para = zeros(numElements,1);
                for i = 1:numElements
                    if i < numElements
                        TVD_group{i} = TVD((inv_station_start+(i - 1) * InvGroupPts):(inv_station_start+i*InvGroupPts)-1);
                    else
                        TVD_group{i} = TVD((inv_station_start+(i - 1) * InvGroupPts):inv_station_end);
                    end
                end
                if runConfig.parallel_run
                    for ii = 1:numElements  % parfor
                        i = (ii-1)*InvGroupPts+inv_station_start;
                        fprintf("There are %d inversion station left.\n",inv_station_end-i-InvGroupPts+1);
                        [init, mesh] = initialize(TVD(i),config,model);
                    
                        % run the inversion control
                        if i+InvGroupPts-1<=inv_station_end
                            [result_para{step,ii}, info_out] = inversion1D_para(init, data, ...
                                measurement_par(:,i:i+InvGroupPts-1), repmat(w_par,1,InvGroupPts), ...
                                model, runConfig, tool_para, TVD_group{ii}, mesh);
                        else
                            [result_para{step,ii}, info_out] = inversion1D_para(init, data, ...
                                measurement_par(:,i:i+InvGroupPts-1), repmat(w_par,1,InvGroupPts), ...
                                model, runConfig, tool_para, TVD_group{ii}, mesh);
                        end
                    
                        misfit_para(ii) = info_out(3);
                        if model.nParamsAnis>0
                            anis_para(ii) = result_para{step,ii}(end);
                        else
                            anis_para(ii) = model.anisotropy;
                        end
                        fprintf("The anisotropy is %f \n", anis_para(ii));
                    end %end of parfor numElements
                else
                    for ii = 1:numElements
                        i = (ii-1)*InvGroupPts+inv_station_start;
                        fprintf("There are %d inversion station left.\n",inv_station_end-i-InvGroupPts+1);
                        [init, mesh] = initialize(TVD(i),config,model);
                    
                        % run the inversion control
                        if i+InvGroupPts-1<=inv_station_end
                            [result_para{step,ii}, info_out] = inversion1D_para(init, data, ...
                                measurement_par(:,i:i+InvGroupPts-1), repmat(w_par,1,InvGroupPts), ...
                                model, runConfig, tool_para, TVD_group{ii}, mesh);
                        else
                            [result_para{step,ii}, info_out] = inversion1D_para(init, data, ...
                                measurement_par(:,i:i+InvGroupPts-1), repmat(w_par,1,InvGroupPts), ...
                                model, runConfig, tool_para, TVD_group{ii}, mesh);
                        end
                    
                        misfit_para(ii) = info_out(3);
                        if model.nParamsAnis>0
                            anis_para(ii) = result_para{step,ii}(end);
                        else
                            anis_para(ii) = model.anisotropy;
                        end
                        fprintf("The anisotropy is %f \n", anis_para(ii));
                    end %end of serial run numElements
                end
                % reorganize the data
                for ii = 1:numElements
                    i = (ii-1)*InvGroupPts+inv_station_start;
                    misfit(i) = misfit_para(ii);
                    anis(i) = anis_para(ii);
                    result(step,i) = result_para(step,ii);
                end
                    
            end % iter
        end %step

        timeOffset = toc(time0);
        fprintf('Total running time(s): %16.4f\n', timeOffset);

        % delete(gcp('nocreate'));

        save(runConfig.matfile);

        diary off
    end

end