classdef Occam < handle
    properties (SetAccess=public)
        maxOccamIterations = 40    % Just what is says.
        currentIteration   = 0      % Current Occam iteration number
        occamPrintLevel    = 1      % 0: print nothing, 1: also print logfile statements to the terminal window.
        modelLog10Mu             % input and output model mu, large default
        targetRMS                   % target RMS misfit
        modelRoughness              % output model roughness
        modelRMS                    % output model RMS misfit
    end

    % Definition for convergenceFlag:
    %  0 - Successful iteration: Model RMS was lowered but target RMS has not been obtained
    %  1 - Successful iteration: Target RMS has been obtained and Occam smoothing is in progress
    %  2 - Normal convergence:   Target RMS has been obtained, model has been smoothed until the step size is below a threshold
    %  3 - Unusual convergence:  A perfectly smooth model has been found. Occam is done.
    %  4 - Convergence problem:  Occam could not find a model with an RMS less than the last iteration
    %  5 - Convergence problem:  Occam found the target misfit, but the model roughness couldn't be decreased.
    %  6 - Convergence problem:  Maximum number of iterations achieved.
    properties (SetAccess=public)
        convergenceFlag
    end
    properties
        nd = 0;          % Number of data
        d = [];          % Data array d(nd)
        sd = [];         % Standard error array sd(nd)
        dp = [];         % Data parameters (nd x nDataParams) lookup table of parameter indices (frequencies, positions, etc)
        d_wt = [];       % Data weights applied with sd but not to misfit, used to balance the influence of different data types
        % for example, a few MT data with LOTS of CSEM, use data weights to upweight the MT influence.

        nParams = 0;     % Number of model parameters
        nParamsAnis = 1; % number of anisotropic ratio in model parametes
        pm = [];         % Model parameter array pm(nParams)
        dm = [];         % Model response vector dm(nd)
        dm_0 = [];       % Model response of input starting mode for each iteration

        wj = [];         % Model response Jacobian matrix
        premod = [];     % Model preference values premod(nParams)
        prewts = [];     % Weights for model preference prewts(nParams). Set to 0 for no preference

        nDiff = 0;       % Number of parameter difference preference values
        preDiff = [];    % Model parameter difference preference values
        preDiffwts = []; % Weights for model parameter difference. Set to 0 for no preference. preDiffwts(nParams)
        ijDiff = [];     % nDiff x 2 array of i,j indices of model parameters to difference. ijDiff(nParams,2). Set to 0 for no difference

        npenalty = 0;    % Number of rows in the penalty matrix
        penaltywts = []; % Vector of penalty weights to apply
        ijpenalty = [];  % npenalty x 2 array of i,j indices of model parameters to difference
    end
    properties
        pen = PenaltyMatrix()       % penalty matrix in sparse matrix format
    end
    properties
        lMGS = false;           % Option for minimum gradient support regularization
        beta_mgs = 1e-2;        % Beta parameter for MGS regularization
        lrough_with_prej = false; % If true, spatial roughness term is ||R(m-m_prej)||

        npm_assoc = 0;          % Number of associate (pass through) parameters
        pm_assoc;          % Associated parameter arrays pm_assoc(npm_assoc)

        % David Myer's autoconverge feature (my implementation is still a work in progress):
        lAutoConverge = true;
        nAutoConvergeFactor = 1.5;

        % David Myer's large RMS decrease escape hatch. Minimization stops if RMS decreases
        % significantly before the minimum is found, so that the next iteration can then proceed
        % with a new Jacobian. Iteration ends when rms < rmsThreshold*startingRMS
        rmsThreshold = 0.85; % (0 <= rmsThreshold < 1), 0.7-0.9 works well

        % Option to slowly obtain the target misfit setting the target misfit to be
        % max(targetRMS, startingRMS*rmsThreshold) so that Occam finds the smoothest
        % model at each step.
        lConvergeSlowly = false;

        % Non-linear transformation for constraining parameters with upper and lower bounds:
        bandPassFactor;       % factor in exp(bandPassFactor/(a-b)*x)
        cBoundsTransform; % 'exponential' or 'bandpass'

        % Can be modified by an optional parameter in the iteration file.
        % 'exponential' - from Habashy and Abubakar (2004), Commer and Newman (2008)
        % 'bandpass' - designed by K. Key in March 2011
        lowerBound;
        upperBound;
        lBoundMe;

        % If true, saves Occam's dense linear system for the winning mu value at each iteration.
        lSaveOccamSystem = false;
        domainSet;
        tool;
        model;
    end
    properties (SetAccess=private)
        % Parameters controlling Occam's behavior:
        rmsTol;       % Tolerance for hitting target RMS
        % iteration complete if: abs(RMS-targetRMS) < rmsTol
        maxNumStepCuts = 8;  % don't cut the step-size in half more than this
        stepSizeTol;         % Stop if targetRMS obtained and stepSize is smaller than this
        numStepCuts = 0;
        fdstep;

        nParallelProc;
        MinProcScaLAPACK = 4;
        lUseScaLAPACK = false;

        % Working arrays and variables:
        lTargetHit=false;          % true if input model is at target RMS
        startingRMS;
        startingRoughness;
        stepCut;             % current reduction in model update size
        stepSize;

        % Matrices and arrays for the inversion process
        wjtwj;    % product of weighted jacobian matrix and its transpose
        wjtwd;    % product of weighted jacobian matrix and weighted translated data
        resid;    % model fit residual, normalized by sd
        amat;     % matrix to invert for model update

        % Arrays to keep track of optimal model and response
        pm_b; dm_b; pm_assoc_b;
        lmu_b; rms_b; rough_b;

        % Used during intercept search
        pm_a; dm_a; pm_assoc_a;
        lmu_a; rms_a; rough_a;

        pm_c; dm_c; pm_assoc_c;
        lmu_c; rms_c; rough_c;

        % Used in misfitOfMu
        pm_test;
        rough_test;

        tracker1_rms; tracker1_mu;
        tracker_Dummy = 1e15; % dummy initialization flag
        lTrackerOn;

        numForwardCalls; numForwardCallsCumulative;
        timeIter0; timeIterEnd; timeOccamStart; timeOccamEnd;

        % For ScaLAPACK, we need storage for the local subarrays of wjtwj:
        desc_wjtwj;
        wjtwj_loc;
    end


    methods
        function obj = Occam(data, param, nDiff, npenalty, npm_assoc, domainSet, tool, model)
            % Occam constructor: Initializes the properties with provided values.
            % If no arguments are provided, the properties will remain as empty arrays.
            if nargin > 0
                obj.d = data.measurement(:);
                obj.nd = numel(data.measurement);
                obj.sd = data.stderr*ones(obj.nd, 1);
                obj.dp = zeros(obj.nd, 4);
                obj.d_wt = data.weight(:);

                %input model is bounded
                obj.pm = param;
                obj.nParams = size(param,1);
                obj.premod = zeros(obj.nParams, 1);
                obj.prewts = zeros(obj.nParams, 1);
                obj.nDiff = nDiff;
                obj.preDiff = zeros(obj.nDiff, 1);
                obj.preDiffwts = zeros(obj.nDiff, 1);
                obj.ijDiff = zeros(obj.nDiff, 2);
                obj.npenalty = npenalty;
                obj.penaltywts = zeros(obj.npenalty, 1);
                obj.ijpenalty = zeros(obj.npenalty, 2);
                obj.npm_assoc = npm_assoc;
                obj.pm_assoc = zeros(obj.npm_assoc, 1);

                obj.bandPassFactor = domainSet.bandPassFactor;
                obj.cBoundsTransform = domainSet.cBoundsTransform;
                obj.nParamsAnis = domainSet.nParamsAnis;

                if domainSet.lowerBoundGlobal>-inf && domainSet.upperBoundGlobal<inf
%                     obj.lowerBound = ones(obj.nParams, 1)*log10(domainSet.lowerBoundGlobal);
%                     obj.upperBound = ones(obj.nParams, 1)*log10(domainSet.upperBoundGlobal);
                    obj.lowerBound = ones(obj.nParams, 1)*(domainSet.lowerBoundGlobal);
                    if obj.nParamsAnis>0
                        obj.lowerBound(obj.nParams-obj.nParamsAnis+1:obj.nParams) = domainSet.lowerBoundAnisGlobal;
                    end
                    obj.upperBound = ones(obj.nParams, 1)*(domainSet.upperBoundGlobal);
                    if obj.nParamsAnis>0
                        obj.upperBound(obj.nParams-obj.nParamsAnis+1:obj.nParams) = domainSet.upperBoundAnisGlobal;
                    end
                    obj.lBoundMe = true(obj.nParams, 1);
                end

                obj.domainSet = domainSet;
                obj.tool = tool;
                obj.targetRMS = domainSet.targetRMS;
                obj.rmsTol = domainSet.rmsTol;
                obj.fdstep = domainSet.fdstep;
                obj.model = model;
            end
            obj.stepSizeTol = 2^(1 - obj.maxNumStepCuts);
            obj.numForwardCallsCumulative = 0;
            obj.modelLog10Mu = domainSet.modelLog10Mu;

            % transform model to unbounded
            obj.pm = obj.transformToUnbound(param, domainSet.lowerBoundGlobal,domainSet.upperBoundGlobal,obj.lBoundMe,false); % transform Rh
            if domainSet.nParamsAnis
                obj.pm = obj.transformToUnbound(obj.pm, domainSet.lowerBoundAnisGlobal,domainSet.upperBoundAnisGlobal,obj.lBoundMe,true); % transform anisotropy
            end
        end
    end

    methods (Access=public)
        function openOccamLog(obj)
            fprintf('Format:     Inversion2D OccamLog.2023.4\n')
            currentTimestamp = datetime('now');
            disp(currentTimestamp);
            poolObj = gcp('nocreate');
            if ~isempty(poolObj)
                fprintf('Number of parallel processors: %8d\n', poolObj.NumWorkers)
            end
            fprintf('Number of input data: %8d\n', obj.nd)
            fprintf('Number of model parameters: %8d\n', obj.nParams)
            obj.timeOccamStart = tic;
        end

        function closeOccamLog(obj)
            currentTimestamp = datetime('now');
            fprintf(' End time: %s\n', currentTimestamp)
            obj.timeOccamEnd = toc(obj.timeOccamStart);
            fprintf(' Total time for all iterations(s): %16.4f\n', obj.timeOccamEnd)
        end
        
        function printOccamExitMessage(obj)
            % Messages written at the end of an Occam iteration

            % Stop the timer:
            obj.timeIterEnd = toc(obj.timeIter0);
            timeOccamCumulative = toc(obj.timeOccamStart);
            obj.numForwardCallsCumulative = obj.numForwardCallsCumulative + obj.numForwardCalls;

            % Print out the summary information about this iteration:
            fprintf(' \n');

            if (obj.convergenceFlag ~= 4 && obj.convergenceFlag ~= 5)
                cStr = 'Occam iteration completed successfully: ';
                obj.printOccamLog(cStr);
                cStr = ' ';
                obj.printOccamLog(cStr);
                cStr = sprintf('Target Misfit: %16.4f', obj.targetRMS);
                obj.printOccamLog(cStr);
                cStr = sprintf('Model Misfit: %16.4f', obj.modelRMS);
                obj.printOccamLog(cStr);
                cStr = sprintf('Roughness: %16.4f', obj.modelRoughness);
                obj.printOccamLog(cStr);
                cStr = sprintf('Optimal Mu: %16.4f', obj.modelLog10Mu);
                obj.printOccamLog(cStr);
                cStr = sprintf('Stepsize: %16.4f', obj.stepSize);
                obj.printOccamLog(cStr);
                cStr = sprintf('Convergence Status: %1d', obj.convergenceFlag);
                obj.printOccamLog(cStr);
                cStr = sprintf('# Forward Calls, Cumulative: %4d %4d', obj.numForwardCalls, obj.numForwardCallsCumulative);
                obj.printOccamLog(cStr);
                cStr = sprintf('Iteration Time, Cumulative (s): %16.4f %16.4f', obj.timeIterEnd, timeOccamCumulative);
                obj.printOccamLog(cStr);
                cStr = ' ';
                obj.printOccamLog(cStr);
            end

            % Special messages based on convergence status:
            switch obj.convergenceFlag
                case 0
                    % Do nothing since target misfit not yet obtained
                case 1
                    % Do nothing since Occam is still smoothing the model
                case 2
                    cStr = 'Stopping on normal convergence.';
                    obj.printOccamLog(cStr);
                case 3
                    cStr = 'Perfectly smooth model found, stopping.';
                    obj.printOccamLog(cStr);
                case 4
                    cStr = 'RMS misfit can not be decreased further, stopping.';
                    obj.printOccamLog(cStr);
                    cStr = 'May have reached minimum possible RMS misfit level!';
                    obj.printOccamLog(cStr);
                case 5
                    cStr = 'Model roughness can not be decreased further, stopping.';
                    obj.printOccamLog(cStr);
                case 6
                    cStr = 'Maximum number of iterations achieved, stopping.';
                    obj.printOccamLog(cStr);
            end
        end

        function saveIterationResults(obj)
            dataname = sprintf('results_it_%03d.mat', obj.currentIteration);
            outval = obj.transformToBound(obj.pm, obj.lowerBound, obj.upperBound, obj.lBoundMe);
            save(dataname, 'outval');
        end

        
        function deallocateOccam(obj)
            % Deallocates arrays used during the entire Occam inversion
            obj.pm = [];
            obj.prewts = [];
            obj.premod = [];
            obj.preDiff = [];
            obj.preDiffwts = [];
            obj.ijDiff = [];
            obj.d = [];
            obj.sd = [];
            obj.dp = [];
            obj.dm = [];
            obj.dm_0 = [];
            obj.pm_assoc = [];
            obj.penaltywts = [];
            obj.ijpenalty = [];
            obj.pen.colind = [];
            obj.pen.val = [];
            obj.pen.rowptr = [];
            obj.lowerBound = [];
            obj.upperBound = [];
        end

        function computeOccamIteration(obj, lSaveJacobian, lSaveSensitivity)
            % Computes an Occam iteration.

            % Say hello and initialize a few quantities
            obj.printOccamIntroMessage();

            % Compute forward response and Jacobian matrix, also allocate arrays
            % used in the model update equations during Occam's lagrange multiplier search:
            obj.computeJacobian(lSaveJacobian, lSaveSensitivity);

            % If using Slow Occam, reset the target misfit:
            if obj.lConvergeSlowly
                targetRMS_input = obj.targetRMS;
                obj.targetRMS = max(obj.startingRMS * obj.rmsThreshold, obj.targetRMS);

                if obj.targetRMS ~= targetRMS_input
                    cStr = sprintf('Occam set to smoothly converge, for this iteration the target RMS is: %g\n', obj.targetRMS);
                    obj.printOccamLog(cStr);
                end
            end

            % Is the starting model at the target RMS already? If so, let the smoothing begin.
            if obj.startingRMS < obj.targetRMS + obj.rmsTol
                obj.lTargetHit = true;
            end

            % Allocate some working arrays used throughout the rest of the iteration
            % This is done after the Jacobian computation in order to keep the memory usage overlap as low as possible
            obj.allocateOccamIteration(obj.nParams, obj.nd, obj.npm_assoc);

            % Get minimum RMS as a function of mu:
            obj.modelLog10Mu = obj.getMinimum(obj.modelLog10Mu); % optimal model variables are returned in _b arrays

            if obj.convergenceFlag == 4 % too many step size cuts, give up
                obj.deallocateOccamIteration();
                return;
            end

            % Minimum has been obtained, now analyze it:
            % Is the minimum less than the target RMS? If so, get the intercept with the target RMS
            if obj.rms_b <= obj.targetRMS + obj.rmsTol
                 obj.convergenceFlag = 1;
                 lGetIntercept = true;

                 while lGetIntercept
                     obj.getIntercept();

                     if obj.lTargetHit
                         if obj.rough_b == 0
                             obj.convergenceFlag = 3;
                             break;
                         elseif obj.rough_b < obj.startingRoughness
                             obj.convergenceFlag = 1;
                             break;
                         else
                             obj.stepCut = obj.stepCut ./ 2;
                             obj.numStepCuts = obj.numStepCuts + 1;
                             obj.initializeTrackers();

                             if obj.numStepCuts < obj.maxNumStepCuts
                                 fprintf('Cutting step size due to model roughness, new step size is: %16.4f\n', obj.stepCut);
                                 obj.lmu_b = obj.getMinimum(obj.lmu_b);
                                 continue;
                             else
                                 obj.convergenceFlag = 5;
                                 fprintf('Roughness not decreased by using small steps, giving up.\n');
                                 break;
                             end
                         end
                     else
                         obj.convergenceFlag = 1;
                         break;
                     end
                 end
             end

             % Compute step size (how much did the model change):
             obj.pm = obj.pm_b - obj.pm;
             obj.stepSize = sqrt(dot(obj.pm, obj.pm) ./ obj.nParams);

             if obj.convergenceFlag == 1 && obj.lTargetHit
                 if obj.stepSize < obj.stepSizeTol
                     obj.convergenceFlag = 2;
                 end
             end

             % Move optimal model to output:
             obj.pm = obj.pm_b;
             obj.dm = obj.dm_b;
             obj.modelRoughness = obj.rough_b;
             obj.modelLog10Mu = obj.lmu_b;
             obj.modelRMS = obj.rms_b;
             if numel(obj.pm_assoc) > 0
                 obj.pm_assoc = obj.pm_assoc_b;
             end

             % Are we at the last allowable iteration?
             if obj.currentIteration >= obj.maxOccamIterations
                 obj.convergenceFlag = 6;
             end

             if obj.lConvergeSlowly
                 obj.targetRMS = targetRMS_input;
                 if obj.modelRMS > obj.targetRMS + obj.rmsTol
                     obj.convergenceFlag = 0;
                 end
             end

             % Save dense linear system if requested by user:
             if obj.lSaveOccamSystem
                constructLHS(obj.modelLog10Mu) 
                obj.save_lhs()     
                obj.constructRHS(obj.modelLog10Mu)    
                obj.save_rhs()
                obj.save_stepsize()
             end

             obj.printOccamExitMessage()
             obj.deallocateOccamIteration()
             % obj.saveIterationResults()
        end

        function m = transformToBound(obj, xin, bin, ain, lbind)
            % Converts the unbound parameter x to the bound parameter b < m < a

            % Convert input to double variables
            a = ain;
            b = bin;
            x = xin;

            if lbind
                switch obj.cBoundsTransform
                    case 'linear'
                        x = (a.*exp(x)+b.*exp(-x))./(exp(x)+exp(-x));
                    case 'exponential'
                        x = x-(a+b)/2;
                        x = (a .* exp(x) + b) ./ (exp(x) + 1);
                    case 'bandpass'
                        c = obj.bandPassFactor ./ (a - b);
                        if x <= 0
                            q = log((exp(c .* x) + exp(c .* b)) ./ (exp(c .* x) + exp(c .* a)));
                        else
                            q = log((1 + exp(c .* (b - x))) ./ (1 + exp(c .* (a - x))));
                        end
                        x = q ./ c + a;
                end
            end
            % keep the anisotropic ratios unchanged
            % x(obj.nParams-obj.nParamsAnis+1:obj.nParams) = xin(obj.nParams-obj.nParamsAnis+1:obj.nParams);
            m = x;
        end

        function addPenalty(obj,model,weightx,weightz)
            % Weighting matrix Ws for model structure
            % Lateral direction
            indxi = zeros(obj.nParams,2);
            indxj = indxi;
            valx = indxj;
            for i = 1:obj.nParams
                if (i <= obj.nParams-obj.nParamsAnis-model.numY)
                    indxi(i,1) = i;
                    indxi(i,2) = i;
                    indxj(i,1) = i;
                    indxj(i,2) = i+model.numY;
                    valx(i,1) = -1;
                    valx(i,2) = 1;
                else
                    indxi(i,1) = i;
                    indxi(i,2) = i;
                    indxj(i,1) = i;
                    indxj(i,2) = i;
                    valx(i,1) = 0;
                    valx(i,2) = 0;
                end        
            end
            Wx = sparse(indxi(:),indxj(:),valx(:)*weightx);

            % Vertical direction
            indzi = zeros(obj.nParams,2);
            indzj = indzi;
            valz = indzj;
            for i = 1:obj.nParams
                if i==obj.nParams-obj.nParamsAnis || i==obj.nParams
                    indzi(i,1) = i;
                    indzi(i,2) = i;
                    indzj(i,1) = i;
                    indzj(i,2) = i;
                    valz(i,1) = 0;
                    valz(i,2) = 0;
                elseif mod(i,model.numY)==0
                    indzi(i,1) = i;
                    indzi(i,2) = i;
                    indzj(i,1) = i;
                    indzj(i,2) = i+1;
                    valz(i,1) = 1e-2;
                    valz(i,2) = 1e-2;
                else
                    indzi(i,1) = i;
                    indzi(i,2) = i;
                    indzj(i,1) = i;
                    indzj(i,2) = i+1;
                    valz(i,1) = -1;
                    valz(i,2) = 1;
                end
            end
            Wz = sparse(indzi(:),indzj(:),valz(:)*weightz);
            W = Wx+Wz;

            obj.nDiff = nnz(W);
            fprintf('Number of difference penalties: %d\n', obj.nDiff);
            obj.ijDiff = zeros(obj.nDiff,2);
            obj.preDiff = zeros(obj.nDiff,1);

            [obj.ijDiff(:,1), obj.ijDiff(:,2), obj.preDiffwts] = find(W);

            if any(obj.ijDiff(:) > obj.nParams-obj.nParamsAnis)
                error(' Error, difference penalty matrix points to parameter numbers larger than the input free parameters!\nLargest index: %d, Number of free parameters: %d\n Stopping!\n', max(obj.ijDiff(:)), obj.nParams-1);
            end
        end

        function addPenalty_fix(obj,weightv)
            % Weighting matrix Ws for model structure
            % direction 1
            indxi = zeros(obj.nParams-obj.nParamsAnis,1);
            indxj = indxi;
            valx = indxj;
            for i = 1:obj.nParams-obj.nParamsAnis
                indxi(i) = i;
                if i==obj.nParams-obj.nParamsAnis
                    indxj(i) = i;
                else
                    indxj(i) = i+1;
                end
                valx(i) = 1*weightv;
                if i>=obj.nParams-obj.nParamsAnis
                    valx(i) = 1e-5;
                else
                    valx(i) = 1*weightv;
                end
            end

            obj.nDiff = obj.nParams-obj.nParamsAnis;
            fprintf('Number of difference penalties: %d\n', obj.nDiff);
            obj.ijDiff = [indxi indxj];
            obj.preDiff = zeros(obj.nDiff,1);
            obj.preDiffwts = valx;

            if any(obj.ijDiff(:) > obj.nParams-obj.nParamsAnis) || any(obj.ijDiff(:) == 0)
                error(' Error, difference penalty matrix points to parameter numbers larger than the input free parameters!\nLargest index: %d, Number of free parameters: %d\n Stopping!\n', max(obj.ijDiff(:)), obj.nParams-1);
            end
        end
    end

    methods (Access = private)
        function printOccamIntroMessage(obj)
            cStr = repmat('-', 1, 127);
            obj.printOccamLog(cStr);

            cStr = sprintf('** Iteration %5d **', obj.currentIteration);
            obj.printOccamLog(cStr);

            cStr = ' ';
            obj.printOccamLog(cStr);

            cStr = ' Constructing derivative dependent matrices...';
            obj.printOccamLog(cStr);

            obj.timeIter0 = tic;
            obj.numStepCuts = 0;
            obj.stepCut = 1.0;
            obj.numForwardCalls = 0;   
            obj.convergenceFlag = 0;

            obj.initializeTrackers();
        end
        
        function allocateOccamIteration(obj, nParams, nd, npm_assoc)
            % Allocates arrays during an Occam iteration:
            obj.pm_b = zeros(nParams, 1);
            obj.pm_c = zeros(nParams, 1);
            obj.pm_test = zeros(nParams, 1);
            obj.dm_b = zeros(nd, 1);
            obj.dm_c = zeros(nd, 1);

            % Allocate matrix globally if PC or only a few processors
            obj.amat = zeros(nParams, nParams);

            % Allocate arrays for associated parameters
            if npm_assoc > 0
                obj.pm_assoc_b = zeros(npm_assoc, 1);
                obj.pm_assoc_c = zeros(npm_assoc, 1);
            else
                obj.pm_assoc_b = [];
                obj.pm_assoc_c = [];
            end

        end

        function deallocateOccamIteration(obj)
            % Deallocates arrays used during an Occam iteration

            if ~isempty(obj.wjtwj), obj.wjtwj = []; end
            if ~isempty(obj.wjtwd), obj.wjtwd = []; end
            if ~isempty(obj.resid), obj.resid = []; end
            if ~isempty(obj.amat), obj.amat = []; end
            if ~isempty(obj.pm_b), obj.pm_b = []; end
            if ~isempty(obj.dm_b), obj.dm_b = []; end
            if ~isempty(obj.pm_test), obj.pm_test = []; end
            if ~isempty(obj.pm_assoc_b), obj.pm_assoc_b = []; end
            if ~isempty(obj.pm_c), obj.pm_c = []; end
            if ~isempty(obj.dm_c), obj.dm_c = []; end
            if ~isempty(obj.pm_assoc_c), obj.pm_assoc_c = []; end
        end
        
        function set_scalapack(obj)
        % Check for reasons to not use scalapack parallel solver: 
            if (obj.lSaveOccamSystem) 
                obj.lUseScaLAPACK = false;  % form linear system on root process only so easy to write to file
            end
            if (obj.nParallelProc < obj.MinProcScaLAPACK)
                obj.lUseScaLAPACK = false;
            end
        end

        function computeStaticMuIteration(obj, lSaveJacobian, lSaveSensitivity)

            % Computes Gauss-Newton iteration with a static fixed value of mu

            % Initialize a few quantities
            obj.printOccamIntroMessage();

            % Compute forward response and Jacobian matrix, also allocate arrays
            % used in the model update equations during Occam's lagrange multiplier search
            obj.computeJacobian(lSaveJacobian, lSaveSensitivity);

            % If using Slow Occam, reset the target misfit
            if obj.lConvergeSlowly
                targetRMS_input = obj.targetRMS;
                obj.targetRMS = max(obj.startingRMS * obj.rmsThreshold, obj.targetRMS);

                if obj.targetRMS ~= targetRMS_input
                    fprintf('Occam set to smoothly converge, for this iteration the target RMS is: %g\n', obj.targetRMS);
                end
            end

            % Is the starting model at the target RMS already? If so, let the smoothing begin.
            if obj.startingRMS < obj.targetRMS + obj.rmsTol
                obj.lTargetHit = true;
            else
                obj.lTargetHit = false;
            end

            % Allocate some working arrays used throughout the rest of the iteration
            % This is done after the Jacobian computation in order to keep the memory usage overlap as low as possible
            obj.allocateOccamIteration(); % Replace [...] with output variables if needed

            % Loop that cuts step size until a better fitting model has been obtained (or it gives up)
            lReduceMisfit = true;

            fprintf('... Computing model update step:\n');
            fprintf('%16s %16s %16s %16s %16s %16s %16s %16s\n', 'Misfit', ...
                'Roughness', 'Log10(mu)', 'Min log10(rho)', 'Max log10(rho)', ...
                'Mean Anisotropy', 'Fwd Call (s)', 'Matrix Ops. (s)');

            while lReduceMisfit
                rms = obj.misfitOfMu(obj.modelLog10Mu);

                if rms >= obj.startingRMS
                    % Minimum isn't as good as starting model, cut step size
                    obj.stepCut = obj.stepCut ./ 2.0;
                    obj.numStepCuts = obj.numStepCuts + 1;

                    if obj.numStepCuts < obj.maxNumStepCuts
                        fprintf('Cutting step size in attempt to reduce misfit, new step size is: %g\n', obj.stepCut);
                    end

                    % Have we cut the step size too many times to no avail?
                    if obj.numStepCuts > obj.maxNumStepCuts
                        obj.convergenceFlag = 4;
                        lReduceMisfit = false;
                    end
                else
                    lReduceMisfit = false;
                end
            end

            % Move results to _b arrays
            obj.updateB(obj.modelLog10Mu, rms);

            if obj.rms_b <= obj.targetRMS + obj.rmsTol
                obj.convergenceFlag = 2;
            end

            % Compute step size (how much did the model change)
            pm_diff = obj.pm_b - obj.pm;
            obj.nParams = length(obj.pm);
            obj.stepSize = sqrt(dot(pm_diff, pm_diff) ./ obj.nParams);

            % Move optimal model to output
            obj.pm = obj.pm_b;
            obj.dm = obj.dm_b;
            obj.modelRoughness = obj.rough_b;
            obj.modelLog10Mu = obj.lmu_b;
            obj.modelRMS = obj.rms_b;

            if ~isempty(obj.pm_assoc)
                obj.pm_assoc = obj.pm_assoc_b;
            end

            % Are we at the last allowable iteration?
            if obj.currentIteration >= obj.maxOccamIterations
                obj.convergenceFlag = 6;
            end

            if obj.lConvergeSlowly
                obj.targetRMS = targetRMS_input;
                if obj.modelRMS > obj.targetRMS + obj.rmsTol
                    obj.convergenceFlag = 0;
                end
            end

            % Print out some summary information
            obj.printOccamExitMessage();

            % Deallocate working arrays
            obj.deallocateOccamIteration();

        end

        function computeJacobian(obj, lSaveJacobian, lSaveSensitivity)
            % Computes the forward response and Jacobian matrix for the input model at the start of an Occam iteration.
            % Inserts these into various arrays required by the Occam model update equations.

            % Allocate arrays
            [obj.wj, obj.resid] = deal(zeros(obj.nd, obj.nParams), zeros(obj.nd, 1));

            % Compute the forward response and the Jacobian matrix: results returned in arrays dm and wj
            t0 = tic;
%             obj.computeFwd(true, obj.transformToBound(obj.pm, obj.lowerBound, obj.upperBound, obj.lBoundMe));
            obj.dm = forward_1d(obj.pm,obj.model);
            fun = @(x)forward_1d(x,obj.model);
            obj.wj = obj.NumJ(fun,obj.pm,obj.fdstep,obj.dm,obj.nParams,obj.nd); 
            t1 = toc(t0);

            % Save Jacobian if requested
            if lSaveJacobian
                obj.writeJacobian(obj.currentIteration);
            end

            % Save sensitivity if requested
            if lSaveSensitivity
                obj.writeSensitivity(obj.currentIteration);
            end

            % Save the input model response to dm_0 array
            obj.dm_0 = obj.dm;

            t0 = tic;

            % Scale Jacobian for nonlinear transformation to bound model parameters
            % obj.transformWJ();

            % Compute residual and RMS misfit
            obj.resid = (obj.d - obj.dm) ./ obj.sd;
            obj.startingRMS = sqrt(sum(obj.resid .* obj.resid) ./ obj.nd);
            obj.startingRoughness = obj.getRoughness(obj.pm);

            % Weight the Jacobian matrix by the data uncertainty, also apply optional data weights
            for i = 1:obj.nd
                obj.wj(i, :) = obj.wj(i, :) * obj.d_wt(i) / obj.sd(i);
            end

            % Form arrays needed for the Occam model update equations
            fprintf(' ... Forming matrix products... \n');

            % Form (WJ)^T W \hat(d)
            obj.wjtwd = zeros(obj.nParams, 1);
            beta = 1.0;
            dhat = obj.resid .* obj.d_wt;
            dhat = obj.wj*obj.pm+beta*dhat; % dhat = wj * pm + beta * dhat
            beta = 0.0;
            obj.wjtwd = obj.wj'*dhat; % wjtwd = (wj)^T * dhat
            dhat = [];

            % Form (WJ)^T WJ
            obj.wjtwj = zeros(obj.nParams, obj.nParams);
            obj.wjtwj = obj.wj'*obj.wj;

            % Display the input model information
            t2 = toc(t0);
            pm_temp = obj.transformToBound(obj.pm, obj.lowerBound, obj.upperBound, obj.lBoundMe);
            minv = min(pm_temp(1:end-obj.nParamsAnis));
            maxv = max(pm_temp(1:end-obj.nParamsAnis));
            if obj.nParamsAnis
                anisMean = mean(pm_temp(end-obj.nParamsAnis+1:end));
            else
                anisMean = NaN;
            end

            fprintf(' ... Starting model and matrix assembly:\n');

            fprintf('%16s %16s %16s %16s %16s %16s %16s %16s\n', 'Misfit', 'Roughness', 'Log10(mu)', ...
                'Min log10(rho)', 'Max log10(rho)', 'Mean Anisotroy', 'Jacobian (s)', 'Matrix Ops. (s)');

            fprintf('%16.5g %16.5g %16.5g %16.5g %16.5g %16.5g %16.5g %16.5g\n', obj.startingRMS, ...
                obj.startingRoughness, obj.modelLog10Mu, minv, maxv, anisMean, t1, t2);

        end

        function misfit = misfitOfMu(obj,alogmu)
            % Solves the Occam model update equations for a given log10(mu)

            % Input argument
            % alogmu: log10 of mu value

            t0 = tic;

            % Check for very large or small mu, if so set large misfit and return
            if abs(alogmu) > 20
                fprintf(' Skipping forward call due to extremely large or small mu in misfitOfMu ...\n');
                misfit = 1000 + abs(alogmu);  % return large misfit that grows with small or large mu
                return;
            end

            % Construct rhs vector
            obj.constructRHS(alogmu);

            % Solve the linear system
            istat = 0;
            if ispc
                obj.constructLHS(alogmu);
                try
                    obj.pm_test = obj.solveLinearSystem(obj.amat, obj.pm_test);
                catch exception
                    fprintf('Error: %s\n', exception.message);
                    istat = 1;
                end
            end

            if ~ispc
                if ~obj.lUseScaLAPACK
                    % this is a windows PC or only a few processors are being used, solve using LAPACK:
                    % Construct LHS matrix:
                    obj.constructLHS(alogmu);
    
                    % solve the system using lapack:
                    try
                        obj.pm_test = obj.solveLinearSystem(obj.amat, obj.pm_test);
                    catch exception
                        fprintf('Error: %s\n', exception.message);
                        istat = 1;
                    end
                else
                    obj.pm_test = obj.solveLinearSystem_scalapack(10^alogmu);
                end
            end
            % obj.pm_test(end) = 1;
            % obj.pm_test = obj.transformToUnbound(obj.pm_test, 0.5,10,obj.lBoundMe,true); % transform anisotropy

            if istat > 0
                fprintf(' Solving Ax=b failed for this mu, returning large misfit...\n');
                misfit = 1000 + abs(alogmu);  % return large misfit that grows with small or large mu
                return;
            end

            % Cut step size if necessary
            if obj.stepCut < 1.0
                obj.pm_test = (1 - obj.stepCut) * obj.pm + obj.stepCut * obj.pm_test;
            end

            t1 = toc(t0);

            % Compute forward response
            t0 = tic;
%             obj.computefwd(false, obj.transformToBound(obj.pm_test, obj.lowerBound, obj.upperBound, obj.lBoundMe)); % don't forget to convert back to bounded
            [obj.dm] = forward_1d(obj.pm_test,obj.model);
            t2 = toc(t0);

            obj.numForwardCalls = obj.numForwardCalls + 1;

            % Compute RMS misfit
            obj.resid = (obj.d - obj.dm) ./ obj.sd;
            % chi2 = dot(obj.d_wt.*obj.resid, obj.d_wt.*obj.resid);
            chi2 = dot(obj.resid, obj.resid);
            misfit = sqrt(chi2 ./ double(obj.nd));

            if isnan(misfit)  % If the misfit is nan then problem encountered in FWD code. Can occur for extremely small MU values.
                fprintf('Error: nan misfit. Returning artificially large misfit to steer inversion back to the good place...\n');

                % Make artificially large misfit
                misfit = 1e2 * max(abs(alogmu), 1);
            end

            % Compute test model roughness
            obj.rough_test = obj.getRoughness(obj.pm_test);

            % Update the (mu, rms) tracker
            obj.updateTrackers(alogmu, misfit);

            % Report to terminal and logfile
            pm_test_temp = obj.transformToBound(obj.pm_test, obj.lowerBound, obj.upperBound, obj.lBoundMe);
            minv = min(pm_test_temp(1:end-obj.nParamsAnis));
            maxv = max(pm_test_temp(1:end-obj.nParamsAnis));
            if obj.nParamsAnis
                ratioMean = mean(pm_test_temp(end-obj.nParamsAnis+1:end));
            else
                ratioMean = NaN;
            end

            fprintf('%16.5g %16.5g %16.5g %16.5g %16.5g %16.5g %16.5g %16.5g\n', misfit, obj.rough_test, alogmu, minv, maxv, ratioMean, t2, t1);
        end

        function constructRHS(obj, alogmu)
            % Build RHS vector used in Occam model update equation

            % Input argument
            % alogmu: log10 of mu value

            % Local variables
            mu = 10^alogmu;

            obj.pm_test = obj.wjtwd;

            if obj.lrough_with_prej
                for irow = 1:obj.pen.nrows_spatial
                    istart = obj.pen.rowptr(irow);
                    iend = obj.pen.rowptr(irow + 1) - 1;

                    if obj.lMGS
                        w_mgs = 1 ./ (dot(obj.pen.val(istart:iend), obj.pm(obj.pen.colind(istart:iend))).^2 + obj.beta_mgs.^2);
                    else
                        w_mgs = 1;
                    end

                    for i1 = istart:iend
                        for i2 = istart:iend
                            i = obj.pen.colind(i1);
                            j = obj.pen.colind(i2);

                            w_i = obj.pen.val(i1);
                            w_j = obj.pen.val(i2);

                            obj.pm_test(i) = obj.pm_test(i) + mu * w_i * w_j * w_mgs * obj.premod(j);
                        end
                    end
                end
            else
                obj.pm_test = obj.pm_test + mu .* (obj.prewts.^2) .* obj.premod;
            end

            % Add on parameter difference preferences, if any given
%             for irow = 1:obj.nDiff
%                 i = obj.ijDiff(irow, 1);
%                 j = obj.ijDiff(irow, 2);
%                 w2 = obj.preDiffwts(irow)^2;
% 
%                 obj.pm_test(i) = obj.pm_test(i) + mu * w2 * obj.preDiff(irow);
%                 obj.pm_test(j) = obj.pm_test(j) - mu * w2 * obj.preDiff(irow);
%             end

        end

        function constructLHS(obj, alogmu)
            % Construct matrix A

            % Input argument
            % alogmu: log10 of mu value

            obj.amat = zeros(obj.nParams, obj.nParams);

            % First insert penalty terms for del^T del
            if ~isempty(obj.pen.val)
                for irow = 1:obj.pen.nrows
                    istart = obj.pen.rowptr(irow);
                    iend = obj.pen.rowptr(irow + 1) - 1;

                    if obj.lMGS
                        w_mgs = 1 ./ (dot(obj.pen.val(istart:iend), obj.pm(obj.pen.colind(istart:iend)))^2 + obj.beta_mgs^2);
                    else
                        w_mgs = 1;
                    end

                    for i1 = istart:iend
                        for i2 = istart:iend
                            i = obj.pen.colind(i1);
                            j = obj.pen.colind(i2);

                            w_i = obj.pen.val(i1);
                            w_j = obj.pen.val(i2);

                            obj.amat(i, j) = obj.amat(i, j) + w_i * w_j * w_mgs;
                        end
                    end
                end
            else
                for ipen = 1:obj.npenalty
                    i = obj.ijpenalty(ipen, 1);
                    j = obj.ijpenalty(ipen, 2);
                    w2 = obj.penaltywts(ipen)^2;

                    obj.amat(i, j) = obj.amat(i, j) + w2;
                end
            end

            if ~obj.lrough_with_prej
                for i = 1:obj.nParams
                    if obj.prewts(i) ~= 0
                        obj.amat(i, i) = obj.amat(i, i) + obj.prewts(i)^2;
                    end
                end
            end

            for irow = 1:obj.nDiff
                i = obj.ijDiff(irow, 1);
                j = obj.ijDiff(irow, 2);
                w2 = obj.preDiffwts(irow)^2;

                obj.amat(i, j) = obj.amat(i, j) - w2;
                obj.amat(j, i) = obj.amat(j, i) - w2;
                obj.amat(i, i) = obj.amat(i, i) + w2;
                obj.amat(j, j) = obj.amat(j, j) + w2;
            end

            mu = 10^alogmu;
            obj.amat = mu * obj.amat + obj.wjtwj;

        end

        function initializeTrackers(obj)
            % The trackers keep track of the best fitting mu and the mu immediately to the right, so that if Occam meets the target RMS,
            % it will already have bounding mu's for the right side intercept.
            %
            % The two trackers need to be reinitialized from a few places, so they are placed in this subroutine to hide the distracting
            % details from the rest of the code.
            %
            obj.tracker1_mu  = -obj.tracker_Dummy;  % left and right bounds on mu
            obj.lmu_c         =  obj.tracker_Dummy;
            obj.tracker1_rms =  obj.tracker_Dummy;  % large positive misfits for dummy values
            obj.rms_c        =  obj.tracker_Dummy;
            obj.lTrackerOn   = true;
        end
        
        function updateTrackers(obj, alogmu, rms)
            % Update trackers

            % Input arguments
            % alogmu: log10 of mu value
            % rms: root mean square value

            if ~obj.lTrackerOn
                return;
            end

            if rms < obj.tracker1_rms
                mOld = obj.tracker1_mu;

                obj.tracker1_mu = alogmu;
                obj.tracker1_rms = rms;

                if mOld > obj.tracker1_mu
                    obj.lmu_c = obj.lmu_b;
                    obj.rms_c = obj.rms_b;
                    obj.rough_c = obj.rough_b;
                    obj.dm_c = obj.dm_b;
                    obj.pm_c = obj.pm_b;
                    if obj.npm_assoc > 0
                        obj.pm_assoc_c = obj.pm_assoc_b;
                    end
                end
            elseif alogmu > obj.tracker1_mu && alogmu < obj.lmu_c
                obj.lmu_c = alogmu;
                obj.rms_c = rms;
                obj.rough_c = obj.rough_test;
                obj.dm_c = obj.dm;
                obj.pm_c = obj.pm_test;
                if obj.npm_assoc > 0
                    obj.pm_assoc_c = obj.pm_assoc;
                end
            end
        end

        function escape = lEscapeFromMinimization(obj, rms)
            % Test various conditions for which the Occam minimization should be terminated early.

            escape = false;

            % Condition 1: the rms is less than or equal to the target
            if (rms <= obj.targetRMS + obj.rmsTol)
                escape = true;

                % Condition 2: a significant decrease in RMS has occurred
            elseif (rms <= obj.rmsThreshold * obj.startingRMS)
                escape = true;
            end
        end

        function roughness = getRoughness(obj, model)
            % Computes the roughness for the input model

            model_size = numel(model);
            if model_size == obj.nParams
                % fprintf('The size of the model matches nParams (%d).\n', obj.nParams);
            else
                error('The size of the model does not match nParams (%d).\n', obj.nParams);
            end

            roughness = 0;

            % || R*m ||^2
            for i = 1:obj.pen.nrows
                istart = obj.pen.rowptr(i);
                iend = obj.pen.rowptr(i+1) - 1;

                if obj.lrough_with_prej
                    rtemp = dot(obj.pen.val(istart:iend), (model(obj.pen.colind(istart:iend)) - obj.premod(obj.pen.colind(istart:iend))));
                else
                    rtemp = dot(obj.pen.val(istart:iend), model(obj.pen.colind(istart:iend)));
                end

                % Re-weight if using minimum gradient support regularization:
                if obj.lMGS
                    w_mgs = 1.0 ./ sqrt( (dot(obj.pen.val(istart:iend), obj.pm(obj.pen.colind(istart:iend))) )^2 + obj.beta_mgs^2 );
                    rtemp = rtemp * w_mgs;
                end

                roughness = roughness + rtemp^2;
            end

            % Now add on the norm for the preference model (which is 0 if not used since prewts=0 then):
            if ~obj.lrough_with_prej
                roughness = roughness + sum((obj.prewts .* (obj.premod - model)).^2);
            end

            % Add on norm for model parameter difference preferences, if any given:
            for irow = 1:obj.nDiff
                i = obj.ijDiff(irow, 1);
                j = obj.ijDiff(irow, 2);
                roughness = roughness + (obj.preDiffwts(irow) * (model(i) - model(j)))^2;
            end
        end

        function lastMu = getMinimum(obj, lastMu)
            % Sweeps through mu to find optimal RMS

            % Local variables
            lGetMinimum = true;

            fprintf(' Searching Lagrange multiplier...\n');

            while lGetMinimum
                % Bracket the minimum using parabolic interpolation or golden section search
                mu1 = lastMu - 1.0;
                mu2 = lastMu;

                mu3 = obj.bracketMinimum(mu1, mu2); % best model is returned in _b arrays

                % Now find the minimum if the RMS is still big
                if obj.rms_b <= obj.targetRMS + obj.rmsTol % RMS is good enough already
                    % nothing to do...

                elseif obj.rms_b <= obj.rmsThreshold * obj.startingRMS
                    % nothing to do...

                elseif obj.rms_b > obj.targetRMS + obj.rmsTol
                    % Find the minimum using Brent's method
                    rms2 = obj.rms_b;
                    obj.findMinimum(mu1, mu2, mu3, rms2);
                end

                % Analyze minimum or whatever rms is returned if shortcuts were taken
                if ((obj.rms_b >= obj.startingRMS) && ~obj.lTargetHit) || (obj.lTargetHit && (obj.rms_b > obj.targetRMS + obj.rmsTol))
                    % Minimum isn't as good as starting model, cut step size
                    obj.stepCut = obj.stepCut / 2.0;
                    obj.numStepCuts = obj.numStepCuts + 1;
                    lastMu = obj.lmu_b; % save last minimum mu for next call to bracketMinimum
                    obj.initializeTrackers();

                    % Have we cut the step size too many times to no avail?
                    if obj.numStepCuts > obj.maxNumStepCuts
                        obj.convergenceFlag = 4;
                        lGetMinimum = false;
                    else
                        fprintf(' Cutting the size of the model update due to divergence, new fractional size: %16.4g\n', obj.stepCut);
                    end

                elseif obj.rms_b <= obj.targetRMS + obj.rmsTol
                    lGetMinimum = false;

                elseif obj.rms_b <= obj.rmsThreshold * obj.startingRMS
                    fprintf(' Large misfit decrease detected, ending minimization search. Decrease in misfit: %5.2f%%\n', ...
                        (obj.startingRMS - obj.rms_b) ./ obj.startingRMS * 100);
                    lGetMinimum = false;
                else % no special case, the minimum rms is less than or equal to the starting model so exit the while loop
                    lGetMinimum = false;
                end
            end % while lGetMinimum
        end

        function getIntercept(obj)
            % Given input lmu_b with rms_b < targetRMS, this routine finds a larger mu above the intercept
            % and then calls the NR root finding routine to find the intercept to within a specified tolerance

            % Check to see if pm_c or pm_b are close enough to the targetRMS
            if abs(obj.rms_c - obj.targetRMS) < obj.rmsTol % _c is from larger mu, so give it precedence
                obj.rms_b = obj.rms_c;
                obj.lmu_b = obj.lmu_c;
                obj.pm_b = obj.pm_c;
                obj.dm_b = obj.dm_c;
                obj.rough_b = obj.rough_c;
                if numel(obj.pm_assoc) > 0
                    obj.pm_assoc_b = obj.pm_assoc_c;
                end
                return;
            elseif abs(obj.rms_b - obj.targetRMS) < obj.rmsTol
                return;
            end

            % Allocate arrays for carrying around models associated with a (b uses _b arrays from module)
            obj.pm_a = zeros(size(obj.pm_b));
            obj.dm_a = zeros(size(obj.dm_b));
            if numel(obj.pm_assoc) > 0
                obj.pm_assoc_a = zeros(size(obj.pm_assoc_b));
            end

            fprintf(' ... Finding intercept:\n');
            fprintf('%16s %16s %16s %16s %16s %16s %16s %16s\n', 'Misfit', 'Roughness', ...
                'Log10(mu)', 'Min log10(rho)', 'Max log10(rho)', 'Mean Anisotropy', 'Fwd Call (s)', 'Matrix Ops. (s)');

            obj.lTrackerOn = false;
            lSkipIntercept = false;

            % If rms_c has been set, use it to bound intercept, otherwise find a mu with an rms above the intercept
            if (obj.rms_c < obj.tracker_Dummy) && (obj.rms_c > obj.targetRMS)
                obj.rms_a = obj.rms_b;
                obj.lmu_a = obj.lmu_b;
                obj.pm_a = obj.pm_b;
                obj.dm_a = obj.dm_b;
                obj.rough_a = obj.rough_b;
                if numel(obj.pm_assoc) > 0
                    obj.pm_assoc_a = obj.pm_assoc_b;
                end

                obj.rms_b = obj.rms_c;
                obj.lmu_b = obj.lmu_c;
                obj.pm_b = obj.pm_c;
                obj.dm_b = obj.dm_c;
                obj.rough_b = obj.rough_c;
                if numel(obj.pm_assoc) > 0
                    obj.pm_assoc_b = obj.pm_assoc_c;
                end
            else % find lmu_c with rms_c above the intercept
                fac = 0;
                lmu_b_start = obj.lmu_b;

                while obj.rms_b < obj.targetRMS % Keep on increasing mu until we cross the targetRMS
                    % Update arrays keep track of models near intercept (_b goes to _a before updating _b)
                    obj.rms_a = obj.rms_b;
                    obj.lmu_a = obj.lmu_b;
                    obj.pm_a = obj.pm_b;
                    obj.dm_a = obj.dm_b;
                    obj.rough_a = obj.rough_b;
                    if numel(obj.pm_assoc) > 0
                        obj.pm_assoc_a = obj.pm_assoc_b;
                    end
                    % Test new lmu_b:
                    fac = fac + 1.0;
                    obj.lmu_b = obj.lmu_b + fac * 0.699; % Move up by 1/2 decade, then full decade...

                    if obj.lmu_b < 12.0
                        obj.rms_b = obj.misfitOfMu(obj.lmu_b);

                        % Save results in _b arrays:
                        obj.updateB(obj.lmu_b, obj.rms_b);

                    else
                        lSkipIntercept = true;
                        fprintf(' Intercept not found by increasing mu, giving up...\n');
                        obj.lmu_b = lmu_b_start; % reset mu to reasonable value

                        break; % exit while loop since we've gone far enough
                    end

                end
            end

            % lmu_a and lmu_b bound the intercept, now find the intercept more precisely
            if ~lSkipIntercept
                interceptMu = obj.findIntercept();

                % Returning from findIntercept(), the intercept model is in pm_b
                fprintf(' Intercept is at mu: %16.4g\n', interceptMu);
            end
        end
        
        function x = transformToUnbound(obj, mmin, bin, ain, lbind, lanis)
            % Converts the bound parameter b < m < a to the unbound parameter x
            % lanis: true if transform anisotropy, false if not transform
            % anisotropy
            % Convert input to double variables
            a = ain;
            b = bin;
            m = mmin;

            pmtol = (a - b) / 232.0;

            if ~lanis
                for i=1:obj.nParams-obj.nParamsAnis
                    if m(i)<b || m(i)>a
                        warning(['The resistivity parameter which has been designated a free parameter, ' ...
                        'exceeds either the global or local bounds on resistivity!'])
                    end
                    if m(i)<=b
                        m(i) = b+pmtol;
                    elseif m(i)>=a
                        m(i) = a-pmtol;
                    end
                end
            elseif obj.nParamsAnis
                for i=obj.nParams-obj.nParamsAnis+1:obj.nParams
                    if m(i)<b || m(i)>a
                        warning(['The anisotropy parameter which has been designated a free parameter, ' ...
                        'exceeds either the global or local bounds on anisotropy ratio!'])
                    end
                    if m(i)<=b
                        m(i) = b+pmtol;
                    elseif m(i)>=a
                        m(i) = a-pmtol;
                    end
                end
            end

            if lbind
                switch obj.cBoundsTransform
                    case 'linear'
                        m = 0.5*(log(m - b) - log(a - m));

                    case 'exponential'
                        m = log(m - b) - log(a - m)+(a+b)/2;

                    case 'bandpass'
                        c = obj.bandPassFactor ./ (a - b);
                        if m <= 0
                            q = log((exp(c .* (m - b)) - 1) ./ (1 - exp(c .* (m - a))));
                        else
                            q = log((exp(-c .* b) - exp(-c .* m)) ./ (exp(-c .* m) - exp(-c .* a)));
                        end
                        m = q ./ c + b;
                end
            end
            if any(isinf(m), 'all')
                error('Error: model has value exccedding or equatl to the bounds!')
            end
            
            if ~lanis
                % keep the anisotropic ratios unchanged
                m(obj.nParams-obj.nParamsAnis+1:obj.nParams) = mmin(obj.nParams-obj.nParamsAnis+1:obj.nParams);
                x = m;
            elseif obj.nParamsAnis
                % keep the resistivity unchanged
                m(1:obj.nParams-obj.nParamsAnis) = mmin(1:obj.nParams-obj.nParamsAnis);
                x = m;
            end
        end

        function transformWJ(obj)
            % Converts WJ from bound parameter m to unbound parameter x: dF/dx = dm/dx *dF/dm
            % for j = 1:obj.nParams-obj.nParamsAnis % last parameters are global anisotropy, DO NOT apply dm/dx
            for j = 1:obj.nParams
                if obj.lBoundMe(j)
                    a = obj.upperBound(j);
                    b = obj.lowerBound(j);

                    pdp = obj.pm(j);

                    switch obj.cBoundsTransform
                        case 'linear'
                            dmdx = 2*(a-b)./(exp(pdp)+exp(-pdp)).^2;
                            obj.wj(:, j) = obj.wj(:, j) * dmdx;
                        case 'exponential'
                            p = exp(pdp-(a+b)/2);
                            dmdx = (a - b) * p ./ (1 + p).^2;
                            obj.wj(:, j) = obj.wj(:, j) * dmdx;

                        case 'bandpass'
                            c = obj.bandPassFactor / (a - b);

                            if pdp <= 0
                                p = exp(c * (pdp - b));
                                q = exp(c * (pdp - a));
                                dmdx = p * (1 - exp(-obj.bandPassFactor)) / ((1 + p) * (1 + q));
                            elseif pdp > 0
                                p = exp(-c * (pdp - b));
                                q = exp(-c * (pdp - a));
                                dmdx = q * (1 - exp(-obj.bandPassFactor)) / ((1 + p) * (1 + q));
                            end
                            obj.wj(:, j) = obj.wj(:, j) * dmdx;
                    end
                end
            end
        end

        function cx = bracketMinimum(obj, ax, bx)
            % BracketMinimum - Modified from NR's mnbrak() routine
            % Given a function handle func, and given distinct initial points ax and bx,
            % this function searches in the downhill direction (defined by the function as
            % evaluated at the initial points) and returns new points ax, bx, cx that
            % bracket a minimum of the function. Also returned are the function values
            % at the three points, fa, fb, and fc.
            % Parameters: GOLD is the default ratio by which successive intervals are
            % magnified; GLIMIT is the maximum magnification allowed for a
            % parabolic-fit step.

            fprintf(' ... Bracketing minimum:\n')
            fprintf('%16s %16s %16s %16s %16s %16s %16s %16s\n', 'Misfit', 'Roughness', ...
                'Log10(mu)', 'Min log10(rho)', 'Max log10(rho)', 'Mean Anisotropy', 'Fwd Call (s)', 'Matrix Ops. (s)');
            gold = 1.618034;
            glimit = 100.0;
            tiny = 1.0e-20;
            cx = 0;

            fb = obj.misfitOfMu(bx);
            obj.updateB(bx,fb)
            if (obj.lEscapeFromMinimization(fb))
                return
            end

            fa = obj.misfitOfMu(ax);

            if fa < fb  % Switch roles of a and b so that we can go downhill in the direction from a to b.
                obj.updateB(ax,fa)
                [ax, bx] = deal(bx, ax);
                [fa, fb] = deal(fb, fa);
                if (obj.lEscapeFromMinimization(fb))
                    return
                end
            end

            cx = bx + gold * (bx - ax);     % First guess for c.
            fc = obj.misfitOfMu(cx);

            while true          % Do-while-loop: Keep returning here until we bracket.
                if fb < fc      % ie fa > fb < fc, so we've bracketed a minimum
                    break
                end

                obj.updateB(cx,fc) % cx/fc has lower misfit, so save it
                if (obj.lEscapeFromMinimization(fc))
                    bx = cx;
                    fb = fc;
                    break
                end
                % Compute u by parabolic extrapolation from a, b, c. TINY is used to prevent any possible division by zero.
                r = (bx - ax) * (fb - fc);
                q = (bx - cx) * (fb - fa);
                u = bx - ((bx - cx) * q - (bx - ax) * r) ./ (2 * max(abs(q - r), tiny)*obj.fortran_sign(q - r));
                ulim = bx + glimit * (cx - bx);
                %  We won’t go farther than this. Test various possibilities:
                if (bx - u) * (u - cx) > 0          % Parabolic u is between b and c: try it.
                    fu = obj.misfitOfMu(u);
                    if fu < fc                      % Got a minimum between b and c.
                        ax = bx;
                        fa = fb;
                        bx = u;
                        fb = fu;
                        obj.updateB(bx,fb)  % u/fu model saved
                        break
                    elseif fu > fb                  % Got a minimum between a and u.
                        cx = u;
                        fc = fu;
                        break
                    end
                    u = cx + gold * (cx - bx);      % Parabolic fit was no use. Use default magnification.
                    fu = obj.misfitOfMu(u);
                elseif (cx - u) * (u - ulim) > 0    % Parabolic fit is between c and its allowed limit.
                    fu = obj.misfitOfMu(u);
                    if fu < fc
                        obj.updateB(u,fu)   % u/fu model saved
                        bx = cx;
                        cx = u;
                        u = cx + gold * (cx - bx);
                        [fb, fc, fu] = obj.shft(fc, fu, obj.misfitOfMu(u));
                    end
                elseif (u - ulim) * (ulim - cx) >= 0        % Limit parabolic u to maximum allowed value.
                    u = ulim;
                    fu = obj.misfitOfMu(u);
                else                    % Reject parabolic u, use default magnification.
                    u = cx + gold * (cx - bx);
                    fu = obj.misfitOfMu(u);
                end
                [ax, bx, cx] = obj.shft(bx, cx, u);    % Eliminate oldest point and continue.
                [fa, fb, fc] = obj.shft(fb, fc, fu);
            end
        end

        function findMinimum(obj, ax, bx, cx, fbx)
            % Modified from NR's brent() routine
            fprintf(' ... Finding minimum:\n')
            fprintf('%16s %16s %16s %16s %16s %16s %16s %16s\n', 'Misfit', 'Roughness', ...
                'Log10(mu)', 'Min log10(rho)', 'Max log10(rho)', 'Mean Anisotropy', 'Fwd Call (s)', 'Matrix Ops. (s)');
            
            % Tolerance
            tol = 0.1;
            itmax = 100;
            cgold = 0.3819660;
            zeps = 1.0e-3 * eps(ax);

            a = min(ax, cx);
            b = max(ax, cx);
            v = bx;
            w = v;
            x = v;
            e = 0.0;
            fx = fbx;
            fv = fx;
            fw = fx;

            for iter = 1:itmax
                xm = 0.5 * (a + b);
                tol1 = tol * abs(x) + zeps;
                tol2 = 2.0 * tol1;

                if abs(x - xm) <= (tol2 - 0.5 * (b - a))
                    return;
                end

                if abs(e) > tol1
                    r = (x - w) * (fx - fv);
                    q = (x - v) * (fx - fw);
                    p = (x - v) * q - (x - w) * r;
                    q = 2.0 * (q - r);

                    if q > 0.0
                        p = -p;
                    end

                    q = abs(q);
                    etemp = e;
                    e = dd;

                    if abs(p) >= abs(0.5 * q * etemp) || ...
                            p <= q * (a - x) || ...
                            p >= q * (b - x)
                        if x >= xm
                            e = a - x;
                        else
                            e = b - x;
                        end
                        dd = cgold * e;
                    else
                        dd = p / q;
                        u = x + dd;
                        if u - a < tol2 || b - u < tol2
                            dd = abs(tol1)*obj.fortran_sign(xm - x);
                        end
                    end
                else
                    if x >= xm
                        e = a - x;
                    else
                        e = b - x;
                    end
                    dd = cgold * e;
                end

                if abs(dd) >= tol1
                    u = x + dd;
                else
                    u = x + abs(tol1)*obj.fortran_sign(dd);
                end
                fu = obj.misfitOfMu(u);

                if fu <= fx
                    obj.updateB(u, fu)
                    if obj.lEscapeFromMinimization(fu)
                        return      % return if current misfit is good enough
                    end

                    if u >= x
                        a = x;
                    else
                        b = x;
                    end
                    [v, w, x] = obj.shft(w, x, u);
                    [fv, fw, fx] = obj.shft(fw, fx, fu);
                else
                    if u < x
                        a = u;
                    else
                        b = u;
                    end
                    if fu <= fw || w == x
                        v = w;
                        fv = fw;
                        w = u;
                        fw = fu;
                    elseif fu <= fv || v == x || v == w
                        v = u;
                        fv = fu;
                    end
                end
                % David Myer's edit that ends the minimization based on misfit flattening, rather than by mu changing below the tolerance:
                if  (abs(fw-fx) < obj.rmsTol ) 
                    return
                end
                
            end %end for
        end

        function xintercept = findIntercept(obj)
            % Modified from NR's zbrent() routine

            itmax = 100;
            tol = 0.001;
            tolRMS = 0.005; % RMS tolerance for intercept, stop if at least this close to target
            eps0 = eps(tolRMS);

            a = obj.lmu_a; % x1
            b = obj.lmu_b; % x2
            fa = obj.rms_a - obj.targetRMS; % func(a)
            fb = obj.rms_b - obj.targetRMS; % func(b)

            if ((fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0))
                fprintf('Error in function findIntercept: root must be bracketed on entry!\n');
            end

            c = b;
            fc = fb;

            % b was moved to c:
            % Update arrays keep track of models near intercept:
            obj.rms_c   = obj.rms_b;
            obj.lmu_c   = obj.lmu_b;
            obj.pm_c    = obj.pm_b;
            obj.dm_c    = obj.dm_b;
            obj.rough_c = obj.rough_b;
            if (obj.npm_assoc > 0)
                obj.pm_assoc_c = obj.pm_assoc_a;
            end

            for iter = 1:itmax

                if ((fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0))
                    c = a;
                    fc = fa;
                    dd = b - a;
                    e = dd;

                    % a was moved to c:
                    % Update arrays keep track of models near intercept:
                    obj.rms_c   = obj.rms_a;
                    obj.lmu_c   = obj.lmu_a;
                    obj.pm_c    = obj.pm_a;
                    obj.dm_c    = obj.dm_a;
                    obj.rough_c = obj.rough_a;
                    if (obj.npm_assoc > 0)
                        obj.pm_assoc_c = obj.pm_assoc_a;
                    end

                end

                if (abs(fc) < abs(fb))
                    a = b;
                    b = c;
                    c = a;
                    fa = fb;
                    fb = fc;
                    fc = fa;

                    % b moved to a, c moved to b, a moved to c (so a=c...)
                    % Update arrays keep track of models near intercept:
                    obj.rms_a   = obj.rms_b;
                    obj.lmu_a   = obj.lmu_b;
                    obj.pm_a    = obj.pm_b;
                    obj.dm_a    = obj.dm_b;
                    obj.rough_a = obj.rough_b;
                    if (obj.npm_assoc > 0)
                        obj.pm_assoc_a = obj.pm_assoc_b;
                    end

                    obj.rms_b   = obj.rms_c;
                    obj.lmu_b   = obj.lmu_c;
                    obj.pm_b    = obj.pm_c;
                    obj.dm_b    = obj.dm_c;
                    obj.rough_b = obj.rough_c;
                    if (obj.npm_assoc > 0)
                        obj.pm_assoc_b = obj.pm_assoc_c;
                    end

                    obj.rms_c   = obj.rms_a;
                    obj.lmu_c   = obj.lmu_a;
                    obj.pm_c    = obj.pm_a;
                    obj.dm_c    = obj.dm_a;
                    obj.rough_c = obj.rough_a;
                    if (obj.npm_assoc > 0)
                        obj.pm_assoc_c = obj.pm_assoc_a;
                    end

                end

                tol1 = 2.0 * eps0 * abs(b) + 0.5 * tol;
                xm = 0.5 * (c - b);
                if (abs(xm) <= tol1 || abs(fb) <= tolRMS) % fb == 0.0
                    xintercept = b;
                    return;
                end

                if (abs(e) >= tol1 && abs(fa) > abs(fb))
                    s = fb / fa;
                    if (a == c)
                        p = 2.0 * xm * s;
                        q = 1.0 - s;
                    else
                        q = fa / fc;
                        r = fb / fc;
                        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
                        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
                    end

                    if (p > 0.0)
                        q = -q;
                    end

                    p = abs(p);
                    if (2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)))
                        e = dd;
                        dd = p / q;
                    else
                        dd = xm;
                        e = dd;
                    end
                else
                    dd = xm;
                    e = dd;
                end

                a = b;
                fa = fb;

                % b moved to a
                % Update arrays keep track of models near intercept:
                obj.rms_a   = obj.rms_b;
                obj.lmu_a   = obj.lmu_b;
                obj.pm_a    = obj.pm_b;
                obj.dm_a    = obj.dm_b;
                obj.rough_a = obj.rough_b;
                if (obj.npm_assoc > 0)
                    obj.pm_assoc_a = obj.pm_assoc_b;
                end

                if abs(dd)>tol1
                    b = b + dd;
                else
                    b = b + abs(tol1)*obj.fortran_sign(xm);
                end
                rb = obj.misfitOfMu(b); % func call
                obj.updateB(b, rb);
                fb = rb - obj.targetRMS;

            end

            fprintf('Error in function findIntercept: exceeded maximum iterations trying to find intercept\n');
            xintercept = b;

        end

        function save_lhs(obj)
            nParams = obj.nParams;
            amat = obj.amat;
            filename = sprintf('occma_lhs.%03d.mat',obj.currentIteration);
            save(filename, 'nParams', 'amat', '-mat', '-v7.3');
        end

        function save_rhs(obj)
            nParams = obj.nParams;
            pm_test = obj.pm_test;
            filename = sprintf('occma_rhs.%03d.mat',obj.currentIteration);
            save(filename, 'nParams', 'pm_test', '-mat', '-v7.3');
        end

        function save_stepsize(obj)
            % saves Occam iteration's stepsize value to a text file

            % Create the filename
            filename = sprintf('occam_stepsize.%03d.txt', obj.currentIteration);

            % Open the file
            fid = fopen(filename, 'w');

            % Check for error opening the file
            if fid == -1
                error('Error opening Occam stepsize file, stopping!');
            end

            % Write stepCut to the file
            fprintf(fid, '%f\n', obj.stepCut);

            % Close the file
            fclose(fid);
        end

        function writeJacobian(obj, nCurrentIter)
            wj = obj.wj;
            filename = sprintf('jacobianBin.%03d.mat',nCurrentIter-1);
            save(filename, 'wj', '-mat', '-v7.3');
        end

        function printOccamLog(obj, cStr)
            % Write the message to the terminal if the print level is greater than 0
            if obj.occamPrintLevel > 0
                fprintf('%s\n', strtrim(cStr));
            end
        end

        function updateB(obj, mu, rms)
            %Helper routine for updating the *_b arrays that keep track of the optimal model and its response.
            obj.pm_b = obj.pm_test;
            obj.dm_b = obj.dm;
            obj.rms_b = rms;
            obj.lmu_b = mu;
            obj.rough_b = obj.rough_test;

            if obj.npm_assoc > 0 % Check if pm_assoc is provided
                obj.pm_assoc_b = obj.pm_assoc;
            end
        end

    end

    methods (Static)
        function [a, b, c] = shft(b, c, dd)
            a = b;
            b = c;
            c = dd;
        end

        function [a, b] = swap(a,b)
            dum = a;
            a = b;
            b= dum;
        end

        function val = fortran_sign(a)
            val = sign(a);
            if val==0
                val=1;
            end
        end

        function c = multiplyAx(A, x, beta, trans)
            % Multiply Ax or A^t*x

            % Input arguments
            % A: Input matrix
            % x: Input vector
            % beta: Scalar value
            % trans: 'N' for A*x, 'T' or 'C' for A^t*x

            % Output arguments
            % c: Resultant vector

            if isa(A, 'double')
                if trans == 'T' || trans == 'C'
                    c = beta*x + alpha * (A' * x);
                else
                    c = beta*x + alpha * (A * x);
                end
            elseif isa(A, 'single')
                if trans == 'T' || trans == 'C'
                    c = single(beta*x + alpha * (A' * x));
                else
                    c = single(beta*x + alpha * (A * x));
                end
            end
        end

        function C = multiplyATA(A)
            % Multiply A^t * A

            % Input arguments
            % A: Input matrix

            % Output arguments
            % C: Resultant matrix

            if isa(A, 'double')
                C = A' * A;
            elseif isa(A, 'single')
                C = single(A' * A);
            end
        end

        function x = solveLinearSystem(A,b)
            x = A\b;
        end

        function  [J, flag_stop] =NumJ(fun,x,d,f,n,nc)
            J=zeros(nc,n);
            xx=x;
            for ii=1:n
                if x(ii)==0
                    xp=d^2;
                else
                    xp=x(ii)+d*abs(x(ii));
                end
                xx(ii)=xp;
                fp=feval(fun,xx);
                J(:,ii)=(fp-f)./(xp-x(ii));
                xx(ii)=x(ii);
            end
            if ~isreal(J)||any(isnan(J(:)))||any(isinf(J(:)))
                flag_stop=-6;
            else
                flag_stop=0;
            end
        end       

    end


end



