function [outval,infoOut] = inversion1D_para(Para_init, data, measurement, weight, model, runConfig, tool, TVD, Zbed)

    switch model.solver
    case 'occam'
        domainSet.lBoundsTransform = true;
        domainSet.cBoundsTransform = runConfig.cBoundsTransform;
        domainSet.bandPassFactor = runConfig.bandPassFactor;
        domainSet.lBoundMe = true;
        if runConfig.cBoundsTransform~='linear'
            domainSet.lowerBoundGlobal = log10(runConfig.lowerBoundGlobal);
            domainSet.upperBoundGlobal = log10(runConfig.upperBoundGlobal);
            domainSet.lowerBoundAnisGlobal = log10(runConfig.lowerBoundAnisGlobal);
            domainSet.upperBoundAnisGlobal = log10(runConfig.upperBoundAnisGlobal);
        else
            domainSet.lowerBoundGlobal = runConfig.lowerBoundGlobal;
            domainSet.upperBoundGlobal = runConfig.upperBoundGlobal;
            domainSet.lowerBoundAnisGlobal = runConfig.lowerBoundAnisGlobal;
            domainSet.upperBoundAnisGlobal = runConfig.upperBoundAnisGlobal;
        end
        domainSet.targetRMS = runConfig.targetRMS;
        domainSet.rmsTol = runConfig.rmsTol;
        domainSet.modelLog10Mu = runConfig.modelLog10Mu;
        domainSet.fdstep = runConfig.fdstep;
        domainSet.nParamsAnis = runConfig.nParamsAnis;
        model.TVD =TVD;
        model.Zbed = Zbed;
        data.measurement = measurement;
        data.weight = weight;
        invObj = Occam(data, Para_init, 0, 0, 0, domainSet, tool, model); %Para_init is the model in log10
        invObj.maxOccamIterations = runConfig.maxItr;

        % add penalty for roughness term
        invObj.addPenalty_fix(runConfig.weightv)
        invObj.openOccamLog()
        lrunMARE1DEM = true;

        while lrunMARE1DEM
            invObj.currentIteration = invObj.currentIteration + 1;

            % Compute an Occam iteration:

            switch runConfig.inversion_method
                case 'occam' % occam:
                    invObj.computeOccamIteration(runConfig.SaveJacobian,runConfig.SaveSensitivity);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %                 [outval,info] = alg_occam_new(Para_init, data, model, domainSet, tool, weight, options);
                    %                 inv_Rh = outval(1:end-1);
                    %                 anis_Ratio = outval(end);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                case 'staticmu'  % static mu gauss-newton:
                    invObj.computeStaticMuIteration(runConfig.SaveJacobian,runConfig.SaveSensitivity)
            end

            if invObj.convergenceFlag~=4 && invObj.convergenceFlag~=5
                % Convert pm from unbound to bound space and insert into nRhoParams:
                rho = invObj.transformToBound(invObj.pm,invObj.lowerBound,invObj.upperBound,invObj.lBoundMe);
                if (invObj.lSaveOccamSystem)
                    invObj.write_paramlist(invObj.currentIteration)
                end
            end

            if invObj.convergenceFlag>1
                break
            end
        end

        invObj.closeOccamLog()
    end

    outval = invObj.transformToBound(invObj.pm,invObj.lowerBound,invObj.upperBound,invObj.lBoundMe);
    infoOut = [invObj.currentIteration, invObj.modelLog10Mu, invObj.rms_b, invObj.rough_b];

    fprintf('Deallocating memory...\n');
    invObj.deallocateOccam();

    % output a result in linear space
    % nL = length(outval);
    % rx = outval(1:nL);
    
    % m1= (model.r_ub*exp(rx)+model.r_lb*exp(-rx))./(exp(rx)+exp(-rx));
    % outval = m1;

end