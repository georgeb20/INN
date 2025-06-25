 function [domainSet, rhoh, anis_ratio] = Init_gridinv_INVanis(tool, skdRatio, prior, type)
    % Generate mesh for inversion from uniform background grid
    % Constant
    ft = 0.3048;
    MU_0 = pi*4.0e-07;
    freq = tool.freq;

    gSigma = [1/100,1/0.5]; % suitable range for formation: [1, 100];
    skinDepth_min = sqrt(2/2/pi/max(freq)/MU_0/gSigma(2));
    skinDepth_max = sqrt(2/2/pi/min(freq)/MU_0/gSigma(1));

    % forward computational domain
    dimX_fwd = 5*skinDepth_max;
    dimY_fwd = 5*skinDepth_max;
    scope(1,1:2) = [dimX_fwd,dimY_fwd]; 

    tvd_min = min(tool.TVD);
    md_min = min(tool.MD);
    tvd_max = max(tool.TVD);
    md_max = max(tool.MD);

    % dimX(1) = max(-dimX_fwd/2,(md_min - 15*ft)); dimX(2) = min(dimX_fwd/2,(md_max + 15*ft));
    % dimY(1) = max(-dimY_fwd/2,(tvd_min - 60*ft)); dimY(2) = min(dimY_fwd/2,(tvd_max + 50*ft));
    
    
%     dimX(1) = max(md_min-dimX_fwd/2,(md_min - 20*ft)); dimX(2) = min(md_max+dimX_fwd/2,(md_max + 20*ft));
%     dimY(1) = max(tvd_min-dimY_fwd/2,(tvd_min - 80*ft)); dimY(2) = min(tvd_max+dimY_fwd/2,(tvd_max + 20*ft));
    
    %% Layer 2
    dimX(1) = max(md_min-dimX_fwd/2,(md_min - 20*ft)); dimX(2) = min(md_max+dimX_fwd/2,(md_max + 20*ft));
    dimY(1) = max(tvd_min-dimY_fwd/2,(tvd_min - 50*ft)); dimY(2) = min(tvd_max+dimY_fwd/2,(tvd_max + 30*ft));
    
    
    scope(2,1:2) = [dimX(1), dimX(2)];
    scope(3,1:2) = [dimY(1), dimY(2)];

    % Generate uniform background grid
    iniDeltaX = skinDepth_min/skdRatio; % minimal grid size along X direction
    iniDeltaY = skinDepth_min/skdRatio; % flexible - minimal grid size along Y direction
    xvec = dimX(1):iniDeltaX:dimX(2);
    if isempty(find(xvec == dimX(1),1))
        xvec = [xvec, dimX(1)];
    elseif isempty(find(xvec == dimX(2),1))
        xvec = [xvec, dimX(2)];
    end
    yvec = dimY(1):iniDeltaY:dimY(2);
    if isempty(find(yvec == dimY(1),1))
        yvec = [yvec, dimY(1)];
    elseif isempty(find(yvec == dimY(2),1))
        yvec = [yvec, dimY(2)];
    end

    % Extract the grid for inversion
    % Rh_inv: x-horizontal; z-vertical
    [Rh_inv,anis_ratio,xinv,yinv] = grid_inv(tool,xvec, yvec, type,prior);
%     figure;surf(xinv(1:end-1),yinv(1:end-1),Rh_inv);view(2);
    mesh_inv.xinv = xinv;
    mesh_inv.yinv = yinv;
    rhoh = Rh_inv(:);
    domainSet.mesh_inv = mesh_inv;
    domainSet.scope = scope;
end

function [Rh_inv,anis_ratio,xinv,yinv] = grid_inv(tool,xtemp, ytemp, type,prior)

    xinv = xtemp;
    yinv = ytemp;

    % assign material to the inversion grid
    if prior == 0
        [Rh_inv,anis_ratio] = model_2D_generator_new(xinv,yinv,type,2);
        Rh_inv = Rh_inv'; %gGridY*gGridX
    elseif prior == 1
        % based on UH1D initial
        feet = 0.3048;
        UH1D = load('UH1D_initial.mat');
        [xdir,ydir] = meshgrid(xinv(1:end-1), yinv(1:end-1));
        md = UH1D.LD(1:end-1);
        tvd = UH1D.gridY(2:end)*feet;
        [xv,yv] = meshgrid(md.',UH1D.gridY);
        
        % Rh
        Rh_inv = interp2(xv, -yv*feet, UH1D.R_UH1D, xdir, ydir,'spline');
        Rh_inv(Rh_inv<0) = abs(Rh_inv(Rh_inv<0));
        ind1 = find(xinv<=md(1),1,'last');
        n = length(find(xinv<md(1)));
        Rh_inv(:,xinv<=md(1)) = repmat(Rh_inv(:,ind1+1),1,n);
        ind2 = find(xinv>=md(end),1,'first');
        n = length(find(xinv(1:end-1)>md(end)));
        Rh_inv(:,xinv(1:end-1)>=md(end)) = repmat(Rh_inv(:,ind2-1),1,n);

        ind3 = find(yinv<=tvd(1),1,'last');
        n = length(find(yinv<tvd(1)));
        Rh_inv(yinv<=tvd(1),:) = repmat(Rh_inv(ind3+1,:),n,1);
        ind4 = find(yinv>=tvd(end),1,'first');
        n = length(find(yinv(1:end-1)>tvd(end)));
        Rh_inv(yinv(1:end-1)>=tvd(end),:) = repmat(Rh_inv(ind4-1,:),n,1);
        Rh_inv(Rh_inv<tool.lb) = tool.lb+0.5;
        Rh_inv(Rh_inv>tool.ub) = tool.ub-0.5;
        
        % Rv
        anis_ratio = 2;
%         anis_ratio = mean(UH1D.anis_UH1D);
    end
end