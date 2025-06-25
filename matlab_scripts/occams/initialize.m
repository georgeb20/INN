function [X0, mesh] = initialize(TVD, config, model)
%INITIALIZE key functionality generates an initial model

% fetch out the configuration
pixel = config.resolution;
meshType = config.meshType;
initType = config.initType;
% Set up the vertical range (tool is in the symmetrical center)
ZRange = TVD + config.scope;
% Select a mesh type, this is very important, think about further methods
switch meshType
    case 'equal'    % this type equally pixelize the ZRange space
        mesh = ZRange(1):pixel:ZRange(2);
        mesh = mesh';
    case 'unequal'
        hinc = cumsum(pixel:30);
        hequ = -30:pixel:30;
        mesh = sort([hequ(2:end-1), hequ(2)-hinc, hequ(end-1)+hinc])+TVD;
        ind = mesh>ZRange(2);
        mesh(ind) = [];
        ind = mesh<ZRange(1);
        mesh(ind) = [];
        mesh = mesh';
end
% Select the initial model type, think about other way except for homo
nL = length(mesh)+1;
switch initType
    case 'homo'     % initialize a homogeneous model
        Rh_init = config.homo_Rh;
        X_temp = ones(nL,1)*Rh_init;
        if model.nParamsAnis>0
            aniso_init = ones(model.nParamsAnis,1);
            X0 = [X_temp;aniso_init];
        else
            X0 = X_temp;
        end
        % X_rh = 0.5*(log(X_temp(1:nL)-model.r_lb)-log(model.r_ub-X_temp(1:nL)));
        % anis = 0.5*(log(aniso_init-model.anis_lb)-log(model.anis_ub-aniso_init));
        % X0 = [X_rh;anis];
        
end
end