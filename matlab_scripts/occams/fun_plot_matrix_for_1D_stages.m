function fun_plot_matrix_for_1D_stages(configfile,taperWidth)
%%-------------------- Read in config file -------------------------------%
runConfig = yaml.loadFile(configfile); % load configuration from YAML file
% disp(yaml.dump(runConfig, "block")); % show yaml file
%% The script generates entire curtain (map)

load(runConfig.matfile);

config.scope = str2num(runConfig.tool_investigation_scope{1});          % the tool investigation range
config.resolution = runConfig.resolution_of_pixel;              % the resolution of pixel
config.meshType = runConfig.mesh_type;        % initial mesh type: equal or unequal

LD = tool.MD(1:runConfig.InvGroupPts:end);
LD(end+1) = LD(end);
TVD = TVD(1:runConfig.InvGroupPts:end);
TVD(end+1) = TVD(end);

numStages = size(result, 1); % Assuming result is 3x34
R_UH1D_AllStages = []; % Initialize for all stages

pixel = config.resolution;
for stage = 1:numStages
    inv = cell2mat(result(stage, :)); % Process each stage
    [rw, clm] = size(inv);
    np = rw - 1;
    R_UH1D = []; % Initialize for current stage

    % Your existing processing loop
    for j = 1:length(TVD)-1
        ZRange = TVD(j) + config.scope;
        switch config.meshType
            case 'equal'
                mesh = ZRange(1):pixel:ZRange(2);
                mesh = mesh';
            case 'unequal'
                hinc = cumsum(pixel:30);
                hequ = -30:pixel:30;
                mesh = sort([hequ(2:end-1), hequ(2)-hinc, hequ(end-1)+hinc])+TVD(j);
                ind = find(mesh>ZRange(2));
                mesh(ind) = [];
                ind = find(mesh<ZRange(1));
                mesh(ind) = [];
                mesh = mesh';
        end
        X = repmat([LD(j), LD(j+1)],np,1);
        Y = repmat([ZRange(1)-0.01;mesh],1,2);
        Zr = repmat(log(inv(1:np,j)),1,2);
%         surf(X,Y*0.3048,Zr,'EdgeColor','none');
%         view([-180,-90]);
%         hold on
        
        gridY = TVD(1)+config.scope(1):TVD(end)+config.scope(2);
        R_gridy = interp1(Y(:,1),inv(1:np,j),gridY).';
        R_gridy(gridY<Y(1,1)) = median(inv(1:4,j));
        R_gridy(gridY>Y(end,1)) = median(inv(np,j));
        R_UH1D = [R_UH1D, R_gridy];
    end

    R_UH1D_AllStages = cat(3, R_UH1D_AllStages, R_UH1D); % Accumulate or overlap data
end

%% plot inverted Rh and misfit
% R_UH1D(:,4) = (R_UH1D(:,3) + R_UH1D(:,5))/2;
fontsize = 18;

figure;
% subplot(2,2,1);
subplot('Position', [0.05, 0.4, 0.4, 0.5]) % [left bottom width height]
stageWght = ones(size(R_UH1D_AllStages)); 
% taperWidth = 30; % Width of the taper zone
for idx = 1:size(stageWght,2)
    tvd_value = TVD(idx);

    % Define the taper zones
    upperTaperZone = tvd_value + 75 - taperWidth : tvd_value + 75;
    lowerTaperZone = tvd_value - 75 : tvd_value - 75 + taperWidth;

    % First set weights for non-taper zones
    stageWght(gridY >= tvd_value + 75 | gridY <= tvd_value - 75, idx, 1) = 0.2;
    stageWght(gridY < tvd_value + 75 | gridY > tvd_value - 75, idx, 2:3) = 0.0;
    stageWght(gridY >= tvd_value + 75 | gridY <= tvd_value - 75, idx, 2:3) = 0.4;
    stageWght(gridY >= tvd_value + 150 | gridY <= tvd_value - 150, idx, 1) = 0;
    stageWght(gridY >= tvd_value + 150 | gridY <= tvd_value - 150, idx, 2) = 0.2;
    stageWght(gridY >= tvd_value + 150 | gridY <= tvd_value - 150, idx, 3) = 0.8;

    % Adjust weights within the taper zones
    for t = upperTaperZone
        taperFactor = (t - (tvd_value + 75 - taperWidth)) / taperWidth;
        stageWght(gridY == t, idx, :) = (1 - taperFactor) * 0.2 + taperFactor * 0.4; % Example tapering formula
    end

    for t = lowerTaperZone
        taperFactor = (t - (tvd_value - 75)) / taperWidth;
        stageWght(gridY == t, idx, :) = (1 - taperFactor) * 0.2 + taperFactor * 0.4; % Example tapering formula
    end
end

% R_UH1D = sum(R_UH1D_AllStages.*stageWght,3);
for stage=numStages:-1:3
    R_UH1D = R_UH1D_AllStages(:,:,stage);
    hsurf = surf(LD(1:end-1)/feet, gridY, log(R_UH1D),'edgecolor','none', 'FaceColor', 'interp');
    hold on;
end
view([-180 -90]);
set(gca,'Xdir','reverse');
set(gca,'Ydir','reverse');
box off
plot3(LD/feet,TVD,-10*ones(length(TVD),1),'-.*k','linewidth',3);
colormap(jet(256));
% xlabel('X (ft)','fontsize',fontsize);
ylabel('TVD (ft)','fontsize',fontsize);
xlim([LD(1)/feet, LD(end)/feet]);
ylim([min(gridY) max(gridY)])
R = [0.1,1,10,100];
caxis(log([R(1) R(length(R))]));
h = colorbar('FontSize',9,'YTick',log(R),'YTickLabel',R);
set(h,'Position', [.46 .4 .02 .5]);
set(gcf,'unit','centimeters','position',[8 6 11.5 9]);
title('inverted R_h','FontSize',fontsize);

subplot('Position', [0.05, 0.1, 0.4, 0.28]) % [left bottom width height]
yyaxis left
plot(LD(1:end-1)/feet,misfit(1:runConfig.InvGroupPts:end),'-*');grid on;hold on
xlabel('X(ft)','fontsize',fontsize);
ylabel('Misfit','fontsize',fontsize);
xlim([LD(1)/feet, LD(end)/feet]);

yyaxis right
plot(LD(1:end-1)/feet,anis(1:runConfig.InvGroupPts:end),'-*');grid on;hold on
ylabel('Anisotropy Ratio','fontsize',fontsize);

subplot('Position', [0.55, 0.1, 0.4, 0.28]) % [left bottom width height]
for stage = numStages:-1:1
    R_UH1D = R_UH1D_AllStages(:, :, stage);
    surf(LD(1:end-1)/feet, gridY, (R_UH1D),'edgecolor','none');
    hold on;
end
xlabel('X (ft)','fontsize',fontsize);
ylabel('Inverted Rh(linear)','fontsize',fontsize);
xlim([LD(1)/feet, LD(end)/feet]);
ylim([min(gridY) max(gridY)])
view([-180 -90]);
set(gca,'Xdir','reverse');
set(gca,'Ydir','reverse');
box off
hold on;
plot3(LD/feet,TVD,-10*ones(length(TVD),1),'-.*k','linewidth',3);

%% plot true Rh
prior = 0;  %1: use UH1D as initial model; 
            %0: not use, initial model is homogeneous
type = runConfig.formation_type;
skdRatio = 2;
[domainSet, ~, ~] = Init_gridinv_INVanis(tool, skdRatio, prior, type);
% model.ZbedInput
ref_rh = model.Rh;
ref_rh(end+1) = ref_rh(end);

load model.mat Rel_X TVD;
% ind_tvd = 24:34;
ind_tvd = 29:2:43;
TVD_org = TVD(ind_tvd);
tvd_mid = mean(TVD_org);

gAxisX = domainSet.mesh_inv.xinv/feet;
gAxisY = model.ZbedInput;% - tvd_mid;
gAxisY(end+1) = gAxisY(1)-200;
gAxisY(end+1) = gAxisY(end-1)+200;
gAxisY = sort(gAxisY);

Rtick = [1,10,100];

subplot('Position', [0.55, 0.4, 0.4, 0.5]) % [left bottom width height]
surf(gAxisX,-1*gAxisY,log(repmat(ref_rh,length(gAxisX),1)).','EdgeColor','none');
colormap(jet(256));
% set(gca,'fontsize',fontsize);
box off;
set(gca,'Xdir','reverse');
set(gca,'Ydir', 'reverse');
view([-180 -90]);
% xlabel('X (ft)','fontsize',fontsize);
ylabel('TVD (ft)','fontsize',fontsize);
% xlim([min(gAxisX), max(gAxisX)]);ylim([min(gAxisY), max(gAxisY)]);
xlim([LD(1)/feet, LD(end)/feet]);
ylim([min(gridY) max(gridY)]);
R = [0.1,1,10,100];
caxis(log([R(1) R(length(R))]));
h = colorbar('FontSize',9,'YTick',log(R),'YTickLabel',R);
hold on
plot3(tool.MD/feet, -tool.TVD/feet, -20*ones(1,length(tool.TVD)), '-.*k','linewidth', 2)
title('true R_h','FontSize',fontsize);

set(gca, 'box', 'off')

return;

%% save UH1D inverse result
dip = model.Dip;
gridY = gridY;
init_TVD = TVD;
R_UH1D = R_UH1D;
save('UH1D_initial','gridY','R_UH1D','init_TVD','dip','LD','gAxisX','gAxisY','ref_rh');
end