function fun_plot_matrix_for_1D(configfile)
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
inv = cell2mat(result);
[rw,clm] = size(inv);
np = rw-runConfig.nParamsAnis;
R_UH1D = [];
E_UH1D = [];
% figure
pixel = config.resolution;
for j = runConfig.inv_station_start:runConfig.inv_station_end
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
        Zr = repmat(log(inv(1:np,j-runConfig.inv_station_start+1)),1,2);
%         surf(X,Y*0.3048,Zr,'EdgeColor','none');
%         view([-180,-90]);
%         hold on
        
        gridY = TVD(1)+config.scope(1):TVD(end)+config.scope(2);
        R_gridy = interp1(Y(:,1),(inv(1:np,j-runConfig.inv_station_start+1)),gridY).';
        R_gridy(gridY<Y(1,1)) = median(inv(1:4,j-runConfig.inv_station_start+1));
        R_gridy(gridY>Y(end,1)) = median(inv(np-4:np,j-runConfig.inv_station_start+1));
        R_UH1D = [R_UH1D, R_gridy];       

end
%% plot inverted Rh and misfit
% R_UH1D(:,4) = (R_UH1D(:,3) + R_UH1D(:,5))/2;
fontsize = 18;

fig = figure;
% subplot(2,2,1);
subplot('Position', [0.05, 0.4, 0.4, 0.5]) % [left bottom width height]
surf(LD(runConfig.inv_station_start:runConfig.inv_station_end)/feet, gridY, log(R_UH1D),'edgecolor','none','FaceColor','interp');
% view([-180 -90]);
view(0, 90);
% set(gca,'Xdir','reverse');
set(gca,'Ydir','reverse');
box off
hold on;
plot3(LD/feet,TVD,10000*ones(length(TVD),1),'-.*k','linewidth',3);
colormap(jet(256));
% xlabel('X (ft)','fontsize',fontsize);
ylabel('TVD (ft)','fontsize',fontsize);
xlim([LD(1)/feet, LD(end)/feet]);
ylim([min(gridY) max(gridY)])
R = [0.1,1,10,100];
caxis(log([R(1) R(length(R))]));
h = colorbar('FontSize',9,'YTick',log(R),'YTickLabel',R);
% caxis([0,50]);
% h = colorbar;
set(h,'Position', [.46 .4 .02 .5]);
set(gcf,'unit','centimeters','position',[8 6 11.5 9]);
title('inverted R_h','FontSize',fontsize);

subplot('Position', [0.05, 0.1, 0.4, 0.28]) % [left bottom width height]
yyaxis left
plot(LD(runConfig.inv_station_start:runConfig.InvGroupPts:runConfig.inv_station_end)/feet,misfit(runConfig.inv_station_start:runConfig.InvGroupPts:runConfig.inv_station_end),'-*');grid on;hold on
xlabel('X(ft)','fontsize',fontsize);
ylabel('Misfit','fontsize',fontsize);
xlim([LD(1)/feet, LD(end)/feet]);

yyaxis right
plot(LD(runConfig.inv_station_start:runConfig.InvGroupPts:runConfig.inv_station_end)/feet,anis(runConfig.inv_station_start:runConfig.InvGroupPts:runConfig.inv_station_end),'-*');grid on;hold on
ylabel('Anisotropy Ratio','fontsize',fontsize);

subplot('Position', [0.55, 0.1, 0.4, 0.28]) % [left bottom width height]
surf(LD(runConfig.inv_station_start:runConfig.inv_station_end)/feet, gridY, (R_UH1D),'edgecolor','none');
colormap(jet(256));

minValue = min(R_UH1D,[],'all');
maxValue = max(R_UH1D,[],'all');
caxis([minValue maxValue]);
h1 = colorbar('FontSize',9);
cbpos = get(h1, 'Position');
% text(cbpos(1) + cbpos(3) + 0.01, cbpos(2), num2str(minValue), 'FontSize', 9, 'VerticalAlignment', 'bottom');
% text(cbpos(1) + cbpos(3) + 0.01, cbpos(2) + cbpos(4), num2str(maxValue), 'FontSize', 9, 'VerticalAlignment', 'top');
xlabel('X (ft)','fontsize',fontsize);
ylabel('Inverted Rh(linear)','fontsize',fontsize);
xlim([LD(1)/feet, LD(end)/feet]);
ylim([min(gridY) max(gridY)])
view([0 90]);
% set(gca,'Xdir','reverse');
set(gca,'Ydir','reverse');
box off
hold on;
plot3(LD/feet,TVD,100000*ones(length(TVD),1),'-.*k','linewidth',3);

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
surf(gAxisX,gAxisY,log(repmat(ref_rh,length(gAxisX),1)).','EdgeColor','none');
colormap(jet(256));
% set(gca,'fontsize',fontsize);
box off;
% set(gca,'Xdir','reverse');
set(gca,'Ydir', 'reverse');
view([0 90]);
% xlabel('X (ft)','fontsize',fontsize);
ylabel('TVD (ft)','fontsize',fontsize);
% xlim([min(gAxisX), max(gAxisX)]);ylim([min(gAxisY), max(gAxisY)]);
xlim([LD(1)/feet, LD(end)/feet]);
ylim([min(gridY) max(gridY)]);
R = [0.1,1,10,100];
caxis(log([R(1) R(length(R))]));
h = colorbar('FontSize',9,'YTick',log(R),'YTickLabel',R);
hold on
plot3(tool.MD/feet, tool.TVD/feet, 10000*ones(1,length(tool.TVD)), '-.*k','linewidth', 2)
title('true R_h','FontSize',fontsize);

set(gca, 'box', 'off')

% save figure
if isunix
        screenSize = [35, 24];
        set(fig, 'Position', [-1 0 screenSize]);
        saveas(gcf,'UH1D_results','epsc');
        saveas(gcf,'UH1D_results','png');
end
return;

%% save UH1D inverse result
dip = model.Dip;
gridY = gridY;
init_TVD = TVD;
R_UH1D = R_UH1D;
save('UH1D_initial','gridY','R_UH1D','init_TVD','dip','LD','gAxisX','gAxisY','ref_rh');
end