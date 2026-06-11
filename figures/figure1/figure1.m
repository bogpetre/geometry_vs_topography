close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));

addpath('../../matlab_libraries');
addpath('../../resources/neuromaps');

fs=config.matlab_disp_scheme.fontsize;

dc_color = config.matlab_disp_scheme.color_main(3,:);
dc_color_light = config.matlab_disp_scheme.color_light(3,:);

% we use these for color coordinate wiring costs with parcel colors for
% visual consistency
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall

cmap = zeros(length(atlas_labels),3);
for i = 1:length(atlas_labels)
    cmap(i,:) = atlas_labels(i).rgba(1:3);
end

% color choices are arbitrary, but chosen to evoke depth correspondences
% when juxtaposed against brain results
%{
layer_color_src = {'Ctx_V1_L', 'Ctx_IFSp_L'; ...
'Ctx_V2_L', 'Ctx_8C_L'; ...
'Ctx_V3_L', 'Ctx_8Av_L'; ...
'Ctx_V4_L', 'Ctx_8Ad_L'; ...
'Ctx_V4t_L', 'Ctx_8BL_L'; ...
'Ctx_V8_L', 'Ctx_9m_L'; ...
'Ctx_FFC_L', 'Ctx_d32_L'; ...
'Ctx_VVC_L', 'Ctx_a24_L'};

layer_colors = cell(size(layer_color_src));
for i = 1:size(layer_colors,1)
    for j = 1:size(layer_colors,2)
        ind = find(contains({atlas_labels.name}, layer_color_src{i,j}));
        layer_colors{i,j} = atlas_labels(ind).rgba(1:3);
    end
end
%}

layer_colors = repmat({[0.2471,0.0196,1], [0.3804, 0.3843, 0.3412]},8,1);

%% Plot similarity metrics

tbl = readtable('../../derivatives/models/tdann_v_resnet_similarities_wide.csv');

layers = find(contains(tbl.Properties.VariableNames,'layer'));

layer_names = {'layer1.0','layer1.1','layer2.0','layer2.1',...
    'layer3.0','layer3.1','layer4.0','layer4.1'};

figure(1);

tdann_ind = find(strcmp(tbl.type,'TDANN') & strcmp(tbl.metric,'cosim'));
resnet_ind = find(strcmp(tbl.type,'ResNet') & strcmp(tbl.metric,'cosim'));

subplot(1,2,1);
cla
h1 = plot(table2array(tbl(tdann_ind, layers'))','-','color',layer_colors{1,1},'LineWidth',2);
hold on
for i = 1:length(layers), plot(i, tbl.(strrep(layer_names{i},'.','_'))(tdann_ind), ...
        'o', 'MarkerFaceColor', layer_colors{i,1}, 'color', layer_colors{1,1}); end
h2 = plot(table2array(tbl(resnet_ind, layers'))','-','color',layer_colors{1,2},'LineWidth',2);
hold on
for i = 1:length(layers), plot(i, tbl.(strrep(layer_names{i},'.','_'))(resnet_ind), ...
        'o', 'MarkerFaceColor', layer_colors{i,2}, 'color', layer_colors{1,2}); end
%plot(table2array(tbl(cross_ind(1:7), layers'))','o-','color','g')
xlim([1,8])
set(gca,'XTick',[1:8],'XTickLabels',layer_names,'FontSize',fs)
ylabel('cos\theta of activations')
ylim([0,1.2])
xlim([1,9])
title({'Topographic','Similarity'},'fontweight','normal','fontsize',fs)
box off
legend off

tdann_ind = find(strcmp(tbl.type,'TDANN') & strcmp(tbl.metric,'CKA'));
resnet_ind = find(strcmp(tbl.type,'ResNet') & strcmp(tbl.metric,'CKA'));

subplot(1,2,2);
cla
h1 = plot(table2array(tbl(tdann_ind, layers'))','-','color',layer_colors{1,1},'LineWidth',2);
hold on
for i = 1:length(layers), plot(i, tbl.(strrep(layer_names{i},'.','_'))(tdann_ind), ...
        'o', 'MarkerFaceColor', layer_colors{i,1}, 'color', layer_colors{1,1}); end
h2 = plot(table2array(tbl(resnet_ind, layers'))','-','color',layer_colors{1,2},'LineWidth',2);
hold on
for i = 1:length(layers), plot(i, tbl.(strrep(layer_names{i},'.','_'))(resnet_ind), ...
        'o', 'MarkerFaceColor', layer_colors{i,2}, 'color', layer_colors{1,2}); end
%plot(table2array(tbl(cross_ind(1:6), layers'))','o-','color','g')
xlim([1,8])
set(gca,'XTick',[1:8],'XTickLabels',layer_names,'YAxisLocation','right','FontSize',fs);
ylabel('CKA of activations')
xlim([1,9])
ylim([0,1.2])
title({'Geometric','Similarity'},'fontweight','normal','fontsize',fs)
box off
l = legend([h1(1),h2(1)],sprintf('TDANN\ndyad\n(wiring cost)'),sprintf('ResNet-18\ndyad\n(no cost)'),'location','southwest','fontsize',fs, 'box','off')


pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2), 450, 360])

pos = get(l,'position')
pos(1) = 0.57;
pos(2) = 0.18;
set(l,'Position',pos);

sgtitle({'Representations (geometries) are similar despite','idiosyncratic topographic implementations'},'FontWeight','bold','fontsize',fs+1);

export_fig(gcf,'panels/similarity_metrics.png','-png','-r300','-transparent')

%% do inferential stats using mixed models
[arch, seed1, seed2, selfsimilar, val, layer, metric] = deal([]);
col_names = tbl.Properties.VariableNames;
layer_cols = find(contains(col_names,'layer'));
for i = 1:height(tbl)
    for j = 1:length(layer_cols)
        if strcmp(tbl.type(i),'TDANN')
            arch(end+1) = 1/2;
            selfsimilar(end+1) = 1/3;
        elseif strcmp(tbl.type(i),'ResNet')
            arch(end+1) = -1/2;
            selfsimilar(end+1) = 1/3;
        else
            arch(end+1) = 0;
            selfsimilar(end+1) = -2/3;
        end
        if strcmp(tbl.metric(i),'cosim')
            metric(end+1) = -0.5;
        else
            metric(end+1) = 0.5;
        end
        seed1(end+1) = tbl.seed1(i);
        seed2(end+1) = tbl.seed2(i);
        layer(end+1) = (j - mean(1:8))/7;
        val(end+1) = tbl.(col_names{layer_cols(j)})(i);
    end
end
metric_v_arch = metric.*arch;
metric_v_layer = metric.*layer;

tbl_long = table(arch', selfsimilar', metric', seed1', seed2', layer', metric_v_layer', metric_v_arch', val', ...
    'VariableNames', {'spatial','selfsimilar','metric','seed1','seed2','layer',...
    'metric_v_layer','metric_v_arch','val'});


% first we ignore the TDANN vs. ResNet network similarities
tbl_short = tbl_long(tbl_long.spatial ~=0,:);

% zscore similarity measures within network-pairs
networks = unique(tbl_short.seed1);
n_networks = length(networks);
zval = nan(height(tbl_short),1);
for i = 1:n_networks
    this_net_ind = tbl_short.seed1 == networks(i);
    this_net_topo_ind = this_net_ind & tbl_short.metric == 0.5;
    this_net_geom_ind = this_net_ind & tbl_short.metric == -0.5;
    zval(this_net_topo_ind) = zscore(tbl_short.val(this_net_topo_ind)); 
    zval(this_net_geom_ind) = zscore(tbl_short.val(this_net_geom_ind));
end
tbl_short.zval = zval;

topo_short = tbl_short(tbl_short.metric == -0.5,:);
m = fitlme(topo_short,'val ~ spatial*layer + (spatial*layer + 1 | seed1)','FitMethod','REML');
[~,~,topo_short_STATS] = fixedEffects(m,'DFMethod','Satterthwaite')

geom_short = tbl_short(tbl_short.metric == 0.5,:);
m = fitlme(geom_short,'val ~ spatial*layer + (spatial*layer + 1 | seed1)','FitMethod','REML');
[~,~,geom_short_STATS] = fixedEffects(m,'DFMethod','Satterthwaite')

m = fitlme(tbl_short, 'zval ~ metric*layer + spatial + (metric*layer + spatial | seed1)','FitMethod','REML');
[~,~,short_layer_int_STATS] = fixedEffects(m,'DFMethod','Satterthwaite')

m = fitlme(tbl_short, 'zval ~ metric*spatial + layer + (metric*spatial + layer | seed1)','FitMethod','REML');
[~,~,short_spatial_int_STATS] = fixedEffects(m,'DFMethod','Satterthwaite')

%{
% topographic similarity
tbl1 = tbl_long(tbl_long.metric == 0.5,:);

m = fitlme(tbl1,'val ~ spatial*layer + selfsimilar*layer + (spatial*layer + selfsimilar*layer + 1 | seed1)','FitMethod','REML');
[~,~,STATS] = fixedEffects(m,'DFMethod','Satterthwaite')

%tbl2 = tbl_long(tbl_long.arch ~= 0 & strcmp(tbl_long.metric, 'cka'),:);
tbl2 = tbl_long(strcmp(tbl_long.metric, 'cka'),:);
tbl2.seed1 = categorical(tbl2.seed1);

m = fitlme(tbl2,'val ~ spatial*layer + selfsimilar*layer + (spatial*layer + selfsimilar*layer + 1 | seed1)');
[~,~,STATS] = fixedEffects(m,'DFMethod','Satterthwaite')
%}

%% Estimate fixed effects models
% We reestimate the above model parameters to ensure approxiate equivalence
% for plotting

[topo_layer_B, topo_spatial_B, ...
    geom_layer_B, geom_spatial_B, ...
    layer_metric_int_B, spatial_metric_int_B] = deal([]);
for i = 1:n_networks
    this_subj = tbl_long(networks(i) == tbl_long.seed1,:);

    topo = this_subj(this_subj.metric == -0.5,:);
    topo_within_arch = topo(topo.spatial ~= 0,:);
    if height(topo_within_arch) > 0
        y = topo_within_arch.val;

        % layer
        X = topo_within_arch.layer;
        X = [X, ones(length(X),1)];
    
        b = (X'*X)\X'*y;
        topo_layer_B(end+1) = b(1);

        % spatial
        X = topo_within_arch.spatial;
        X = [X, ones(length(X),1)];
    
        b = (X'*X)\X'*y;
        topo_spatial_B(end+1) = b(1);
    end

    % geom
    geom = this_subj(this_subj.metric == 0.5,:);
    geom_within_arch = geom(geom.spatial ~= 0,:);
    if height(geom_within_arch) > 0
        y = geom_within_arch.val;

        % layer
        X = geom_within_arch.layer;
        X = [X, ones(length(X),1)];
    
        b = (X'*X)\X'*y;
        geom_layer_B(end+1) = b(1);

        % spatial
        X = geom_within_arch.spatial;
        X = [X, ones(length(X),1)];
    
        b = (X'*X)\X'*y;
        geom_spatial_B(end+1) = b(1);

        % metric x layer interaction
        y = [zscore(topo_within_arch.val); zscore(geom_within_arch.val)];
        X = [-0.5*ones(height(topo_within_arch),1); 0.5*ones(height(geom_within_arch),1)]; % metric (cka > cos)
        X = [X, ...
            [topo_within_arch.layer; geom_within_arch.layer], ... % layer
            [topo_within_arch.spatial; geom_within_arch.spatial], ... % TDANN vs. ResNet
            ones(size(X,1),1)]; % intercept
        X1 = [X(:,1).*X(:,2),X]; % metric vs. layer
        b = (X1'*X1)\X1'*y;
        layer_metric_int_B(end+1) = b(1); % fx of metric x z(layer) on average across TDANNs and ResNets


        % metric x spatial interaction
        X2 = [X(:,1).*X(:,3),X]; % metric vs. spatial
        b = (X2'*X2)\X2'*y;
        spatial_metric_int_B(end+1) = b(1); % fx of metric x z(layer) on average across TDANNs and ResNets
    end
end

% get OLS stats
fprintf('\nTopographic Similarity\n') % compare these with topo_short_STATS for mixed effects versions
[~,topo_spatial_p,~,STATS] = ttest(topo_spatial_B);
fprintf('loss (TDANN > ResNet): B = %0.3f, t(%d) = %0.3f, p = %0.2e\n', mean(topo_spatial_B), STATS.df, STATS.tstat, topo_spatial_p);
[~,topo_layer_p,~,STATS] = ttest(topo_layer_B);
fprintf('layer (1.0 > 4.1): B = %0.3f, t(%d) = %0.3f, p = %0.2e\n', mean(topo_layer_B), STATS.df, STATS.tstat, topo_layer_p);

fprintf('\nGeometric Similarity\n') % compare these with geom_short_STATS for mixed effects versions
[~,geom_spatial_p,~,STATS] = ttest(geom_spatial_B);
fprintf('loss (TDANN > ResNet): B = %0.3f, t(%d) = %0.3f, p = %0.2e\n', mean(geom_spatial_B), STATS.df, STATS.tstat, geom_spatial_p);
[~,geom_layer_p,~,STATS] = ttest(geom_layer_B);
fprintf('layer (1.0 > 4.1): B = %0.3f, t(%d) = %0.3f, p = %0.2e\n', mean(geom_layer_B), STATS.df, STATS.tstat, geom_layer_p);

fprintf('\nGeometric vs. Topographic Similarity (standardized siimlarities)\n')
% compare the below with short_layer_int_STATSt
[~,p,~,STATS] = ttest(layer_metric_int_B);
fprintf('metric x layer (geom > topo): B = %0.3f, t(%d) = %0.3f, p = %0.2e\n', mean(layer_metric_int_B), STATS.df, STATS.tstat, p);
% compare the below with short_spatial_int_STATS
[~,p,~,STATS] = ttest(spatial_metric_int_B);
fprintf('metric x spatial (geom > topo x TDANN > ResNet): B = %0.3f, t(%d) = %0.3f, p = %0.2e\n', mean(spatial_metric_int_B), STATS.df, STATS.tstat, p);


%% Plot layer contrasts

figure(2);
clf
t1 = tiledlayout(1,3,'TileSpacing','compact','padding','none');
s1 = nexttile();
% order atters here. must match topo_short_STATS order
h1 = barplot_columns({topo_spatial_B(:), topo_layer_B(:)}, 'nofig', ...
    'custom_p',topo_short_STATS.pValue(2:3)', ...
    'colors',dc_color_light, ...
    'MarkerSize',5);
set(h1.star_handles,'fontsize',config.matlab_disp_scheme.fontsize-3,'horizontalalign','center')
set(gca,'XTickLabels',{'Wiring Cost','Layer Depth'})
xlabel([])
ylabel({'\Deltasimilarity (\beta)'})
yl = ylim;
ylim([yl(1)-diff(yl)*0.4,yl(2)+diff(yl)*0.5])
pos = get(h1.star_handles,'pos');
for i = 1:length(pos)
    pos{i}(2) = yl(2)+diff(ylim)*0.02; 
    set(h1.star_handles(i),'pos',pos{i});
end
box off
annot{1,1} = text(0.95, yl(1)-diff(yl)*0.4,'ResNet','FontSize',fs-3,'HorizontalAlignment','left','Rotation',90);
annot{1,2} = text(0.95, yl(2)+diff(yl)*0.5,'TDANN','FontSize',fs-3,'HorizontalAlignment','right','Rotation',90);
annot{2,1} = text(1.95, yl(1)-diff(yl)*0.4,'layer1.0','FontSize',fs-3,'HorizontalAlignment','left','Rotation',90);
annot{2,2} = text(1.95, yl(2)+diff(yl)*0.5,'layer4.1','FontSize',fs-3,'HorizontalAlignment','right','Rotation',90);
title({'Topography','(cos\theta)'},'fontsize',fs,'fontweight','normal')

s2 = nexttile();
% order atters here. must match geom_short_STATS order
h2 = barplot_columns({geom_spatial_B(:), geom_layer_B(:)}, 'nofig', ...
    'custom_p',geom_short_STATS.pValue(2:3)', ...
    'colors',dc_color_light, ...
    'MarkerSize',5);
set(h2.star_handles,'fontsize',fs-3,'horizontalalign','center')
set(gca,'XTickLabels',{'Wiring Cost','Layer Depth'},'XTickLabelRotation',45)
xlabel([])
ylabel({'\Deltasimilarity (\beta)'})
yl = ylim;
ylim([yl(1)-diff(yl)*0.4,yl(2)+diff(yl)*0.5])
pos = get(h2.star_handles,'pos');
for i = 1:length(pos)
    pos{i}(2) = yl(2)+diff(ylim)*0.02; 
    set(h2.star_handles(i),'pos',pos{i});
end
box off
annot{1,1} = text(0.95, yl(1)-diff(yl)*0.4,'ResNet','FontSize',fs-3,'HorizontalAlignment','left','Rotation',90);
annot{1,2} = text(0.95, yl(2)+diff(yl)*0.5,'TDANN','FontSize',fs-3,'HorizontalAlignment','right','Rotation',90);
annot{2,1} = text(1.95, yl(1)-diff(yl)*0.4,'layer1.0','FontSize',fs-3,'HorizontalAlignment','left','Rotation',90);
annot{2,2} = text(1.95, yl(2)+diff(yl)*0.5,'layer4.1','FontSize',fs-3,'HorizontalAlignment','right','Rotation',90);
title({'Geometry','(CKA)'},'fontsize',fs,'fontweight','normal')

s3 = nexttile();
h3 = barplot_columns({spatial_metric_int_B(:), layer_metric_int_B(:)}, 'nofig',...
    'custom_p',[short_spatial_int_STATS.pValue(end), short_layer_int_STATS.pValue(end)], ...
    'colors',dc_color_light, ...
    'MarkerSize',5);
set(h3.star_handles,'fontsize',fs-3,'horizontalalign','center')
set(gca,'XTickLabels',{'Wiring Cost','Layer Depth'},'XTickLabelRotation',45)
xlabel([])
ylabel({'\Deltasimilarity (\beta_{std})'})
yl = ylim;
ylim([yl(1)-diff(yl)*0.4,yl(2)+diff(yl)*0.5])
pos = get(h3.star_handles,'pos');
for i = 1:length(pos)
    pos{i}(2) = yl(2)+diff(ylim)*0.02; 
    set(h3.star_handles(i),'pos',pos{i});
end
annot{1,1} = text(0.95, yl(1)-diff(yl)*0.4,'ResNet','FontSize',fs-3,'HorizontalAlignment','left','Rotation',90);
annot{1,2} = text(0.95, yl(2)+diff(yl)*0.5,'TDANN','FontSize',fs-3,'HorizontalAlignment','right','Rotation',90);
annot{2,1} = text(1.95, yl(1)-diff(yl)*0.4,'layer1.0','FontSize',fs-3,'HorizontalAlignment','left','Rotation',90);
annot{2,2} = text(1.95, yl(2)+diff(yl)*0.5,'layer4.1','FontSize',fs-3,'HorizontalAlignment','right','Rotation',90);

box off
title({'zGeo - zTopo'},'fontsize',fs,'fontweight','normal')

sgtitle({'Factors associated with model variation','(regression coefficients)'},...
    'fontsize',fs+1,'fontweight','bold')

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2), 350, 360])

t1.Position(1) = 0.18
%t1.Position(2) = 0.25;
t1.Position(3:4) = [0.68,0.53];
set(s1,'XTickLabelRotation',45)

export_fig(gcf,'panels/similarity_coefs.png','-png','-r300','-transparent')

%% Plot model aggreement

tdann_ag = table2array(readtable('../../derivatives/models/tdann_agreement.csv'))'*100;
resnet_ag = table2array(readtable('../../derivatives/models/resnet_agreement.csv'))'*100;

[~,p,~,STATS] = ttest(resnet_ag(:,1), tdann_ag(:,1));
fprintf('ResNet > TDANN 1000-cat agreement: t(%0.1f)=%0.3f, %0.2e\n',STATS.df,STATS.tstat,p)

[~,p,~,STATS] = ttest(resnet_ag(:,2), tdann_ag(:,2));
fprintf('ResNet > TDANN 50-cat agreement: t(%0.1f)=%0.3f, %0.2e\n',STATS.df,STATS.tstat,p)

figure(5);
clf
t1 = tiledlayout(1,2,'TileSpacing','compact','padding','none');
s1 = nexttile();
% order atters here. must match topo_short_STATS order
h1 = barplot_columns({tdann_ag(:,1), resnet_ag(:,1)}, 'nofig', ...
    'nostars', ...
    'colors',dc_color_light, ...
    'MarkerSize',5, 'MarkerAlpha', 0.5, ...
    'nobars','noviolins');
plot([0,3],[0,0],'color',[0.5,0.5,0.5])
set(gca,'XTickLabels',{'TDANN','ResNet-18'},'YTickLabels',[], ...
    'XTickLabelRotation',90,'YTick',0:20:100)
ax = findall(gcf,'type','axes');
pause(1);
ax.YRuler.Axle.Visible = 'off';
ax.YAxis.TickLength = [0,0];
xlabel([])
ylabel({'Mdl pred. agreement (1000 categories)'})
ylim([0,100]);
box off
grid on

s2 = nexttile();
h1 = barplot_columns({tdann_ag(:,2), resnet_ag(:,2)}, 'nofig', ...
    'nostars', ...
    'colors',dc_color_light, ...
    'MarkerSize',5, 'MarkerAlpha', 0.5, ...
    'nobars','noviolins','fontsize',fs);
plot([0,3],[0,0],'color',[0.5,0.5,0.5])
set(gca,'XTickLabels',{'TDANN','ResNet-18'},...
    'YTick',0:20:100,'YTickLabels',[],'XTickLabelRotation',90)
pause(1)
ax = findall(gcf,'type','axes');
ax(1).YRuler.Axle.Visible = 'off';
ax(1).YAxis.TickLength = [0,0];
xlabel([])
grid on;
ylim([0,100]);
ycolor = ax(2).YColor;
ylabel({['Mdl pred. agreement (50 hypernyms)']})
yyaxis('right');
ax = findall(gcf,'type','axes');
ax(1).YRuler.Axle.Visible = 'off';
ax(1).YAxis(2).TickLength = [0,0];
ax(1).YAxis(2).TickValues = 0:20:100;
ax(1).YTickLabelRotation = 90;
ylim([0,100]);
box off

for i=1:length(ax)
    set(ax(i),'fontsize', fs);
end

ax(1).YColor = ycolor;
ax(1).YAxis(1).Color = ycolor;
ax(1).YAxis(2).Color = ycolor;

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),150,380])

export_fig(gcf,'panels/model_agreement.png','-png','-r300','-transparent')

%% Plot performance metrics
[resnet_top1, tdann_top1, resnet_top5, tdann_top5] = deal(nan(2*n_networks,1));
for s = 1:2*n_networks
    data = readlines(sprintf('../../derivatives/models/checkpoints/linear_eval/simclr_spatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_%d_linear_eval/metrics.json',s-1));
    json = jsondecode(data(end-1,:));
    tdann_top1(s) = json.test_accuracy_list_meter.top_1.x0;
    tdann_top5(s) = json.test_accuracy_list_meter.top_5.x0;

    data = readlines(sprintf('../../derivatives/models/checkpoints/linear_eval/simclr_nonspatial_resnet18_swappedon_SineGrating2019_isoswap_3_seed_%d_linear_eval/metrics.json',s-1));
    json = jsondecode(data(end-1,:));
    resnet_top1(s) = json.test_accuracy_list_meter.top_1.x0;
    resnet_top5(s) = json.test_accuracy_list_meter.top_5.x0;
end

[~,p,~,STATS] = ttest(tdann_top1, resnet_top1);
fprintf('ResNet > TDANN top1 accuracy: t(%0.1f)=%0.3f, %0.2e\n',STATS.df,STATS.tstat,p)

[~,p,~,STATS] = ttest(tdann_top5, resnet_top5);
fprintf('ResNet > TDANN top5 accuracy: t(%0.1f)=%0.3f, %0.2e\n',STATS.df,STATS.tstat,p)

figure(3);
clf
t1 = tiledlayout(1,2,'TileSpacing','compact','padding','none');
s1 = nexttile();
% order atters here. must match topo_short_STATS order
h1 = barplot_columns({tdann_top1(:), resnet_top1(:)}, 'nofig', ...
    'nostars', ...
    'colors',dc_color_light, ...
    'MarkerSize',5, 'MarkerAlpha', 0.5,...
    'nobars','noviolins');
h1.point_han{1,1}.MarkerEdgeColor = dc_color;
h1.point_han{1,1}.MarkerFaceColor = 'none';
h1.point_han{1,2}.MarkerEdgeColor = dc_color;
h1.point_han{1,2}.MarkerFaceColor = 'none';
plot([0,3],[0,0],'color',[0.5,0.5,0.5])
set(gca,'XTickLabels',{'TDANN','ResNet-18'},'YTickLabels',[], ...
    'XTickLabelRotation',90,'YTick',0:20:100)
ax = findall(gcf,'type','axes');
pause(1);
ax.YRuler.Axle.Visible = 'off';
ax.YAxis.TickLength = [0,0];
xlabel([])
ylabel({'Top-1 Accuracy (chance: 0.1)'})
ylim([0,100]);
box off
grid on

s2 = nexttile();
h1 = barplot_columns({tdann_top5(:), resnet_top5(:)}, 'nofig', ...
    'nostars', ...
    'colors',dc_color_light, ...
    'MarkerSize',5, 'MarkerAlpha', 0.5, ...
    'nobars','noviolins','fontsize',fs);
h1.point_han{1,1}.MarkerEdgeColor = dc_color;
h1.point_han{1,1}.MarkerFaceColor = 'none';
h1.point_han{1,2}.MarkerEdgeColor = dc_color;
h1.point_han{1,2}.MarkerFaceColor = 'none';
plot([0,3],[0,0],'color',[0.5,0.5,0.5])
set(gca,'XTickLabels',{'TDANN','ResNet-18'},...
    'YTick',0:20:100,'YTickLabels',[],'XTickLabelRotation',90)
pause(1)
ax = findall(gcf,'type','axes');
ax(1).YRuler.Axle.Visible = 'off';
ax(1).YAxis.TickLength = [0,0];
xlabel([])
grid on;
ylim([0,100]);
ycolor = ax(2).YColor;
ylabel({['Top-5 Accuracy (chance: 0.5)']})
yyaxis('right');
ax = findall(gcf,'type','axes');
ax(1).YRuler.Axle.Visible = 'off';
ax(1).YAxis(2).TickLength = [0,0];
ax(1).YAxis(2).TickValues = 0:20:100;
ax(1).YTickLabelRotation = 90;
ylim([0,100]);
box off

for i=1:length(ax)
    set(ax(i),'fontsize', fs);
end

ax(1).YColor = ycolor;
ax(1).YAxis(1).Color = ycolor;
ax(1).YAxis(2).Color = ycolor;

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),150,360])

export_fig(gcf,'panels/model_acc.png','-png','-r300','-transparent')

%% Spatial gradients
medial_mask_R = gifti('../../resources/100307.R.atlasroi.32k_fs_LR.shape.gii').cdata == 1;

myelin_map = gifti('source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-R_feature.func.gii');
thickness = gifti('source-hcps1200_desc-thickness_space-fsLR_den-32k_hemi-R_feature.func.gii');
marg_rh_1 = gifti('source-margulies2016_desc-fcgradient01_space-fsaverage_den-32k_hemi-R_feature.func.gii');

evo_exp1 = gifti('source-hill2010_desc-evoexp_space-fsaverage_den-32k_hemi-R_feature.func.gii');
evo_exp2 = gifti('source-xu2020_desc-evoexp_space-fsaverage_den-32k_hemi-R_feature.func.gii');
fchomology = gifti('source-xu2020_desc-FChomology_space-fsaverage_den-32k_hemi-R_feature.func.gii');

devexp1 = gifti('source-hill2010_desc-devexp_space-fsLR_den-32k_hemi-R_feature.func.gii');
devexp2 = gifti('source-reardon2018_desc-scalinghcp_space-fsLR_dens-32k_hemi-R_feature.func.gii');

genepc1 = gifti('source-abagen_desc-genepc1_space-fsaverage_den-32k_hemi-R_feature.func.gii');
cogpc1 = gifti('source-neurosynth_desc-cogpc1_space-fsLR_dens-32k_hemi-R_feature.func.gii');
cbf1 = gifti('source-raichle_desc-cbf_space-fsaverage_den-32k_hemi-R_feature.func.gii');
cbf2 = gifti('source-satterthwaite2014_desc-meancbf_space-fsaverage_den-32k_hemi-R_feature.func.gii');


% zscore to have them be on a common scale
zscore_fun = @(x)((x - nanmean(x))/nanstd(x));

myelin_map.cdata(medial_mask_R) = zscore_fun(myelin_map.cdata(medial_mask_R));
thickness.cdata(medial_mask_R) = zscore_fun(thickness.cdata(medial_mask_R));
marg_rh_1.cdata(medial_mask_R) = zscore_fun(marg_rh_1.cdata(medial_mask_R));

evo_exp1.cdata(medial_mask_R) = zscore_fun(evo_exp1.cdata(medial_mask_R));
evo_exp2.cdata(medial_mask_R) = zscore_fun(evo_exp2.cdata(medial_mask_R));
fchomology.cdata(medial_mask_R) = zscore_fun(fchomology.cdata(medial_mask_R));

devexp1.cdata(medial_mask_R) = zscore_fun(devexp1.cdata(medial_mask_R));
devexp2.cdata(medial_mask_R) = zscore_fun(devexp2.cdata(medial_mask_R));

genepc1.cdata(medial_mask_R) = zscore_fun(genepc1.cdata(medial_mask_R));
cogpc1.cdata(medial_mask_R) = zscore_fun(cogpc1.cdata(medial_mask_R));
cbf1.cdata(medial_mask_R) = zscore_fun(cbf1.cdata(medial_mask_R));
cbf2.cdata(medial_mask_R) = zscore_fun(cbf2.cdata(medial_mask_R));


figure(4)
clf
t7 = tiledlayout(4,3,'Padding','compact','TileSpacing','tight');
sgtitle(t7,{'Factors hypothetically associated with','brain topography variation'},'FontWeight','bold','FontSize',fs+1);


ax1 = nexttile(t7);
ax1.Layout.Tile=1;
myelin_plot = fmridisplay();

jet = colormap(ax1,'jet');

myelin_plot = surface(myelin_plot, 'axes', ax1, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
h = plot_to_surf(myelin_map.cdata(:),myelin_plot.surface{1}.object_handle,'sourcespace','MNI152NLin6Asym','targetsurface','fsLR_32k','nolegend', ...
    'colormap', 'turbo');

title(ax1, {'Myelination','(Wiring)'},'FontWeight','normal','FontSize',fs)


ax2 = nexttile(t7);
ax2.Layout.Tile=2;
thick_plot = fmridisplay();

thick_plot = surface(thick_plot, 'axes', ax2, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
plot_to_surf(thickness.cdata(:), thick_plot.surface{1}.object_handle,'colormap',colormap(ax2,'turbo'));

title(ax2, {'Thickness','(~Differentiation)'},'FontWeight','normal','FontSize',fs)


ax3 = nexttile(t7);
ax3.Layout.Tile=3;
net_plot = fmridisplay();

net_plot = surface(net_plot, 'axes', ax3, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
plot_to_surf(marg_rh_1.cdata(:), net_plot.surface{1}.object_handle,'nolegend',...
    'colormap','turbo');

title(ax3, {'Network','Hierarchy'},'FontWeight','normal','FontSize',fs)

ax4 = nexttile(t7);
ax4.Layout.Tile=4;
evo1_plot = fmridisplay();

evo1_plot = surface(evo1_plot, 'axes', ax4, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(evo_exp1.cdata(:), evo1_plot.surface{1}.object_handle,'colormap','turbo');

title(ax4, {'Evolutionary', 'Expansion 1'},'FontWeight','normal','FontSize',fs)


ax5 = nexttile(t7)
ax5.Layout.Tile = 5;
evo2_plot = fmridisplay();

evo2_plot = surface(evo2_plot, 'axes', ax5, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
[~,cbar1,cbar2] = plot_to_surf(evo_exp2.cdata, evo2_plot.surface{1}.object_handle,'colormap','turbo');
%{
cbar1.Location = 'southoutside';
cbar1.TickLabels = {'',''};
cbar1.Label.String = 'trans';
cbar1.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) + [0.15,-0.06];
cbar1.Position(3) = 0.15;
cbar1.Position(4) = 0.02;
cbar2.Location = 'south';
cbar2.TickLabels = {'',''};
cbar2.Label.String = 'uni';
cbar2.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) - [0,0.06];
cbar2.Position(3) = 0.15;
cbar2.Position(4) = 0.02;
%}

title(ax5, {'Evolutionary','Expansion 2'},'FontWeight','normal','FontSize',fs)


ax6 = nexttile(t7);
ax6.Layout.Tile=6;
fchomo_plot = fmridisplay();

fchomo_plot = surface(fchomo_plot, 'axes', ax6, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
% we add a small increment to the fchomology data because it's all positive
% valued except for the medial wall and a small patch of the posterior
% parietal cortex. plot_to_surface is designed to map zero values to
% grayscale for the medial wall, so to avoid a hole in the lateral view
% we're looking for we increment things slightly to circumvent the mapping
% of zero values to to grayscale
%[~,cbar1,cbar2] = plot_to_surf(fchomology.cdata + 0.000001,fchom.surface{1}.object_handle,'colormap','turbo');
[~,cbar1,cbar2] = plot_to_surf(fchomology.cdata, fchomo_plot.surface{1}.object_handle,'colormap','turbo');
%{
cbar1.Location = 'southoutside';
cbar1.TickLabels = {'',''};
cbar1.Label.String = 'trans';
cbar1.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) + [0.15,-0.05];
cbar1.Position(3) = 0.15;
cbar1.Position(4) = 0.02;
cbar2.Location = 'southoutside';
cbar2.TickLabels = {'',''};
cbar2.Label.String = 'uni';
cbar2.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) - [0,0.05];
cbar2.Position(3) = 0.15;
cbar2.Position(4) = 0.02;
%}

title(ax6, {'Funntional Conn.','Homology'},'FontWeight','normal','FontSize',fs)


ax7 = nexttile(t7);
ax7.Layout.Tile=7;
dev1_plot = fmridisplay();

dev1_plot = surface(dev1_plot, 'axes', ax7, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(devexp1.cdata(:)), dev1_plot.surface{1}.object_handle,'colormap','turbo');

title(ax7, {'Developmental','Expansion 1'},'FontWeight','normal','FontSize',fs)

ax8 = nexttile(t7);
ax8.Layout.Tile=8;
dev2_plot = fmridisplay();

dev2_plot = surface(dev2_plot, 'axes', ax8, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(devexp2.cdata(:)), dev2_plot.surface{1}.object_handle,'colormap','turbo');

title(ax8, {'Developmental','Expansion 2'},'FontWeight','normal','FontSize',fs)

ax9 = nexttile(t7);
ax9.Layout.Tile=9;
gene_plot = fmridisplay();

gene_plot = surface(gene_plot, 'axes', ax9, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(genepc1.cdata(:)), gene_plot.surface{1}.object_handle,'colormap','turbo');

title(ax9, {'GenePC1','(transcriptomic)'},'FontWeight','normal','FontSize',fs)


ax10 = nexttile(t7);
ax10.Layout.Tile=10;
cog_plot = fmridisplay();

cog_plot = surface(cog_plot, 'axes', ax10, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(cogpc1.cdata(:)), cog_plot.surface{1}.object_handle,'colormap','turbo');

title(ax10, {'CogPC1','(neurosynth)'},'FontWeight','normal','FontSize',fs)


ax11 = nexttile(t7);
ax11.Layout.Tile=11;
cbf1_plot = fmridisplay();

cbf1_plot = surface(cbf1_plot, 'axes', ax11, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(cbf1.cdata(:)), cbf1_plot.surface{1}.object_handle,'colormap','turbo');

title(ax11, {'Cerebral Blood','Flow 1'},'FontWeight','normal','FontSize',fs)


ax12 = nexttile(t7);
ax12.Layout.Tile=12;
cbf2_plot = fmridisplay();

cbf2_plot = surface(cbf2_plot, 'axes', ax12, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(cbf2.cdata(:)), cbf2_plot.surface{1}.object_handle,'colormap','turbo');

title(ax12, {'Cerebral','Blood Flow 2'},'FontWeight','normal','FontSize',fs)

%

pos = get(gcf,'Position');
%set(gcf,'Position',[pos(1:2),750,288]); % wide
set(gcf,'Position',[pos(1:2),430,523]); %tall

export_fig(gcf,'panels/gradients_tall.png','-png','-r300','-transparent')
