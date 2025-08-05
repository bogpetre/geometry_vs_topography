close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));

addpath('../../matlab_libraries');
addpath('../../resources/neuromaps');

fs=config.matlab_disp_scheme.fontsize;

f = figure;
cm = colormap(f,'hot');
close(f)

dc_color = config.matlab_disp_scheme.color_main;
dc_color_light = config.matlab_disp_scheme.color_light;

%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii.cortex_left, atlas_cii.cortex_right, atlas_cii.volumes])) - 1;

%% import between subject similarity measures for unrelated individuals
sid = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);

tsnr = zeros(height(sid), n_roi);
wi_cosim = nan(height(sid), n_roi);
for s = 1:height(sid)
    try
        tsnr1 = readmatrix(sprintf('../../derivatives/restingstate/hcp50/results/%d/tsnr.csv',sid.Var1(s)));
        tsnr2 = readmatrix(sprintf('../../derivatives/restingstate/hcp50/results/%d/tsnr.csv',sid.Var2(s)));
        tsnr(s,:) = mean([tsnr1, tsnr2],2);

        wi_cosim1 = readmatrix(sprintf('../../derivatives/restingstate/hcp50/results/%d/standardized_betas/standardized_similarity.csv',sid.Var1(s)),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('../../derivatives/restingstate/hcp50/results/%d/standardized_betas/standardized_similarity.csv',sid.Var1(s)),'FileType','text');
        wi_cosim(s,:) = mean(mean(cat(3,wi_cosim1, wi_cosim2),3));
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = diag(readmatrix(sprintf('../../derivatives/restingstate/hcp50/bsc/standardized_betas/cosine/%d_v_%d_wuc.tsv',sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',','));
    catch
        warning('Could not import pair %d', s);
    end
end

cosim = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        cosim(s,:) = mean(readmatrix(sprintf('../../derivatives/restingstate/hcp50/bsc/standardized_betas/cosine/%d_v_%d_cosim.tsv',sid.Var1(s), sid.Var2(s)),...
            'FileType','text'),1);
    catch
        warning('Could not import pair %d', s);
    end
end

has_data = any(wuc_md,2) & any(cosim,2) & any(~isnan(wuc_md),2) & any(~isnan(cosim),2) & any(~isnan(wi_cosim),2);
cosim = cosim(has_data,:);
wuc_md = wuc_md(has_data,:);
tsnr = tsnr(has_data,:);
wi_cosim = wi_cosim(has_data,:);

good_rois = find(10*sum(isnan(wuc_md),1) < size(wuc_md,1));

confounds = {tsnr, wi_cosim};

%% plot topographic and geometric similarities

B = mean(cosim(:,good_rois));
cmaprange = prctile(B,[2.5,97.5]);
cmaprange(1) = eps;
T = {'Between subject topographic similarity',['(RSN Spatial cos\theta, ',sprintf('N=%d',sum(~all(cosim == 0,2))), ')'],''};
plot_to_brain(B, good_rois, cmaprange, T, fs+2);
exportgraphics(gcf,'panels/topographic_similarity.png','ContentType','image','Resolution',300);

B = nanmean(wuc_md(:,good_rois));
cmaprange = prctile(B,[2.5,97.5]);
cmaprange(1) = eps;
T = {'Between subject geometric similarity',['(RSN WUC, ',sprintf('N=%d',sum(~all(wuc_md == 0,2))), ')'],''};
plot_to_brain(B, good_rois, cmaprange, T, fs+2);
exportgraphics(gcf,'panels/geometric_similarity.png','ContentType','image','Resolution',300);

%% compute similarity of geometry and topography
% We use these statistics in the main text of the results
sid_ind = repmat(1:size(wuc_md,1)',1,length(good_rois));
roi_ind = kron(helmertCoding(1:length(good_rois)),ones(size(wuc_md,1),1));
nanzscore = @(x1)((x1 - nanmean(x1,2))./nanstd(x1,0,2));
zwuc = nanzscore(wuc_md(:,good_rois));
cosim_ctx = nanzscore(cosim(:,good_rois));
ztsnr = nanzscore(tsnr(:,good_rois));
zwi_cosim = nanzscore(wi_cosim(:,good_rois));

% model (within participant) standardized effect of wuc on cosim while
% controlling for test-retest reliability (wi_cosim), tSNR, and fixed ROI
% effects. Model participant specific wuc, test-retest reliability and tsnr
% random effects along with a random subject specific intercept, but no
% group level intercept (because mean cosim across group is 0 due to
% z-scoring). Helmert coding of roi_ind produces coefficients that are
% averaged across ROIs.
mdl = fitlmematrix([zwuc(:), ztsnr(:), zwi_cosim(:), roi_ind], ...
    cosim_ctx(:), ...
    [zwuc(:), ztsnr(:), zwi_cosim(:), ones(length(sid_ind(:)),1)], ...
    sid_ind(:));
[~,~,STATS] = fixedEffects(mdl,'dfmethod','satterthwaite');

disp('Dependence of topography on geometry, brainwide:')
disp(STATS)

%% Plot contrast of relative similarities
[cosim_ctx, zwuc] = deal(nan(size(wuc_md)));

d = nan(size(zwuc));
for i = 1:size(zwuc,1)
    this_wuc = wuc_md(i,:)';
    this_cosim = cosim(i,:)';

    good_roi = ~isnan(this_wuc) & ~isnan(this_cosim);

    d(i,good_roi) = [zscore(this_wuc(good_roi)) - zscore(this_cosim(good_roi))];
end

B = nanmean(d(:,good_rois),1);
cmaprange = prctile(B,[2.5,97.5]);

T = {'Difference in relative geometric similarity','and relative topographic similarity',['(RSN: WUC_{std} - cos\theta_{std}, ',sprintf('N = %d)',size(d,1))]};
plot_to_brain(B, good_rois, cmaprange, T, fs+1);
exportgraphics(gcf,'panels/relative_dif_wuc_cosim.png','ContentType','image','Resolution',300);

%% estimate neuromap associations

% note that the margulies gradient map from the neuromaps database
% differs slightly from the one shared by margulies et al. To the naked
% eye it looks like the official margulies map is smoother than the 
% neuromaps version, and there are slight numerical differences. We 
% use the neuromaps version here for consistency across spatial models. 
% If we use Margulies' version it strengths the effect of polysynaptic 
% depth on implementations of common representations, so using the 
% neuromaps version is conservative with respect to our conclusions.
maps = [{'abagen', 'genepc1','GenePC1','(-)','(+)'};...
    {'hill2010', 'evoexp','EvoExp1','old','new'};...
    {'xu2020', 'evoexp','EvoExp2','old','new'};...
    {'xu2020', 'FChomology','FCHomology','diff','same'}; ...
    {'reardon2018', 'scalinghcp','DevExp1','early','late'};...
    {'hill2010', 'devexp','DevExp2','early','late'};...
    {'neurosynth', 'cogpc1','CogPC1','(-)','(+)'};...
    {'hcps1200', 'myelinmap','Myelin','min','max'};...
    {'hcps1200', 'thickness','Thickness','thin','thick'};...
    {'margulies2016', 'fcgradient01','NetHierarchy','uni.','trans.'};...
    {'raichle', 'cbf', 'CBF1','low','high'};...
    {'satterthwaite2014', 'meancbf', 'CBF2','low','high'}];

mapvals = dir('../../resources/neuromaps/canlab2024_parcel_vals/');
mapvals(1:2) = []; % remove '.' and '..' refs

keep = zeros(length(mapvals),1);
for i = 1:length(maps)
    keep(i) = find(contains({mapvals.name}, maps(i,1)) & contains({mapvals.name}, maps(i,2)));
end
keep(keep==0) = [];
mapvals = mapvals(keep);

vals = zeros(358, length(mapvals));
mapname = {};
[wucD, wucDStd, cosimD, cosimDStd, ...
    wucb, wucp, wucstd, cosimb, cosimp, cosimstd] = deal(zeros(length(mapvals),1));
[wucb_CI, wucstd_CI, cosim_CI, cosimstd_CI] = deal(zeros(length(mapvals),2));
[mainInt, mainStd, mainD, mainDStd] = deal(zeros(length(mapvals),8));
[mainInt_CI, mainStd_CI] = deal(zeros(length(mapvals),8,2));
for i = 1:length(mapvals)
    mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');

    vals(:,i) = csvread(fullfile(mapvals(i).folder, mapvals(i).name));
    
    these_good_rois = good_rois;
    these_good_rois(these_good_rois > 358) = [];
    if contains(mapname{i},{'hill2010'})
        these_good_rois(these_good_rois < 180) = [];
    end

    fprintf('Evaluating %s\n', mapname{i})

    map_val = vals(these_good_rois, i);

    map_mu = mean(map_val);
    map_sd = std(map_val);
    
    map_val = (map_val - map_mu)./map_sd;
    
    n_tests = size(vals,2);
    sidak_95CI = [(1-0.95^(1/n_tests))/2,1-(1-0.95^(1/n_tests))/2];
    
    confounds_good_rois = cell(1,length(confounds));
    for j = 1:length(confounds)
        confounds_good_rois{j} = confounds{j}(:,these_good_rois)';
    end

    %eval wuc_md
    obs_val = atanh(wuc_md(:, these_good_rois))';
    [wucb(i), ~, bootstat, wucstd(i), ~, bootstat_std, wucD(i), wucDStd(i)] = neuromaps_corr_fx(obs_val, ...
        map_val, confounds_good_rois);
    wucb_CI(i,:) = prctile(bootstat, 100*sidak_95CI);
    wucstd_CI(i,:) = prctile(bootstat_std, 100*sidak_95CI);

    %eval cosim
    obs_val = atanh(cosim(:,these_good_rois))';
    [cosimb(i), ~, bootstat, cosimstd(i), ~, bootstat_std, cosimD(i), cosimDStd(i)] = neuromaps_corr_fx(obs_val, ...
        map_val, confounds_good_rois);
    cosim_CI(i,:) = prctile(bootstat, 100*sidak_95CI);
    cosimstd_CI(i,:) = prctile(bootstat_std, 100*sidak_95CI);

    %eval cosim & wuc interaction
    obs_val1 = atanh(wuc_md(:, these_good_rois))';
    obs_val2 = atanh(cosim(:,these_good_rois))';
    [mainInt(i,:), mainInt_CI(i,:,:), bootstat, mainStd(i,:), ~, bootstat_std, mainD(i,:), mainDStd(i,:)] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, ...
        map_val, confounds_good_rois);
    mainInt_CI(i,:,:) = prctile(bootstat, 100*sidak_95CI)';
    mainStd_CI(i,:,:) = prctile(bootstat_std, 100*sidak_95CI)';
end


%% plot neurmap associations
% The tables we print below contribute to the tables in the manuscript

abr_mapname = maps(:,3);

disp('Effects (betas):')
cosimb_str = {};
for i = 1:length(cosimb)
    cosimb_str{i} = sprintf('%0.3f±%0.3f',cosimb(i), mean([cosim_CI(i,2) - cosimb(i), cosimb(i) - cosim_CI(i,1)],2));
end
wucb_str = {};
for i = 1:length(wucb)
    wucb_str{i} = sprintf('%0.3f±%0.3f',wucb(i), mean([wucb_CI(i,2) - wucb(i), wucb(i) - wucb_CI(i,1)],2));
end
int_str = {};
for i = 1:length(wucb)
    int_str{i} = sprintf('%0.3f±%0.3f',mainStd(i,3), mean([mainStd_CI(i,3,2) - mainStd(i,3), mainStd(i,3) - mainStd_CI(i,3,1)],2));
end
disp(table(cosimb_str', wucb_str', int_str', ...
    'VariableNames', {'Topo', 'Geo', 'zGeo-zTopo'}))

disp('Cohens Ds:');
disp(table(cosimD(:), wucD(:), mainDStd(:,3), ...
    'VariableNames',{'cosim', 'wuc', 'zGeo-zTopo'},...
    'RowNames', abr_mapname));

buffer = 3; % how many times do we scale the x axis to fit anchor labels on either end?

figure(4);
clf
t5 = tiledlayout(1,3,'Padding','none','TileSpacing','compact');
ax1 = nexttile();
cla
hold on;
pos_err = cosim_CI(:,2) - cosimb;
neg_err = cosimb - cosim_CI(:,1);
errorbar(cosimb, 1:length(cosimb), neg_err, pos_err, '.', 'horizontal', 'capsize', 0, 'color', dc_color_light)
plot(cosimb, 1:length(cosimb),'o','MarkerFaceColor',dc_color_light,'color', dc_color);

set(gca,'YTick',1:length(mapname), 'YTickLabels', abr_mapname,'FontSize',fs-3, 'YDir', 'rev','YGrid','on');
ylim([0.5,length(mapname)+0.5])
title('Topography (cos\theta)','fontweight','normal', 'fontsize',fs);
xlabel('Mean \beta', 'fontsize',fs)
box off
xl = max(max(abs(cosim_CI))).*[-1,1];
xlim([buffer*xl(1), xl(2)*buffer])
yl = ylim;
annot = cell(length(maps),2);
for i = 1:length(maps)
    try, delete(annot{i,1}); end
    annot{i,1} = text(buffer*xl(1)*0.95,i,maps{i,4},'FontSize',fs-4);
end
for i = 1:length(maps)
    try, delete(annot{i,2}); end
    annot{i,2} = text(buffer*xl(2)*0.95,i,maps{i,5},'FontSize',fs-4,'HorizontalAlignment','right');
end

sig = cosim_CI(:,1).*cosim_CI(:,2) > 0;
if any(sig)
    x = sign(cosimb(sig)).*(pos_err(sig) + abs(cosimb(sig)) + xl(2) * 0.2);
    text(x, find(sig),'*','HorizontalAlignment','center');
end
xline(0, 'color', [0.5,0.5,0.5]);

ax2 = nexttile();
cla
hold on;
pos_err = wucb_CI(:,2) - wucb;
neg_err = wucb - wucb_CI(:,1);
errorbar(wucb, 1:length(wucb), neg_err, pos_err, '.', 'horizontal', 'capsize', 0, 'color', dc_color_light)
plot(wucb, 1:length(wucb), '^','MarkerFaceColor',dc_color_light,'color', dc_color);

set(gca,'YTick',1:length(mapname), 'YTickLabels', [],'FontSize',fs-3, 'YDir', 'rev','YGrid','on');
ylim([0.5,length(mapname)+0.5])
title('Geometry (WUC)','fontweight','normal', 'fontsize',fs);
xlabel('Mean \beta', 'fontsize',fs)
box off
xl = max(max(abs(wucb_CI))).*[-1,1];
xlim([buffer*xl(1), xl(2)*buffer])
yl = ylim;
annot = cell(length(maps),2);
for i = 1:length(maps)
    try, delete(annot{i,1}); end
    annot{i,1} = text(buffer*xl(1)*0.95,i,maps{i,4},'FontSize',fs-4);
end
for i = 1:length(maps)
    try, delete(annot{i,2}); end
    annot{i,2} = text(buffer*xl(2)*0.95,i,maps{i,5},'FontSize',fs-4,'HorizontalAlignment','right');
end

sig = wucb_CI(:,1).*wucb_CI(:,2) > 0;
if any(sig)
    x = sign(wucb(sig)).*(pos_err(sig) + abs(wucb(sig)) + xl(2) * 0.15);
    text(x, find(sig),'*','HorizontalAlignment','center','VerticalAlignment','middle');
end
xline(0, 'color', [0.5,0.5,0.5]);

ax3 = nexttile();
cla
hold on;
pos_err = mainStd_CI(:,3,2) - mainStd(:,3);
neg_err = mainStd(:,3,1) - mainStd_CI(:,3,1);
errorbar(mainStd(:,3), 1:size(mainStd,1), neg_err, pos_err, '.', 'horizontal', 'capsize', 0, 'color', dc_color_light)
plot(mainStd(:,3),1:size(mainStd,1),'s','MarkerFaceColor',dc_color_light,'color', dc_color);

set(gca,'YTick',1:length(mapname), 'YTickLabels', [],'FontSize',fs-3, 'YDir', 'rev','YGrid','on');
ylim([0.5,length(mapname)+0.5])

title('zGeo - zTopo','fontweight','normal', 'fontsize',fs);
xlabel('Mean \Delta\beta_{std}', 'fontsize',fs)
box off
xl = max(max(abs(squeeze(mainStd_CI(:,3,:)))))'.*[-1,1];
xlim([buffer*xl(1), xl(2)*buffer])
yl = ylim;
annot = cell(length(maps),2);
for i = 1:length(maps)
    try, delete(annot{i,1}); end
    annot{i,1} = text(buffer*xl(1)*0.95,i,maps{i,4},'FontSize',fs-4);
end
for i = 1:length(maps)
    try, delete(annot{i,2}); end
    annot{i,2} = text(buffer*xl(2)*0.95,i,maps{i,5},'FontSize',fs-4,'HorizontalAlignment','right');
end

sig = mainStd_CI(:,3,1).*mainStd_CI(:,3,2) > 0;
if any(sig)
    x = sign(mainStd(sig,3)).*(pos_err(sig) + abs(mainStd(sig,3)) + xl(2) * 0.25);
    text(x, find(sig),'*','HorizontalAlignment','center');
end
xline(0, 'color', [0.5,0.5,0.5]);


pos = get(gcf,'Position');
set(gcf,'Position', [pos(1:2), 600,285]);

sgtitle({'Specific factors are associated with','flexible implementation of shared representations'},'fontweight','bold','fontsize',fs+1)

exportgraphics(gcf,'panels/gradient_barplots_nostd.png','ContentType','image','Resolution',300);



%% plot RDM & topographic correlations vs. margulies 1, MSMAll, unstandardized
% find margulies gradient 1
map_ind = find(contains(mapname,'fcgradient01'));

cmap = zeros(length(atlas_labels),3);
for i = 1:length(atlas_labels)
    cmap(i,:) = atlas_labels(i).rgba(1:3);
end

good_rois_ctx = good_rois;
good_rois_ctx(good_rois_ctx > 358) = [];

wuc_md_ctx = wuc_md(:, good_rois_ctx);
cosim_ctx = cosim(:, good_rois_ctx);
good_grad_roi = vals(ismember(1:358, good_rois_ctx), map_ind);

figure;
clf
ax1 = subplot(1,2,1);
hold on;
ax2 = subplot(1,2,2);
hold on;
for i = 1:size(wuc_md_ctx,2)
    color = cmap(good_rois_ctx(i),:);
    if good_rois_ctx(i) < 358 
        s1 = plot(ax2, good_grad_roi(i), mean(wuc_md_ctx(:,i)), '^', 'color', color);
        s2 = plot(ax1, good_grad_roi(i), mean(cosim_ctx(:,i)), 'o', 'color', color);
    else
        continue;
    end
end
xl = [min(good_grad_roi), max(good_grad_roi)];
yl = [min([ylim(ax2),ylim(ax1)]), max([ylim(ax2),ylim(ax1)])];
xlim(ax2,xl);
ylim(ax2,yl);
xlim(ax1,xl);
ylim(ax1,yl);
set(ax2, 'YGrid', 'on', 'box', 'off', 'fontsize', fs,...
    'XTick', prctile(good_grad_roi, [10,90]), 'XTickLabels', {'Uni', 'Trans'}, 'TickLength', [0,0.025],'XTickLabelRotation',0)
ylabel(ax2,{'Mean regional','geometric similarity', '(WUC of RDMs)'});
set(ax1, 'YGrid', 'on', 'box', 'off', 'fontsize', fs, ...
    'XTick', prctile(good_grad_roi, [10,90]), 'XTickLabels', {'Uni', 'Trans'}, 'TickLength', [0,0.025],'XTickLabelRotation',0);
ylabel(ax1,{'Mean regional','topographic similarity', '(cos\theta of spatial patterns)'});

% fit group level mean slope
b_topo = zeros(size(cosim_ctx,1),2);
b_geo = zeros(size(wuc_md_ctx,1),2);
for i = 1:size(cosim_ctx,1)
    x_geo = wuc_md_ctx(i,:)';
    x_topo = cosim_ctx(i,:)';
    y = good_grad_roi;

    m_rep = fitlm(y, x_geo);
    b_geo(i,:) = m_rep.Coefficients.Estimate;
    m_topo = fitlm(y, x_topo);
    b_topo(i,:) = m_topo.Coefficients.Estimate;
end
b_geo = mean(b_geo);
b_topo = mean(b_topo);

y = b_geo(2)*xlim' + b_geo(1);
l = plot(ax2, xlim',y,'-','color',dc_color);
l.LineWidth = 2;

y = b_topo(2)*xlim' + b_topo(1);
l = plot(ax1, xlim',y,'-','color',dc_color);
l.LineWidth = 2;

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),642,345]);

title(ax2, {'Representations converge','across cortical hierarchy'},'FontWeight','normal');
title(ax1, {'Topographies diverge','along cortical hierarchy'},'FontWeight','normal')

sgtitle({'Transmodal representations are similar','but implemented more idiosyncratically'},'FontWeight','bold','fontsize',fs+1)


% add ROI legend
a1 = axes();
a1.Position = [0.33,0.56,0.15,0.15];
a1.Visible = 'off';

overlay = canlab_get_underlay_image;
o2 = fmridisplay('overlay', which(overlay));
o2 = surface(o2, 'axes', a1, 'direction', 'hcp inflated left', 'orientation', 'lateral');     

this_map = regexp(mapname{map_ind},'(.*)-(.*)','tokens');
maps = dir('../../resources/neuromaps/');
file_ind = find(contains({maps.name}, this_map{1}{1}) & contains({maps.name}, this_map{1}{2}) & contains({maps.name}, 'hemi-L'));

grayord_surf_L = gifti(fullfile(maps(file_ind).folder, maps(file_ind).name));
plot_to_surf(grayord_surf_L.cdata,o2.surface{1}.object_handle);

exportgraphics(gcf,'panels/margulies_01.png','ContentType','image','Resolution',300);
