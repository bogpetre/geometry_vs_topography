close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));
addpath(genpath(fullfile(config.matlab_libraries.npm)));

addpath('../../matlab_libraries');
addpath('../../resources/neuromaps');

fs=config.matlab_disp_scheme.fontsize;

f = figure;
cm = colormap(f,'hot');
close(f)

colors = config.matlab_disp_scheme.color_main;
colors_light = config.matlab_disp_scheme.color_light;

noise='whitened';
%noise='standardized';

data_root = '../../derivatives_bak/restingstate/hcp25';
seed = 1;

%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii_data = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii_data.cortex_left, atlas_cii_data.cortex_right, atlas_cii_data.volumes])) - 1;

%% import between subject similarity measures for unrelated individuals
sid = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);

tsnr = zeros(height(sid), n_roi);
wi_cosim = nan(height(sid), n_roi);
for s = 1:height(sid)
    try
        tsnr1 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/tsnr.csv',sid.Var1(s)));
        tsnr2 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/tsnr.csv',sid.Var2(s)));
        tsnr(s,:) = mean([tsnr1, tsnr2],2);

        wi_cosim1 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/%s_betas/%s_similarity.csv',sid.Var1(s),noise,noise),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/%s_betas/%s_similarity.csv',sid.Var1(s),noise,noise),'FileType','text');
        wi_cosim(s,:) = mean(mean(cat(3,wi_cosim1, wi_cosim2),3));
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = readmatrix(sprintf('../../derivatives_bak/restingstate/hcp25/bsc_null_%d/%s_betas/cosine/%d_v_%d_wuc.tsv',seed, noise, sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',',');
    catch
        warning('Could not import pair %d', s);
    end
end

n_subj = height(sid);

left_ctx_roi = unique(atlas_cii_data.cortex_left);
right_ctx_roi = unique(atlas_cii_data.cortex_right);
subctx_roi = unique(atlas_cii_data.volumes);
left_ctx_roi(left_ctx_roi == 0) = [];
right_ctx_roi(right_ctx_roi == 0) = [];
subctx_roi(subctx_roi == 0) = [];

% load topographies
topos1 = cell(n_roi,n_subj);
good_topo = true(1,n_subj);
parfor i = 1:height(sid)
    try
        contrast_file = dir([data_root, '/results/', ...
                sprintf('%d/%s_betas/cifti_math_results.dscalar.nii', ...
                sid.Var1(i), noise)]);

        these_contrasts = get_cifti_data(fullfile(contrast_file.folder, contrast_file.name));
        
        for r = 1:n_roi
            if ismember(r, left_ctx_roi)
                struct = 'cortex_left';
            elseif ismember(r, right_ctx_roi)
                struct = 'cortex_right';
            elseif ismember(r, subctx_roi)
                struct = 'volumes';
            else
                error('Could not identify structure for region %d',r);
            end

            assert(size(atlas_cii_data.(struct),2) == size(these_contrasts.(struct),2));

            ind = atlas_cii_data.(struct) == r;
            topos1{r,i} = these_contrasts.(struct)(:,ind);
            s = zeros(size(topos1{r,i},1),1);
            for j = 1:size(topos1{r,i},1)
                s(j) = norm(topos1{r,i}(j,:));
            end
            topos1{r,i} = topos1{r,i}./s;
        end
    catch
        good_topo(i) = false;
        warning('Failed to import topographies for subject %d', sid.Var1(i));
    end
end

topos2 = cell(n_roi,n_subj);
good_topo = true(1,n_subj);
parfor i = 1:height(sid)
    try
        contrast_file = dir([data_root, '/results/', ...
                sprintf('%d/%s_betas/cifti_math_results.dscalar.nii', ...
                sid.Var2(i), noise)]);

        these_contrasts = get_cifti_data(fullfile(contrast_file.folder, contrast_file.name));
        
        for r = 1:n_roi
            if ismember(r, left_ctx_roi)
                struct = 'cortex_left';
            elseif ismember(r, right_ctx_roi)
                struct = 'cortex_right';
            elseif ismember(r, subctx_roi)
                struct = 'volumes';
            else
                error('Could not identify structure for region %d',r);
            end

            assert(size(atlas_cii_data.(struct),2) == size(these_contrasts.(struct),2));

            ind = atlas_cii_data.(struct) == r;
            topos2{r,i} = these_contrasts.(struct)(:,ind);
            s = zeros(size(topos2{r,i},1),1);
            for j = 1:size(topos2{r,i},1)
                s(j) = norm(topos2{r,i}(j,:));
            end
            topos2{r,i} = topos2{r,i}./s;
        end
    catch
        good_topo(i) = false;
        warning('Failed to import topographies for subject %d', sid.Var2(i));
    end
end

% compute similarities
rand_ordering = csvread([data_root, sprintf('/bsc_null_%d/shuffle_ind.csv',seed)]);
rand_ordering = rand_ordering+1; % from python to matlab indexing

cosim = nan(n_subj, n_roi);
for r = 1:n_roi
    fprintf('Compute topographic similarity for region %d\n',r);
    for i = 1:n_subj
        if ~good_topo(i)
            continue;
        end

        % balance with oversampling rather than weights
        X = topos1{r,i};
        Y = topos2{r,i};

        X = X(rand_ordering,:);

        cosim(i,r) = mean(diag(X*Y'));
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

% confound correction
cosim_corr = nan(size(cosim));
for i = 1:size(cosim,1)
    Y = cosim(i,good_rois)';
    X = [];
    for j = 1:length(confounds)
        X = [X, confounds{j}(i,good_rois)'];
    end
    X = X - nanmean(X);
    X = [ones(length(Y),1), X];
    good_roi = ~isnan(Y) & all(~isnan(X),2);
    X = X(good_roi,:);
    Y = Y(good_roi);
    B = (X'*X)\X'*Y;
    cosim_corr(i,good_roi) = (Y - X*B) + B(1);
end

wuc_corr = nan(size(wuc_md));
for i = 1:size(wuc_md,1)
    Y = wuc_md(i,good_rois)';
    X = [];
    for j = 1:length(confounds)
        X = [confounds{j}(i,good_rois)'];
    end
    X = X - nanmean(X);
    X = [ones(length(Y),1), X];
    good_roi = ~isnan(Y) & all(~isnan(X),2);
    X = X(good_roi,:);
    Y = Y(good_roi);
    B = (X'*X)\X'*Y;
    wuc_corr(i,good_roi) = (Y - X*B) + B(1);
end

B = mean(cosim_corr(:,good_rois));
cmaprange = prctile(B,[2.5,97.5]);
cmaprange(1) = eps;
T = {'Between subject topographic similarity',['(RSN Spatial cos\theta | tSNR, rel., ',sprintf('N=%d',sum(~all(cosim == 0,2))), ')'],''};
plot_to_brain(B, good_rois, cmaprange, T, fs+2);
exportgraphics(gcf,sprintf('panels_%s/topographic_similarity.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('mean_topo_similarity_%s.dscalar.nii',noise));

B = nanmean(wuc_corr(:,good_rois));
cmaprange = prctile(B,[2.5,97.5]);
cmaprange(1) = eps;
T = {'Between subject geometric similarity',['(RSN WUC | tSNR, rel., ',sprintf('N=%d',sum(~all(wuc_md == 0,2))), ')'],''};
plot_to_brain(B, good_rois, cmaprange, T, fs+2);
exportgraphics(gcf,sprintf('panels_%s/geometric_similarity.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('mean_geom_similarity_%s.dscalar.nii',noise));

%% compute similarity of geometry and topography
% We use these statistics in the main text of the results, but this
% significantly slows things down and is best commented out in most cases.
%{
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
%}
%% Plot contrast of relative similarities
[cosim_ctx, zwuc] = deal(nan(size(wuc_md)));

d = nan(size(zwuc));
for i = 1:size(zwuc,1)
    this_wuc = wuc_corr(i,:)';
    this_cosim = cosim_corr(i,:)';

    good_roi = ~isnan(this_wuc) & ~isnan(this_cosim);

    d(i,good_roi) = [zscore(this_wuc(good_roi)) - zscore(this_cosim(good_roi))];
end

B = nanmean(d(:,good_rois),1);
cmaprange = prctile(B,[2.5,97.5]);

T = {'Difference in relative geometric similarity','and relative topographic similarity',['(RSN: WUC_{std} - cos\theta_{std} | tSNR, rel., ',sprintf('N = %d)',size(d,1))]};
plot_to_brain(B, good_rois, cmaprange, T, fs+1);
exportgraphics(gcf,sprintf('panels_%s/relative_dif_wuc_cosim.png',noise),'ContentType','image','Resolution',300);for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('mean_zgeom_vs_ztopo_moderation_%s.dscalar.nii',noise));

%% estimate neuromap associations

% note that the margulies gradient map from the neuromaps database
% differs slightly from the one shared by margulies et al. To the naked
% eye it looks like the official margulies map is smoother than the 
% neuromaps version, and there are slight numerical differences. We 
% use the neuromaps version here for consistency across spatial models. 
% If we use Margulies' version it strengths the effect of polysynaptic 
% depth on implementations of common representations, so using the 
% neuromaps version is conservative with respect to our conclusions.
maps = [{'hill2010', 'evoexp','EvoExp1','old','new',1};...
    {'xu2020', 'evoexp','EvoExp2','old','new',1};...
    {'xu2020', 'FChomology','FCHomology','diff','same',1}; ...
    {'reardon2018', 'scalinghcp','DevExp1','early','late',2};...
    {'hill2010', 'devexp','DevExp2','early','late',2};...
    {'hcps1200', 'myelinmap','Myelin','min','max',3};...
    {'hcps1200', 'thickness','Thickness','thin','thick',3};...
    {'margulies2016', 'fcgradient01','NetHierarchy','uni.','trans.',3};...
    {'abagen', 'genepc1','GenePC1','(-)','(+)',4};...
    {'neurosynth', 'cogpc1','CogPC1','(-)','(+)',4};...
    {'raichle', 'cbf', 'CBF1','low','high',4};...
    {'satterthwaite2014', 'meancbf', 'CBF2','low','high',4}];

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
[wucb, wucp, wucD, cosimb, cosimp, cosimD] = deal(zeros(length(mapvals),1));
[wucb_CI, cosim_CI] = deal(zeros(length(mapvals),2));
[mainStd, mainStdP, mainDStd] = deal(zeros(length(mapvals),6));
mainStd_CI = zeros(length(mapvals),6,2);
for i = 1:length(mapvals)
    mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');

    vals(:,i) = csvread(fullfile(mapvals(i).folder, mapvals(i).name));
    
    these_good_rois = good_rois;
    these_good_rois(these_good_rois > 358) = [];
    if contains(mapname{i},{'hill2010'})
        these_good_rois(these_good_rois < 180) = [];
    end

    fprintf('Evaluating %s\n', mapname{i})

    randgrad = csvread(fullfile('../../resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
    randgrad(randgrad == 0) = nan; % medial wall

    map_val = vals(these_good_rois, i);
    perm_map = randgrad(these_good_rois,:);

    map_mu = mean(map_val);
    map_sd = std(map_val);
    
    map_val = (map_val - map_mu)./map_sd;
    perm_map = (perm_map - nanmean(perm_map))./map_sd;
        
    confounds_good_rois = cell(1,length(confounds));
    for j = 1:length(confounds)
        confounds_good_rois{j} = confounds{j}(:,these_good_rois)';
    end

    %eval wuc_md
    obs_val = atanh(wuc_md(:, these_good_rois))';
    [wucb(i), wucb_CI(i,:), wucp(i) wucD(i)] = neuromaps_corr_fx(obs_val, ...
        map_val, perm_map, confounds_good_rois);

    %eval cosim
    obs_val = atanh(cosim(:,these_good_rois))';
    [cosimb(i), cosim_CI(i,:), cosimp(i), cosimD(i)] = neuromaps_corr_fx(obs_val, ...
        map_val, perm_map, confounds_good_rois);

    %eval cosim & wuc interaction
    obs_val1 = atanh(wuc_md(:, these_good_rois))';
    obs_val2 = atanh(cosim(:,these_good_rois))';
    [mainStd(i,:), mainStd_CI(i,:,:), mainStdP(i,:), mainDStd(i,:)] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, ...
        map_val, perm_map, confounds_good_rois);
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
    int_str{i} = sprintf('%0.3f±%0.3f',mainStd(i,2), mean([mainStd_CI(i,2,2) - mainStd(i,2), mainStd(i,2) - mainStd_CI(i,2,1)],2));
end
disp(table(cosimb_str', wucb_str', int_str', ...
    'VariableNames', {'Topo', 'Geo', 'zGeo-zTopo'}))

disp('Cohens Ds:');
disp(table(cosimD(:), wucD(:), mainDStd(:,2), ...
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
for i = 1:length(maps)
    hold on;
    errorbar(cosimb(i), i, neg_err(i), pos_err(i), '.', 'horizontal', ...
        'capsize', 0, 'color', colors_light(maps{i,6},:), 'linewidth', 2)
    plot(cosimb(i), i,'o','MarkerFaceColor', colors_light(maps{i,6},:),'color', colors(maps{i,6},:));
end

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

pthresh = FDR(cosimp, 0.05);
if any(pthresh)
    sig = cosimp <= pthresh;
    x = sign(cosimb(sig)).*(pos_err(sig) + abs(cosimb(sig)) + xl(2) * 0.25);
    text(x, find(sig),'*','HorizontalAlignment','center');
end
xline(0, 'color', [0.5,0.5,0.5]);

ax2 = nexttile();
cla
hold on;
pos_err = wucb_CI(:,2) - wucb;
neg_err = wucb - wucb_CI(:,1);
for i = 1:length(maps)
    hold on;
    errorbar(wucb(i), i, neg_err(i), pos_err(i), '.', 'horizontal', ...
        'capsize', 0, 'color', colors_light(maps{i,6},:), 'linewidth', 2)
    plot(wucb(i), i,'o','MarkerFaceColor', colors_light(maps{i,6},:),'color', colors(maps{i,6},:));
end

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

pthresh = FDR(wucp, 0.05);
if any(pthresh)
    sig = wucp <= pthresh;
    x = sign(wucb(sig)).*(pos_err(sig) + abs(wucb(sig)) + xl(2) * 0.25);
    text(x, find(sig),'*','HorizontalAlignment','center','VerticalAlignment','middle');
end
xline(0, 'color', [0.5,0.5,0.5]);

ax3 = nexttile();
cla
hold on;
pos_err = mainStd_CI(:,2,2) - mainStd(:,2);
neg_err = mainStd(:,2) - mainStd_CI(:,2,1);
for i = 1:length(maps)
    hold on;
    errorbar(mainStd(i,2), i, neg_err(i), pos_err(i), '.', 'horizontal', ...
        'capsize', 0, 'color', colors_light(maps{i,6},:), 'linewidth', 2)
    plot(mainStd(i,2), i,'o','MarkerFaceColor', colors_light(maps{i,6},:),'color', colors(maps{i,6},:));
end

set(gca,'YTick',1:length(mapname), 'YTickLabels', [],'FontSize',fs-3, 'YDir', 'rev','YGrid','on');
ylim([0.5,length(mapname)+0.5])

title('zGeo - zTopo','fontweight','normal', 'fontsize',fs);
xlabel('Mean \Delta\beta_{std}', 'fontsize',fs)
box off
xl = max(max(abs(squeeze(mainStd_CI(:,2,:)))))'.*[-1,1];
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

pthresh = FDR(mainStdP(:,2),0.05);
if any(pthresh)
    sig = mainStdP(:,2) <= pthresh;
    x = sign(mainStd(sig,2)).*(pos_err(sig) + abs(mainStd(sig,2)) + xl(2) * 0.25);
    text(x, find(sig),'*','HorizontalAlignment','center');
end
xline(0, 'color', [0.5,0.5,0.5]);


pos = get(gcf,'Position');
set(gcf,'Position', [pos(1:2), 600,285]);

sgtitle({'Specific factors dissociate resting-state network','topographic and representational similarity'},'fontweight','bold','fontsize',fs+1)

exportgraphics(gcf,sprintf('panels_%s/gradient_barplots_nostd.png',noise),'ContentType','image','Resolution',300);



%% plot RDM & topographic correlations vs. margulies 1, MSMAll, unstandardized
% find margulies gradient 1
map_ind = find(contains(mapname,'fcgradient01'));

cmap = zeros(length(atlas_labels),3);
for i = 1:length(atlas_labels)
    cmap(i,:) = atlas_labels(i).rgba(1:3);
end

good_rois_ctx = good_rois;
good_rois_ctx(good_rois_ctx > 358) = [];

wuc_md_ctx = wuc_corr(:, good_rois_ctx);
cosim_ctx = cosim_corr(:, good_rois_ctx);
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
ylabel(ax2,{'Mean regional','geometric similarity', '(WUC of RDMs | tSNR, rel.)'});
set(ax1, 'YGrid', 'on', 'box', 'off', 'fontsize', fs, ...
    'XTick', prctile(good_grad_roi, [10,90]), 'XTickLabels', {'Uni', 'Trans'}, 'TickLength', [0,0.025],'XTickLabelRotation',0);
ylabel(ax1,{'Mean regional','topographic similarity', '(cos\theta of spatial patterns | tSNR, rel.)'});

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
l = plot(ax2, xlim',y,'-','color',colors(3,:));
l.LineWidth = 2;

y = b_topo(2)*xlim' + b_topo(1);
l = plot(ax1, xlim',y,'-','color',colors(3,:));
l.LineWidth = 2;

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),642,345]);

title(ax2, {'Representations converge','across cortical hierarchy'},'FontWeight','normal');
title(ax1, {'Topographic trends','along cortical hierarchy'},'FontWeight','normal')

sgtitle({'Transmodal representations are most similar','but also implemented most idiosyncratically'},'FontWeight','bold','fontsize',fs+1)


% add margulies map
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


% add ROI legend

atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall

a2 = axes();
a2.Position = [0.33,0.40,0.15,0.15];
a2.Visible = 'off';

overlay = canlab_get_underlay_image;
o3 = fmridisplay('overlay', which(overlay));
o3 = surface(o3, 'axes', a2, 'direction', 'hcp inflated left', 'orientation', 'lateral');     

atlas_cii = get_cifti_data(config.canlab2024.path);
cdata = atlas_cii.cortex_left;
plot_to_surf(cdata',o3.surface{1}.object_handle, 'indexmap', 'colormap', [cmap(1:358,:); [0,0,0]]);

exportgraphics(gcf,sprintf('panels_%s/margulies_01.png',noise),'ContentType','image','Resolution',300);
