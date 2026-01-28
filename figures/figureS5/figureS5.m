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

%% import atlas in cifti space and get region names
atlas_cii = cifti_read('../../resources/Gordon_333Cort.32k.dlabel.nii');
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii_data = get_cifti_data('../../resources/Gordon_333Cort.32k.dlabel.nii');
n_roi = length(unique([atlas_cii_data.cortex_left, atlas_cii_data.cortex_right, atlas_cii_data.volumes])) - 1;

n_roi_ctx = length(unique([atlas_cii_data.cortex_left, atlas_cii_data.cortex_right])) - 1;
n_roi_lctx = length(unique([atlas_cii_data.cortex_left]))-1;

%% import between subject similarity measures for unrelated individuals
sid = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

% multiplying by this vector will balances conditions across tasks
balanced_mean_op = [1/7*repmat(1/2,1,10), 1/7*repmat(1/5,1,5), 1/7*repmat(1/8,1,8)];

tsnr = zeros(height(sid), n_roi);
wi_cosim = nan(height(sid), n_roi);
for s = 1:height(sid)
    try
        tsnr1 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm_gordon/results/%d/tsnr.csv',sid.Var1(s)));
        tsnr2 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm_gordon/results/%d/tsnr.csv',sid.Var2(s)));
        tsnr(s,:) = mean([tsnr1, tsnr2],2);

        wi_cosim1 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm_gordon/results/%d/all_tasks/%s_contrasts/%s_similarity.csv',sid.Var1(s),noise,noise),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm_gordon/results/%d/all_tasks/%s_contrasts/%s_similarity.csv',sid.Var2(s),noise,noise),'FileType','text');
        comb_cosim = mean(cat(3,wi_cosim1, wi_cosim2),3);
        wi_cosim(s,:) = balanced_mean_op*comb_cosim;
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = diag(readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm_gordon/bsc/%s_betas/cosine/%d_v_%d_wuc.tsv',noise,sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',','));
    catch
        warning('Could not import pair %d', s);
    end
end

cosim = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        cosim(s,:) = balanced_mean_op*dlmread(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm_gordon/bsc/%s_betas/cosine/%d_v_%d_cosim.tsv', noise, sid.Var1(s), sid.Var2(s)), '\t');
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
        X = [X, confounds{j}(i,good_rois)'];
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
T = {'Between subject topographic similarity',['(Task Spatial cos\theta | tSNR, rel., ',sprintf('N=%d',sum(~all(cosim_corr == 0,2))), ')'],''};
plot_to_brain(B, good_rois, cmaprange, T, fs+2, atlas_cii);
exportgraphics(gcf,sprintf('panels_%s/topographic_similarity.png',noise),'ContentType','image','Resolution',300);
new_cii_data = nan(size(atlas_cii.cdata));
for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('mean_topo_similarity_%s.dscalar.nii',noise));

B = mean(wuc_corr(:,good_rois));
cmaprange = prctile(B,[2.5,97.5]);
cmaprange(1) = eps;
T = {'Between subject geometric similarity',['(Task WUC | tSNR, rel., ',sprintf('N=%d',sum(~all(wuc_corr == 0,2))), ')'],''};
plot_to_brain(B, good_rois, cmaprange, T, fs+2, atlas_cii);
exportgraphics(gcf,sprintf('panels_%s/geometric_similarity.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('mean_geom_similarity_%s.dscalar.nii',noise));


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

T = {'Difference in relative geometric similarity','and relative topographic similarity',['(Task: WUC_{std} - cos\theta_{std} | tSNR, rel., ',sprintf('N = %d)',size(d,1))]};
plot_to_brain(B, good_rois, cmaprange, T, fs+1, atlas_cii);
exportgraphics(gcf,sprintf('panels_%s/relative_dif_wuc_cosim.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('mean_zgeom_gt_ztopo_moderation_%s.dscalar.nii',noise));


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

mapvals = dir('../../resources/neuromaps/gordon_parcel_vals/');
mapvals(1:2) = []; % remove '.' and '..' refs

keep = zeros(length(mapvals),1);
for i = 1:length(maps)
    keep(i) = find(contains({mapvals.name}, maps(i,1)) & contains({mapvals.name}, maps(i,2)));
end
keep(keep==0) = [];
mapvals = mapvals(keep);

vals = zeros(n_roi_ctx, length(mapvals));
mapname = {};
[wucb, wucp, wucD, cosimb, cosimp, cosimD] = deal(zeros(length(mapvals),1));
[wucb_CI, cosim_CI] = deal(zeros(length(mapvals),2));
[mainStd, mainStdP, mainDStd] = deal(zeros(length(mapvals),6));
mainStd_CI = zeros(length(mapvals),6,2);
for i = 1:length(mapvals)
    mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');

    vals(:,i) = csvread(fullfile(mapvals(i).folder, mapvals(i).name));
    
    these_good_rois = good_rois;
    these_good_rois(these_good_rois > n_roi_ctx) = [];
    if contains(mapname{i},{'hill2010'})
        these_good_rois(these_good_rois <= n_roi_lctx) = [];
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
    [cosimb(i), cosim_CI(i,:), cosimp(i), cosimD(i), sampling_var, perm_var] = neuromaps_corr_fx(obs_val, ...
        map_val, perm_map, confounds_good_rois);

    %eval cosim & wuc interaction
    obs_val1 = atanh(wuc_md(:, these_good_rois))';
    obs_val2 = atanh(cosim(:,these_good_rois))';
    [mainStd(i,:), mainStd_CI(i,:,:), mainStdP(i,:), mainDStd(i,:), ...
        sampling_var, perm_var] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, ...
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

pthresh = FDR(cosimp,0.05);
if any(pthresh)
    sig = cosimp <= pthresh;
    x = sign(cosimb(sig)).*(pos_err(sig) + abs(cosimb(sig)) + xl(2) * 0.2);
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

pthresh = FDR(wucp,0.05);
if any(pthresh)
    sig = wucp <= pthresh;
    x = sign(wucb(sig)).*(pos_err(sig) + abs(wucb(sig)) + xl(2) * 0.15);
    text(x, find(sig),'*','HorizontalAlignment','center','VerticalAlignment','middle');
end
xline(0, 'color', [0.5,0.5,0.5]);

ax3 = nexttile();
cla
hold on;
pos_err = mainStd_CI(:,2,2) - mainStd(:,2);
neg_err = mainStd(:,2,1) - mainStd_CI(:,2,1);
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

%sig = mainStd_CI(:,3,1).*mainStd_CI(:,3,2) > 0;
pthresh = FDR(mainStdP(:,2),0.05);
if any(pthresh)
    sig = mainStdP(:,2) <= pthresh;
    x = sign(mainStd(sig,2)).*(pos_err(sig) + abs(mainStd(sig,2)) + xl(2) * 0.25);
    text(x, find(sig),'*','HorizontalAlignment','center');
end
xline(0, 'color', [0.5,0.5,0.5]);


pos = get(gcf,'Position');
set(gcf,'Position', [pos(1:2), 600,285]);

sgtitle({'Specific factors dissociate','topographic and representational similarity'},'fontweight','bold','fontsize',fs+1)

exportgraphics(gcf,sprintf('panels_%s/gradient_barplots_nostd.png',noise),'ContentType','image','Resolution',300);



%% plot RDM & topographic correlations vs. margulies 1, MSMAll, unstandardized
% find margulies gradient 1
map_ind = find(contains(mapname,'fcgradient01'));

cmap = zeros(length(atlas_labels),3);
for i = 1:length(atlas_labels)
    cmap(i,:) = atlas_labels(i).rgba(1:3);
end

good_rois_ctx = good_rois;
good_rois_ctx(good_rois_ctx > n_roi_ctx) = [];

wuc_md_ctx = wuc_corr(:, good_rois_ctx);
cosim_ctx = cosim_corr(:, good_rois_ctx);
good_grad_roi = vals(ismember(1:n_roi_ctx, good_rois_ctx), map_ind);

figure;
clf
ax1 = subplot(1,2,1);
hold on;
ax2 = subplot(1,2,2);
hold on;
for i = 1:size(wuc_md_ctx,2)
    color = cmap(good_rois_ctx(i),:);
    if good_rois_ctx(i) < n_roi_ctx 
        s1 = plot(ax2, good_grad_roi(i), mean(wuc_md_ctx(:,i)), '^', 'color', color);
        s2 = plot(ax1, good_grad_roi(i), mean(cosim_ctx(:,i)), 'o', 'color', color);
    else
        continue;
    end
end
xl = [min(good_grad_roi), max(good_grad_roi)];
yl = [min([ylim(ax2),ylim(ax1)]), max([ylim(ax2),ylim(ax1)])];
xlim(ax1,xl);
xlim(ax2,xl);
if strcmp(noise,'standardized')
    ylim(ax1,yl);
    ylim(ax2,yl);
end
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


% add gradient
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

a2 = axes();
a2.Position = [0.33,0.10,0.15,0.15];
a2.Visible = 'off';

overlay = canlab_get_underlay_image;
o3 = fmridisplay('overlay', which(overlay));
o3 = surface(o3, 'axes', a2, 'direction', 'hcp inflated left', 'orientation', 'lateral');     

cdata = atlas_cii_data.cortex_left;
plot_to_surf(cdata',o3.surface{1}.object_handle, 'indexmap', 'colormap', [cmap; [0,0,0]]);

exportgraphics(gcf,sprintf('panels_%s/margulies_01.png',noise),'ContentType','image','Resolution',300);
