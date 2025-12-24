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

noise = 'whitened';
%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii.cortex_left, atlas_cii.cortex_right, atlas_cii.volumes])) - 1;

atlas_cii = cifti_read(config.canlab2024.path);
%% import between subject similarity measures for unrelated individuals
sid = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

% multiplying by this vector will balances conditions across tasks
balanced_mean_op = [1/7*repmat(1/2,1,10), 1/7*repmat(1/5,1,5), 1/7*repmat(1/8,1,8)];

tsnr = zeros(height(sid), n_roi);
wi_cosim = nan(height(sid), n_roi);
for s = 1:height(sid)
    try
        tsnr1 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/tsnr.csv',sid.Var1(s)));
        tsnr2 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/tsnr.csv',sid.Var2(s)));
        tsnr(s,:) = mean([tsnr1, tsnr2],2);

        wi_cosim1 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_betas/%s_similarity.csv',sid.Var1(s),noise,noise),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_betas/%s_similarity.csv',sid.Var2(s),noise,noise),'FileType','text');
        comb_cosim = mean(cat(3,wi_cosim1, wi_cosim2),3);
        wi_cosim(s,:) = balanced_mean_op*comb_cosim;
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = diag(readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/bsc/%s_betas/cosine/%d_v_%d_wuc.tsv',noise,sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',','));
    catch
        warning('Could not import pair %d', s);
    end
end
%{
for s = 1:height(sid)
    nan_regions = imag(wuc_md(s,:)) ~= 0;
    wuc_md(s, nan_regions) = nan;
end
%}

cosim = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        cosim(s,:) = balanced_mean_op*dlmread(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/bsc/%s_betas/cosine/%d_v_%d_cosim.tsv', noise, sid.Var1(s), sid.Var2(s)), '\t');
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

%% Plot associations 

[r,assocB] = deal(nan(1,size(wuc_md,2)));
topo = atanh(cosim(:, good_rois));
rdm = atanh(wuc_md(:, good_rois));

[n,p] = size(topo);

rois = repmat((1:p), n, 1);
sid = repmat((1:n)', 1, p);

% censor invalid entries and vectorize
isgood = find(~isnan(rdm) & ~isnan(topo));
rdm = rdm(isgood);
topo = topo(isgood);
rois = rois(isgood);
sid = sid(isgood);

X = [rdm.*dummyvar(rois), dummyvar(rois), helmertCoding(sid)];
Y = topo;

B0 = (X'*X)\X'*Y;
assocB(good_rois) = B0(1:p);

cmaprange = prctile(assocB(~isnan(assocB)),[2.5,90]);
cmaprange(1) = eps;

T = {'Dependence of topographic','on geometric similarity',['(across subject \beta, ',sprintf('N=%d',sum(~all(wuc_md == 0,2))), ')']};
plot_to_brain(assocB, 1:length(assocB), cmaprange, T, fs+2);
exportgraphics(gcf,sprintf('panels_%s/association_map.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(good_rois)
    new_cii_data(atlas_cii.cdata == good_rois(i)) = assocB(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('regional_ztopo_regression_on_zgeom_%s.dscalar.nii',noise));
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
[Bb, Bp, cohensf2] = deal(zeros(length(mapvals),1));
[Bb_CI] = deal(zeros(length(mapvals),2));
for i = 1:length(mapvals)
    tic
    mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');

    vals(:,i) = csvread(fullfile(mapvals(i).folder, mapvals(i).name));
    
    these_good_rois = find(10*sum(isnan(wuc_md),1) < size(wuc_md,1));
    these_good_rois(these_good_rois > 358) = [];
    if contains(mapname{i},{'hill2010'})
        these_good_rois(these_good_rois < 180) = [];
    end

    fprintf('Evaluating %s\n', mapname{i})

    randgrad = csvread(fullfile('../../resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
    randgrad(randgrad == 0) = nan; % medial wall

    % eval B wuc ~ rdm
    map_val = vals(these_good_rois, i);
    perm_map = randgrad(these_good_rois,:);
    
    map_mu = mean(map_val);
    map_sd = std(map_val);
    
    map_val = (map_val - map_mu)./map_sd;
    perm_map = (perm_map - nanmean(perm_map))./map_sd;
    
    topo = atanh(cosim(:,these_good_rois));
    rdm = atanh(wuc_md(:,these_good_rois));

    [Bb(i), Bb_CI(i,:), Bp(i), cohensD(i)] = neuromaps_corr(topo, rdm, map_val, perm_map, {confounds{1}(:,these_good_rois), confounds{2}(:,these_good_rois)});
    toc
end



%% plot association scatterplots
abr_mapname = maps(:,3);

disp('Effects (betas):')
bb_str = {};
for i = 1:length(Bb)
    bb_str{i} = sprintf('%0.3f±%0.3f',Bb(i), mean([Bb_CI(i,2) - Bb(i), Bb(i) - Bb_CI(i,1)],2));
end
disp(table(bb_str', ...
    'VariableNames', {'neuromap_modulation'}))

disp('Cohens D:');
disp(table(cohensD(:), ...
    'VariableNames',{'neuromap_modulation'},...
    'RowNames', abr_mapname));

cmap = zeros(length(atlas_labels),3);
for i = 1:length(atlas_labels)
    cmap(i,:) = atlas_labels(i).rgba(1:3);
end

figure(2);
clf

t2 = tiledlayout(2,1,'TileSpacing','compact');

nexttile()
cla
uni_label = 'Ctx_V1_L';
uni = find(contains(roi_labels, uni_label),1,'first');

x = wuc_md(:,uni);
y = cosim(:, uni);
% seems prudent to just show the data rather than interpolated estimates
%x = regress_out(x, [confounds{1}(:,uni), confounds{2}(:,uni)]);
%y = regress_out(y, [confounds{1}(:,uni), confounds{2}(:,uni)]);
p1 = plot(x, y,'h', 'color',cmap(uni,:));
hold on;

% get bootstrap ci
B_CI = bootci(5000,@get_regression_B,[x,y])
fprintf('V1: topo ~ geom \beta = %0.3f +- [%0.3f, %0.3f]\n', get_regression_B([x,y]), B_CI(1), B_CI(2));

m = fitlm(x, y);
y = m.predict(xlim');
l = plot(xlim',y,'-','color',colors(3,:));
l.LineWidth = 2;

title([strrep(uni_label, '_',' '), ' (unimodal)'], 'fontweight','normal','fontsize',fs)
xlabel('geo (WUC)')
ylabel({'topo','(cos\theta)'})
set(gca,'FontSize',fs)
box off;
axis image

nexttile()
cla

trans_label = 'Ctx_p9_46v_R';
trans = find(contains(roi_labels, trans_label),1,'first');

x = wuc_md(:,trans);
y = cosim(:, trans);
% seems prudent to just show the data rather than interpolated estimates
%x = regress_out(x, [confounds{1}(:,trans), confounds{2}(:,trans)]);
%y = regress_out(y, [confounds{1}(:,trans), confounds{2}(:,trans)]);
p2 = plot(x, y,'h', 'color',cmap(trans,:));
hold on;
xl = xlim;

B_CI = bootci(5000,@get_regression_B,[x,y])
fprintf('p9-46v: topo ~ geom \beta = %0.3f +- [%0.3f, %0.3f]\n', get_regression_B([x,y]), B_CI(1), B_CI(2));

m = fitlm(x, y);
y = m.predict(xlim');
l = plot(xlim',y,'-','color',colors(3,:));
l.LineWidth = 2;
xlim(xl);

title({[strrep(trans_label, '_',' '), ' (transmodal)']}, 'fontweight','normal','fontsize',fs)
xlabel({'geo (WUC)'})
ylabel({'topo','(cos\theta)'})
set(gca,'FontSize',fs,'YTick', [0.2,0.4])
box off;
axis image

leg1 = legend([p1,p2],{'V1 dyad', 'p9-46v dyad'},'fontsize',fs);

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),326,407]);

t2.Position(2) = 0.18;
t2.Position(4) = 0.635;

leg1.Position(1) = -0.09;
leg1.Position(2) = -0.02;

sgtitle({'Topography is an inconsistent','measure of geometry'},'FontWeight','bold','fontsize',fs+2)

exportgraphics(gcf,sprintf('panels_%s/scatterplots.png',noise),'ContentType','image','Resolution',300);

%% plot association 2nd level regression and barplots
abr_mapname = maps(:,3);

example_perm_data = csvread(fullfile('../../resources/neuromaps/canlab2024_permuted_annotations',mapvals(1).name));
nperms = size(example_perm_data,2);

figure;
clf
t2 = tiledlayout(1,2,'TileSpacing','compact');

nexttile()
[~,I] = sort(Bp, 'ascend');
sig = Bp <= FDR(Bp, 0.05);
%{
I = I(ismember(I,find(Bp <= pthresh)));
if length(I) > 0
    map_ind = I(1);
end
% if there's a tie, prioritize margulies gradient
if length(I) == 0 || Bp(10) == Bp(map_ind)
    map_ind = 10;
end
%}
map_ind = 8;

% has more significant subjects than holm-sidak threshold
good_grad_roi = vals(ismember(1:358, good_rois), map_ind);
hold on;
for i = 1:length(good_grad_roi)
    color = cmap(good_rois(i),:);
    if good_rois(i) < 358 
        plot(good_grad_roi(i), assocB(good_rois(i)), 'h', 'color', color);
    else
        continue;
    end
end
set(gca, 'YGrid', 'on', 'box', 'off', 'fontsize', fs);
ylabel({'\beta','topo ~ geo'});
title({'Topographic sensitivity ', 'to geometric similarity'},'FontWeight','normal','fontsize',fs-1);
set(gca,'XTick',[-5,6],'XTickLabels',{'Uni','Trans'},'FontSize',fs)
xl = xlim;

m_rep = fitlm(good_grad_roi, assocB(good_rois < 359));
y = m_rep.predict(xlim');
l = plot(xlim',y,'-','color',colors(3,:));
l.LineWidth = 2;
leg2 = legend('\beta across dyads','fontsize',fs);
set(gca,'FontSize',fs)


ax4 = nexttile();
buffer = 2.5;

cla
hold on;

pos_err = Bb_CI(:,2) - Bb;
neg_err = Bb - Bb_CI(:,1);
for i = 1:length(mapname)
    errorbar(Bb(i), i, neg_err(i), pos_err(i), '.', 'horizontal', 'capsize', 0,'color',colors_light(maps{i,6},:), 'linewidth',2)
    plot(Bb(i),i, 'h','MarkerFaceColor', colors_light(maps{i,6},:),'color', colors(maps{i,6},:));
end

set(gca,'YTick',1:length(mapname), 'YTickLabels', abr_mapname,'FontSize',fs-1, 'YDir', 'rev','YGrid','on');
ylim([0.5,length(mapname)+0.5])

title({'Cortical gradients associated with','coupled geometry and topography'},'fontweight','normal', 'fontsize', fs+1);
xlabel({'\beta_1','topo ~ \beta_0geo + \beta_1geo*map'}, 'fontsize', fs)
box off
xl = xlim;
xlim([buffer*xl(1), xl(2)*buffer])
yl = ylim;
annot = cell(length(maps),2);
for i = 1:length(maps)
    try, delete(annot{i,1}); end
    annot{i,1} = text(buffer*xl(1)*0.95,i,maps{i,4},'FontSize',fs-3);
end
for i = 1:length(maps)
    try, delete(annot{i,2}); end
    annot{i,2} = text(buffer*xl(2)*0.95,i,maps{i,5},'FontSize',fs-3,'HorizontalAlignment','right');
end

sig = Bp <= FDR(Bp, 0.05);
if any(sig)
    x = sign(Bb(sig)).*(pos_err(sig) + abs(Bb(sig)) + xl(2) * 0.2);
    text(x, find(sig),'*','HorizontalAlignment','center');
end
xline(0, 'color', [0.5,0.5,0.5]);

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),530,407]);

t2.Position(4) = 0.53;
t2.Position(1) = 0.15;

leg2.Position(1) = 0.15;
leg2.Position(2) = 0.036;



% add ROI legend
a1 = axes();
a1.Position = [0.28,0.61,0.15,0.15];
a1.Visible = 'off';

overlay = canlab_get_underlay_image;
o2 = fmridisplay('overlay', which(overlay));
o2 = surface(o2, 'axes', a1, 'direction', 'hcp inflated left', 'orientation', 'lateral');     

map_files = dir('../../resources/neuromaps/');
map_tokens = regexp(mapname,'(.*)-(.*)','tokens');
file_ind = find(contains({map_files.name}, map_tokens{map_ind}{1}{1}) & ...
    contains({map_files.name}, map_tokens{map_ind}{1}{2}) & ...
    contains({map_files.name}, 'hemi-L'));

grayord_surf_L = gifti(fullfile(map_files(file_ind).folder, map_files(file_ind).name));
plot_to_surf(grayord_surf_L.cdata,o2.surface{1}.object_handle);

sgtitle({'Topographic similarity only indicates','geometric similarity in unimodal areas'},'FontWeight','Bold','fontsize',fs+2)


exportgraphics(gcf,sprintf('panels_%s/second_level_associations.png',noise),'ContentType','image','Resolution',300);

%% post hoc eval of cerebellum
ind = find(contains({atlas_labels.name},{'Cblm_V_','Cblm_VI_'}));
x = mean(wuc_md(:,ind),2);
y = mean(cosim(:,ind),2);

B_CI = bootci(5000,@get_regression_B,[x,y])
B = get_regression_B([x,y])

ind = find(contains({atlas_labels.name},{'Cblm_Crus'}));
x = mean(wuc_md(:,ind),2);
y = mean(cosim(:,ind),2);

B_CI = bootci(5000,@get_regression_B,[x,y])
B = get_regression_B([x,y])