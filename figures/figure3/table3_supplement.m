% This script produces the uncorrected estimates and sensitivity analysis 
% values for table 3

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

noise='whitened';

%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii_data = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii_data.cortex_left, atlas_cii_data.cortex_right, atlas_cii_data.volumes])) - 1;

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

        wi_cosim1 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/%s_similarity.csv',sid.Var1(s),noise,noise),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/%s_similarity.csv',sid.Var2(s),noise,noise),'FileType','text');
        comb_cosim = mean(cat(3,wi_cosim1, wi_cosim2),3);
        wi_cosim(s,:) = balanced_mean_op*comb_cosim;
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/bsc/%s_betas/cosine/%d_v_%d_wuc.tsv',noise,sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',',');
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
[wucb, wucp, cosimb, cosimp] = deal(zeros(length(mapvals),1));
[wucb_CI, cosim_CI] = deal(zeros(length(mapvals),2));
[mainStd, mainStdP, mainDStd] = deal(zeros(length(mapvals),2));
mainStd_CI = zeros(length(mapvals),2,2);
[Bb, Bp, BD] = deal(zeros(length(mapvals),1));
[Bb_CI] = deal(zeros(length(mapvals),2));
for i = 1:length(mapvals)
    mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');

    vals(:,i) = csvread(fullfile(mapvals(i).folder, mapvals(i).name));
    
    these_good_rois = good_rois;
    these_good_rois(these_good_rois > 358) = [];
    if contains(mapname{i},{'hill2010'})
        these_good_rois(these_good_rois < 180) = [];
    end

    fprintf('Evaluating %s\n', mapname{i})

    % do coupling strength test with and without confounds
    randgrad = csvread(fullfile('../../resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
    randgrad(randgrad == 0) = nan; % medial wall

    map_val = vals(these_good_rois, i);
    perm_map = randgrad(these_good_rois,:);
    
    map_mu = mean(map_val);
    map_sd = std(map_val);
    
    map_val = (map_val - map_mu)./map_sd;
    perm_map = (perm_map - nanmean(perm_map))./map_sd;
    
    topo = atanh(cosim(:,these_good_rois));
    rdm = atanh(wuc_md(:,these_good_rois));

    [Bb(i), Bb_CI(i,:), Bp(i), BD(i)] = neuromaps_corr(topo, rdm, map_val, perm_map);

    % estimate uncorrected gradient similarities
    
    %eval wuc_md
    obs_val = atanh(wuc_md(:, these_good_rois))';
    [wucb(i), wucb_CI(i,:), wucp(i), wucD(i)] = neuromaps_corr_fx(obs_val, ...
        map_val, perm_map);

    %eval cosim
    obs_val = atanh(cosim(:,these_good_rois))';
    [cosimb(i), cosim_CI(i,:), cosimp(i), cosimD(i)] = neuromaps_corr_fx(obs_val, ...
        map_val, perm_map);

    %eval cosim & wuc interaction
    obs_val1 = atanh(wuc_md(:, these_good_rois))';
    obs_val2 = atanh(cosim(:,these_good_rois))';
    [mainStd(i,:), mainStd_CI(i,:,:), mainStdP(i,:), mainDStd(i,:)] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, ...
        map_val, perm_map);
end


%% print neurmap associations
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
bb_str = {};
for i = 1:length(Bb)
    bb_str{i} = sprintf('%0.3f±%0.3f',Bb(i), mean([Bb_CI(i,2) - Bb(i), Bb(i) - Bb_CI(i,1)],2));
end
%disp(table(cosimb_str', wucb_str', int_str', bb_str', ...
%    'VariableNames', {'Topo', 'Geo', 'zGeo-zTopo', 'coupling'}))

disp(table(cosimb_str', wucb_str', int_str', bb_str', ...
    'VariableNames', {'Topo', 'Rep', 'zRep-zTopo', 'coupling'}, 'RowNames', abr_mapname))
disp(table(cosimD', wucD', mainDStd(:,2), BD, ...
    'VariableNames', {'Topo', 'Rep', 'zRep-zTopo', 'coupling'}, 'RowNames', abr_mapname))
disp(table(cosimp, wucp, mainStdP(:,2), Bp, ...
    'VariableNames', {'Topo', 'Rep', 'zRep-zTopo', 'coupling'}, 'RowNames', abr_mapname))