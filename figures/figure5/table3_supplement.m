% This script produces the uncorrected estimates for table 4 and the
% coupling strength analysis for table 4 and 5

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
        tsnr1 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/tsnr.csv',sid.Var1(s)));
        tsnr2 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/tsnr.csv',sid.Var2(s)));
        tsnr(s,:) = mean([tsnr1, tsnr2],2);

        wi_cosim1 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/standardized_betas/standardized_similarity.csv',sid.Var1(s)),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('../../derivatives/restingstate/hcp25/results/%d/standardized_betas/standardized_similarity.csv',sid.Var1(s)),'FileType','text');
        wi_cosim(s,:) = mean(mean(cat(3,wi_cosim1, wi_cosim2),3));
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = diag(readmatrix(sprintf('../../derivatives/restingstate/hcp25/bsc/standardized_betas/cosine/%d_v_%d_wuc.tsv',sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',','));
    catch
        warning('Could not import pair %d', s);
    end
end

cosim = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        cosim(s,:) = mean(readmatrix(sprintf('../../derivatives/restingstate/hcp25/bsc/standardized_betas/cosine/%d_v_%d_cosim.tsv',sid.Var1(s), sid.Var2(s)),...
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
[Bb, Bp, cohensD, Bb_corr, Bp_corr, cohensD_corr] = deal(zeros(length(mapvals),1));
[Bb_CI, Bb_CI_corr] = deal(zeros(length(mapvals),2));
[wucD, wucDStd, cosimD, cosimDStd, ...
    wucb, wucp, wucstd, cosimb, cosimp, cosimstd] = deal(zeros(length(mapvals),1));
[wucb_CI, wucstd_CI, cosim_CI, cosimstd_CI] = deal(zeros(length(mapvals),2));
[mainInt, mainStd, mainD, mainDStd] = deal(zeros(length(mapvals),2));
[mainInt_CI, mainStd_CI] = deal(zeros(length(mapvals),2,2));
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

    % mean imputation within dyad
    for j = 1:size(rdm,1)
        rdm(j,isnan(rdm(j,:))) = nanmean(rdm(j,:));
    end

    [Bb(i), Bb_CI(i,:), Bp(i), cohensD(i)] = neuromaps_corr(topo, rdm, map_val, perm_map);

    [Bb_corr(i), Bb_CI_corr(i,:), Bp_corr(i), cohensD_corr(i)] = neuromaps_corr(topo, rdm, map_val, perm_map, {confounds{1}(:,these_good_rois), confounds{2}(:,these_good_rois)});


    % estimate uncorrected gradient similarities
    
    %eval wuc_md
    obs_val = atanh(wuc_md(:, these_good_rois))';
    [wucb(i), wucb_CI(i,:), wucp(i) wucD(i)] = neuromaps_corr_fx(obs_val, ...
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
disp(table(cosimb_str', wucb_str', int_str', bb_str', ...
    'VariableNames', {'Topo', 'Geo', 'zGeo-zTopo', 'coupling'}))

disp('Cohens Ds:');
disp(table(cosimD(:), wucD(:), mainDStd(:,2), cohensD(:),...
    'VariableNames',{'cosim', 'wuc', 'zGeo-zTopo', 'coupling'},...
    'RowNames', abr_mapname));

disp('Coupling Effects (betas, corrected):')
bb_corr_str = {};
for i = 1:length(Bb_corr)
    bb_corr_str{i} = sprintf('%0.3f±%0.3f',Bb_corr(i), mean([Bb_CI_corr(i,2) - Bb_corr(i), Bb(i) - Bb_CI_corr(i,1)],2));
end
disp(table(bb_corr_str', ...
    'VariableNames', {'neuromap_modulation'}))

disp('Cohens D (corrected):');
disp(table(cohensD_corr(:), ...
    'VariableNames',{'neuromap_modulation'},...
    'RowNames', abr_mapname));