close all; clear all;

analysisRoot = '../../';
config = jsondecode(fileread([analysisRoot, '/config.json']));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));
addpath(genpath(fullfile(config.matlab_libraries.npm)));

addpath(genpath([analysisRoot, '/matlab_libraries']));
addpath([analysisRoot, '/resources/neuromaps']);


fs=config.matlab_disp_scheme.fontsize;

f = figure;
cm = colormap(f,'hot');
close(f)

dc_color = config.matlab_disp_scheme.color_main;
dc_color_light = config.matlab_disp_scheme.color_light;

data_root = [analysisRoot, '/derivatives/hcp_glm_msmall_grayord_spm/'];

unrestricted = readtable(config.hcp_participant_data.unrestricted);

noise = 'whitened';

bootstrap_sample_size = 207;
n_bs = 1000;
parpool(64)

%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii.cortex_left, atlas_cii.cortex_right, atlas_cii.volumes])) - 1;

%% import confounds for iid participants
sid = readtable([analysisRoot, '/resources/paired_sid.csv'], 'ReadVariableNames',false);
sid = sort([sid.Var1; sid.Var2]);

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

% multiplying by this vector will balances conditions across tasks
balanced_mean_op = [1/7*repmat(1/2,1,10), 1/7*repmat(1/5,1,5), 1/7*repmat(1/8,1,8)];

tsnr = zeros(height(sid), n_roi);
wi_cosim = nan(height(sid), n_roi);
for s = 1:height(sid)
    try
        tsnr(s,:) = readmatrix(sprintf([analysisRoot, '/derivatives/hcp_glm_msmall_grayord_spm/results/%d/tsnr.csv'],sid(s)));

        wi_cosim(s,:) = balanced_mean_op*readmatrix(sprintf([analysisRoot, '/derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/%s_similarity.csv'],sid(s),noise,noise),'FileType','text');
    catch
        warning('Could not import pair %d', s);
    end
end

%% load exhaustive cosine similarities and goemetric similarities of tasks
% this lets us bootstrap new dyads, not used in the main analysis
load([analysisRoot, sprintf('/figures/figure6/task_topographic_similarities_%s.mat',noise)],'roi_topo','good_topo');

geom_bsc = dir(fullfile(data_root,sprintf('bsc_all/%s_betas/cosine/*v_all.tsv',noise)));
roi_geom0 = zeros(length(geom_bsc),length(geom_bsc),518);
for i = 1:length(geom_bsc)
    csv = readmatrix(fullfile(geom_bsc(i).folder, geom_bsc(i).name),'FileType','text');
    [roi_geom0(i,(i+1):end,:), roi_geom0((i+1):end,i,:)] = deal(csv);
    roi_geom0(i,i) = 1;
end

% check for correspondence between roi_topo and roi_geom
geom_sid = arrayfun(@(x1)(str2double(strrep(x1.name,'_v_all.tsv',''))), geom_bsc);
topo_sid = unrestricted.Subject(good_topo);
for i = 1:length(geom_sid)
    assert(geom_sid(i) == topo_sid(i));
end

% expand roi_geom to match topo_roi (and more importantly restricted)
roi_geom = nan(size(roi_topo));
roi_geom(good_topo, good_topo,:) = roi_geom0;
clear roi_geom0

%% filter full matrices for iid participants
iid_ind = find(ismember(unrestricted.Subject, sid));
roi_topo = roi_topo(iid_ind, iid_ind, :);
roi_geom = roi_geom(iid_ind, iid_ind, :);

t0 = tic;
%% FPR analysis (gradient test)
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

mapvals = dir([analysisRoot, '/resources/neuromaps/canlab2024_parcel_vals/']);
mapvals(1:2) = []; % remove '.' and '..' refs

keep = zeros(length(mapvals),1);
for i = 1:length(maps)
    keep(i) = find(contains({mapvals.name}, maps(i,1)) & contains({mapvals.name}, maps(i,2)));
end
keep(keep==0) = [];
mapvals = mapvals(keep);

[Bp_null_grd, Bp_int_null_grd, spin_var_null_grd, spin_var_int_null_grd, ...
    jk_var_null_grd, jk_var_int_null_grd] = deal(zeros(length(mapvals), n_bs));
load('stats.mat')
k = 1;
while k <= n_bs
    try
        fprintf('Evaluating gradient effect jackknife-spin test null %d\n',k)
        [topo_bs, geom_bs, tsnr_bs, wi_cosim_bs] = get_bs_sample(bootstrap_sample_size, roi_topo, roi_geom, tsnr, wi_cosim);
        
        mapname = {};
        for i = 1:length(mapvals)
            mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');
        
            these_good_rois = find(10*sum(isnan(geom_bs),1) < size(geom_bs,1));
            these_good_rois(these_good_rois > 358) = [];
            if contains(mapname{i},{'hill2010'})
                these_good_rois(these_good_rois < 180) = [];
            end
        
            fprintf('Evaluating %s\n', mapname{i})
        
            randgrad = csvread(fullfile(analysisRoot, '/resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
            randgrad(randgrad == 0) = nan; % medial wall
            
            % pick a random rotated map to use as the main map
            null_ind = randi(5000,1);
            map_val = randgrad(these_good_rois,null_ind);
            map_val(isnan(map_val)) = nanmean(map_val);
            randgrad(:,null_ind) = [];
            perm_map = randgrad(these_good_rois,:);
        
            map_mu = mean(map_val);
            map_sd = std(map_val);
            
            map_val = (map_val - map_mu)./map_sd;
            perm_map = (perm_map - nanmean(perm_map))./map_sd;
            
            %eval wuc_md
            obs_val = atanh(geom_bs(:, these_good_rois))';
            [~, ~, Bp_null_grd(i,k), ~, jk_var_null_grd(i,k), ...
                spin_var_null_grd(i,k)] = neuromaps_corr_fx(obs_val, ...
                map_val, perm_map, {tsnr_bs(:,these_good_rois)', wi_cosim_bs(:,these_good_rois)'});
        
            %eval cosim & wuc interaction
            obs_val1 = atanh(geom_bs(:, these_good_rois))';
            obs_val2 = atanh(topo_bs(:,these_good_rois))';
            [~, ~, mainStdP, ~, jk_var_int_null_grd, spin_var_int_null_grd] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, ...
                map_val, perm_map,  {tsnr_bs(:,these_good_rois)', wi_cosim_bs(:,these_good_rois)'});
            Bp_int_null_grd(i,k) = mainStdP(2);
        end
        k = k+1;
    catch
        % this can happen if we resample a participant pair for which RDM
        % whitening failed in some cortical region. It's rare, but can
        % happen occasionally.
        warning('Iteration %d failed, repeating', k)
    end
end
%{

%% FPR analysis using sampling CI alone (gradient test)

[Bp_null_grd_jk, Bp_int_null_grd_jk] = deal(zeros(length(mapvals), n_bs));
k = 1;
while k <= n_bs
    try
        fprintf('Evaluating gradient effect jackknife CI null %d\n',k)
        [topo_bs, geom_bs, tsnr_bs, wi_cosim_bs] = get_bs_sample(bootstrap_sample_size, roi_topo, roi_geom, tsnr, wi_cosim);
        
        mapname = {};
        for i = 1:length(mapvals)
            mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');
        
            these_good_rois = find(10*sum(isnan(geom_bs),1) < size(geom_bs,1));
            these_good_rois(these_good_rois > 358) = [];
            if contains(mapname{i},{'hill2010'})
                these_good_rois(these_good_rois < 180) = [];
            end
        
            fprintf('Evaluating %s\n', mapname{i})
        
            randgrad = csvread(fullfile(analysisRoot, '/resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
            randgrad(randgrad == 0) = nan; % medial wall
            
            % pick a random rotated map to use as the main map
            null_ind = randi(5000,1);
            map_val = randgrad(these_good_rois,null_ind);
            map_val(isnan(map_val)) = nanmean(map_val);
            randgrad(:,null_ind) = [];
            perm_map = randgrad(these_good_rois,:);
        
            map_mu = mean(map_val);
            map_sd = std(map_val);
            
            map_val = (map_val - map_mu)./map_sd;
            perm_map = (perm_map - nanmean(perm_map))./map_sd;
            
            %eval wuc_md
            obs_val = atanh(geom_bs(:, these_good_rois))';
            [~, ~, Bp_null_grd_jk(i,k), ~] = neuromaps_corr_fx_jk(obs_val, ...
                map_val, perm_map, {tsnr_bs(:,these_good_rois)', wi_cosim_bs(:,these_good_rois)'});
        
            %eval cosim & wuc interaction
            obs_val1 = atanh(geom_bs(:, these_good_rois))';
            obs_val2 = atanh(topo_bs(:,these_good_rois))';
            [~, ~, mainStdP] = neuromaps_corr_interaction_fx_jk(obs_val1, obs_val2, ...
                map_val, perm_map,  {tsnr_bs(:,these_good_rois)', wi_cosim_bs(:,these_good_rois)'});
            Bp_int_null_grd_jk(i,k) = mainStdP(2);
        end
        k = k+1;
    catch
        % this can happen if we resample a participant pair for which RDM
        % whitening failed in some cortical region. It's rare, but can
        % happen occasionally.
        warning('Iteration %d failed, repeating', k)
    end
end


%% FPR analysis using spin test alone (gradient test)

[Bp_null_spin, Bp_int_null_spin] = deal(zeros(length(mapvals), n_bs));
k = 1;
while k <= n_bs
    try
        fprintf('Evaluating gradient effect spin-test null %d\n',k)
        [topo_bs, geom_bs, tsnr_bs, wi_cosim_bs] = get_bs_sample(bootstrap_sample_size, roi_topo, roi_geom, tsnr, wi_cosim);
        
        mapname = {};
        for i = 1:length(mapvals)
            mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');
        
            these_good_rois = find(10*sum(isnan(geom_bs),1) < size(geom_bs,1));
            these_good_rois(these_good_rois > 358) = [];
            if contains(mapname{i},{'hill2010'})
                these_good_rois(these_good_rois < 180) = [];
            end
        
            fprintf('Evaluating %s\n', mapname{i})
        
            randgrad = csvread(fullfile(analysisRoot, '/resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
            randgrad(randgrad == 0) = nan; % medial wall
            
            % pick a random rotated map to use as the main map
            null_ind = randi(5000,1);
            map_val = randgrad(these_good_rois,null_ind);
            map_val(isnan(map_val)) = nanmean(map_val);
            randgrad(:,null_ind) = [];
            perm_map = randgrad(these_good_rois,:);
        
            map_mu = mean(map_val);
            map_sd = std(map_val);
            
            map_val = (map_val - map_mu)./map_sd;
            perm_map = (perm_map - nanmean(perm_map))./map_sd;
            
            %eval wuc_md
            obs_val = atanh(geom_bs(:, these_good_rois))';
            [~, ~, Bp_null_spin(i,k)] = neuromaps_corr_fx_spin(obs_val, ...
                map_val, perm_map, {tsnr_bs(:,these_good_rois)', wi_cosim_bs(:,these_good_rois)'});
        
            %eval cosim & wuc interaction
            obs_val1 = atanh(geom_bs(:, these_good_rois))';
            obs_val2 = atanh(topo_bs(:,these_good_rois))';
            [~, ~, mainStdP] = neuromaps_corr_interaction_fx_spin(obs_val1, obs_val2, ...
                map_val, perm_map,  {tsnr_bs(:,these_good_rois)', wi_cosim_bs(:,these_good_rois)'});
            Bp_int_null_spin(i,k) = mainStdP(2);
        end
        k = k+1;
    catch
        % this can happen if we resample a participant pair for which RDM
        % whitening failed in some cortical region. It's rare, but can
        % happen occasionally.
        warning('Iteration %d failed, repeating', k)
    end
end



%% FPR analysis (coupling test)

[Bp_null_cpl, spin_var_null_cpl, jk_var_null_cpl] = deal(zeros(length(mapvals), n_bs));
k = 1;

%load('partial_run.mat')
while k <= n_bs
    try
        fprintf('Evaluating coupling effect jackknife spin test moderator null %d\n',k)
        [topo_bs, geom_bs, tsnr_bs, wi_cosim_bs] = get_bs_sample(bootstrap_sample_size, roi_topo, roi_geom, tsnr, wi_cosim);
        
        mapname = {};
        for i = 1:length(mapvals)
            mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');
        
            these_good_rois = find(10*sum(isnan(geom_bs),1) < size(geom_bs,1));
            these_good_rois(these_good_rois > 358) = [];
            if contains(mapname{i},{'hill2010'})
                these_good_rois(these_good_rois < 180) = [];
            end
        
            fprintf('Evaluating %s\n', mapname{i})
        
            randgrad = csvread(fullfile(analysisRoot, '/resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
            randgrad(randgrad == 0) = nan; % medial wall
        
            % pick a random rotated map to use as the main map
            null_ind = randi(5000,1);
            map_val = randgrad(these_good_rois,null_ind);
            map_val(isnan(map_val)) = nanmean(map_val);
            randgrad(:,null_ind) = [];
            perm_map = randgrad(these_good_rois,:);
            
            map_mu = mean(map_val);
            map_sd = std(map_val);
            
            map_val = (map_val - map_mu)./map_sd;
            %perm_map = (perm_map - map_mu)./map_sd;
            perm_map = (perm_map - nanmean(perm_map))./map_sd;
            
            topo = atanh(topo_bs(:,these_good_rois));
            rdm = atanh(geom_bs(:,these_good_rois));
        
            [~, ~, Bp_null_cpl(i,k), ~, spin_var_null_cpl(i,k), jk_var_null_cpl(i,k)] = neuromaps_corr(topo, rdm, map_val, perm_map, {tsnr_bs(:,these_good_rois), wi_cosim_bs(:,these_good_rois)});
        end
        k = k+1;
    catch
        % this can happen if we resample a participant pair for which RDM
        % whitening failed in some cortical region. It's rare, but can
        % happen occasionally.
        warning('Iteration %d failed, repeating', k)
    end
end

%% Test permuting topography-geometry relationships (coupling test)


[Bp_null2_cpl, spin_var_null2_cpl, jk_var_null2_cpl] = deal(zeros(length(mapvals), n_bs));
k = 1;
while k <= n_bs
    try
        fprintf('Evaluating coupling effect jackknife spin test main-effect null %d\n',k)
        [topo_bs, geom_bs, tsnr_bs, wi_cosim_bs] = get_bs_sample(bootstrap_sample_size, roi_topo, roi_geom, tsnr, wi_cosim);
        
        reorder_ind = randperm(size(topo_bs,1),size(topo_bs,1));
        topo_bs = topo_bs(reorder_ind,:);

        mapname = {};
        for i = 1:length(mapvals)
            mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');
        
            vals = csvread(fullfile(mapvals(i).folder, mapvals(i).name));
            
            these_good_rois = find(10*sum(isnan(geom_bs),1) < size(geom_bs,1));
            these_good_rois(these_good_rois > 358) = [];
            if contains(mapname{i},{'hill2010'})
                these_good_rois(these_good_rois < 180) = [];
            end
        
            fprintf('Evaluating %s\n', mapname{i})
        
            randgrad = csvread(fullfile(analysisRoot, '/resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
            randgrad(randgrad == 0) = nan; % medial wall
        
            % eval B wuc ~ rdm
            map_val = vals(these_good_rois);
            perm_map = randgrad(these_good_rois,:);
            
            map_mu = mean(map_val);
            map_sd = std(map_val);
            
            map_val = (map_val - map_mu)./map_sd;
            perm_map = (perm_map - nanmean(perm_map))./map_sd;
            
            topo = atanh(topo_bs(:,these_good_rois));
            rdm = atanh(geom_bs(:,these_good_rois));
        
            [~, ~, Bp_null2_cpl(i,k), ~, spin_var_null2_cpl(i,k), jk_var_null2_cpl(i,k)] = neuromaps_corr(topo, rdm, map_val, perm_map, {tsnr_bs(:,these_good_rois), wi_cosim_bs(:,these_good_rois)});
        end
        k = k+1;
    catch
        % this can happen if we resample a participant pair for which RDM
        % whitening failed in some cortical region. It's rare, but can
        % happen occasionally.
        warning('Iteration %d failed, repeating', k)
    end
end

%% Test basic approach using only spin tests (coupling test)

Bp_null_cpl_basic = zeros(length(mapvals), n_bs);
k = 1;
while k <= n_bs
    try
        fprintf('Evaluating coupling effect spin test moderator null %d\n',k)
        [topo_bs, geom_bs, tsnr_bs, wi_cosim_bs] = get_bs_sample(bootstrap_sample_size, roi_topo, roi_geom, tsnr, wi_cosim);
        
        mapname = {};
        for i = 1:length(mapvals)
            mapname{i} = regexprep(mapvals(i).name,'(.*)_(.*).csv','$1-$2');
        
            these_good_rois = find(10*sum(isnan(geom_bs),1) < size(geom_bs,1));
            these_good_rois(these_good_rois > 358) = [];
            if contains(mapname{i},{'hill2010'})
                these_good_rois(these_good_rois < 180) = [];
            end
        
            fprintf('Evaluating %s\n', mapname{i})
        
            randgrad = csvread(fullfile(analysisRoot, '/resources/neuromaps/canlab2024_permuted_annotations',mapvals(i).name));
            randgrad(randgrad == 0) = nan; % medial wall
        
            % eval B wuc ~ rdm
            null_ind = randi(5000,1);
            map_val = randgrad(these_good_rois,null_ind);
            map_val(isnan(map_val)) = nanmean(map_val);
            randgrad(:,null_ind) = [];
            perm_map = randgrad(these_good_rois,:);
            
            map_mu = mean(map_val);
            map_sd = std(map_val);
            
            map_val = (map_val - map_mu)./map_sd;
            %perm_map = (perm_map - map_mu)./map_sd;
            perm_map = (perm_map - nanmean(perm_map))./map_sd;
            
            topo = atanh(topo_bs(:,these_good_rois));
            rdm = atanh(geom_bs(:,these_good_rois));
        
            [~, ~, Bp_null_cpl_basic(i,k)] = neuromaps_corr_basic(topo, rdm, map_val, perm_map, {tsnr_bs(:,these_good_rois), wi_cosim_bs(:,these_good_rois)});
        end
        k = k+1;
    catch
        % this can happen if we resample a participant pair for which RDM
        % whitening failed in some cortical region. It's rare, but can
        % happen occasionally.
        warning('Iteration %d failed, repeating', k)
    end
end
%}
fprintf('Runtime: %0.3f min\n', toc(t0)/60);
save('stats2b.mat',...
    'Bp_null_grd', 'Bp_int_null_grd', ...
    'spin_var_null_grd', 'spin_var_int_null_grd', 'jk_var_null_grd', 'jk_var_int_null_grd', ...
    'Bp_null_grd_jk', 'Bp_int_null_grd_jk', 'Bp_null_spin', 'Bp_int_null_spin', ...
    'Bp_null_cpl', 'spin_var_null_cpl', 'jk_var_null_cpl', ...
    'Bp_null2_cpl', 'spin_var_null2_cpl', 'jk_var_null2_cpl', ...
    'Bp_null_cpl_basic');
