% computes similarities based on results from permuted stimulus comparisons
% (shuffling participant 1's stimuli and comparing with unshuffled
% participant 2 stimuli)

close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));
addpath(genpath(fullfile(config.matlab_libraries.npm)));

addpath('../../matlab_libraries');
addpath('../../resources/neuromaps');
addpath('../../src/rdm_similarity/matlab')

fs=config.matlab_disp_scheme.fontsize;

f = figure;
cm = colormap(f,'hot');
close(f)

colors = config.matlab_disp_scheme.color_main;
colors_light = config.matlab_disp_scheme.color_light;

data_root = '../../derivatives/restingstate/hcp25';

noise='whitened';
%noise='standardized';


%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii_data = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii_data.cortex_left, atlas_cii_data.cortex_right, atlas_cii_data.volumes])) - 1;

%% import between subject similarity measures for unrelated individuals to identify 'good' rois and confounds
sid = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

tsnr = zeros(height(sid), n_roi);
wi_cosim = nan(height(sid), n_roi);
for s = 1:height(sid)
    try
        tsnr1 = readmatrix(sprintf('%s/results/%d/tsnr.csv',data_root,sid.Var1(s)));
        tsnr2 = readmatrix(sprintf('%s/results/%d/tsnr.csv',data_root,sid.Var2(s)));
        tsnr(s,:) = mean([tsnr1, tsnr2],2);

        wi_cosim1 = readmatrix(sprintf('%s/results/%d/%s_betas/%s_similarity.csv',data_root,sid.Var1(s),noise,noise),'FileType','text');
        wi_cosim2 = readmatrix(sprintf('%s/results/%d/%s_betas/%s_similarity.csv',data_root,sid.Var2(s),noise,noise),'FileType','text');
        comb_cosim = mean(cat(3,wi_cosim1, wi_cosim2),3);
        wi_cosim(s,:) = mean(comb_cosim,1);
    catch
        warning('Could not import pair %d', s);
    end
end

wuc_md = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        wuc_md(s,:) = readmatrix(sprintf('%s/bsc/%s_betas/cosine/%d_v_%d_wuc.tsv',data_root,noise,sid.Var1(s), sid.Var2(s)),...
            'FileType','text','Delimiter',',');
    catch
        warning('Could not import pair %d', s);
    end
end

cosim = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        cosim(s,:) = mean(dlmread(sprintf('%s/bsc/%s_betas/cosine/%d_v_%d_cosim.tsv', data_root, noise, sid.Var1(s), sid.Var2(s)), '\t'),1);
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
%% compute null data
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

% compute confound corrected null similarities
n_seeds = 100;
wuc_null = nan(n_seeds, height(sid), n_roi);
cosim_null = nan(n_seeds, n_subj, n_roi);
for seed = 1:n_seeds
    fprintf('Compute similarities for seed %d\n',seed-1);
    parfor s = 1:height(sid)
        try
            wuc0 = readmatrix([data_root, sprintf('/bsc_null/seed%d/%s_betas/cosine/%d_v_%d_wuc.tsv',seed-1,noise,sid.Var1(s), sid.Var2(s))],...
                'FileType','text','Delimiter',',');
        catch
            warning('Could not import pair %d', s);
        end

        % confound correction
        Y = wuc0(good_rois)';
        X = [];
        for j = 1:length(confounds)
            X = [X, confounds{j}(s,good_rois)'];
        end
        X = X - nanmean(X);
        X = [ones(length(Y),1), X];
        good_roi = ~isnan(Y) & all(~isnan(X),2);
        X = X(good_roi,:);
        Y = Y(good_roi);
        B = (X'*X)\X'*Y;

        values = zeros(length(good_rois),1);
        values(good_rois(good_roi)) = (Y - X*B) + B(1);
        wuc_null(seed, s,:) = values;
    end

    % compute similarities
    rand_ordering = csvread([data_root, sprintf('/bsc_null/seed%d/shuffle_ind.csv',seed-1)]);
    rand_ordering = rand_ordering+1; % from python to matlab indexing
    
    
    parfor i = 1:n_subj
        cosim0 = nan(n_roi,1);
        for r = 1:n_roi
            if ~good_topo(i)
                continue;
            end
    
            % balance with oversampling rather than weights
            X = topos1{r,i};
            Y = topos2{r,i};
    
            X = X(rand_ordering,:);
    
            cosim0(r) = mean(diag(X*Y'));

        end

        % confound correction
        Y = cosim0(good_rois);
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

        values = zeros(length(good_rois),1);
        values(good_rois(good_roi)) = (Y - X*B) + B(1);
        cosim_null(seed,i,:) = values;
    end
end

save(sprintf('%s/bsc_null/perm_nulls_%s.mat',data_root,noise),'cosim_null','wuc_null')