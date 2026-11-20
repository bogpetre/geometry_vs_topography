close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath('../../matlab_libraries');
addpath('../../src/rdm_similarity/matlab')

data_root = '../../derivatives_bak/hcp_glm_msmall_grayord_spm/';

noise = 'whitened';
seed = 0;

%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii.cortex_left, atlas_cii.cortex_right, atlas_cii.volumes])) - 1;

rand_ordering = csvread([data_root, sprintf('/bsc_null_%d/shuffle_ind.csv',seed)]);
rand_ordering = rand_ordering+1; % from python to matlab indexing

%% compute paired cosine similarities of task topographies with shuffling of first participant
sid = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);
n_subj = height(sid);

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

left_ctx_roi = unique(atlas_cii.cortex_left);
right_ctx_roi = unique(atlas_cii.cortex_right);
subctx_roi = unique(atlas_cii.volumes);
left_ctx_roi(left_ctx_roi == 0) = [];
right_ctx_roi(right_ctx_roi == 0) = [];
subctx_roi(subctx_roi == 0) = [];

% load topographies
topos1 = cell(n_roi,n_subj);
good_topo = true(1,n_subj);
parfor i = 1:height(sid)
    try
        contrast_file = dir([data_root, '/results/', ...
                sprintf('%d/all_tasks/%s_contrasts/merged_cifti.dscalar.nii', ...
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

            assert(size(atlas_cii.(struct),2) == size(these_contrasts.(struct),2));

            ind = atlas_cii.(struct) == r;
            topos1{r,i} = these_contrasts.(struct)(:,ind);
            s = zeros(size(topos1{r,i},1),1);
            for j = 1:size(topos1{r,i},1)
                s(j) = norm(topos1{r,i}(j,:));
            end
            topos1{r,i} = topos1{r,i}./s;
        end
    catch
        good_topo(i) = false;
        warning('Failed to import topographies for subject %d', tbl0.Subject(i));
    end
end

topos2 = cell(n_roi,n_subj);
good_topo = true(1,n_subj);
parfor i = 1:height(sid)
    try
        contrast_file = dir([data_root, '/results/', ...
                sprintf('%d/all_tasks/%s_contrasts/merged_cifti.dscalar.nii', ...
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

            assert(size(atlas_cii.(struct),2) == size(these_contrasts.(struct),2));

            ind = atlas_cii.(struct) == r;
            topos2{r,i} = these_contrasts.(struct)(:,ind);
            s = zeros(size(topos2{r,i},1),1);
            for j = 1:size(topos2{r,i},1)
                s(j) = norm(topos2{r,i}(j,:));
            end
            topos2{r,i} = topos2{r,i}./s;
        end
    catch
        good_topo(i) = false;
        warning('Failed to import topographies for subject %d', tbl0.Subject(i));
    end
end

% compute similarities
roi_topo = nan(n_subj, n_roi);
for r = 1:n_roi
    fprintf('Compute topographic similarity for region %d\n',r);
    for i = 1:n_subj
        if ~good_topo(i)
            continue;
        end

        if ~good_topo(j)
             continue;
        end

        % balance with oversampling rather than weights
        X = topos1{r,i};
        Y = topos2{r,i};
        
        % build expansion mapper that replicates rows as needed
        replicate_id = build_replication_info(task_labels);
        X = X(replicate_id,:);
        Y = Y(replicate_id,:);

        perm_id = lift_perm_to_expanded(rand_ordering, replicate_id);
        X = X(perm_id,:);

        roi_topo(i,r) = mean(diag(X*Y'));
    end
end

clear topos1

save(sprintf('task_topographic_similarities_%s.mat', noise),'roi_topo','good_topo','-v7.3');