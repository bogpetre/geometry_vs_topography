close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath('../../matlab_libraries');

tbl0 = readtable(config.hcp_participant_data.restricted);
n_subj = height(tbl0);

data_root = '../../derivatives/hcp_glm_msmall_grayord_spm/';

%% import atlas in cifti space and get region names
atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii = get_cifti_data(config.canlab2024.path);
n_roi = length(unique([atlas_cii.cortex_left, atlas_cii.cortex_right, atlas_cii.volumes])) - 1;


%% compute exhaustive cosine similarities of task topographies

left_ctx_roi = unique(atlas_cii.cortex_left);
right_ctx_roi = unique(atlas_cii.cortex_right);
subctx_roi = unique(atlas_cii.volumes);
left_ctx_roi(left_ctx_roi == 0) = [];
right_ctx_roi(right_ctx_roi == 0) = [];
subctx_roi(subctx_roi == 0) = [];

% load topographies
topos = cell(n_roi,n_subj);
good_topo = true(1,n_subj);
parfor i = 1:height(tbl0)
    try
        contrast_file = dir([data_root, '/results/', ...
                sprintf('%d/all_tasks/standardized_contrasts/merged_cifti.dscalar.nii', ...
                tbl0.Subject(i))]);

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
            topos{r,i} = these_contrasts.(struct)(:,ind);
            s = zeros(size(topos{r,i},1),1);
            for j = 1:size(topos{r,i},1)
                s(j) = norm(topos{r,i}(j,:));
            end
            topos{r,i} = topos{r,i}./s;
        end
    catch
        good_topo(i) = false;
        warning('Failed to import topographies for subject %d', tbl0.Subject(i));
    end
end

balanced_mean_op = [1/7*repmat(1/2,1,10), 1/7*repmat(1/5,1,5), 1/7*repmat(1/8,1,8)];

% compute similarities
roi_topo = nan(n_subj, n_subj, n_roi);
for r = 1:n_roi
    fprintf('Compute topographic similarity for region %d\n',r);
    for i = 1:n_subj
        if ~good_topo(i)
            continue;
        end

        for j = 1:n_subj
            if ~good_topo(j)
                 continue;
            end
            roi_topo(i,j,r) = balanced_mean_op*diag(topos{r,i}*topos{r,j}');
        end
    end
end

clear topos

save('task_topographic_similarities.mat','roi_topo','good_topo','-v7.3');