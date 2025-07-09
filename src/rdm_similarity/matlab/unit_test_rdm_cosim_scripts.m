
config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));

%% load RDMs, whitening matrices and metadata

% these input files are all output from the main nipype pipelines that run first level GLMs and RSA on HCP data
d1_all = readmatrix('../../derivatives/hcp_glm_msmall_grayord_spm/results/100307/all_tasks/rsa/crossnobis/crossnobis_distance.csv');
d2_all = readmatrix('/mnt/external/MyDocuments/canlab/hcp/stats/hcp_glm_msmall_grayord_spm/results/992673/all_tasks/rsa/crossnobis/crossnobis_distance.csv');
fid = fopen('/mnt/external/MyDocuments/canlab/hcp/stats/hcp_glm_msmall_grayord_spm/results/100307/all_tasks/rsa/crossnobis/whitening_matrix_out.bin','r');
V1_all = reshape(fread(fid,'float32'), 253*254/2, 518);
fclose(fid);
fid = fopen('/mnt/external/MyDocuments/canlab/hcp/stats/hcp_glm_msmall_grayord_spm/results/992673/all_tasks/rsa/crossnobis/whitening_matrix_out.bin','r');
V2_all = reshape(fread(fid,'float32'), 253*254/2, 518);
fclose(fid);

fid = fopen('/mnt/external/MyDocuments/canlab/hcp/stats/hcp_glm_msmall_grayord_spm/results/100307/all_tasks/rsa/crossnobis/whitening_matrix_out.json'); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
dof1 = jsondecode(str).dof;

fid = fopen('/mnt/external/MyDocuments/canlab/hcp/stats/hcp_glm_msmall_grayord_spm/results/992673/all_tasks/rsa/crossnobis/whitening_matrix_out.json'); 
raw = fread(fid,inf); 
str = char(raw'); 
fclose(fid); 
dof2 = jsondecode(str).dof;

% task identifiers used for balancing.
% 1 - EMOTION
% 2 - GAMBLING
% 3 - LANGUAGE
% 4 - SOCIAL
% 5 - RELATIONAL
% 6 - MOTOR
% 7 - WM
% Must match order of tasks in RDMs
task_ids = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

%% whiten an ROI
roi_ind = 1;
d1 = d1_all(:,roi_ind);
d2 = d2_all(:,roi_ind);
V1 = V1_all(:,roi_ind);
V2 = V2_all(:,roi_ind);

V1 = squareform_with_diag(V1);
V2 = squareform_with_diag(V2);

% pool whitening matrix
V = (dof1*V1 + dof2*V2)/(dof1 + dof2);

% whiten distances
R = chol(V + 1e-6 * eye(size(V)), 'upper');
d1w = d1' / R;
d2w = d2' / R;

% replicate undersampled tasks
d1w_sq = squareform(d1w);
d2w_sq = squareform(d2w);

[replicate_ids, ~, block_labels] = build_replication_info(task_ids);

d1w_sq_exp = d1w_sq(replicate_ids, replicate_ids);
d2w_sq_exp = d2w_sq(replicate_ids, replicate_ids);

% adjust block diagonal terms
d1r = patch_block_diagonals(d1w_sq, d1w_sq_exp, task_ids, replicate_ids, block_labels);
d2r = patch_block_diagonals(d2w_sq, d2w_sq_exp, task_ids, replicate_ids, block_labels);

d1r*d2r'/(norm(d1r)*norm(d2r));

%% whitten all ROIs
tic;
wuc = zeros(518,518);
parfor roi_ind1 = 1:518
    fprintf('Whitening rois (.,%d)\n',roi_ind1);
    for roi_ind2 = 1:518
        d1 = d1_all(:,roi_ind1);
        d2 = d2_all(:,roi_ind2);
        V1 = V1_all(:,roi_ind1);
        V2 = V2_all(:,roi_ind2);
        
        V1 = squareform_with_diag(V1);
        V2 = squareform_with_diag(V2);
        
        % pool whitening matrix
        V = (dof1*V1 + dof2*V2)/(dof1 + dof2);
        
        % whiten distances
        try
            R = chol(V + 1e-6 * eye(size(V)), 'upper');
            d1w = d1' / R;
            d2w = d2' / R;
            
            % replicate undersampled tasks
            d1w_sq = squareform(d1w);
            d2w_sq = squareform(d2w);
            
            [replicate_ids, ~, block_labels] = build_replication_info(task_ids);
            
            d1w_sq_exp = d1w_sq(replicate_ids, replicate_ids);
            d2w_sq_exp = d2w_sq(replicate_ids, replicate_ids);
            
            % adjust block diagonal terms
            d1r = patch_block_diagonals(d1w_sq, d1w_sq_exp, task_ids, replicate_ids, block_labels);
            d2r = patch_block_diagonals(d2w_sq, d2w_sq_exp, task_ids, replicate_ids, block_labels);
            
            wuc(roi_ind1, roi_ind2) = d1r*d2r'/(norm(d1r)*norm(d2r));
        catch
            wuc(roi_ind1, roi_ind2) = nan;
        end
    end
end
toc