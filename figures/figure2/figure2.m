close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));

addpath('../../matlab_libraries');

cm = colormap('jet');
cm1 = colormap(gcf,'hot');
cmlines = colormap('lines');
cmlines(3:4,:) = [];
colormap('parula');
close all;

% these parameters affect the appearance of the 3D geometries
convexity = 10;
vector_scale = 1.5;
qvarargin = {'MaxHeadSize', 0.5, 'LineWidth', 2};

fs=config.matlab_disp_scheme.fontsize;

dc_color = config.matlab_disp_scheme.color_main;
dc_color_light = config.matlab_disp_scheme.color_light;

noise = 'whitened';

%% import atlas and get region names

atlas_cii = cifti_read(config.canlab2024.path);
atlas_labels = atlas_cii.diminfo{2}.maps.table(2:end); % drop first label, it corresponds to 0-valued vertices, i.e. the medial wall
roi_labels = {atlas_labels.name}; 

atlas_cii = get_cifti_data(config.canlab2024.path);

subj_pair_id = config.exemplary_dyad;

%% import atlas in cifti space
cmap = zeros(length(atlas_labels),3);
for i = 1:length(atlas_labels)
    cmap(i,:) = atlas_labels(i).rgba(1:3);
end

uni_label = 'Ctx_V1_L';
uni = find(contains(roi_labels, uni_label));

trans_label = 'Ctx_p9-46v_R';
trans = find(contains(roi_labels, strrep(trans_label,'-','_')));


hcp_dir = config.python_libraries.hcp_utils;
surf_L = gifti(fullfile(hcp_dir, 'hcp_utils/data/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii'));
surf_R = gifti(fullfile(hcp_dir, 'hcp_utils/data/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii'));


sid1 = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);
sid = sid1;

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

% this order corresponds to the order of first level GLM EVs
task_names = {'Faces','Shapes','Punish','Reward','Random','TOM',...
    'Math','Story','Match','Rel','LFoot','LHand','RFoot','RHand','Tongue',...
    '2BK Body','2BK Face','2BK Place','2BK Tool',...
    '0BK Body','0BK Face','0BK Place','0BK Tool'};

modality = {'Vis','Vis','Vis','Vis','Vis','Vis',...
    'Aud','Aud','Vis','Vis','Mot','Mot','Mot','Mot','Mot',...
    'Vis','Vis','Vis','Vis','Vis','Vis','Vis','Vis'};

% clf labels are in a different order from task_names because they come
% from a different first level GLM. We need to resort them.
clf_resort = [2,1,3,4,6,5,8,7,10,9,12,15,14,11,13,20,18,21,16,17,22,23,19];

tasks = [1,4:5];
cond = ismember(task_labels, tasks);

topo1 = get_cifti_data(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/merged_cifti.dscalar.nii', sid.Var1(subj_pair_id), noise));
topo2 = get_cifti_data(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/merged_cifti.dscalar.nii', sid.Var2(subj_pair_id), noise));

uni_mask = atlas_cii.cortex_left == uni;

trans_mask = atlas_cii.cortex_right == trans;

%% plot regions of interest and atlas
%{
figure(1);
clf
t0 = tiledlayout(1,2,'Padding','none','TileSpacing','compact');
ax1 = nexttile();
atlas_medial = fmridisplay();

atlas_medial = surface(atlas_medial, 'axes', ax1, 'direction', 'hcp inflated left', 'orientation', 'medial', 'disableVis3d');
    
cdata = atlas_cii.cortex_left;
faces = atlas_medial.surface{1}.object_handle.Faces;
tri = atlas_medial.surface{1}.object_handle.Vertices;
[~,exteriorPoints] = findExteriorPoints(faces, tri, uni_mask);
cdata(exteriorPoints) = 359;

plot_to_surf(cdata',atlas_medial.surface{1}.object_handle, 'indexmap', 'colormap', [cmap(1:358,:); [0,0,0]]);

title(ax1, strrep(uni_label,'_',' '),'FontWeight','bold','fontsize',fs);


ax2 = nexttile();
atlas_lateral = fmridisplay();

atlas_lateral = surface(atlas_lateral, 'axes', ax2, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
cdata = atlas_cii.cortex_right;
faces = atlas_lateral.surface{1}.object_handle.Faces;
tri = atlas_lateral.surface{1}.object_handle.Vertices;
[~,exteriorPoints] = findExteriorPoints(faces, tri, trans_mask);
cdata(exteriorPoints) = 359;

plot_to_surf(cdata',atlas_lateral.surface{1}.object_handle, 'indexmap', 'colormap', [cmap(1:358,:); [0,0,0]]);

title(ax2, strrep(trans_label,'_',' '),'FontWeight','bold','fontsize',fs)

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2), 380,143]);

exportgraphics(gcf,sprintf('panels_%s/atlas.png',noise),'ContentType','image','Resolution',300);
%}
%% Unimodal topographies

topo1_uni = topo1.cortex_left;
topo2_uni = topo2.cortex_left;

mtopo1_uni = topo1_uni(:,uni_mask);
mtopo2_uni = topo2_uni(:,uni_mask);

figure(2);
t0 = tiledlayout(3,2,'TileSpacing','compact','padding','none');

uni_topo_tile1 = tiledlayout(t0,3,2, 'TileSpacing', 'compact', 'padding', 'none');
uni_topo_tile1.Layout.Tile = 1;
uni_topo_tile1.Layout.TileSpan = [3,1];

uni_topo_tile2 = tiledlayout(t0,3,2, 'TileSpacing', 'compact', 'padding', 'none');
uni_topo_tile2.Layout.Tile = 2;
uni_topo_tile2.Layout.TileSpan = [3,1];


uni_topo_disp1 = fmridisplay();

t = cell(3,2);
for i = 1:numel(t)
    t{i} = nexttile(uni_topo_tile1);
    axis off;
end

data = cat(3,mtopo1_uni, mtopo2_uni);
cmaprange = prctile(data(:),[2.5,97.5]);
cmaprange = [-1*max(abs(cmaprange)), max(abs(cmaprange))];

cm2 = flip(cm1(:,[3,2,1]));
cm = [cm2(1:2:end,:); cm1(1:2:end,:)]; 

[~, exteriorPoints] = findExteriorPoints(surf_R.faces, surf_R.vertices, uni_mask);

these_tasks = task_labels(ismember(task_labels, tasks));
these_names = task_names(ismember(task_labels, these_tasks));
these_task_ind = find(ismember(task_labels, these_tasks));
for i = 1:numel(t)
    color = cmlines(ceil(i/2),:);

    if floor(i/2) == i/2
        color = min(1.25*color,1);
    else
        color = 0.75*color;
    end

    uni_topo_disp1 = surface(uni_topo_disp1, 'axes', t{i}, 'direction', 'hcp inflated left', 'orientation', 'medial', 'disableVis3d');
    
    [~,cbar1, cbar2] = plot_to_surf(topo1_uni(these_task_ind(i), :)'.*uni_mask(:), uni_topo_disp1.surface{i}.object_handle,...
        'colorbar','cmaprange',cmaprange,'colormap',cm);
    uni_topo_disp1.surface{i}.object_handle.FaceVertexCData(exteriorPoints) = 1;
    delete([cbar1, cbar2]);

    % gray out unused values
    this_cm = colormap(t{i});
    this_cm(1:127,:) = repmat(this_cm(128,:),127,1); 
    this_cm(1:100,:) = repmat(color,100,1);
    
    used_ind = unique(uni_topo_disp1.surface{i}.object_handle.FaceVertexCData);
    unused_ind = 129:size(this_cm,1);
    unused_ind = unused_ind(~ismember(unused_ind, min(floor(used_ind)):max(ceil(used_ind)))); % 128 is the surface gray;
    this_cm(unused_ind,:) = repmat(this_cm(128,:), length(unused_ind), 1);
    colormap(t{i}, this_cm);

    zlim(t{i}, [-15,18]);
    ylim(t{i}, [-103,-53]);

    title(t{i},sprintf('%s', these_names{i}),'Color', color, 'fontsize', fs-2);
end
sgtitle(uni_topo_tile1, sprintf('Participant A'),'fontsize',fs+1);


uni_topo_disp2 = fmridisplay();

t = cell(3,2);
for i = 1:numel(t)
    t{i} = nexttile(uni_topo_tile2);
    axis off;
end

data = cat(3,mtopo1_uni, mtopo2_uni);
cmaprange = prctile(data(:),[2.5,97.5]);
cmaprange = [-1*max(abs(cmaprange)), max(abs(cmaprange))];

cm2 = flip(cm1(:,[3,2,1]));
cm = [cm2(1:2:end,:); cm1(1:2:end,:)]; 

[~, exteriorPoints] = findExteriorPoints(surf_R.faces, surf_R.vertices, uni_mask);

these_tasks = task_labels(ismember(task_labels, tasks));
these_names = task_names(ismember(task_labels, these_tasks));
these_task_ind = find(ismember(task_labels, these_tasks));
for i = 1:numel(t)
    color = cmlines(ceil(i/2),:);

    if floor(i/2) == i/2
        color = min(1.25*color,1);
    else
        color = 0.75*color;
    end

    uni_topo_disp2 = surface(uni_topo_disp2, 'axes', t{i}, 'direction', 'hcp inflated left', 'orientation', 'medial', 'disableVis3d');
    
    [~,cbar1, cbar2] = plot_to_surf(topo2_uni(these_task_ind(i), :)'.*uni_mask(:), uni_topo_disp2.surface{i}.object_handle,...
        'colorbar','cmaprange',cmaprange,'colormap',cm);
    uni_topo_disp2.surface{i}.object_handle.FaceVertexCData(exteriorPoints) = 1;
    if i ~= numel(t)
        delete([cbar1, cbar2]);
    end

    % gray out unused values
    this_cm = colormap(t{i});
    this_cm(1:127,:) = repmat(this_cm(128,:),127,1); 
    this_cm(1:100,:) = repmat(color,100,1);
    
    used_ind = unique(uni_topo_disp2.surface{i}.object_handle.FaceVertexCData);
    unused_ind = 129:size(this_cm,1);
    unused_ind = unused_ind(~ismember(unused_ind, min(floor(used_ind)):max(ceil(used_ind)))); % 128 is the surface gray;
    this_cm(unused_ind,:) = repmat(this_cm(128,:), length(unused_ind), 1);
    colormap(t{i}, this_cm);

    zlim(t{i}, [-15,18]);
    ylim(t{i}, [-103,-53]);

    title(t{i},sprintf('%s', these_names{i}),'Color', color,'fontsize',fs-2);
end
sgtitle(uni_topo_tile2, sprintf('Participant B'),'fontsize',fs+1);

sgtitle(t0,sprintf('Common Topographies (%s)', strrep(uni_label,'_',' ')),'FontWeight','bold','fontsize',fs+2);

t0.Position(2) = 0.15;
t0.Position(4)=0.64;

cbar1.Location= 'southoutside';
cbar1.Position(1) = 0.55;
cbar1.Position(2) = 0.08;
cbar1.Position(3) = 0.4;
cbar1.Position(4) = 0.03;

cbar2.Location= 'southoutside';
cbar2.Position(1) = 0.1;
cbar2.Position(2) = 0.08;
cbar2.Position(3) = 0.4;
cbar2.Position(4) = 0.03;

pos = get(gcf,'Position');
set(gcf,'Position', [932,533,385,315])

exportgraphics(gcf,sprintf('panels_%s/common_topographies.png',noise),'ContentType','image','Resolution',300);

% print descriptive statistics
cosim_uni = zeros(size(mtopo1_uni,1),1);
for i = 1:size(mtopo1_uni,1)
    cosim_uni(i) = mtopo1_uni(i,:)*mtopo2_uni(i,:)'/(norm(mtopo1_uni(i,:))*norm(mtopo2_uni(i,:)));
end
sprintf('Mean unimodal similarity: %0.3f (cosim, all tasks)',mean(cosim_uni))
sprintf('Mean unimodal similarity: %0.3f (cosim, 6 tasks)',mean(cosim_uni(these_task_ind)))
    

% print descriptive statistics
cosim_x_task_uni_A = zeros(size(mtopo1_uni,1));
for i = 1:size(mtopo1_uni,1)
    for j = 1:size(mtopo1_uni,1)
        cosim_x_task_uni_A(i,j) = mtopo1_uni(i,:)*mtopo1_uni(j,:)'/(norm(mtopo1_uni(i,:))*norm(mtopo1_uni(j,:)));
    end
end
disp('Unimodal similarity across tasks, participant A:');
disp(cosim_x_task_uni_A(these_task_ind,these_task_ind))


cosim_x_task_uni_B = zeros(size(mtopo2_uni,1));
for i = 1:size(mtopo2_uni,1)
    for j = 1:size(mtopo2_uni,1)
        cosim_x_task_uni_B(i,j) = mtopo2_uni(i,:)*mtopo2_uni(j,:)'/(norm(mtopo2_uni(i,:))*norm(mtopo2_uni(j,:)));
    end
end
disp('Unimodal similarity across tasks, participant B:');
disp(cosim_x_task_uni_B(these_task_ind,these_task_ind))

%% Transmodal topographies

topo1_trans = topo1.cortex_right;
topo2_trans = topo2.cortex_right;

mtopo1_trans = topo1_trans(:,trans_mask);
mtopo2_trans = topo2_trans(:,trans_mask);

figure(3);
clf
t1 = tiledlayout(3,2,'TileSpacing','compact','padding','none');

trans_topo_tile1 = tiledlayout(t1,3,2, 'TileSpacing', 'compact', 'padding', 'none');
trans_topo_tile1.Layout.Tile = 1;
trans_topo_tile1.Layout.TileSpan = [3,1];

trans_topo_tile2 = tiledlayout(t1,3,2, 'TileSpacing', 'compact', 'padding', 'none');
trans_topo_tile2.Layout.Tile = 2;
trans_topo_tile2.Layout.TileSpan = [3,1];


trans_topo_disp1 = fmridisplay();

t = cell(3,2);
for i = 1:numel(t)
    t{i} = nexttile(trans_topo_tile1);
    axis off;
end

data = cat(3,mtopo1_trans, mtopo2_trans);
cmaprange = prctile(data(:),[2.5,97.5]);
cmaprange = [-1*max(abs(cmaprange)), max(abs(cmaprange))];

cm2 = flip(cm1(:,[3,2,1]));
cm = [cm2(1:2:end,:); cm1(1:2:end,:)]; 

[~, exteriorPoints] = findExteriorPoints(surf_R.faces, surf_R.vertices, trans_mask);

these_tasks = task_labels(ismember(task_labels, tasks));
these_names = task_names(ismember(task_labels, these_tasks));
these_task_ind = find(ismember(task_labels, these_tasks));
for i = 1:numel(t)
    color = cmlines(ceil(i/2),:);

    if floor(i/2) == i/2
        color = min(1.25*color,1);
    else
        color = 0.75*color;
    end

    trans_topo_disp1 = surface(trans_topo_disp1, 'axes', t{i}, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
    [~,cbar1, cbar2] = plot_to_surf(topo1_trans(these_task_ind(i), :)'.*trans_mask(:), trans_topo_disp1.surface{i}.object_handle,...
        'colorbar','cmaprange',cmaprange,'colormap',cm);
    trans_topo_disp1.surface{i}.object_handle.FaceVertexCData(exteriorPoints) = 1;
    delete([cbar1, cbar2]);
    set(t{i}, 'view', [135.9349, 5.9202]);

    % gray out unused values
    this_cm = colormap(t{i});
    this_cm(1:127,:) = repmat(this_cm(128,:),127,1); 
    this_cm(1:100,:) = repmat(color,100,1);
    
    used_ind = unique(trans_topo_disp1.surface{i}.object_handle.FaceVertexCData);
    unused_ind = 129:size(this_cm,1);
    unused_ind = unused_ind(~ismember(unused_ind, min(floor(used_ind)):max(ceil(used_ind)))); % 128 is the surface gray;
    this_cm(unused_ind,:) = repmat(this_cm(128,:), length(unused_ind), 1);
    colormap(t{i}, this_cm);

    zlim(t{i}, [10,40]);
    ylim(t{i}, [30,55]);
    xlim(t{i}, [35,55]);

    title(t{i},sprintf('%s', these_names{i}),'Color', color, 'fontsize', fs-2);
end
sgtitle(trans_topo_tile1, sprintf('Participant A'), 'fontsize', fs+1);




trans_topo_disp2 = fmridisplay();

t = cell(3,2);
for i = 1:numel(t)
    t{i} = nexttile(trans_topo_tile2);
    axis off;
end

data = cat(3,mtopo1_trans, mtopo2_trans);
cmaprange = prctile(data(:),[2.5,97.5]);
cmaprange = [-1*max(abs(cmaprange)), max(abs(cmaprange))];

cm2 = flip(cm1(:,[3,2,1]));
%neg_cmap_space = round(linspace(round(length(cm1)*-1*cmaprange(1)/cmaprange(2)), length(cm1), length(cm1)/2));
%cm = [cm2(neg_cmap_space,:); cm1(1:2:end,:)]; 
cm = [cm2(1:2:end,:); cm1(1:2:end,:)]; 

[~, exteriorPoints] = findExteriorPoints(surf_R.faces, surf_R.vertices, trans_mask);

these_tasks = task_labels(ismember(task_labels, tasks));
these_names = task_names(ismember(task_labels, these_tasks));
these_task_ind = find(ismember(task_labels, these_tasks));
for i = 1:numel(t)
    color = cmlines(ceil(i/2),:);

    if floor(i/2) == i/2
        color = min(1.25*color,1);
    else
        color = 0.75*color;
    end

    trans_topo_disp2 = surface(trans_topo_disp2, 'axes', t{i}, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
    [~,cbar1, cbar2] = plot_to_surf(topo2_trans(these_task_ind(i), :)'.*trans_mask(:), trans_topo_disp2.surface{i}.object_handle,...
        'colorbar','cmaprange',cmaprange,'colormap',cm);
    trans_topo_disp2.surface{i}.object_handle.FaceVertexCData(exteriorPoints) = 1;
    if i ~= numel(t)
        delete([cbar1, cbar2]);
    end
    set(t{i}, 'view', [135.9349, 5.9202]);

    % gray out unused values
    this_cm = colormap(t{i});
    this_cm(1:127,:) = repmat(this_cm(128,:),127,1); 
    this_cm(1:100,:) = repmat(color,100,1);
    
    used_ind = unique(trans_topo_disp2.surface{i}.object_handle.FaceVertexCData);
    unused_ind = 129:size(this_cm,1);
    unused_ind = unused_ind(~ismember(unused_ind, min(floor(used_ind)):max(ceil(used_ind)))); % 128 is the surface gray;
    this_cm(unused_ind,:) = repmat(this_cm(128,:), length(unused_ind), 1);
    colormap(t{i}, this_cm);

    zlim(t{i}, [10,40]);
    ylim(t{i}, [30,55]);
    xlim(t{i}, [35,55]);

    title(t{i},sprintf('%s', these_names{i}),'Color', color, 'fontsize', fs-2);
end
sgtitle(trans_topo_tile2, sprintf('Participant B'), 'fontsize', fs+1);

sgtitle(t1,sprintf('Idiosyncratic topographies (%s)',strrep(trans_label,'_',' ')),'FontWeight','bold', 'fontsize', fs+2);

t1.Position(2) = 0.11;
t1.Position(4)=0.72;

cbar1.Location= 'southoutside';
cbar1.Position(1) = 0.55;
cbar1.Position(2) = 0.06;
cbar1.Position(3) = 0.4;
cbar1.Position(4) = 0.03;

cbar2.Location= 'southoutside';
cbar2.Position(1) = 0.1;
cbar2.Position(2) = 0.06;
cbar2.Position(3) = 0.4;
cbar2.Position(4) = 0.03;

pos = get(gcf,'Position');
set(gcf,'Position',[937,82, 385, 380])

exportgraphics(gcf,sprintf('panels_%s/idiosyncratic_topographies.png',noise),'ContentType','image','Resolution',300);

% print descriptive statistics
cosim_trans = zeros(size(mtopo1_trans,1),1);
for i = 1:size(mtopo1_trans,1)
    cosim_trans(i) = mtopo1_trans(i,:)*mtopo2_trans(i,:)'/(norm(mtopo1_trans(i,:))*norm(mtopo2_trans(i,:)));
end
sprintf('Mean transmodal similarity: %0.3f (cosim, all tasks)',mean(cosim_trans))
sprintf('Mean transmodal similarity: %0.3f (cosim, 6 tasks)',mean(cosim_trans(these_task_ind)))


% print descriptive statistics
cosim_x_task_trans_A = zeros(size(mtopo1_trans,1));
for i = 1:size(mtopo1_trans,1)
    for j = 1:size(mtopo1_trans,1)
        cosim_x_task_trans_A(i,j) = mtopo1_trans(i,:)*mtopo1_trans(j,:)'/(norm(mtopo1_trans(i,:))*norm(mtopo1_trans(j,:)));
    end
end
disp('Transmodal similarity across tasks, participant A:');
disp(cosim_x_task_trans_A(these_task_ind,these_task_ind))


cosim_x_task_trans_B = zeros(size(mtopo2_trans,1));
for i = 1:size(mtopo2_trans,1)
    for j = 1:size(mtopo2_trans,1)
        cosim_x_task_trans_B(i,j) = mtopo2_trans(i,:)*mtopo2_trans(j,:)'/(norm(mtopo2_trans(i,:))*norm(mtopo2_trans(j,:)));
    end
end
disp('Transmodal similarity across tasks, participant B:');
disp(cosim_x_task_trans_B(these_task_ind,these_task_ind))


%% compute unimodal projections
% solve procrustes problem rotating topo1 onto topo2, ignoring vector
% magnitudes (to balance across conditions for visualization purposes)
ntopo1_uni = mtopo1_uni./sqrt(sum(mtopo1_uni.^2,2));
ntopo2_uni = mtopo2_uni./sqrt(sum(mtopo2_uni.^2,2));

[u,~,v] = svd(ntopo2_uni(cond,:)'*ntopo1_uni(cond,:));
rot = u*v';

[u2,s2,v2] = svd(ntopo2_uni(cond,:), 'econ');

% compute projections based on unnormalized data
proj1_uni = mtopo1_uni(cond,:)*rot'*v2;
proj2_uni = mtopo2_uni(cond,:)*v2;

%% compute transmodal projections

% solve procrustes problem rotating topo1 onto topo2, ignoring vector
% magnitudes (to balance across conditions for visualization purposes)
ntopo1_trans = mtopo1_trans./sqrt(sum(mtopo1_trans.^2,2));
ntopo2_trans = mtopo2_trans./sqrt(sum(mtopo2_trans.^2,2));

[u,~,v] = svd(ntopo2_trans(cond,:)'*ntopo1_trans(cond,:));
rot = u*v';

rot_ntopo1 = (rot*ntopo1_trans(cond,:)')';

[u2,s2,v2] = svd(ntopo2_trans(cond,:), 'econ');

% compute projections based on unnormalized data
proj1_trans = mtopo1_trans(cond,:)*rot'*v2;
proj2_trans = mtopo2_trans(cond,:)*v2;

%% import rdms and binary clf performances

rdm_root = '../../derivatives/hcp_glm_msmall_grayord_spm/results/';

if strcmp(noise,'standardized')
    standardized1 = readmatrix(sprintf('%s/%d/all_tasks/rsa/stddist/standardized_distance.csv',rdm_root, sid.Var1(subj_pair_id)));
    standardized2 = readmatrix(sprintf('%s/%d/all_tasks/rsa/stddist/standardized_distance.csv',rdm_root, sid.Var2(subj_pair_id)));
elseif strcmp(noise,'whitened')
    standardized1 = readmatrix(sprintf('%s/%d/all_tasks/rsa/crossnobis/crossnobis_distance.csv',rdm_root, sid.Var1(subj_pair_id)));
    standardized2 = readmatrix(sprintf('%s/%d/all_tasks/rsa/crossnobis/crossnobis_distance.csv',rdm_root, sid.Var2(subj_pair_id)));
else
    error('Did not understand noise model choice "%s"',noise)
end

conf_mats1 = readtable(sprintf('../../derivatives/single_blocks_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/binary_clf_performance.csv', sid.Var1(subj_pair_id),noise),'ReadRowNames',true,'ReadVariableNames',true);
conf_mats2 = readtable(sprintf('../../derivatives/single_blocks_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/binary_clf_performance.csv',sid.Var2(subj_pair_id),noise),'ReadRowNames',true,'ReadVariableNames',true);

uni_rdm1 = squareform(standardized1(:, uni));
uni_rdm2 = squareform(standardized2(:, uni));

trans_rdm1 = squareform(standardized1(:, trans));
trans_rdm2 = squareform(standardized2(:, trans));

uni_confmat1 = squareform(conf_mats1(contains(conf_mats1.Properties.RowNames, strrep(uni_label,'-','_')),:).Variables);
uni_confmat2 = squareform(conf_mats2(contains(conf_mats2.Properties.RowNames, strrep(uni_label,'-','_')),:).Variables);

trans_confmat1 = squareform(conf_mats1(contains(conf_mats1.Properties.RowNames, strrep(trans_label,'-','_')),:).Variables);
trans_confmat2 = squareform(conf_mats2(contains(conf_mats2.Properties.RowNames, strrep(trans_label,'-','_')),:).Variables);

% resort order to match RDMs
uni_confmat1 = uni_confmat1(clf_resort, clf_resort);
uni_confmat2 = uni_confmat2(clf_resort, clf_resort);
trans_confmat1 = trans_confmat1(clf_resort, clf_resort);
trans_confmat2 = trans_confmat2(clf_resort, clf_resort);


disp('Unimodal Std Distances Participant A:')
disp(uni_rdm1(these_task_ind,these_task_ind))

disp('unimodal Std Distances Participant B:')
disp(uni_rdm2(these_task_ind,these_task_ind))

disp('Transmodal Std Distances Participant A:')
disp(trans_rdm1(these_task_ind,these_task_ind))

disp('Transmodal Std Distances Participant B:')
disp(trans_rdm2(these_task_ind,these_task_ind))

%% plot unimodal geometry

figure(4);
clf
uni_rdm_tile = tiledlayout(2,2,'TileSpacing','compact','padding','tight');

v = [-43.2446, 10.7065];

rdm_ind = ismember(task_labels,tasks);
crange = prctile([standardized1(rdm_ind, uni); standardized2(rdm_ind, uni)], [2.5, 97.5]);

ax_ticks = cell(1,sum(rdm_ind));
for i = 1:sum(rdm_ind)
    ind = find(rdm_ind);
    ax_ticks{i} = sprintf('%s (%s)',task_names{ind(i)}, modality{ind(i)});
end

nexttile(uni_rdm_tile);
imagesc(uni_rdm1(rdm_ind,rdm_ind));
caxis(crange);
axis square;
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks,'FontSize',fs-2);
if strcmp(noise ,'whitened')
    title({'Response Dissimilarity', '(Crossnobis Dist)', 'Participant A'},'FontWeight','normal','fontsize',fs)
elseif strcmp(noise, 'standardized')
    title({'Response Dissimilarity', '(Unbiased t-Dist)', 'Participant A'},'FontWeight','normal','fontsize',fs)
end

nexttile(uni_rdm_tile);
imagesc(uni_rdm2(rdm_ind,rdm_ind));
caxis(crange)
axis square
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks, 'FontSize',fs-2);
title({'', '','Participant B'},'FontWeight','normal','fontsize',fs)

ax1 = nexttile(uni_rdm_tile);
view(3)
view(v);
hold on;

% construct faces
shp = alphaShape(proj1_uni(:,1:3), convexity*norm(proj1_uni(:,1:3)));
p = shp.plot();
p.FaceAlpha = 0.3;
p.EdgeColor = [0,0.5,0];
lightRestoreSingle; 
axis image; 
material dull

these_tasks = task_labels(ismember(task_labels, tasks));
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
        else
            this_color = min(1.25*cmlines(i,:),1);
        end
    
        this_data = proj1_uni(this_task(j),1:3);
    
        x = this_data(:,1)*vector_scale;
        y = this_data(:,2)*vector_scale;
        z = this_data(:,3)*vector_scale;
    
        zero = zeros(size(this_data,1),1);
        quiver3(zero, zero, zero, x, y, z, qvarargin{:}, 'color', this_color);
    end
end

xl1 = xlim;
yl1 = ylim;
zl1 = zlim;

grid on;

camlight(135,-60);
camlight(-135,-60);
camlight(0,135);


ax2 = nexttile(uni_rdm_tile);
view(3);
view(v);
hold on;

shp = alphaShape(proj2_uni(:,1:3), convexity*norm(proj2_uni(:,1:3)));
p = shp.plot();
p.FaceAlpha = 0.3;
p.EdgeColor = [0,0,0.5];
p.FaceColor = p.FaceColor([3,2,1]);
lightRestoreSingle; 
axis image; 
material dull

these_tasks = task_labels(ismember(task_labels, tasks));
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
        else
            this_color = min(1.25*cmlines(i,:),1);
        end
    
        this_data = proj2_uni(this_task(j),1:3);
    
        x = this_data(:,1)*vector_scale;
        y = this_data(:,2)*vector_scale;
        z = this_data(:,3)*vector_scale;
    
        zero = zeros(size(this_data,1),1);
        quiver3(zero, zero, zero, x, y, z, qvarargin{:}, 'color', this_color);
    end
end


xl2 = xlim;
yl2 = ylim;
zl2 = zlim;

xl = [min([xl1,xl2]), max([xl1, xl2])];
yl = [min([yl1,yl2]), max([yl1, yl2])];
zl = [min([zl1,zl2]), max([zl1, zl2])];

grid on;

xlabel(ax1, 'EV_1','FontSize',fs-3)
ylabel(ax1, 'EV_2','FontSize',fs-3)
zlabel(ax1, 'EV_3','FontSize',fs-3)

xlabel(ax2, 'R*EV_1','FontSize',fs-3)
ylabel(ax2, 'R*EV_2','FontSize',fs-3)
zlabel(ax2, 'R*EV_3','FontSize',fs-3)

title(ax1,{'Evoked Response', 'Subspace Projection'},'FontWeight','normal','fontsize',fs)
title(ax2,{'','Evoked Response', 'Rotated Subspace Projection'},'FontWeight','normal','fontsize',fs)


set(ax1,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], 'ZTickLabels', [], 'Xlim', xl, 'YLim', yl, 'Zlim', zl);
set(ax2,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], 'ZTickLabels', [], 'Xlim', xl, 'YLim', yl, 'Zlim', zl);

camlight(135,-60);
camlight(-135,-60);
camlight(0,135);

set(ax2, 'view', get(ax1,'View'))

sgtitle(uni_rdm_tile, {'Geometry of common topographies',sprintf('with common representations (%s)',regexprep(strrep(strrep(uni_label,'_',' '),'Ctx ',''),' [LR]',''))},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1329,566, 400, 400]);

exportgraphics(gcf,sprintf('panels_%s/common_topographies_common_representations.png',noise), ...
    'ContentType','image','Resolution',300);


% compute unbiased cosine similarity
x = uni_rdm1(rdm_ind, rdm_ind);
y = uni_rdm2(rdm_ind, rdm_ind);
tril_ind = tril(true(size(x)),-1);
x = x(tril_ind);
y = y(tril_ind);
sprintf('Cosine of unimodal RDMs: %0.3f\n', x'*y/(norm(x)*norm(y)))

%% plot unimodal clf and vector space

figure(6);
clf
uni_clf_tile = tiledlayout(2,2,'TileSpacing','compact','padding','tight');

v = [-43.2446, 10.7065];

rdm_ind = ismember(task_labels,tasks);

nexttile(uni_clf_tile );
imagesc(uni_confmat1(rdm_ind,rdm_ind));
caxis([0.5,1]);
axis square;
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks,'FontSize',fs-2);
title({'Clf Perf', '(Bal. Accuracy)', 'Participant A'},'FontWeight','normal','fontsize',fs)

nexttile(uni_clf_tile);
imagesc(uni_confmat2(rdm_ind,rdm_ind));
caxis([0.5,1])
axis square
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks,'FontSize',fs-2);
title({'Clf Perf', '(Bal. Accuracy)', 'Participant B'},'FontWeight','normal','fontsize',fs)


nexttile(uni_clf_tile);
this_rdm = uni_rdm1(rdm_ind,rdm_ind);
this_clf = uni_confmat1(rdm_ind, rdm_ind);

tril_ind = tril(true(size(this_rdm)),-1);
this_rdm_vec = this_rdm(tril_ind);
this_clf_vec = this_clf(tril_ind);

plot(this_rdm_vec, this_clf_vec, 'o', 'color', cmap(uni,:));
h = lsline;
set(h,'color', dc_color, 'LineWidth',3);
ylim([0.4,1.1]);
ylabel({'Clf Perf', '(Bal. Acc.)'},'fontsize',fs)
switch noise
    case 'whitened'
        xlabel({'Crossnobis dist'},'fontsize',fs)
    case 'standardized'
        xlabel({'t-distance'},'fontsize',fs)
end
box off;
title(sprintf('r = %0.3f', corr(this_rdm_vec, this_clf_vec)),'fontsize',fs)
set(gca,'fontsize',fs)


nexttile(uni_clf_tile);
this_rdm = uni_rdm2(rdm_ind,rdm_ind);
this_clf = uni_confmat2(rdm_ind, rdm_ind);

tril_ind = tril(true(size(this_rdm)),-1);
this_rdm_vec = this_rdm(tril_ind);
this_clf_vec = this_clf(tril_ind);

plot(this_rdm_vec, this_clf_vec, 'o', 'color', cmap(uni,:));
h = lsline;
set(h,'color', dc_color, 'LineWidth',3);
ylim([0.4,1.1]);
ylabel({'Clf Perf', '(Bal. Acc.)'},'fontsize',fs)
switch noise
    case 'whitened'
        xlabel({'Crossnobis dist'},'fontsize',fs)
    case 'standardized'
        xlabel({'t-distance'},'fontsize',fs)
end
box off;
title(sprintf('r = %0.3f', corr(this_rdm_vec, this_clf_vec)),'fontsize',fs)
set(gca,'fontsize',fs)

sgtitle(uni_clf_tile, {'Geometry measures decodability',regexprep(strrep(strrep(uni_label,'_',' '),'Ctx ',''),' [LR]','')},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1733, 572, 400, 400]);

exportgraphics(gcf,sprintf('panels_%s/common_topographies_common_representations_sup.png',noise), ...
    'ContentType','image','Resolution',300);


%% transmodal geometry

figure(5);
clf
trans_rdm_tile = tiledlayout(2,2, 'TileSpacing', 'compact', 'padding', 'none');

v = [-43.2446, 10.7065];


rdm_ind = ismember(task_labels,tasks);
crange = prctile([standardized1(rdm_ind, trans); standardized2(rdm_ind, trans)], [2.5, 97.5]);

nexttile(trans_rdm_tile);
imagesc(trans_rdm1(rdm_ind,rdm_ind));
caxis(crange);
axis square;
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks);
switch noise
    case 'whitened'
        title({'Response Dissimilarity', '(Crossnobis Dist)', 'Participant A'},'FontWeight','normal','fontsize',fs)
    case 'standardized'
        title({'Response Dissimilarity', '(Unbiased t-Dist)', 'Participant A'},'FontWeight','normal','fontsize',fs)
end

nexttile(trans_rdm_tile);
imagesc(trans_rdm2(rdm_ind,rdm_ind));
caxis(crange)
axis square
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks);
switch noise
    case 'whitened'
        title({'Response Dissimilarity', '(Crossnobis Dist)', 'Participant B'},'FontWeight','normal','fontsize',fs)
    case 'standardized'
        title({'Response Dissimilarity', '(Unbiased t-Dist)', 'Participant B'},'FontWeight','normal','fontsize',fs)
end


ax1 = nexttile(trans_rdm_tile);
view(3)
view(v);
hold on;

% construct faces
shp = alphaShape(proj1_trans(:,1:3), convexity*norm(proj1_trans(:,1:3)));
p = shp.plot();
p.FaceAlpha = 0.3;
p.EdgeColor = [0,0.5,0];
lightRestoreSingle; 
axis image; 
material dull

these_tasks = task_labels(ismember(task_labels, tasks));
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
        else
            this_color = min(1.25*cmlines(i,:),1);
        end
    
        this_data = proj1_trans(this_task(j),1:3);
    
        x = this_data(:,1)*vector_scale;
        y = this_data(:,2)*vector_scale;
        z = this_data(:,3)*vector_scale;
    
        zero = zeros(size(this_data,1),1);
        quiver3(zero, zero, zero, x, y, z, qvarargin{:}, 'color', this_color);
    end
end

xl1 = xlim;
yl1 = ylim;
zl1 = zlim;

grid on;

camlight(135,-60);
camlight(-135,-60);
camlight(0,135);


ax2 = nexttile(trans_rdm_tile);
view(3);
view(v);
hold on;

shp = alphaShape(proj2_trans(:,1:3), convexity*norm(proj2_trans(:,1:3)));
p = shp.plot();
p.FaceAlpha = 0.3;
p.EdgeColor = [0,0,0.5];
p.FaceColor = p.FaceColor([3,2,1]);
lightRestoreSingle; 
axis image; 
material dull

these_tasks = task_labels(ismember(task_labels, tasks));
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
        else
            this_color = min(1.25*cmlines(i,:),1);
        end
    
        this_data = proj2_trans(this_task(j),1:3);
    
        x = this_data(:,1)*vector_scale;
        y = this_data(:,2)*vector_scale;
        z = this_data(:,3)*vector_scale;
    
        zero = zeros(size(this_data,1),1);
        quiver3(zero, zero, zero, x, y, z, qvarargin{:}, 'color', this_color);
    end
end


xl2 = xlim;
yl2 = ylim;
zl2 = zlim;

xl = [min([xl1,xl2]), max([xl1, xl2])];
yl = [min([yl1,yl2]), max([yl1, yl2])];
zl = [min([zl1,zl2]), max([zl1, zl2])];

grid on;

xlabel(ax1, 'EV_1')
ylabel(ax1, 'EV_2')
zlabel(ax1, 'EV_3')

xlabel(ax2, 'R*EV_1')
ylabel(ax2, 'R*EV_2')
zlabel(ax2, 'R*EV_3')

title(ax1,{'Evoked Responses', 'Subspace Projection'},'FontWeight','normal','fontsize',fs)
title(ax2,{'', 'Evoked Responses', 'Rotated Subspace Projection'},'FontWeight','normal','fontsize',fs)


set(ax1,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], 'ZTickLabels', [], 'Xlim', xl, 'YLim', yl, 'Zlim', zl);
set(ax2,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], 'ZTickLabels', [], 'Xlim', xl, 'YLim', yl, 'Zlim', zl);

camlight(135,-60);
camlight(-135,-60);
camlight(0,135);

set(ax2, 'view', get(ax1,'View'))

sgtitle(trans_rdm_tile, {'Geometry of idiosyncratic topographies',sprintf('with common representations (%s)',regexprep(strrep(strrep(trans_label,'_',' '),'Ctx ',''),' [LR]',''))},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1322, 69, 400, 400]);

exportgraphics(gcf,sprintf('panels_%s/idiosyncratic_topographies_common_topographies.png',noise), ...
    'ContentType','image','Resolution',300);


% compute unbiased cosine similarity
x = trans_rdm1(rdm_ind, rdm_ind);
y = trans_rdm2(rdm_ind, rdm_ind);
tril_ind = tril(true(size(x)),-1);
x = x(tril_ind);
y = y(tril_ind);
sprintf('Cosine of unimodal RDMs: %0.3f\n', x'*y/(norm(x)*norm(y)))


%% plot classifiers

figure(7);
clf
trans_clf_tile = tiledlayout(2,2,'TileSpacing','compact','padding','tight');

v = [-43.2446, 10.7065];

rdm_ind = ismember(task_labels,tasks);

nexttile(trans_clf_tile);
imagesc(trans_confmat1(rdm_ind,rdm_ind));
caxis([0.5,1]);
axis square;
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks,'FontSize',fs-2);
title({'Clf Perf', '(Bal. Accuracy)', 'Participant A'},'FontWeight','normal','fontsize',fs)

nexttile(trans_clf_tile);
imagesc(trans_confmat2(rdm_ind,rdm_ind));
caxis([0.5,1])
axis square
set(gca,'XTick',1:length(rdm_ind), 'XTickLabels', task_names(rdm_ind),...
    'XTickLabelRotation',90,'FontSize',fs-2);
set(gca,'YTick',1:length(rdm_ind), 'YTickLabels', ax_ticks,'FontSize',fs-2);
title({'Clf Perf', '(Bal. Accuracy)', 'Participant B'},'FontWeight','normal','fontsize',fs)


nexttile(trans_clf_tile);
this_rdm = trans_rdm1(rdm_ind,rdm_ind);
this_clf = trans_confmat1(rdm_ind, rdm_ind);

tril_ind = tril(true(size(this_rdm)),-1);
this_rdm_vec = this_rdm(tril_ind);
this_clf_vec = this_clf(tril_ind);

plot(this_rdm_vec, this_clf_vec, 'o', 'color', cmap(trans,:));
h = lsline;
set(h,'color', dc_color, 'LineWidth',3);
ylim([0.4,1.1]);
ylabel({'Clf Perf', '(Bal. Acc.)'},'fontsize',fs)
switch noise
    case 'whitened'
        xlabel('crossnobis dist', 'fontsize', fs);
    case 'standardized'
        xlabel({'t-distance'},'fontsize',fs)
end
box off;
title(sprintf('r = %0.3f', corr(this_rdm_vec, this_clf_vec)),'fontsize',fs)
set(gca,'fontsize',fs)


nexttile(trans_clf_tile);
this_rdm = trans_rdm2(rdm_ind,rdm_ind);
this_clf = trans_confmat2(rdm_ind, rdm_ind);

tril_ind = tril(true(size(this_rdm)),-1);
this_rdm_vec = this_rdm(tril_ind);
this_clf_vec = this_clf(tril_ind);

plot(this_rdm_vec, this_clf_vec, 'o', 'color', cmap(trans,:));
h = lsline;
set(h,'color', dc_color, 'LineWidth',3);
ylim([0.4,1.1]);
ylabel({'Clf Perf', '(Bal. Acc.)'},'fontsize',fs)
switch noise
    case 'whitened'
        xlabel('crossnobis dist', 'fontsize', fs);
    case 'standardized'
        xlabel({'t-distance'},'fontsize',fs)
end
xl = xlim;
xlim([xl(1),xl(2)*1.1])
box off;
title(sprintf('r = %0.3f', corr(this_rdm_vec, this_clf_vec)),'fontsize',fs)
set(gca,'fontsize',fs)

sgtitle(trans_clf_tile, {'Geometry measures decodability',regexprep(strrep(strrep(trans_label,'_',' '),'Ctx ',''),' [LR]','')},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1733, 75, 400, 400]);

exportgraphics(gcf,sprintf('panels_%s/idiosyncratic_topographies_common_topographies_sup.png',noise), ...
    'ContentType','image','Resolution',300);
