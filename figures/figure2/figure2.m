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

dc_color = config.matlab_disp_scheme.color_main(3,:);
dc_color_light = config.matlab_disp_scheme.color_light(3,:);

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

export_fig(gcf,sprintf('panels_%s/common_topographies.png',noise),'-transparent','-r300');

% print descriptive statistics
cosim_uni = zeros(size(mtopo1_uni,1),1);
for i = 1:size(mtopo1_uni,1)
    cosim_uni(i) = mtopo1_uni(i,:)*mtopo2_uni(i,:)'/(norm(mtopo1_uni(i,:))*norm(mtopo2_uni(i,:)));
end
%sprintf('Mean unimodal similarity: %0.3f (cosim, all tasks)',mean(cosim_uni))
sprintf('Mean unimodal similarity: %0.3f (cosim, 6 tasks)',mean(cosim_uni(these_task_ind)))
    

% print descriptive statistics
cosim_x_task_uni_A = zeros(size(mtopo1_uni,1));
for i = 1:size(mtopo1_uni,1)
    for j = 1:size(mtopo1_uni,1)
        cosim_x_task_uni_A(i,j) = mtopo1_uni(i,:)*mtopo1_uni(j,:)'/(norm(mtopo1_uni(i,:))*norm(mtopo1_uni(j,:)));
    end
end
%disp('Unimodal similarity across tasks, participant A:');
%disp(cosim_x_task_uni_A(these_task_ind,these_task_ind))


cosim_x_task_uni_B = zeros(size(mtopo2_uni,1));
for i = 1:size(mtopo2_uni,1)
    for j = 1:size(mtopo2_uni,1)
        cosim_x_task_uni_B(i,j) = mtopo2_uni(i,:)*mtopo2_uni(j,:)'/(norm(mtopo2_uni(i,:))*norm(mtopo2_uni(j,:)));
    end
end
%disp('Unimodal similarity across tasks, participant B:');
%disp(cosim_x_task_uni_B(these_task_ind,these_task_ind))

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

export_fig(gcf,sprintf('panels_%s/idiosyncratic_topographies.png',noise),'-transparent','-r300');

% print descriptive statistics
cosim_trans = zeros(size(mtopo1_trans,1),1);
for i = 1:size(mtopo1_trans,1)
    cosim_trans(i) = mtopo1_trans(i,:)*mtopo2_trans(i,:)'/(norm(mtopo1_trans(i,:))*norm(mtopo2_trans(i,:)));
end
%sprintf('Mean transmodal similarity: %0.3f (cosim, all tasks)',mean(cosim_trans))
sprintf('Mean transmodal similarity: %0.3f (cosim, 6 tasks)',mean(cosim_trans(these_task_ind)))


% print descriptive statistics
cosim_x_task_trans_A = zeros(size(mtopo1_trans,1));
for i = 1:size(mtopo1_trans,1)
    for j = 1:size(mtopo1_trans,1)
        cosim_x_task_trans_A(i,j) = mtopo1_trans(i,:)*mtopo1_trans(j,:)'/(norm(mtopo1_trans(i,:))*norm(mtopo1_trans(j,:)));
    end
end
%disp('Transmodal similarity across tasks, participant A:');
%disp(cosim_x_task_trans_A(these_task_ind,these_task_ind))


cosim_x_task_trans_B = zeros(size(mtopo2_trans,1));
for i = 1:size(mtopo2_trans,1)
    for j = 1:size(mtopo2_trans,1)
        cosim_x_task_trans_B(i,j) = mtopo2_trans(i,:)*mtopo2_trans(j,:)'/(norm(mtopo2_trans(i,:))*norm(mtopo2_trans(j,:)));
    end
end
%disp('Transmodal similarity across tasks, participant B:');
%disp(cosim_x_task_trans_B(these_task_ind,these_task_ind))


%{
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
%}
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

%{
disp('Unimodal Std Distances Participant A:')
disp(uni_rdm1(these_task_ind,these_task_ind))

disp('unimodal Std Distances Participant B:')
disp(uni_rdm2(these_task_ind,these_task_ind))

disp('Transmodal Std Distances Participant A:')
disp(trans_rdm1(these_task_ind,these_task_ind))

disp('Transmodal Std Distances Participant B:')
disp(trans_rdm2(these_task_ind,these_task_ind))
%}
%% Compute unimodal MDS
% Let's stick to the mds of the 6 tasks we show and just rotate in that 
% space. It's a 2D projection of the 6x6 distance matrix
uni_mds1 = cmdscale(real(sqrt(uni_rdm1(cond,cond))));
uni_mds2 = cmdscale(real(sqrt(uni_rdm2(cond,cond))));

d1 = size(uni_mds1,2);
d2 = size(uni_mds2,2);
if d2 > d1
    [d,Z,transform] = procrustes(uni_mds2, uni_mds1);
    
    proj1_uni = Z;
    proj2_uni = uni_mds2;
else
    [d,Z,transform] = procrustes(uni_mds1, uni_mds2);
    
    proj1_uni = uni_mds1;
    proj2_uni = Z;
end
%% plot unimodal geometry

dim = 2;

figure(4);
clf
uni_rdm_tile = tiledlayout(2,2,'TileSpacing','compact','padding','normal');

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
hold on;
these_tasks = task_labels(ismember(task_labels, tasks));
[xl1, yl1] = deal([0,0]);
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    these_names = task_names(task_labels == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
            name = these_names{1};
        else
            this_color = min(1.25*cmlines(i,:),1);
            name = these_names{2};
        end

        this_data = proj1_uni(this_task(j),1:2);
        x = this_data(1);
        y = this_data(2);

        text(x,y,name,'Color',this_color,'FontWeight','bold','FontSize',fs, ...
            'HorizontalAlign','center')
        
        xl1 = [min([xl1,x]), max([xl1, x])];
        yl1 = [min([yl1,y]), max([yl1, y])];
    end
end

grid on;
axis square
xlabel(' ')

ax2 = nexttile(uni_rdm_tile);
hold on;
these_tasks = task_labels(ismember(task_labels, tasks));
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    these_names = task_names(task_labels == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
            name = these_names{1};
        else
            this_color = min(1.25*cmlines(i,:),1);
            name = these_names{2};
        end
    
        this_data = proj2_uni(this_task(j),1:2);
        x = this_data(1);
        y = this_data(2);

        text(x,y,name,'Color',this_color,'FontWeight','bold','FontSize',fs,'HorizontalAlign','center')
        
        xl1 = [min([xl1,x]), max([xl1, x])];
        yl1 = [min([yl1,y]), max([yl1, y])];
    end
end

grid on;
axis square;

title2 = title(ax1,{'Subspace Projection (MDS)'},'FontWeight','normal','fontsize',fs)

set(ax1,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], ...
    'Xlim', xl1, 'YLim', yl1, ...
    'XColor','none','YColor','none');
set(ax2,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], ...
    'Xlim', xl1, 'YLim', yl1, ...
    'XColor','none','YColor','none');
xlabel(' ')

sgtitle(uni_rdm_tile, {'Geometry of common topographies',sprintf('with common representations (%s)',regexprep(strrep(strrep(uni_label,'_',' '),'Ctx ',''),' [LR]',''))},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1329,566, 455, 463]);

export_fig(gcf,sprintf('panels_%s/common_topographies_common_representations.png',noise), ...
    '-transparent','-r300');


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
title({'', '', 'Participant A'},'FontWeight','normal','fontsize',fs)


nexttile(uni_clf_tile);
this_rdm = uni_rdm1(rdm_ind,rdm_ind);
this_clf = uni_confmat1(rdm_ind, rdm_ind);

tril_ind = tril(true(size(this_rdm)),-1);
this_rdm_vec = this_rdm(tril_ind);
this_clf_vec = this_clf(tril_ind);

plot(this_rdm_vec, this_clf_vec, 'o', 'color', cmap(uni,:));
h = lsline;
set(h,'Color', dc_color, 'LineWidth',3);
ylim([0.4,1.1]);
ylabel({'Clf Perf', '(Bal. Acc.)'},'fontsize',fs)
switch noise
    case 'whitened'
        xlabel({'Crossnobis dist'},'fontsize',fs)
    case 'standardized'
        xlabel({'t-distance'},'fontsize',fs)
end
box off;
axis square;
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
axis square
title(sprintf('r = %0.3f', corr(this_rdm_vec, this_clf_vec)),'fontsize',fs)
set(gca,'fontsize',fs)

sgtitle(uni_clf_tile, {'Geometry measures decodability',regexprep(strrep(strrep(uni_label,'_',' '),'Ctx ',''),' [LR]','')},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1733, 572, 417, 450]);

export_fig(gcf,sprintf('panels_%s/common_topographies_common_representations_sup.png',noise), ...
    '-transparent','-r300');

%% compute transmodal MDS
% we perform mds in the full response space, but only align the 6 tasks of 
% interest (cond) into alignent, with the rotation performed in the space 
% of all tasks.
%{
trans_mds1 = cmdscale(real(sqrt(trans_rdm1)));
trans_mds2 = cmdscale(real(sqrt(trans_rdm2)));

[d,Z,transform] = procrustes(trans_mds1(cond,:), trans_mds2(cond,:));
proj1_trans = trans_mds1(cond,1:2);
proj2_trans = Z(:,1:2);
%}

% the above is overdetermined, and not fair. Let's stick to the mds of the
% 6 tasks we show and just rotate in that space. It's a 2D projection of
% the 6x6 distance matrix
trans_mds1 = cmdscale(real(sqrt(trans_rdm1(cond,cond))));
trans_mds2 = cmdscale(real(sqrt(trans_rdm2(cond,cond))));

d1 = size(trans_mds1,2);
d2 = size(trans_mds2,2);
if d2 > d1
    [d,Z,transform] = procrustes(trans_mds2, trans_mds1);
    
    proj1_trans = Z;
    proj2_trans = trans_mds2;
else
    [d,Z,transform] = procrustes(trans_mds1, trans_mds2);
    
    proj1_trans = trans_mds1;
    proj2_trans = Z;
end

%{
% this is the mds and rotation in the full manifold space. It's not useful
% for the cartoon, but it's more representative of the actual data
trans_mds1 = cmdscale(real(sqrt(trans_rdm1)));
trans_mds2 = cmdscale(real(sqrt(trans_rdm2)));

[d,Z,transform] = procrustes(trans_mds1, trans_mds2);

proj1_trans = trans_mds1(cond,1:2);
proj2_trans = Z(cond,1:2);
%}
%% transmodal geometry

figure(5);
clf
trans_rdm_tile = tiledlayout(2,2, 'TileSpacing', 'compact', 'padding', 'normal');

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
        title({'r', '', 'Participant B'},'FontWeight','normal','fontsize',fs)
    case 'standardized'
        title({'', '', 'Participant B'},'FontWeight','normal','fontsize',fs)
end


ax1 = nexttile(trans_rdm_tile);
hold on;
these_tasks = task_labels(ismember(task_labels, tasks));
[xl1, yl1] = deal([0,0]);
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    these_names = task_names(task_labels == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
            name = these_names{1};
        else
            this_color = min(1.25*cmlines(i,:),1);
            name = these_names{2};
        end

        this_data = proj1_trans(this_task(j),1:2);
        x = this_data(1);
        y = this_data(2);

        text(x,y,name,'Color',this_color,'FontWeight','bold','FontSize',fs, ...
            'HorizontalAlign','center')
        
        xl1 = [min([xl1,x]), max([xl1, x])];
        yl1 = [min([yl1,y]), max([yl1, y])];
    end
end

grid on;
axis square
xlabel(' ')

ax2 = nexttile(trans_rdm_tile);
hold on;
these_tasks = task_labels(ismember(task_labels, tasks));
for i = 1:length(tasks)
    this_task = find(these_tasks == tasks(i));
    these_names = task_names(task_labels == tasks(i));
    assert(length(this_task) == 2); % colormapping won't work otherwise
    for j = 1:2
        if j == 1
            this_color = 0.75*cmlines(i,:);
            name = these_names{1};
        else
            this_color = min(1.25*cmlines(i,:),1);
            name = these_names{2};
        end
    
        this_data = proj2_trans(this_task(j),1:2);
        x = this_data(1);
        y = this_data(2);

        text(x,y,name,'Color',this_color,'FontWeight','bold','FontSize',fs, ...
            'HorizontalAlign','center')
        
        xl1 = [min([xl1,x]), max([xl1, x])];
        yl1 = [min([yl1,y]), max([yl1, y])];
    end
end

grid on;
axis square;

title(ax2,{'Subspace Projection (MDS)'},'FontWeight','normal','fontsize',fs)

set(ax1,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], ...
    'Xlim', xl1, 'YLim', yl1, ...
    'XColor','none','YColor','none');
set(ax2,'TickLength',[0,0], 'XTickLabels',[], 'YTickLabels', [], ...
    'Xlim', xl1, 'YLim', yl1, ...
    'XColor','none','YColor','none');
xlabel(' ')

sgtitle(trans_rdm_tile, {'Geometry of idiosyncratic topographies',sprintf('with common representations (%s)',regexprep(strrep(strrep(trans_label,'_',' '),'Ctx ',''),' [LR]',''))},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1322, 69, 455, 463]);

export_fig(gcf,sprintf('panels_%s/idiosyncratic_topographies_common_representations.png',noise), ...
    '-transparent','-r300');


% compute unbiased cosine similarity
x = trans_rdm1(rdm_ind, rdm_ind);
y = trans_rdm2(rdm_ind, rdm_ind);
tril_ind = tril(true(size(x)),-1);
x = x(tril_ind);
y = y(tril_ind);
sprintf('Cosine of transmodal RDMs: %0.3f\n', x'*y/(norm(x)*norm(y)))


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
title({'', '', 'Participant B'},'FontWeight','normal','fontsize',fs)


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
axis square
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
axis square
title(sprintf('r = %0.3f', corr(this_rdm_vec, this_clf_vec)),'fontsize',fs)
xl = xlim;
set(gca,'fontsize',fs,'XTickLabelRotation',0,'XTick',0:0.3:0.6)

sgtitle(trans_clf_tile, {'Geometry measures decodability',regexprep(strrep(strrep(trans_label,'_',' '),'Ctx ',''),' [LR]','')},'FontWeight','bold','fontsize',fs+2)

pos = get(gcf,'Position');
set(gcf,'Position',[1733, 75, 417, 450]);

export_fig(gcf,sprintf('panels_%s/idiosyncratic_topographies_common_representations_sup.png',noise), ...
    '-transparent','-r300');

%% Estimate mean clf-dist correlation across the sample
[r_u, r_t] = deal(nan(height(sid),2));
lbls = [1,262];
%lbls = randperm(518,2);
uni = lbls(1); % 1
trans = lbls(2); % 262
uni_label = roi_labels(lbls(1));
trans_label = roi_labels(lbls(2));
for i = 1:height(sid)
    try
        rdm_root = '../../derivatives/hcp_glm_msmall_grayord_spm/results/';
        
        if strcmp(noise,'standardized')
            standardized1 = readmatrix(sprintf('%s/%d/all_tasks/rsa/stddist/standardized_distance.csv',rdm_root, sid.Var1(i)));
            standardized2 = readmatrix(sprintf('%s/%d/all_tasks/rsa/stddist/standardized_distance.csv',rdm_root, sid.Var2(i)));
        elseif strcmp(noise,'whitened')
            standardized1 = readmatrix(sprintf('%s/%d/all_tasks/rsa/crossnobis/crossnobis_distance.csv',rdm_root, sid.Var1(i)));
            standardized2 = readmatrix(sprintf('%s/%d/all_tasks/rsa/crossnobis/crossnobis_distance.csv',rdm_root, sid.Var2(i)));
        else
            error('Did not understand noise model choice "%s"',noise)
        end
        
        conf_mats1 = readtable(sprintf('../../derivatives/single_blocks_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/binary_clf_performance.csv', sid.Var1(i),noise),'ReadRowNames',true,'ReadVariableNames',true);
        conf_mats2 = readtable(sprintf('../../derivatives/single_blocks_msmall_grayord_spm/results/%d/all_tasks/%s_contrasts/binary_clf_performance.csv',sid.Var2(i),noise),'ReadRowNames',true,'ReadVariableNames',true);
        
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
    
        ltri = tril(true(size(uni_confmat1)),-1);
        
        % vectorize lower triangles
        uni_clf1 = uni_confmat1(ltri);
        uni_clf2 = uni_confmat2(ltri);
        trans_clf1 = trans_confmat1(ltri);
        trans_clf2 = trans_confmat2(ltri);
    
        uni_rdm1 = uni_rdm1(ltri);
        uni_rdm2 = uni_rdm2(ltri);
        trans_rdm1 = trans_rdm1(ltri);
        trans_rdm2 = trans_rdm2(ltri);
    
        r_u(i,1) = corr(uni_clf1, uni_rdm1);
        r_u(i,2) = corr(uni_clf2, uni_rdm2);
        r_t(i,1) = corr(trans_clf1, trans_rdm1);
        r_t(i,2) = corr(trans_clf2, trans_rdm2);
    catch
        fprintf('Skipped %d\n',i)
    end
end

nanmean([r_u(:), r_t(:)])
nanstd([r_u(:), r_t(:)])