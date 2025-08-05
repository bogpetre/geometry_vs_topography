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
cm = colormap(f,'jet');
close(f)

underlay = which('fsl6_hcp_template.nii.gz');

%% load data

LRJac = fmri_data('../../derivatives/LR_Jacobian.nii.gz');
LRJac.dat = double(LRJac.dat);
RLJac = fmri_data('../../derivatives/RL_Jacobian.nii.gz');
RLJac.dat = double(RLJac.dat);

%% Plot
cmaprange = max(abs(prctile(LRJac.remove_empty.dat,[2.5,97.5])))*[-1,1];

[t0,o2,cbar] = create_montage_figure(LRJac,cm,cmaprange,...
    'Left-to-Right (LR) Aquisition', fs);

exportgraphics(gcf,sprintf('panels/LR_SDC_Log_jac.png'),'ContentType','image','Resolution',300);


cmaprange = max(abs(prctile(RLJac.remove_empty.dat,[2.5,97.5])))*[-1,1];

[t0,o2,cbar] = create_montage_figure(RLJac,cm,cmaprange,...
    'Right-to-Left (RL) Aquisition', fs);

exportgraphics(gcf,sprintf('panels/RL_SDC_Log_jac.png'),'ContentType','image','Resolution',300);