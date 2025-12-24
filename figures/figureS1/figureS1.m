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
cm = colormap(f,'hot');
close(f)

dc_color = config.matlab_disp_scheme.color_main;
dc_color_light = config.matlab_disp_scheme.color_light;

noise = 'whitened';

%% import atlas in cifti space
atlas = get_cifti_data(config.canlab2024.path);

n_roi = length(unique([atlas.cortex_left, atlas.cortex_right, atlas.volumes])) - 1;

atlas_cii = cifti_read(config.canlab2024.path);
%% import test data
sid_all = readtable('../../resources/paired_sid.csv', 'ReadVariableNames',false);

balanced_mean_op = [1/7*repmat(1/2,1,10), 1/7*repmat(1/5,1,5), 1/7*repmat(1/8,1,8)];

task_labels = [1,1,2,2,3,3,4,4,5,5,6,6,6,6,6,7,7,7,7,7,7,7,7];

[wi_cosim1, wi_cosim2, tsnr1, tsnr2] = deal(nan(height(sid_all), n_roi));
for s = 1:height(sid_all)
    try
        tsnr1(s,:) = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/tsnr.csv',sid_all.Var1(s)));
        tsnr2(s,:) = readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/tsnr.csv',sid_all.Var2(s)));
        
        wi_cosim1(s,:) = balanced_mean_op*readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_betas/%s_similarity.csv',sid_all.Var1(s),noise,noise),'FileType','text');
        wi_cosim2(s,:) = balanced_mean_op*readmatrix(sprintf('../../derivatives/hcp_glm_msmall_grayord_spm/results/%d/all_tasks/%s_betas/%s_similarity.csv',sid_all.Var2(s),noise,noise),'FileType','text');        
    catch
        warning('Could not import pair %d', s);
    end
end

has_data = any(~isnan(wi_cosim1) | ~isnan(wi_cosim2),2);

wi_cosim1 = wi_cosim1(has_data,:);
wi_cosim2 = wi_cosim2(has_data,:);
wi_cosim = [wi_cosim1; wi_cosim2];
tsnr1 = tsnr1(has_data,:);
tsnr2 = tsnr2(has_data,:);
tsnr = [tsnr1;tsnr2];

%% import retest data
sid = readtable('../../resources/paired_retest_iid_sid.csv', 'ReadVariableNames',false);
sid = table([sid.Var1;sid.Var2]);

wuc = zeros(height(sid), n_roi);
[wuc, wuc_md_p] = deal(zeros(height(sid), n_roi));
for s = 1:height(sid)
    try
        wuc(s,:) = diag(readmatrix(sprintf('../../derivatives/retest/hcp_glm_msmall_grayord_spm/bsc_retest/%s_betas/cosine/%d_v_%d_wuc.tsv',noise,sid.Var1(s),sid.Var1(s)),'FileType','text'));
    catch
        warning('Could not import pair %d', s);
    end
end
for s = 1:height(sid)
    % Some distance matrices are so noise dominated that their test-retest
    % reliability does not produce a valid RDM whitening matrix. We set
    % those to nans here. If we consider the residual non-imaginary
    % subjects for these regions we'll have biased estimates, but nans make
    % these easy to keep track of so we can mask them out later
    nan_regions = imag(wuc(s,:)) ~= 0;
    wuc(s, nan_regions) = nan;
end

cosim = zeros(height(sid), n_roi);
for s = 1:height(sid)
    try
        cosim(s,:) = balanced_mean_op*dlmread(sprintf('../../derivatives/retest/hcp_glm_msmall_grayord_spm/bsc_retest/%s_betas/cosine/%d_v_%d_cosim.tsv', noise, sid.Var1(s), sid.Var1(s)), '\t');
    catch
        warning('Could not import pair %d', s);
    end
end

has_data = any(cosim,2) & any(~isnan(wuc),2);

cosim = cosim(has_data,:);
wuc = wuc(has_data,:);
sid = sid(has_data,:);

%% plot to brain

B = nanmean(wi_cosim);
cmaprange = prctile(B,[2.5,97.5]);
switch noise
    case 'whitened'
        T = {'Test-Retest Reliability','Within-participant LR-RL', ['(Whitened \beta, cos\theta, N = ', sprintf('%0d)',size(wi_cosim,1))]};
    case 'standardized'
        T = {'Test-Retest Reliability','Within-participant LR-RL', ['(Standardized \beta, cos\theta, N = ', sprintf('%0d)',size(wi_cosim,1))]};
end
plot_to_brain(B,1:length(B),cmaprange,T,fs+2);
exportgraphics(gcf,sprintf('panels_%s/test_retest_cosim.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(B)
    new_cii_data(atlas_cii.cdata == i) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('test_retest_cosim_%s.dscalar.nii',noise));


B = nanmean(tsnr);
cmaprange = prctile(B,[2.5,97.5]);
switch noise
    case 'whitened'
        T = {'Mean task tSNR','across runs and participants',['(Whitened \beta, N = ', sprintf('%0d)',size(tsnr,1))]};
    case 'standardized'
        T = {'Mean task tSNR','across runs and participants',['(Standardized \beta, N = ', sprintf('%0d)',size(tsnr,1))]};
end
plot_to_brain(B,1:length(B),cmaprange,T,fs+2);
exportgraphics(gcf,sprintf('panels_%s/tsnr.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(B)
    new_cii_data(atlas_cii.cdata == i) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('tsnr_%s.dscalar.nii',noise));


B = nanmean(cosim);
cmaprange = prctile(B,[2.5,97.5]);
switch noise
    case 'whitened'
        T = {'Task Topoography Test-Retest Reliability',['(Whitened \beta, cos\theta, ', sprintf('N=%d)',size(cosim,1))]};
    case 'standardized'
        T = {'Task Topoography Test-Retest Reliability',['(t-stat, cos\theta, ', sprintf('N=%d)',size(cosim,1))]};
end
plot_to_brain(B,1:length(B),cmaprange,T,fs+2);
exportgraphics(gcf,sprintf('panels_%s/test_followup_cosim.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(B)
    new_cii_data(atlas_cii.cdata == i) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('topo_cosim_test_v_followup_n22_%s.dscalar.nii',noise));

B = nanmean(wuc);
cmaprange = prctile(B,[2.5,97.5]);
switch noise
    case 'whitened'
        T = {'Geometry Test-Retest Reliability',['(Whitened \beta, cos\theta, ', sprintf('N=%d)',size(wuc,1))]};
    case 'standardized'
        T = {'Geometry Test-Retest Reliability',['(t-stat, cos\theta, ', sprintf('N=%d)',size(wuc,1))]};
end
plot_to_brain(B,1:length(B),cmaprange,T,fs+2);
exportgraphics(gcf,sprintf('panels_%s/test_retest_wu.png',noise),'ContentType','image','Resolution',300);
for i = 1:length(B)
    new_cii_data(atlas_cii.cdata == i) = B(i);
end
cifti_write_from_template(atlas_cii, new_cii_data,sprintf('geom_wuc_test_v_followup_n22_%s.dscalar.nii',noise));

%% test-retest reliability contrasts
%{
figure;
x = wuc(:,1:358);
y = cosim(:,1:358);
hist(mean(x - y,2)./std(x - y,[],2));
%}
%% test within-reliability vs. metric

sid_all_in_retest = [];
for i = 1:height(sid)
    sid_all_in_retest(i) = find(sid.Var1(i) == [sid_all.Var1; sid_all.Var2]);
end

figure(4);
t0 = tiledlayout(3,1,'TileSpacing','compact','padding', 'none');
nexttile()
cla
plot(cosim',wi_cosim(sid_all_in_retest,:)','.')
B = mean_xsubject_B(cosim, wi_cosim(sid_all_in_retest,:));
hold on;
xlim([-0.2,1]);
ylim([-0.2,1]);
h = plot(xlim, xlim*B(1) + B(2),'color',dc_color,'LineWidth',2);
axis square
box off
ylabel({'LR vs. RL Topographic Similarity','(cos\theta within session)'});
xlabel({'Topographic Similarity','(cos\theta between session)'})
legend(h,'LS Line','location','northwest');

stdB = mean_xsubject_B(cosim,wi_cosim(sid_all_in_retest,:),'std');
title(['Mean r^2 = ',sprintf('%0.3f',stdB(1)^2)]);

nexttile()
cla
plot(wuc',wi_cosim(sid_all_in_retest,:)','.')
B = mean_xsubject_B(wuc, wi_cosim(sid_all_in_retest,:));
hold on;
xlim([-0.2,1]);
ylim([-0.2,1]);
h = plot(xlim, xlim*B(1) + B(2),'color',dc_color,'LineWidth',2);
axis square
box off
ylabel({'LR vs. RL Topographic Similarity','(cos\theta within session)'});
xlabel({'Geometric Similarity','(WUC between session)'})
legend(h,'LS Line','location','northwest');

stdB = mean_xsubject_B(wuc,wi_cosim(sid_all_in_retest,:),'std');
title(['Mean r^2 = ',sprintf('%0.3f',stdB(1)^2)]);

nexttile()
cla
plot(wuc',cosim','.')
B = mean_xsubject_B(wuc,cosim);
hold on;
xlim([-0.2,1]);
ylim([-0.2,1]);
h = plot(xlim, xlim*B(1) + B(2),'color',dc_color,'LineWidth',2);
axis square
box off
xlabel('Geometry (WUC)');
ylabel('Topography (cos\theta)')
legend(h,'LS Line','location','northwest')

stdB = mean_xsubject_B(wuc,cosim,'std');
title(['Mean r^2 = ',sprintf('%0.3f',stdB(1)^2)]);

sgtitle(['Test-Retest Reliability',sprintf('(N=%d)',size(wuc,1))]);

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2), 437,834])
exportgraphics(gcf,sprintf('panels_%s/test_retest_scatterjplots_stdb.png',noise),'ContentType','image','Resolution',300);

%%
figure;
p = plot(cosim',wi_cosim(sid_all_in_retest,:)','.')
B = mean_xsubject_B(cosim, wi_cosim(sid_all_in_retest,:));
hold on;
xlim([-0.2,1]);
ylim([-0.2,1]);
h = plot(xlim, xlim*B(1) + B(2),'color',dc_color,'LineWidth',2);
axis square
box off
ylabel({'LR vs. RL Topographic Similarity','(cos\theta within session)'});
xlabel({'Topographic Similarity','(cos\theta between session)'})
leg = legend([h;p(1:4)],{'LS Line','Participant1','Participant2','Participant3','etc'},'location','southeast');
set(gca,'FontSize',fs);

B = mean_xsubject_B(cosim,wi_cosim(sid_all_in_retest,:),'std');
title({'Test-Retest (Visit 1) vs. Visit 1 vs 2', ['Mean r^2 = ',sprintf('%0.3f',B(1)^2)]},'fontsize',fs+2);


pos = get(gcf,'Position');
set(gcf,'Position', [pos(1:2), 400,407])

exportgraphics(gcf,sprintf('panels_%s/test_retest_vs_test_followup_scatterplot.png',noise),'ContentType','image','Resolution',300);