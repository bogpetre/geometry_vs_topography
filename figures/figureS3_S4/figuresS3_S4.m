fs = 12;
coupling = importdata('stats.mat');

% map - randomly spun map, one of 12 source maps
% sample - bootstrapped
% test - jackknife-adjusted spin
p_cd_null = coupling.Bp_null_cpl;

for i = 1:size(p_cd_null,1)
    p_cd_null(i,:) = sort(p_cd_null(i,:));
end

% map - one of 12 source maps
% sample - bootstrapped and shuffled topographic similarities across ROIs
% test - jackknife-adjusted spin
p_cd_null_basic = coupling.Bp_null_cpl_basic;

for i = 1:size(p_cd_null_basic,1)
    p_cd_null_basic(i,:) = sort(p_cd_null_basic(i,:));
end


% map - one of 12 source maps
% sample - bootstrapped
% test - jackknife-adjusted spin
p_cd_null2 = coupling.Bp_null2_cpl;

for i = 1:size(p_cd_null2,1)
    p_cd_null2(i,:) = sort(p_cd_null2(i,:));
end

figure(1);

clf
tiledlayout(1,3,'TileSpacing','compact')

nexttile()
cla
y = (1:size(p_cd_null_basic,2));
plot(p_cd_null_basic,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','(trad) Spin Test'},'FontSize',fs+4,'FontWeight','normal');

nexttile()
cla
y = (1:size(p_cd_null,2));
plot(p_cd_null,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','Jackknife-adjusted spin test'},'FontSize',fs+4,'FontWeight','normal')

nexttile()
cla
y = (1:size(p_cd_null2,2));
plot(p_cd_null(10,:),y,'LineWidth',3)
hold on;
plot(p_cd_null([1:9,11:end],:),y)
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','Shuffled Topo ROIs','Jackknife-adjusted spin test'},'FontSize',fs+4,'FontWeight','normal')

sgtitle('Sensitivity of topographic to representational similarity within-region across-participants: neuromap interactions','FontSize',fs+4,'FontWeight','bold')

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),1400,507]);

exportgraphics(gcf,'fpr_coupling.png','ContentType','image','Resolution',300);

%%

gradient = importdata('stats.mat');

% map - randomly spun map, one of 12 source maps
% sample - bootstrapped
% test - jackknife-adjusted spin
p_cd_null = gradient.Bp_null_grd;

for i = 1:size(p_cd_null,1)
    p_cd_null(i,:) = sort(p_cd_null(i,:));
end

p_cd_null_jk = gradient.Bp_null_grd_jk;

for i = 1:size(p_cd_null_jk,1)
    p_cd_null_jk(i,:) = sort(p_cd_null_jk(i,:));
end

p_cd_null_spin = gradient.Bp_null_spin;

for i = 1:size(p_cd_null_spin,1)
    p_cd_null_spin(i,:) = sort(p_cd_null_spin(i,:));
end

% map - one of 12 source maps
% sample - bootstrapped
% test - jackknife-adjusted spin
p_cd_null2 = gradient.Bp_int_null_grd;

for i = 1:size(p_cd_null2,1)
    p_cd_null2(i,:) = sort(p_cd_null2(i,:));
end

p_cd_null2_jk = gradient.Bp_int_null_grd_jk;

for i = 1:size(p_cd_null2_jk,1)
    p_cd_null2_jk(i,:) = sort(p_cd_null2_jk(i,:));
end

p_cd_null2_spin = gradient.Bp_int_null_spin;

for i = 1:size(p_cd_null2_spin,1)
    p_cd_null2_spin(i,:) = sort(p_cd_null2_spin(i,:));
end



figure(2);

clf
tiledlayout(1,3,'TileSpacing','compact')

nexttile()
cla
y = (1:size(p_cd_null_jk,2));
plot(p_cd_null_jk,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,800,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','Bootstrapped Confidence Interval Test'},'FontSize',fs+4,'FontWeight','normal')

nexttile()
cla
y = (1:size(p_cd_null_spin,2));
plot(p_cd_null_spin,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','(trad) Spin Test'},'FontSize',fs+4,'FontWeight','normal')

nexttile()
cla
y = (1:size(p_cd_null,2));
plot(p_cd_null,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','Jackknife-adjusted spin test'},'FontSize',fs+4,'FontWeight','normal')

sgtitle('Main effects of null neuromaps on representational similarity','FontSize',fs+4,'FontWeight','bold')

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),1400,480]);

exportgraphics(gcf,'fpr_gradient_main_fx.png','ContentType','image','Resolution',300);



figure(3);

clf
tiledlayout(1,3,'TileSpacing','compact')

nexttile()
cla
y = (1:size(p_cd_null2_jk,2));
plot(p_cd_null2_jk,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,800,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','Bootstrapped Confidence Interval Test'},'FontSize',fs+4,'FontWeight','normal')

nexttile()
cla
y = (1:size(p_cd_null2_spin,2));
plot(p_cd_null2_spin,y)
hold on;
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','(trad) Spin Test'},'FontSize',fs+4,'FontWeight','normal')

nexttile()
cla
y = (1:size(p_cd_null2,2));
plot(p_cd_null2(10,:),y,'LineWidth',3)
hold on;
plot(p_cd_null2([1:9,11:end],:),y)
plot([0,1],[0,max(y)],'-','LineWidth',3,'color','bla')
xlabel('\alpha')
ylabel('|P < \alpha|')
t1 = text(0.6,100,'Conservative','FontSize',16','FontWeight','normal');
t2 = text(0.1,950,'Liberal','FontSize',16','FontWeight','normal');
axis square
set(gca,'FontSize',fs)
title({'Spatially Permuted Maps','Bootstrapped Sample','Jackknife-adjusted spin test'},'FontSize',fs+4,'FontWeight','normal')

sgtitle('Moderation effects of null neuromaps on representational - topographic similarity','FontSize',fs+4,'FontWeight','bold')

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),1400,480]);

exportgraphics(gcf,'fpr_gradient_int_fx.png','ContentType','image','Resolution',300);