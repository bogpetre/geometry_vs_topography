close all; clear all;

config = jsondecode(fileread('../../config.json'));

addpath(config.matlab_libraries.spm12);
addpath(genpath(config.matlab_libraries.cifti_matlab));
addpath(genpath(fullfile(config.matlab_libraries.canlabCore, 'CanlabCore')));

addpath('../../matlab_libraries');
addpath('../../resources/neuromaps');

fs=config.matlab_disp_scheme.fontsize;

%dc_color = config.matlab_disp_scheme.color_main;
%dc_color_light = config.matlab_disp_scheme.color_light;

%% Neural Net Diagrams

y_offset = 0;

% Number of nodes in each layer
input_nodes = 5;
hidden_nodes = 6;
output_nodes = 3;

output_label = {'Angry', 'Fearful', 'Shape'};

targets = repmat(output_label,1,2);
n_stim = numel(targets);

x_input = 1; % x-coordinate for input layer
x_hidden = 3; % x-coordinate for hidden layer
x_output = 5; % x-coordinate for output layer

rng(28) % seed matters to get the kinds of activations we want.

output_label = {'Angry', 'Fearful', 'Shape'};
targets = repmat(output_label, 1, 2);
n_stim = numel(targets);

% Numeric target matrix
numeric_targets = cellfun(@(x1) contains(targets, x1), output_label, 'UniformOutput', false);
numeric_targets = double(cat(1, numeric_targets{:}));

% He initialization for weights
weights_hidden = randn(input_nodes, hidden_nodes) * sqrt(2 / input_nodes);
weights_output = randn(hidden_nodes, output_nodes) * sqrt(2 / hidden_nodes);

% Input activations (normalized)
activations_input = rand(input_nodes, n_stim);

% Learning rate for input adjustment
learning_rate = 0.1;

% Convergence criteria
max_iterations = 100000;
tolerance = 1e-6;

for iter = 1:max_iterations
    % Forward pass
    activations_hidden = max(weights_hidden' * activations_input, 0); % ReLU activation for hidden layer
    activations_output = max(weights_output' * activations_hidden, 0); % ReLU activation for output layer
    
    % Compute error
    error = activations_output - numeric_targets;
    loss = sum(error.^2); % Mean squared error
    
    % Check for convergence
    if loss < tolerance
        fprintf('Converged at iteration %d, Loss: %.6f\n', iter, loss);
        break;
    end
    
    % Backpropagate error to adjust inputs
    % Gradient w.r.t output layer
    grad_output = error .* (activations_output > 0);
    
    % Gradient w.r.t hidden layer
    grad_hidden = (weights_output * grad_output) .* (activations_hidden > 0);
    
    % Gradient w.r.t inputs
    grad_inputs = weights_hidden * grad_hidden;
    
    % Update inputs
    activations_input = min(max(0,activations_input - learning_rate * grad_inputs),1);
end

prediction = activations_output == max(activations_output,[],1);


% plot topography 1

figure(1)
clf
t0 = tiledlayout(2,2,'Padding','compact','TileSpacing','normal');
stims = [3,5];

tiles = cell(2,2);
for i = 1:4
    tiles{i} = nexttile();
end
tiles{1,1}.YLabel.String = {'Network (Subject) 1'};
tiles{1,1}.YLabel.FontSize = fs-2;
tiles{1,1}.YLabel.Visible = true;
tiles{1,1}.YLabel.FontWeight = 'bold';
tiles{1,2}.YLabel.String = {'Network (Subject) 2','(permuted hidden units)'};
tiles{1,2}.YLabel.FontSize = fs-2;
tiles{1,2}.YLabel.Visible = true;
tiles{1,2}.YLabel.FontWeight = 'bold';

for s = 1:2
    axes(tiles{s,1});
    hold on;
    
    axis off;
    xlim([1, 5]);
    ylim([-0.5, 0.4]);

    stim = stims(s);
    
    % Input layer positions
    y_input = linspace(-0.4,0.4,input_nodes+1);
    x_input_layer = x_input*ones(input_nodes,1); % x-coordinates
    
    y_hidden = linspace(-0.4,0.4,hidden_nodes+1);
    x_hidden_layer = x_hidden*ones(hidden_nodes,1);
    
    y_output = linspace(-0.4,0.4,output_nodes+1);
    x_output_layer = x_output*ones(output_nodes,1);
    
    % Connect nodes with edges
    for i = 1:input_nodes
        for j = 1:hidden_nodes
            %weight = (max(0,weights_hidden(i,j))*5 + 0.001)^(1/3);
            weight = round(sqrt(max(0,weights_hidden(i,j))*3)) + 0.001;
            if weight < 0.5
                color = [0.8,0.8,0.8];
            else
                color = [0.5,0.5,0.5];
            end
    
            plot([x_input_layer(i), x_hidden_layer(j)], [y_input(i), y_hidden(j)], 'k-',...
                'LineWidth',weight, 'color', color);
            %end
        end
    end
    
    for i = 1:hidden_nodes
        for j = 1:output_nodes
            weight = round(sqrt(3*max(0,weights_output(i,j)))) + 0.001;
            if weight < 0.5
                color = [0.8,0.8,0.8];
            else
                color = [0.5,0.5,0.5];
            end
    
            plot([x_hidden_layer(i), x_output_layer(j)], [y_hidden(i), y_output(j)], ...
                'k-','LineWidth',weight, 'color', color);
            %end
        end
    end
    
    % Draw input layer
    for i = 1:input_nodes
        s1 = scatter(x_input_layer(i), y_input(i), 100, 'b', 'filled');
        s1.MarkerFaceAlpha = activations_input(i,stim);
        s1.MarkerEdgeColor = 'b';
        s1.MarkerFaceColor = [0,0,0];
    end
    %arrayfun(@(y) text(x_input - 0.4, y_input(y), sprintf('Pixel %d', y), 'HorizontalAlignment', 'right'), 1:input_nodes);
    
    % Draw hidden layer
    for i = 1:hidden_nodes
        s2 = scatter(x_hidden_layer(i), y_hidden(i), 100, 'g', 'filled');
        s2.MarkerFaceAlpha = min(activations_hidden(i,stim),1);
        s2.MarkerEdgeColor = 'g';
        s2.MarkerFaceColor = [0,0,0];
    end
    %arrayfun(@(y) text(x_hidden - 0.3, y, 'Hidden', 'HorizontalAlignment', 'right'), y_hidden);
    
    % Draw output layer
    for i = 1:output_nodes
        s3 = scatter(x_output_layer(i), y_output(i), 100, 'r', 'filled');
        s3.MarkerFaceAlpha = min(prediction(i, stim),1);
        s3.MarkerEdgeColor = 'r';
        s3.MarkerFaceColor = [0,0,0];
    end
    %arrayfun(@(y) text(x_output + 0.3, y_output(y), output_label(y), 'HorizontalAlignment', 'left'), 1:output_nodes);
    
    
    % Formatting
    if s == 1
        %title({'Network (Subject) 1',''},'FontWeight','normal','FontSize',fs-2,'FontWeight','bold');
        title({'Condition 1','e.g. ''Faces'''},'FontWeight','normal','FontSize',fs-2,'FontWeight','bold');
    else
        %title({'Network (Subject) 2','(permuted hidden units)'},'FontWeight','normal','FontSize',fs-2,'FontWeight','bold');
        title({'Condition 2','e.g. ''Shapes'''},'FontWeight','normal','FontSize',fs-2,'FontWeight','bold');
    end
    hold off;
    
    text(x_input,-0.55,'Input','HorizontalAlignment','center','FontSize',fs-2)
    text(x_hidden,-0.55,'Hidden','HorizontalAlignment','center','FontSize',fs-2)
    text(x_output,-0.55,'Output','HorizontalAlignment','center','FontSize',fs-2)

    if s == 2
        text(x_output + 1, y_output(3), 'Faces', 'Rotation', 90, 'HorizontalAlignment', 'center','fontsize',fs-2)
        text(x_output + 1.4, y_output(2), 'Shapes', 'Rotation', 90, 'HorizontalAlignment', 'center','fontsize',fs-2)
        text(x_output + 1, y_output(1), 'Places', 'Rotation', 90, 'HorizontalAlignment', 'center','fontsize',fs-2)
    end
end

%perm = [1,6,3,4,5,2];
perm = flip(1:6);
weights_hidden = weights_hidden(:,perm);
weights_output = weights_output(perm,:);
activations_hidden = activations_hidden(perm,:);

for s = 1:2
    axes(tiles{s,2});
    hold on;
    
    axis off;
    xlim([1, 5]);
    ylim([-0.5, 0.4]);

    stim = stims(s);
    
    % Input layer positions
    y_input = linspace(-0.4,0.4,input_nodes+1);
    x_input_layer = x_input*ones(input_nodes,1); % x-coordinates
    
    y_hidden = linspace(-0.4,0.4,hidden_nodes+1);
    x_hidden_layer = x_hidden*ones(hidden_nodes,1);
    
    y_output = linspace(-0.4,0.4,output_nodes+1);
    x_output_layer = x_output*ones(output_nodes,1);
    
    % Connect nodes with edges
    for i = 1:input_nodes
        for j = 1:hidden_nodes
            %weight = (max(0,weights_hidden(i,j))*5 + 0.001)^(1/3);
            weight = round(sqrt(max(0,weights_hidden(i,j))*3)) + 0.001;
            if weight < 0.5
                color = [0.8,0.8,0.8];
            else
                color = [0.5,0.5,0.5];
            end
    
            plot([x_input_layer(i), x_hidden_layer(j)], [y_input(i), y_hidden(j)], 'k-',...
                'LineWidth',weight, 'color', color);
            %end
        end
    end
    
    for i = 1:hidden_nodes
        for j = 1:output_nodes
            weight = round(sqrt(3*max(0,weights_output(i,j)))) + 0.001;
            if weight < 0.5
                color = [0.8,0.8,0.8];
            else
                color = [0.5,0.5,0.5];
            end
    
            plot([x_hidden_layer(i), x_output_layer(j)], [y_hidden(i), y_output(j)], ...
                'k-','LineWidth',weight, 'color', color);
            %end
        end
    end
    
    % Draw input layer
    for i = 1:input_nodes
        s1 = scatter(x_input_layer(i), y_input(i), 100, 'b', 'filled');
        s1.MarkerFaceAlpha = activations_input(i,stim);
        s1.MarkerEdgeColor = 'b';
        s1.MarkerFaceColor = [0,0,0];
    end
    %arrayfun(@(y) text(x_input - 0.4, y_input(y), sprintf('Pixel %d', y), 'HorizontalAlignment', 'right'), 1:input_nodes);
    
    % Draw hidden layer
    for i = 1:hidden_nodes
        s2 = scatter(x_hidden_layer(i), y_hidden(i), 100, 'g', 'filled');
        s2.MarkerFaceAlpha = min(activations_hidden(i,stim),1);
        s2.MarkerEdgeColor = 'g';
        s2.MarkerFaceColor = [0,0,0];
    end
    %arrayfun(@(y) text(x_hidden - 0.3, y, 'Hidden', 'HorizontalAlignment', 'right'), y_hidden);
    
    % Draw output layer
    for i = 1:output_nodes
        s3 = scatter(x_output_layer(i), y_output(i), 100, 'r', 'filled');
        s3.MarkerFaceAlpha = min(prediction(i, stim),1);
        s3.MarkerEdgeColor = 'r';
        s3.MarkerFaceColor = [0,0,0];
    end
    %arrayfun(@(y) text(x_output + 0.3, y_output(y), output_label(y), 'HorizontalAlignment', 'left'), 1:output_nodes);
    
    
    % Formatting
    hold off;
    
    text(x_input,-0.55,'Input','HorizontalAlignment','center','fontsize',fs-2)
    text(x_hidden,-0.55,'Hidden','HorizontalAlignment','center','fontsize',fs-2)
    text(x_output,-0.55,'Output','HorizontalAlignment','center','fontsize',fs-2)

    if s == 2
        text(x_output + 1, y_output(3), 'Faces', 'Rotation', 90, 'HorizontalAlignment', 'center','fontsize',fs-2)
        text(x_output + 1.4, y_output(2), 'Shapes', 'Rotation', 90, 'HorizontalAlignment', 'center','fontsize',fs-2)
        text(x_output + 1, y_output(1), 'Place', 'Rotation', 90, 'HorizontalAlignment', 'center','fontsize',fs-2)
    end
end
sgtitle({'A neural network can implement the same','functional representation in multiple ways'},'FontWeight','bold','fontsize',fs+1)

t0.Position(3) = 0.68;
t0.Position(1) = 0.17;
t0.Position(2) = 0.05;

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),415,415])

exportgraphics(gcf,'panels/neural_networks.png','ContentType','image','Resolution',300);

%% Spatial gradients
medial_mask_R = gifti('../../resources/100307.R.atlasroi.32k_fs_LR.shape.gii').cdata == 1;

myelin_map = gifti('source-hcps1200_desc-myelinmap_space-fsLR_den-32k_hemi-R_feature.func.gii');
thickness = gifti('source-hcps1200_desc-thickness_space-fsLR_den-32k_hemi-R_feature.func.gii');
marg_rh_1 = gifti('source-margulies2016_desc-fcgradient01_space-fsaverage_den-32k_hemi-R_feature.func.gii');

evo_exp1 = gifti('source-hill2010_desc-evoexp_space-fsaverage_den-32k_hemi-R_feature.func.gii');
evo_exp2 = gifti('source-xu2020_desc-evoexp_space-fsaverage_den-32k_hemi-R_feature.func.gii');
fchomology = gifti('source-xu2020_desc-FChomology_space-fsaverage_den-32k_hemi-R_feature.func.gii');

devexp1 = gifti('source-hill2010_desc-devexp_space-fsLR_den-32k_hemi-R_feature.func.gii');
devexp2 = gifti('source-reardon2018_desc-scalinghcp_space-fsLR_dens-32k_hemi-R_feature.func.gii');

genepc1 = gifti('source-abagen_desc-genepc1_space-fsaverage_den-32k_hemi-R_feature.func.gii');
cogpc1 = gifti('source-neurosynth_desc-cogpc1_space-fsLR_dens-32k_hemi-R_feature.func.gii');
cbf1 = gifti('source-raichle_desc-cbf_space-fsaverage_den-32k_hemi-R_feature.func.gii');
cbf2 = gifti('source-satterthwaite2014_desc-meancbf_space-fsaverage_den-32k_hemi-R_feature.func.gii');


% zscore to have them be on a common scale
zscore_fun = @(x)((x - nanmean(x))/nanstd(x));

myelin_map.cdata(medial_mask_R) = zscore_fun(myelin_map.cdata(medial_mask_R));
thickness.cdata(medial_mask_R) = zscore_fun(thickness.cdata(medial_mask_R));
marg_rh_1.cdata(medial_mask_R) = zscore_fun(marg_rh_1.cdata(medial_mask_R));

evo_exp1.cdata(medial_mask_R) = zscore_fun(evo_exp1.cdata(medial_mask_R));
evo_exp2.cdata(medial_mask_R) = zscore_fun(evo_exp2.cdata(medial_mask_R));
fchomology.cdata(medial_mask_R) = zscore_fun(fchomology.cdata(medial_mask_R));

devexp1.cdata(medial_mask_R) = zscore_fun(devexp1.cdata(medial_mask_R));
devexp2.cdata(medial_mask_R) = zscore_fun(devexp2.cdata(medial_mask_R));

genepc1.cdata(medial_mask_R) = zscore_fun(genepc1.cdata(medial_mask_R));
cogpc1.cdata(medial_mask_R) = zscore_fun(cogpc1.cdata(medial_mask_R));
cbf1.cdata(medial_mask_R) = zscore_fun(cbf1.cdata(medial_mask_R));
cbf2.cdata(medial_mask_R) = zscore_fun(cbf2.cdata(medial_mask_R));


figure(3)
clf
t7 = tiledlayout(2,6,'Padding','compact','TileSpacing','tight');
sgtitle(t7,{'Factors hypothetically associated with biological circuit flexibility'},'FontWeight','bold','FontSize',fs+1);


ax1 = nexttile(t7);
ax1.Layout.Tile=1;
myelin_plot = fmridisplay();

jet = colormap(ax1,'jet');

myelin_plot = surface(myelin_plot, 'axes', ax1, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
h = plot_to_surf(myelin_map.cdata(:),myelin_plot.surface{1}.object_handle,'sourcespace','MNI152NLin6Asym','targetsurface','fsLR_32k','nolegend', ...
    'colormap', 'turbo');

title(ax1, {'Myelination','(Wiring)'},'FontWeight','normal','FontSize',fs)


ax2 = nexttile(t7);
ax2.Layout.Tile=2;
thick_plot = fmridisplay();

thick_plot = surface(thick_plot, 'axes', ax2, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
plot_to_surf(thickness.cdata(:), thick_plot.surface{1}.object_handle,'colormap',colormap(ax2,'turbo'));

title(ax2, {'Thickness','(~Differentiation)'},'FontWeight','normal','FontSize',fs)


ax3 = nexttile(t7);
ax3.Layout.Tile=3;
net_plot = fmridisplay();

net_plot = surface(net_plot, 'axes', ax3, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
plot_to_surf(marg_rh_1.cdata(:), net_plot.surface{1}.object_handle,'nolegend',...
    'colormap','turbo');

title(ax3, {'Network','Hierarchy'},'FontWeight','normal','FontSize',fs)

ax4 = nexttile(t7);
ax4.Layout.Tile=4;
evo1_plot = fmridisplay();

evo1_plot = surface(evo1_plot, 'axes', ax4, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(evo_exp1.cdata(:), evo1_plot.surface{1}.object_handle,'colormap','turbo');

title(ax4, {'Evolutionary', 'Expansion 1'},'FontWeight','normal','FontSize',fs)


ax5 = nexttile(t7)
ax5.Layout.Tile = 5;
evo2_plot = fmridisplay();

evo2_plot = surface(evo2_plot, 'axes', ax5, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
[~,cbar1,cbar2] = plot_to_surf(evo_exp2.cdata, evo2_plot.surface{1}.object_handle,'colormap','turbo');
%{
cbar1.Location = 'southoutside';
cbar1.TickLabels = {'',''};
cbar1.Label.String = 'trans';
cbar1.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) + [0.15,-0.06];
cbar1.Position(3) = 0.15;
cbar1.Position(4) = 0.02;
cbar2.Location = 'south';
cbar2.TickLabels = {'',''};
cbar2.Label.String = 'uni';
cbar2.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) - [0,0.06];
cbar2.Position(3) = 0.15;
cbar2.Position(4) = 0.02;
%}

title(ax5, {'Evolutionary','Expansion 2'},'FontWeight','normal','FontSize',fs)


ax6 = nexttile(t7);
ax6.Layout.Tile=6;
fchomo_plot = fmridisplay();

fchomo_plot = surface(fchomo_plot, 'axes', ax6, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
% we add a small increment to the fchomology data because it's all positive
% valued except for the medial wall and a small patch of the posterior
% parietal cortex. plot_to_surface is designed to map zero values to
% grayscale for the medial wall, so to avoid a hole in the lateral view
% we're looking for we increment things slightly to circumvent the mapping
% of zero values to to grayscale
%[~,cbar1,cbar2] = plot_to_surf(fchomology.cdata + 0.000001,fchom.surface{1}.object_handle,'colormap','turbo');
[~,cbar1,cbar2] = plot_to_surf(fchomology.cdata, fchomo_plot.surface{1}.object_handle,'colormap','turbo');
%{
cbar1.Location = 'southoutside';
cbar1.TickLabels = {'',''};
cbar1.Label.String = 'trans';
cbar1.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) + [0.15,-0.05];
cbar1.Position(3) = 0.15;
cbar1.Position(4) = 0.02;
cbar2.Location = 'southoutside';
cbar2.TickLabels = {'',''};
cbar2.Label.String = 'uni';
cbar2.Position(1:2) = margulies1.surface{1}.axis_handles.Position(1:2) - [0,0.05];
cbar2.Position(3) = 0.15;
cbar2.Position(4) = 0.02;
%}

title(ax6, {'Funntional Conn.','Homology'},'FontWeight','normal','FontSize',fs)


ax7 = nexttile(t7);
ax7.Layout.Tile=11;
dev1_plot = fmridisplay();

dev1_plot = surface(dev1_plot, 'axes', ax7, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(devexp1.cdata(:)), dev1_plot.surface{1}.object_handle,'colormap','turbo');

title(ax7, {'Developental','Expansion 1'},'FontWeight','normal','FontSize',fs)

ax8 = nexttile(t7);
ax8.Layout.Tile=12;
dev2_plot = fmridisplay();

dev2_plot = surface(dev2_plot, 'axes', ax8, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(devexp2.cdata(:)), dev2_plot.surface{1}.object_handle,'colormap','turbo');

title(ax8, {'Developmental','Expansion 2'},'FontWeight','normal','FontSize',fs)

ax9 = nexttile(t7);
ax9.Layout.Tile=7;
gene_plot = fmridisplay();

gene_plot = surface(gene_plot, 'axes', ax9, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(genepc1.cdata(:)), gene_plot.surface{1}.object_handle,'colormap','turbo');

title(ax9, {'GenePC1','(transcriptomic)'},'FontWeight','normal','FontSize',fs)


ax10 = nexttile(t7);
ax10.Layout.Tile=8;
cog_plot = fmridisplay();

cog_plot = surface(cog_plot, 'axes', ax10, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(cogpc1.cdata(:)), cog_plot.surface{1}.object_handle,'colormap','turbo');

title(ax10, {'CogPC1','(neurosynth)'},'FontWeight','normal','FontSize',fs)


ax11 = nexttile(t7);
ax11.Layout.Tile=9;
cbf1_plot = fmridisplay();

cbf1_plot = surface(cbf1_plot, 'axes', ax11, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(cbf1.cdata(:)), cbf1_plot.surface{1}.object_handle,'colormap','turbo');

title(ax11, {'Cerebral Blood','Flow 1'},'FontWeight','normal','FontSize',fs)


ax12 = nexttile(t7);
ax12.Layout.Tile=10;
cbf2_plot = fmridisplay();

cbf2_plot = surface(cbf2_plot, 'axes', ax12, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');
    
plot_to_surf(double(cbf2.cdata(:)), cbf2_plot.surface{1}.object_handle,'colormap','turbo');

title(ax12, {'Cerebral','Blood Flow 2'},'FontWeight','normal','FontSize',fs)

%

pos = get(gcf,'Position');
set(gcf,'Position',[pos(1:2),750,288]);

export_fig(gcf,'panels/gradients_wide.png','-png','-r300','-transparent')
