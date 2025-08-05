function [t0,o2,bar1axis] = create_montage_figure(fmri_data_obj, cmap, cmaprange, title_text, fontsize)
    underlay = which('fsl6_hcp_template.nii.gz');    
    grayord_xyz = [-24,-12,-6,6,12,24; -36,-17,-12,-5,0,14; -50,-34,-17,-10,6,10]';
    
    figure;
    o2 = fmridisplay('overlay', underlay);
    
    t0 = tiledlayout(3,6,'Padding','none','TileSpacing','tight');
    
    % create underlay panels
    t = cell(1,18);
    orient = {'saggital','coronal','trans'};
    ax_lbl = {'X','Y','Z'};
    for i = 1:18
        t{i} = nexttile();
        this_orient = orient{ceil(i/6)};
        [o2, dat] = montage(o2, this_orient, 'wh_slice', grayord_xyz(mod(i-1,6)+1,:), ...
            'onerow', 'noverbose', 'existing_axes', t{i});
        switch this_orient
            case 'saggital'
                ax = 1;
            case 'coronal'
                ax = 2;
            case 'trans'
                ax = 3;
        end
        title(sprintf('%s = %d', ax_lbl{ax}, grayord_xyz(mod(i-1,6)+1, ax)), ...
            'FontSize', fontsize);
    end
    
    % overlay data
    o2 = fmri_data_obj.montage(o2, 'colormap', cmap, 'cmaprange', cmaprange, ...
        'overlay', underlay);
    delete(o2.activation_maps{1}.legendhandle(1));
    
    % styling
    sgtitle(title_text,'FontSize',fontsize+4,'FontWeight','bold')
    
    pos = get(gcf,'Position');
    set(gcf,'Position', [pos(1:2), 1240, 475]);

    t0.Position(3) = 0.94;

    % add colorbar
    bar1axis = axes('Position', [.77 .25 .2 .4]);
    colorbar1_han = colorbar(bar1axis);
    colormap(bar1axis, cmap);
    colorbar1_han.Ticks = [0,1];
    colorbar1_han.TickLabels = arrayfun(@(x1)sprintf('%0.2f',x1),cmaprange,'UniformOutput',false);
    set(bar1axis, 'Visible', 'off','FontSize',fontsize);
end