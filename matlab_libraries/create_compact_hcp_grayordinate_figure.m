function [o2, o3, t0, t1, t2, t3, t4, t5, t6, t7, t8] = create_compact_hcp_grayordinate_figure(varargin)   
    overlay = canlab_get_underlay_image;
    for i = 1:size(varargin)
        if ischar(varargin{i})
            switch varargin{i}
                case 'overlay'
                    overlay = varargin{i+1};
            end
        end
    end

    grayord_xyz = [-24,-12,-6,6,12,24; -36,-17,-12,-5,0,14; -50,-34,-17,-10,6,10]';

    o2 = fmridisplay('overlay', which(overlay));

    t0 = tiledlayout(5,5,'Padding','none','TileSpacing','tight');
    t1 = nexttile();
    t1.Layout.TileSpan = [2,2];
    axis off;
    t2 = nexttile();
    t2.Layout.TileSpan = [2,2];
    axis off;
    t3 = nexttile();
    t3.Layout.TileSpan = [2,2];
    t3.Layout.Tile = 11;
    axis off;
    t4 = nexttile();
    t4.Layout.TileSpan = [2,2];
    t4.Layout.Tile = 13;
    axis off;
    o2 = surface(o2, 'axes', t1, 'direction', 'hcp inflated right', 'orientation', 'medial', 'disableVis3d');     
    o2 = surface(o2, 'axes', t2, 'direction', 'hcp inflated left', 'orientation', 'lateral', 'disableVis3d');
    o2 = surface(o2, 'axes', t3, 'direction', 'hcp inflated left', 'orientation', 'medial', 'disableVis3d');     
    o2 = surface(o2, 'axes', t4, 'direction', 'hcp inflated right', 'orientation', 'lateral', 'disableVis3d');
    
    o3 = fmridisplay('overlay', which(overlay));
    t5 = nexttile();
    t5.Layout.Tile = 21;
    axis off
    t6 = nexttile();
    t6.Layout.Tile = 22;
    axis off;
    t7 = nexttile();
    t7.Layout.Tile = 23;
    axis off;
    t8 = nexttile();
    t8.Layout.Tile = 24;
    axis off

    [o3, dat] = montage(o3, 'saggital', 'wh_slice', grayord_xyz(1,:), 'onerow', 'noverbose', 'existing_axes',t5);
    [o3, dat] = montage(o3, 'saggital', 'wh_slice', grayord_xyz(3,:), 'onerow', 'noverbose', 'existing_axes',t6);
    [o3, dat] = montage(o3, 'saggital', 'wh_slice', grayord_xyz(5,:), 'onerow', 'noverbose', 'existing_axes',t7);
    [o3, dat] = montage(o3, 'axial', 'wh_slice', grayord_xyz(2,:), 'onerow', 'noverbose', 'existing_axes',t8);

    title(t5,'X = -24','FontSize',10)
    title(t6,'X = -6','FontSize',10)
    title(t7,'X = 12','FontSize',10)
    title(t8, 'Z = -34','FontSize',10)
    for t = [t5,t6,t7]
        set(t,'XLim',[-100,30],'YLim',[-70,25]);
    end
    set(t8,'XLim',[-60,60],'YLim',[-100,30]);

    wh_surfaces = [1:4];
end