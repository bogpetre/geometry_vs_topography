function p = render_surface(vertices, faces, ax, varargin)
    viewAngle = [270,0];
    for i = 1:length(varargin)
        if ischar(varargin{i})
            switch varargin{i}
                case 'flip'
                    viewAngle = [90,0];
                case 'flat'
                    viewAngle = [0,90];
                otherwise
                    warning('Did not recognize argument %s',varargin{i});
            end
        end
    end

    axes(ax);
    cla;

    p = patch('Faces',faces,'Vertices',vertices,'FaceColor',[.5 .5 .5], ...
        'EdgeColor','none','SpecularStrength',.2,'FaceAlpha',1,'SpecularExponent',200);

    view(viewAngle(1),viewAngle(2))
    camlight right;

    lighting gouraud
    axis vis3d image tight

    axis off;
    axis image;
    lightRestoreSingle(gca);
    material dull;
end