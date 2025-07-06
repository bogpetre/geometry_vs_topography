function dat = get_cifti_data(path)
    dat = cifti_read(path);
    
    lind = dat.diminfo{1}.models{1}.start + (1:dat.diminfo{1}.models{1}.count) - 1;
    ctxl = zeros(size(dat.cdata,2), dat.diminfo{1}.models{1}.numvert);
    ctxl(:,dat.diminfo{1}.models{1}.vertlist+1) = dat.cdata(lind,:)';
    
    rind = dat.diminfo{1}.models{2}.start + (1:dat.diminfo{1}.models{2}.count) - 1;
    ctxr = zeros(size(dat.cdata,2), dat.diminfo{1}.models{2}.numvert);
    ctxr(:,dat.diminfo{1}.models{2}.vertlist+1) = dat.cdata(rind,:)';
    
    this_vol = [];
    for j = 3:length(dat.diminfo{1}.models)
        this_mdl = dat.diminfo{1}.models{j};
        vol_ind = this_mdl.start + (1:this_mdl.count) - 1;
        this_vol = [this_vol; dat.cdata(vol_ind,:)];
    end
    vol = this_vol';
    
    dat = struct('cortex_left', ctxl, 'cortex_right', ctxr, 'volumes', vol);
end