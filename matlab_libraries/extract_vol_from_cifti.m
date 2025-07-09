function nifti_data = extract_vol_from_cifti(cifti)
    brain_model = cifti.diminfo{1};
    volume_metadata = brain_model.vol;

    volume_size = volume_metadata.dims;
    affine = volume_metadata.sform + [zeros(4,3), [2;-2;-2;0]];
    nifti_volume = false(volume_size);

    vol_struct_ind = find(~contains(cellfun(@(x1)(x1.struct), brain_model.models, 'UniformOutput', false),'CORTEX'));
    all_voxlist = [];
    nifti_vec = [];
    for i = 1:length(vol_struct_ind)
        this_struct = brain_model.models{vol_struct_ind(i)};
        voxlist = this_struct.voxlist + 1; % change 0-indexing to 1-indexing
        nifti_vec = [nifti_vec; cifti.cdata(this_struct.start + (0:this_struct.count-1),:)];
        for j = 1:size(voxlist)
            nifti_volume(voxlist(1,j), voxlist(2,j), voxlist(3,j)) = true;
        end
        all_voxlist = [all_voxlist; voxlist'];
    end

    nifti_struct = image_vector;
    nifti_struct.volInfo = struct();
    nifti_struct.volInfo.dim = size(nifti_vec);
    nifti_struct.volInfo.dt = [16, 0];
    nifti_struct.volInfo.mat = affine;   
    nifti_struct.volInfo.xyzlist = all_voxlist;
    nifti_struct.volInfo.n_inmask = size(nifti_struct.volInfo.xyzlist,1);
    nifti_struct.volInfo.nvox = prod(volume_size);
    nifti_struct.volInfo.image_indx = false(prod(volume_size),1);
    nifti_struct.volInfo.image_indx = nifti_volume(:);
    nifti_struct.volInfo.wh_inmask = find(nifti_volume);
    nifti_struct.volInfo.fname = [];
    nifti_struct.dat = double(nifti_vec);

    nifti_data = fmri_data(nifti_struct);
end