function fig = plot_to_brain(data, roi_ind, cmaprange, title, fs, varargin)

if nargin == 5
    atlas_cii = cifti_read(which('CANLab2024_MNI152NLin6Asym_coarse_2mm.dlabel.nii'));
    %atlas_nii = load_atlas('canlab2024');
else
    atlas_cii = varargin{1};
atlas_nii = fmri_data(extract_vol_from_cifti(atlas_cii));

B = zeros(1,518);
B(roi_ind) = data;
good_rois = roi_ind;

figure;
cm = colormap('hot');
[o2,o3,t0] = create_compact_hcp_grayordinate_figure();


left_ctx_ind = cellfun(@(x1)(strcmp(x1.struct,'CORTEX_LEFT')), atlas_cii.diminfo{1}.models);
left_ctx_mdl = atlas_cii.diminfo{1}.models{left_ctx_ind};
cortex_left = zeros(1,left_ctx_mdl.numvert);
cortex_left(left_ctx_mdl.vertlist+1) = atlas_cii.cdata(left_ctx_mdl.start+(1:left_ctx_mdl.count)-1);
cortex_left_val = nan(size(cortex_left));
uniq_rois = unique(cortex_left);
uniq_rois(uniq_rois == 0 | ~ismember(uniq_rois, good_rois)) = [];
for i = 1:length(uniq_rois)
    this_roi = uniq_rois(i);
    if ~isnan(B(this_roi)) && abs(imag(B(this_roi))) < eps
        cortex_left_val(cortex_left == this_roi) = real(B(this_roi));
    end
end
[~,cbar1, cbar2] = plot_to_surf(cortex_left_val',o2.surface{2}.object_handle,...
    'colorbar','cmaprange',cmaprange,'colormap',cm);
plot_to_surf(cortex_left_val',o2.surface{3}.object_handle,...
    'cmaprange',cmaprange,'colormap',cm);


right_ctx_ind = cellfun(@(x1)(strcmp(x1.struct,'CORTEX_RIGHT')), atlas_cii.diminfo{1}.models);
right_ctx_mdl = atlas_cii.diminfo{1}.models{right_ctx_ind};
cortex_right = zeros(1,right_ctx_mdl.numvert);
cortex_right(right_ctx_mdl.vertlist+1) = atlas_cii.cdata(right_ctx_mdl.start+(1:right_ctx_mdl.count)-1);
cortex_right_val = nan(size(cortex_right));
uniq_rois = unique(cortex_right);
uniq_rois(uniq_rois == 0 | ~ismember(uniq_rois, good_rois)) = [];
for i = 1:length(uniq_rois)
    this_roi = uniq_rois(i);
    if ~isnan(B(this_roi)) && abs(imag(B(this_roi))) < eps
        cortex_right_val(cortex_right == this_roi) = real(B(this_roi));
    end
end
plot_to_surf(cortex_right_val',o2.surface{1}.object_handle,...
    'cmaprange',cmaprange,'colormap',cm);
plot_to_surf(cortex_right_val',o2.surface{4}.object_handle,...
    'cmaprange',cmaprange,'colormap',cm);

ref = fmri_data(atlas_nii);
ref.dat = double(ref.dat);
ctx_ind = find(ismember(ref.dat, unique([cortex_left, cortex_right])));
ref.dat(ctx_ind) = 0;
subctx_ind = find(~ismember(1:length(B), unique([cortex_left, cortex_right]))); % any indices not in cortex left or right
uniq_rois = unique(ref.dat);
uniq_rois(uniq_rois == 0 | ~ismember(uniq_rois, good_rois)) = [];
ref_val = ref;
ref_val.dat(:) = nan;
for i = 1:length(uniq_rois)
    this_roi = uniq_rois(i);
    if ~isnan(B(this_roi)) && abs(imag(B(this_roi))) < eps
        ref_val.dat(ref.dat == this_roi) = real(B(this_roi));
    end
end
o3 = ref_val.montage(o3,'hcp grayordinates subcortex','cmaprange',cmaprange,'colormap',cm,'overlay',which('fsl6_hcp_template.nii.gz'));
try, delete(o3.activation_maps{1}.legendhandle); end

t0.Position(4) = 0.73;
try cbar1.Position = [cbar1.Position(1), 0.5, cbar1.Position(3), 0.25]; end
try cbar2.Position = [cbar2.Position(1), cbar2.Position(2) - 0.05, cbar2.Position(3), 0.25]; end
try set(cbar1,'FontSize',fs); end
try set(cbar2,'FontSize',fs); end
if min(cmaprange) >= 0, try delete(cbar2); end; end
sgtitle(title,'FontWeight','bold','FontSize',fs+1)
pos = get(gcf,'Position');
set(gcf,'Position', [pos(1:2), 400,407])

end