% this version doesn't fit an ACE model. It just estimates BCa CI for
% family clusters

function [family_clusters_bs, family_clusters_jk, family_clusters] = bootstrap5(data_mat, nperm, groups, mz, dz, fs, hs, unr, perm_inds)
    family_clusters_bs = zeros(5,nperm);
    parfor i = 1:nperm    
        this_group_ind = groups(perm_inds(i,:));
        ind = cat(1,this_group_ind{:});
        
        % topography
    
        this_data = data_mat(ind, ind);
        this_data(logical(eye(size(this_data)))) = nan;
        
        [this_mz, this_dz, this_fs, this_hs, this_unr] = get_cluster_means( ...
            this_data, mz(ind, ind), dz(ind, ind), ...
            fs(ind, ind), hs(ind, ind), unr(ind, ind));
        family_clusters_bs(:,i) = [this_mz, this_dz, this_fs, this_hs, this_unr]
    end

    parfor i = 1:length(groups)    
        this_group_ind = groups(~ismember(1:length(groups), i));
        ind = cat(1,this_group_ind{:});
        
        % topography
    
        this_data = data_mat(ind, ind);
        this_data(logical(eye(size(this_data)))) = nan;
    
        [this_mz, this_dz, this_fs, this_hs, this_unr] = get_cluster_means( ...
            this_data, mz(ind, ind), dz(ind, ind), ...
            fs(ind, ind), hs(ind, ind), unr(ind, ind));
        family_clusters_jk(:,i) = [this_mz, this_dz, this_fs, this_hs, this_unr];
    end
    
    [r_mz, r_dz, r_fs, r_hs, r_unr] = get_cluster_means( ...
            data_mat, mz, dz, fs, hs, unr);
    family_clusters = [r_mz, r_dz, r_fs, r_hs, r_unr];
end