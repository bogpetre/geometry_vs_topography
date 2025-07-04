function rdm_vec_patched = patch_block_diagonals(rdm_orig, rdm_full, task_ids, replicate_ids, block_labels)
    for b = 1:length(block_labels)
        task = block_labels(b);
        inds = find(task_ids == task);
        block = rdm_orig(inds, inds);
        mean_off_diag = mean(block(~eye(length(inds))));

        for i = inds
            this_off_diag = find(replicate_ids == i);
            rdm_full(this_off_diag, this_off_diag) = mean_off_diag;
        end
    end
    rdm_full(logical(eye(size(rdm_full)))) = 0;

    rdm_vec_patched = squareform(rdm_full);
end