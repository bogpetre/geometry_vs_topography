function [B, p, Bstd] = neuromaps_corr(obs_val, obs_map, perm_map)
    perm = zeros(size(perm_map,2), 1);
    for j = 1:size(perm_map,2)
        X = [perm_map(:, j), ones(size(perm_map,1),1)];
        Y = obs_val;
        b0 = (X'*X)\X'*Y(:);
        perm(j) = b0(1);
    end
    perm = mean(perm,2);
    X = [obs_map, ones(size(obs_map,1),1)];
    Bb0 = (X'*X)\X'*Y(:);
    B = Bb0(1);
    p = sum(abs(perm - mean(perm)) >= abs(B - mean(perm)))/length(perm);

    X = zscore(obs_map);
    Y = zscore(obs_val);
    Bstd = (X'*X)\X'*Y(:);
end