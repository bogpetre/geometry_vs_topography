function [B, CI, p, effectsize, sampling_var, perm_var, nu] = neuromaps_corr_fx(obs_val, obs_map, perm_map, varargin)
    [p, n] = size(obs_val);
    [p_perm, m] = size(perm_map);
    [p_map, n_map] = size(obs_map);

    assert(p == p_perm & p == p_map, 'obs_val, obs_map and perm_map must all have the same number of features (rows).');
    assert(n_map == 1, 'obs_map must be a column vector');

    B_perm = zeros(m,1);
    parfor i = 1:m
        subj_spin_betas = get_mean_B(obs_val, perm_map(:,i), varargin{:});
        B_perm(i) = mean(subj_spin_betas(:,1));
    end
    perm_var = var(B_perm);
    
    subj_B = get_mean_B(obs_val, obs_map, varargin{:});
    B = mean(subj_B(:,1));

    % we could just use sampling_var = var(subj_B(:,1))/n, but for 
    % consistency with coupling analysis let's do the trivial jackknife 
    % estimate of the standard error of B.
    jk = nan(size(subj_B,1),1);
    parfor i = 1:size(subj_B,1)
        ind = 1:size(subj_B,1);
        ind(i) = [];
        jk(i) = mean(subj_B(ind,1));
    end
    sampling_var = (n-1)/n*sum((jk - mean(jk)).^2);

    se = sqrt(perm_var + sampling_var);
    z = B/se;

    CI = icdf('norm', [0.025, 0.975], B, se);
    p = 2*normcdf(-abs(z));

    effectsize = B / sqrt(perm_var + n*sampling_var);
end

function B = get_mean_B(obs_val, obs_map, varargin)
    design_vars = 2;
    if nargin > 2
        design_vars = design_vars + size(varargin{1},2);
    end
    
    B = zeros(size(obs_val,2),design_vars);
    for j = 1:size(obs_val,2)
        X = [obs_map, ones(size(obs_map,1),1)];
        if ~isempty(varargin)
            for i = 1:length(varargin{1})
                % deal with confounds
                X = [X, varargin{1}{i}(:,j)];
            end
        end
        Y = obs_val(:,j);

        good_roi = ~isnan(Y) & ~any(isnan(X),2);
        Y = Y(good_roi);
        X = X(good_roi,:);

        B(j,:) = ((X'*X)\X'*Y(:))';
    end
end