function [B, CI, p, effectsize, sampling_var, perm_var, nu] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, obs_map, perm_map, varargin)
    assert(size(obs_val1,1) == size(obs_val2,1))
    
    [p, n] = size(obs_val1);
    [p_, n_] = size(obs_val2);
    [p_perm, m] = size(perm_map);
    [p_map, n_map] = size(obs_map);

    assert(p == p_perm & p == p_map & p == p_, ...
        'obs_val1, obs_val2, obs_map and perm_map must all have the same number of features (rows).');
    assert(n == n_, 'observed maps must have the same number of dyads')
    assert(n_map == 1, 'obs_map must be a column vector');

    subj_B = get_mean_B(obs_val1, obs_val2, obs_map, varargin{:});
    B = mean(subj_B);

    B_perm = zeros(m,size(B,2));
    parfor i = 1:m
        % impute nan values
        subj_spin_betas = get_mean_B(obs_val1, obs_val2, perm_map(:,i), varargin{:});
        B_perm(i,:) = mean(subj_spin_betas);
    end
    perm_var = var(B_perm);
    perm_var(:,3:end) = 0;

    jk = nan(size(subj_B));
    for i = 1:size(subj_B,1)
        ind = 1:size(subj_B,1);
        ind(i) = [];
        jk(i,:) = mean(subj_B(ind,:));
    end
    sampling_var = (n-1)/n*sum((jk - mean(jk)).^2);
    
    %{
    % Adjust for covariance (minor adjustment, very slow)
    jk_null = nan(size(subj_B));
    for i = 1:size(subj_B,1)
        ind = 1:size(subj_B,1);
        ind(i) = [];
        this_B_perm = zeros(1000,size(B,2));
        parfor j = 1:1000
            % impute nan values
            confounds = cell(1,length(varargin));
            if ~isempty(varargin)
                for k = 1:length(varargin{1})
                    confounds{k} = varargin{1}{k}(:,ind);
                end
            end
            subj_spin_betas = get_mean_B(obs_val1(:,ind), obs_val2(:,ind), perm_map(:,j), confounds);
            this_B_perm(j,:) = mean(subj_spin_betas);
        end
        jk_null(i,:) = mean(this_B_perm);
    end
    sampling_perm_cov = (n-1)/n*sum((jk - mean(jk)).*(jk_null - mean(jk_null)));
    sampling_perm_cov(3:end) = 0;
    perm_var = perm_var - sampling_perm_cov;
    sampling_var = sampling_var - sampling_perm_cov;
    %}

    se = sqrt(perm_var + sampling_var);
    %nu = (perm_var + sampling_var).^2 ./ (sampling_var.^2/(n-1) + perm_var.^2/(n-1));
    nu = n-1;
    t = B./se;

    CI = zeros(size(B,2),2);
    p = nan(size(B,2),1);
    for i = 1:length(B)
        CI(i,:) = sqrt(sampling_var(i))*icdf('t', [0.025, 0.975], nu) + B(i);
        if i < 3
            % assuming we haven't done a spin test on confounds
            p(i) = 2*tcdf(-abs(t(i)), nu);
        end
    end

    effectsize = B ./ sqrt(perm_var + n*sampling_var);
    effectsize(3:end) = nan;
end


function B = get_mean_B(obs_val1, obs_val2, obs_map, varargin)
    design_vars = 2*size(obs_map,2);
    if nargin > 3
        design_vars = design_vars + 2*size(varargin{1},2);
    end
    
    valMain = [0.5*ones(size(obs_val1,1),1); -0.5*ones(size(obs_val2,1),1)];
    
    xMain = [obs_map; obs_map];
    
    B = zeros(size(obs_val1,2),design_vars);
    for j = 1:size(obs_val1,2)
        %X = [zscore(xMain), zscore(xMain).*valMain];
        X = xMain;
        if ~isempty(varargin)
            for k = 1:length(varargin{1})
                % deal with confounds
                this_cfd = [varargin{1}{k}(:,j); varargin{1}{k}(:,j)];
                X = [X, this_cfd];
            end
        end
        Y = [obs_val1(:,j); obs_val2(:,j)];

        good_roi = find(~isnan(Y) & ~any(isnan(X),2));
        % we want each outcome standardized separately
        Y = [zscore(Y(good_roi(1:size(obs_val1,1)))); ...
            zscore(Y(good_roi(size(obs_val1,1)+1:end)))];
        % The top half of X is identical to the bottom half, so we don't
        % need to separate centering like we do for Y. Note, that we're
        % also adding interaction effects for confounds here, which allows
        % for the confound effects to vary by similarity metric.
        X = X(good_roi,:);
        X = X - mean(X);
        X = [X, X.*valMain(good_roi,:)];



        B(j,:) = ((X'*X)\X'*Y(:))';
    end
    % reshape to be [main1, int1, main2, int2, etc.]
    ind = reshape(1:design_vars,design_vars/2,2)';
    ind = ind(:);

    B = B(:,ind);
end