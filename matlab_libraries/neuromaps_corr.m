function [B, CI, p, effectsize, perm_var, jk_var, nu] = neuromaps_corr(DV, IV, obs_map, perm_map, varargin)
    assert(all(size(IV) == size(DV)));

    [n,p] = size(IV);
    [~,m] = size(perm_map);
    
    assert(size(obs_map,1) == p);
    assert(size(obs_map,2) == 1);
    assert(size(perm_map,1) == p);

    % get jacknife variance estimate of B
    jk = neuromaps_corr_jk(DV, IV, obs_map, varargin{:});
    jk_var = (n-1)/n * sum((jk - mean(jk)).^2);

    rois = repmat((1:p), n, 1);
    sid = repmat((1:n)', 1, p);

    obs_map = repmat(obs_map', n, 1);
    perm_map = repmat(perm_map, n, 1);

    isgood = find(~isnan(IV) & ~isnan(DV) & ~isnan(obs_map));
    IV = IV(isgood);
    DV = DV(isgood);
    rois = rois(isgood);
    sid = sid(isgood);
    obs_map = obs_map(isgood);
    
    new_perm_map = zeros(length(isgood),size(perm_map,2));
    for i = 1:size(perm_map,2)
        this_perm_map = perm_map(:,i);

        medial_wall = isnan(this_perm_map) & isgood;
        % use mean imputation. Entire design is centered
        this_perm_map(medial_wall) = nanmean(this_perm_map(isgood));

        new_perm_map(:,i) = this_perm_map(isgood);
    end
    perm_map = new_perm_map;
    clear new_perm_map

    % center for interpretable interaction coefficients
    % superfluous if you've already subtracted out subject means. If all
    % subjects are mean zero, all data is mean zero too
    IV = IV - mean(IV);
    DV = DV - mean(DV);

    sid = helmertCoding(sid);
    rois = dummyvar(categorical(rois));

    X0 = [];
    if ~isempty(varargin)
        for i = 1:length(varargin{1})
            % deal with confounds
            normed_arg = zscore(varargin{1}{i},[],2);
            X0 = [X0, normed_arg(isgood).*sid];
        end
    end

    % We have many permuted maps to test, but each test involves 
    % modifying only a single column of a large matrix. This can be made
    % significantly more efficient by precomputing the common covariance
    % elements and then using a rank-1 update for each map rather than a
    % full OLS.
    Xf = [IV, rois, sid, X0]; % (N x k), shared elements
    Yg = DV;          % (N x 1)

    intx = IV .* perm_map; % precompute all interaction effects
    
    % Precompute fixed terms
    A = Xf' * Xf;     % (k x k)
    b = Xf' * Yg;     % (k x 1)
    
    % Allocate output
    perm = zeros(size(perm_map,2), 1);
    
    % compute all the OLS using rank-1 updates of the above A and b
    % matrices
    B = Xf' * intx;             % (k x n_perms)
    C = sum(intx.^2, 1);        % (1 x n_perms)
    d = intx' * Yg;             % (n_perms x 1)

    parfor i = 1:m
        % parallelization only increases speed a bit since the code below
        % parallelizes pretty well at the level of the linalg libraries.
        % Run serially if memory limited.
        Bi = B(:, i); 
        ci = C(i); 
        di = d(i);
        
        % Build augmented system adding in covariances in the rightmost
        % and bottommost off diagonal columns/rows. Add variance of 
        % the new term in the bottom right corner. This gives us (X'*X) 
        % of the ols formula (X'*X)^-1*X'*Y.
        M = [A, Bi; Bi', ci];
        % we get the cross covariance of X'*Y from b in a similar 
        % way, by adding the missing covariance term.
        rhs = [b; di]; 
        
        beta = M \ rhs;  % solve (k+1 x k+1) system
        perm(i) = beta(end);
    end
    perm_var = var(perm);

    intx = IV.*obs_map;
    B = Xf' * intx;
    c = sum(intx.^2,1);
    d = intx' * Yg;

    M = [A, B; B', c];
    rhs = [b; d];

    beta = M \ rhs;

    B = beta(end);

    
    % compute df of se using Welch-Satterthaite approximation
    %nu = (perm_var + jk_var)^2 ./ (jk_var^2/(n-1) + perm_var^2/(m-1));
    nu = n -1;
    se = sqrt(perm_var + jk_var);
    t = B/se;

    %CI = icdf('norm', [0.025, 0.975], 0, sqrt(jk_var)) + B;
    CI = sqrt(jk_var)*icdf('t', [0.025, 0.975], nu) + B;
    p = 2*tcdf(-abs(t), nu);
        
    effectsize = B / sqrt(perm_var + n*jk_var);
end

function B_jk = neuromaps_corr_jk(DV, IV, obs_map, varargin)
    
    jk_samples = zeros(size(DV,1)-1, size(DV,1));
    for i = 1:size(DV,1)
        this_jk = 1:size(DV,1);
        this_jk(this_jk == i) = [];
        jk_samples(:,i) = this_jk;
    end
    
    B_jk = resample(jk_samples, DV, IV, obs_map, varargin{:});
end

function B = resample(samples, DV, IV, obs_map, varargin)
    assert(all(size(IV) == size(DV)));
    
    assert(all(all(~isnan(IV))) && all(all(~isnan(DV))));

    [n,p] = size(IV);
    
    assert(size(obs_map,1) == p);
    assert(size(obs_map,2) == 1);

    rois = repmat((1:p), n-1, 1);
    sid = repmat((1:n-1)', 1, p);

    obs_map = repmat(obs_map', n-1, 1);

    isgood = find(~isnan(obs_map));
    %IV = IV(isgood);
    %DV = DV(isgood);
    rois = rois(isgood);
    sid = sid(isgood);
    obs_map = obs_map(isgood);

    sid = helmertCoding(sid);
    rois = dummyvar(categorical(rois));

    % because all participants are matched on number of entries we can
    % reuse everything avove when we resample
    B = nan(size(samples,2),1);
    parfor i = 1:size(samples,2)
        this_IV = IV(samples(:,i),:);
        this_DV = DV(samples(:,i),:);

        this_IV = this_IV(isgood);
        this_DV = this_DV(isgood);

        % center for interpretable interaction coefficients
        % superfluous if you've already subtracted out subject means. If all
        % subjects are mean zero, all data is mean zero too
        this_IV = this_IV - mean(this_IV);
        this_DV = this_DV - mean(this_DV);


        X0 = [];
        if ~isempty(varargin)
            for j = 1:length(varargin{1})
                % deal with confounds
                this_arg = varargin{1}{j}(samples(:,i),:);
                normed_arg = zscore(this_arg,[],2);
                X0 = [X0, normed_arg(isgood).*sid];
            end
        end

    
        X = [this_IV, rois, sid, X0, this_IV.*obs_map]; % (N x k), shared elements
        Yg = this_DV;          % (N x 1)
        
        beta = (X'*X)\X'*Yg;  % solve (k+1 x k+1) system
        B(i) = beta(end);
    end
end