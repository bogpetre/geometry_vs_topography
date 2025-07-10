function [B, CI, p, cohensf2] = neuromaps_corr(DV, IV, obs_map, perm_map, varargin)
    assert(all(size(IV) == size(DV)));

    [n,p] = size(IV);
    
    assert(size(obs_map,1) == p);
    assert(size(obs_map,2) == 1);
    assert(size(perm_map,1) == p);

    rois = repmat((1:p), n, 1);
    sid = repmat((1:n)', 1, p);

    obs_map = repmat(obs_map', n, 1);

    % let's assume that perm_map is good everywhere map is
    isgood = find(~isnan(IV) & ~isnan(DV) & ~isnan(obs_map));
    IV = IV(isgood);
    DV = DV(isgood);
    rois = rois(isgood);
    sid = sid(isgood);
    obs_map = obs_map(isgood);

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

    intx = IV .* repmat(perm_map,n,1); % precompute all interaction effects
    
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

    X = [Xf, IV.*obs_map];
    TSS = sum((Yg - mean(Yg)).^2);
    parfor i = 1:size(intx,2)
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
        r2_perm(i) = 1 - sum((Yg - X*beta).^2)./TSS;
    end


    intx = IV.*obs_map;
    B = Xf' * intx;
    c = sum(intx.^2,1);
    d = intx' * Yg;

    M = [A, B; B', c];
    rhs = [b; d];

    beta = M \ rhs;

    B = beta(end);
    CI = prctile(perm,[2.5,97.5]) + B;
    p = sum(abs(perm - mean(perm)) >= abs(B - mean(perm)))/length(perm);
    
    R2_full = 1 - sum((Yg - X*beta).^2)/TSS;

    % get reduced model to compute effect size
    %beta_reduced = A \ b;
    %R2_reduced = 1 - sum((Yg - Xf*beta_reduced).^2)./TSS;

    % compute Cohen's f^2
    %cohensf2 = (R2_full - R2_reduced)/(1-R2_full);
    cohensf2 = (R2_full - mean(r2_perm))/(1-mean(r2_perm));
    
    %cohensf2 = (B - mean(perm))/std(perm);
end