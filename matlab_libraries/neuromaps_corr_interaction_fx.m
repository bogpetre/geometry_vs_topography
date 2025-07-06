function [B, B_95_range, bootstat, Bstd, Bstd_95_range, bootstat_std, cohensD, cohensDStd] = neuromaps_corr_interaction_fx(obs_val1, obs_val2, obs_map, varargin)
    design_vars = 2 + 2*size(obs_map,2);
    if nargin > 3
        design_vars = design_vars + 2*size(varargin{1},2);
    end

    assert(size(obs_val1,1) == size(obs_val2,1))
    valMain = [0.5*ones(size(obs_val1,1),1); -0.5*ones(size(obs_val2,1),1)];
    
    Bb0 = zeros(size(obs_val1,2),design_vars);
    xMain = [obs_map; obs_map];
    for j = 1:size(obs_val1,2)
        X = [xMain, valMain, xMain.*valMain, ones(size(xMain,1),1)];
        if ~isempty(varargin)
            for k = 1:length(varargin{1})
                % deal with confounds
                this_cfd = [varargin{1}{k}(:,j); varargin{1}{k}(:,j)];
                this_cfd = this_cfd - nanmean(this_cfd);
                this_cfd = [this_cfd, this_cfd.*valMain]; % let confounds have different effects on different metrics
                X = [X, this_cfd];
            end
        end
        Y = [obs_val1(:,j); obs_val2(:,j)];

        good_roi = ~isnan(Y) & ~any(isnan(X),2);
        Y = Y(good_roi);
        X = X(good_roi,:);

        Bb0(j,:) = ((X'*X)\X'*Y(:))';
    end
    B = nanmean(Bb0,1);
    B_95_range = prctile(Bb0, [2.5, 97.5], 1)';
    [~,bootstat] = bootci(50000, {@mean, Bb0}, 'type', 'bca');
    cohensD = nanmean(Bb0)./nanstd(Bb0);
    %p = sum(abs(perm - mean(perm)) >= abs(B - mean(perm)))/length(perm);

    Bstd0 = zeros(size(obs_val1,2),design_vars-2);
    for j = 1:size(obs_val1,2)
        X = [zscore(xMain), zscore(xMain).*valMain];
        if ~isempty(varargin)
            for k = 1:length(varargin{1})
                % deal with confounds
                this_cfd = [zscore(varargin{1}{k}(:,j)); zscore(varargin{1}{k}(:,j))];
                this_cfd = [this_cfd, this_cfd.*valMain]; % let confounds have different effects on different metrics
                X = [X, this_cfd];
            end
        end
        Y = [zscore(obs_val1(:,j)); zscore(obs_val2(:,j))];
        Bstd0(j,:) = ((X'*X)\X'*Y(:))';
    end
    % main effect and intercept are both zero because we z-score each
    % observed map (our Y) individually so they're mean 0 and have no
    % difference in magnitude. We're just testing if one map is more
    % correlated with neuromaps of interest than the other.
    Bstd = [nanmean(Bstd0(:,1:size(obs_map,2)),1), 0, nanmean(Bstd0(:,size(obs_map,2)+1:end),1), 0];
    cohensDStd = nanmean(Bstd0)./nanstd(Bstd0);
    cohensDStd = [cohensDStd(1:size(obs_map,2)), 0, cohensDStd(size(obs_map,2)+1:end), 0];

    Bstd_95_range = prctile(Bstd0, [2.5,97.5], 1)';
    [~,bootstat_std0] = bootci(50000, {@mean, Bstd0}, 'type', 'bca');
    bootstat_std = [bootstat_std0(:,1:size(obs_map,2)), zeros(size(bootstat_std0,1),1), bootstat_std0(:,size(obs_map,2)+1:end), zeros(size(bootstat_std0,1),1)];
end