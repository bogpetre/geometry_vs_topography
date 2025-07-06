function [B, B_95_range, bootstat, Bstd, Bstd_95_range, bootstat_std, cohenD, cohenDStd] = neuromaps_corr_fx(obs_val, obs_map, varargin)
    design_vars = 2;
    if nargin > 2
        design_vars = design_vars + size(varargin{1},2);
    end
    
    Bb0 = zeros(size(obs_val,2),design_vars);
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

        Bb0(j,:) = ((X'*X)\X'*Y(:))';
    end
    B = nanmean(Bb0(:,1),1);
    cohenD = B./nanstd(Bb0(:,1));
    B_95_range = prctile(Bb0(:,1),[2.5,97.5],1)';
    [~,bootstat] = bootci(50000, {@mean, Bb0(:,1)}, 'type', 'bca');
    %p = (sum(abs(perm - mean(perm)) >= abs(B - mean(perm))) + 1)/(length(perm) + 1);

    Bstd0 = zeros(size(obs_val,2),design_vars-1);
    for j = 1:size(obs_val,2)
        X = (obs_map - nanmean(obs_map)) / nanstd(obs_map);
        if ~isempty(varargin)
            for i = 1:length(varargin{1})
                % deal with confounds
                X = [X, zscore(varargin{1}{i}(:,j))];
            end
        end
        Y = (obs_val(:,j) - nanmean(obs_val(:,j))) / nanstd(obs_val(:,j));

        good_roi = ~isnan(Y) & ~any(isnan(X),2);
        Y = Y(good_roi);
        X = X(good_roi,:);

        Bstd0(j,:) = ((X'*X)\X'*Y(:))';
    end
    Bstd = nanmean(Bstd0(:,1));
    cohenDStd = Bstd./nanstd(Bstd0(:,1));

    Bstd_95_range = prctile(Bstd0(:,1), [2.5, 97.5], 1)';
    [~,bootstat_std] = bootci(50000, {@mean, Bstd0(:,1)}, 'type', 'bca');
end