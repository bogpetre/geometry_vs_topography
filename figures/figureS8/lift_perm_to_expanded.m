function rand_ordering_expanded = lift_perm_to_expanded(rand_ordering, replicate_ids, n_cond)
% rand_ordering: 1xN permutation of conditions (1-based)
% replicate_ids: n_exp x 1 vector of condition ids (1-based) used to build the expansion
% n_cond: number of original conditions (optional, inferred if omitted)

    if nargin < 3
        n_cond = max(replicate_ids);
    end

    rand_ordering = rand_ordering(:)';        % row
    replicate_ids = replicate_ids(:);         % col

    if numel(rand_ordering) ~= n_cond
        error('rand_ordering length (%d) must equal n_cond (%d).', numel(rand_ordering), n_cond);
    end

    % Validate it's a permutation of 1..n_cond
    if ~isequal(sort(rand_ordering), 1:n_cond)
        error('rand_ordering must be a permutation of 1..n_cond.');
    end

    % Build list of expanded positions per condition
    pos = cell(n_cond, 1);
    for k = 1:numel(replicate_ids)
        c = replicate_ids(k);
        if c < 1 || c > n_cond
            error('replicate_ids contains out-of-range condition id %d.', c);
        end
        pos{c}(end+1,1) = k; %#ok<AGROW>
    end

    % Concatenate expanded indices in the permuted condition order
    rand_ordering_expanded = vertcat(pos{rand_ordering});
end
