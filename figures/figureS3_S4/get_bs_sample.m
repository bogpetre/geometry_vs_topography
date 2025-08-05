function [topo, geom, varargout] = get_bs_sample(n_dyads, topo_mat, geom_mat, varargin)
    [n,~,r] = size(topo_mat);

    assert(all(size(topo_mat) == size(geom_mat)));

    [topo, geom] = deal(nan(n_dyads,r));

    if nargin > 3
        varargout = cell(1,length(varargin));
        for i = 1:length(varargout)
            varargout{i} = nan(n_dyads, r);
        end
    end

    ind = randi(n, n_dyads);
    pair_ind = nan(n_dyads,2);
    for i = 1:length(ind)
        this_ind = ind(i);
        % pick a second participant who is different from the first
        while this_ind == ind(i)
            this_ind = randi(n,1);
        end
        pair_ind(i,:) = [ind(i), this_ind];

        topo(i,:) = squeeze(topo_mat(ind(i), this_ind, :));
        geom(i,:) = squeeze(geom_mat(ind(i), this_ind, :));

        if nargin > 3
            for j = 1:length(varargin)
                conf1 = varargin{j}(ind(i),:);
                conf2 = varargin{j}(this_ind,:);

                varargout{j}(i,:) = mean([conf1; conf2]);
            end
        end
    end
end