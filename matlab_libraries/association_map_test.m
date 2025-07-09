function [Binteraction, varargout] = association_map_test(topo,rdm,map,varargin)
% rdm - n x p matrix of observed geometric similarities
% topo - n x p matrix of topographies
% map - 1 x p matrix of map parcel values
%
% Computes association between rdm and topo across n observations for each
% parcel p, then tests whether or not there's an interaction between this
% and the map parcel values. The latter is standardized so that estimates
% can be compared across maps with different scales

assert(all(size(rdm) == size(topo)));

[n,p] = size(rdm);

assert(size(map,1) == p);
assert(size(map,2) == 1);
map = repmat(map', n, 1);

rois = repmat((1:p), n, 1);
sid = repmat((1:n)', 1, p);

% center rdm within roi for interaction interpretability
% equivalent to adding roi dummy codes to design, but faster
%rdm = rdm - nanmean(rdm);
%topo = topo - nanmean(topo);

% censor invalid entries and vectorize
isgood = find(~isnan(rdm) & ~isnan(topo) & ~isnan(map));
rdm = rdm(isgood);
topo = topo(isgood);
rois = rois(isgood);
sid = sid(isgood);
map = map(isgood);


% center for interpretable interaction coefficients
% superfluous if you've already subtracted out subject means. If all
% subjects are mean zero, all data is mean zero too
rdm = rdm - mean(rdm);
topo = topo - mean(topo);

% Map does not vary within ROIs, so we dont need to model the main effect 
% of map. Design would be rank deficient.
X = [rdm, rdm.*map, dummyvar(categorical(rois)), helmertCoding(sid)];
Y = topo;

if ~isempty(varargin)
    for i = 1:length(varargin{1})
        % deal with confounds
        normed_arg = zscore(varargin{1}{i},[],2);
        mean_arg = repmat(mean(varargin{1}{i}),size(varargin{1}{i},1),1);
        X = [X, normed_arg(isgood).*helmertCoding(sid)];
    end
end

if nargout > 1
    m = fitlm(X,Y,'Intercept',false);
    CI = m.coefCI;
    B = m.Coefficients.Estimate;
    varargout{1} = CI(2,:);
    varargout{2} = m.Coefficients.pValue(2);
else
    B = (X'*X)\X'*Y;
end
Binteraction = B(2);

end