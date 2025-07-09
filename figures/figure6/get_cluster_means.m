function [mz, dz, fs, hs, unr] = get_cluster_means(SM, MZ, DZ, FS, HS, UNR)
% returns mean value of monozygotic (mz), dizygotic (dz) twins, full
%     siblings (fs), half siblings (hs) and unrelated individuals (unr) 
%     given a similarity matrix (SM) and indicator matrices of MZ, DZ, FS, 
%     HS and Unr pairs.


% find independent subset of unrelated individuals, excluding those used
% for mz, dz, fs, hs

triu_ind = triu(true(size(SM)),1);

mz = SM;
mz(~MZ) = nan;
mz(~triu_ind) = nan;

dz = SM;
dz(~DZ) = nan;
dz(~triu_ind) = nan;

fs = SM;
fs(~FS) = nan;
fs(~triu_ind) = nan;

hs = SM;
hs(~HS) = nan;
hs(~triu_ind) = nan;

% filter unrelated
unr = SM;
unr(~UNR | diag(true(1,length(UNR))) | isinf(unr)) = nan;
unr = unr(tril(triu(ones(size(unr)),1),1) == 1);
unr = unr(~isnan(unr));

mz = nanmean(mz(:));
dz = nanmean(dz(:));
fs = nanmean(fs(:));
hs = nanmean(hs(:));
unr = nanmean(unr(:));

end