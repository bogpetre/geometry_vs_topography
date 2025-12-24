function [mz, dz, fs, hs, unr] = get_cluster_corr(SM, MZ, DZ, FS, HS, UNR)
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

% this estimates total variance as the sum of all variance fractions,
% within and between families, which is not exactly correct (Falconer 1990 
% pp 170), but close.
mz_unr = SM;
mz_unr(~any(MZ) | ~any(MZ,2)) = nan;
mz_unr(isinf(SM)) = nan;
mz_unr(~triu_ind) = nan;


dz = SM;
dz(~DZ) = nan;
dz(~triu_ind) = nan;

dz_unr = SM;
dz_unr(~any(DZ) | ~any(DZ,2)) = nan;
dz_unr(isinf(SM)) = nan;
dz_unr(~triu_ind) = nan;


fs = SM;
fs(~FS) = nan;
fs(~triu_ind) = nan;

fs_unr = SM;
fs_unr(~any(FS) | ~any(FS,2)) = nan;
fs_unr(isinf(SM)) = nan;
fs_unr(~triu_ind) = nan;


hs = SM;
hs(~HS) = nan;
hs(~triu_ind) = nan;

hs_unr = SM;
hs_unr(~any(HS) | ~any(HS,2)) = nan;
hs_unr(isinf(SM)) = nan;
hs_unr(~triu_ind) = nan;


% filter unrelated
unr = SM;
unr(~UNR | diag(true(1,length(UNR))) | isinf(unr)) = nan;
unr = unr(tril(triu(ones(size(unr)),1),1) == 1);
unr = unr(~isnan(unr));

mz = sqrt(1 - nanmean(mz(:))/(nanmean(mz_unr(:))));
dz = sqrt(1 - nanmean(dz(:))/(nanmean(dz_unr(:))));
fs = sqrt(1 - nanmean(fs(:))/(nanmean(fs_unr(:))));
hs = sqrt(1 - nanmean(hs(:))/(nanmean(hs_unr(:))));
unr = sqrt(1 - nanmean(unr(:))/nanmean(unr(:)));

end