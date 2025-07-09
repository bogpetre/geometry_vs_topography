function [mz, dz, fs, hs, unr] = get_independent_obs(SM, MZ, DZ, FS, HS, UNR)
% returns example observations of pairwise similarity measures that are 
%   independent (i.e. each pair's 2 members are non-intersecting with the 
%   observations of remaining pairs) for each type of family cluster (mz, 
%   dz, fs, hs, unr; monozygotic and dizygotic twins, full siblings, half 
%   siblings and unrelated individuals, respectively) based on a similarity 
%   matrix SM and indicator matrices MZ, DZ, FS, HS, UNR which indicate 
%   whether an entry in SM corresponds to a relationship between 
%   monozygotic twins, dizygotic twins, full siblings, half siblings or 
%   unrelated individuals


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

% filter fs for repeats
[fs_1, fs_2] = find(~isnan(fs));

fs_ind = [fs_1, fs_2];
for i = flip(1:length(fs_ind))
    if ismember(fs_1(i,1), fs_1(1:i-1,1))
        fs_ind(i,:) = [];
    end
end

% filter hs for repeats
[hs_1, hs_2] = find(~isnan(hs));

hs_ind = [hs_1, hs_2];
for i = flip(1:length(hs_ind))
    if ismember(hs_1(i,1), hs_1(1:i-1,1))
        hs_ind(i,:) = [];
    end
end

% apply filters
mz = mz(~isnan(mz));
dz = dz(~isnan(dz));

geom_r_fs_ = nan(size(fs));
for i = 1:size(fs_ind)
    geom_r_fs_(fs_ind(i,1), fs_ind(i,2)) = fs(fs_ind(i,1), fs_ind(i,2));
end
fs = geom_r_fs_(~isnan(geom_r_fs_));

geom_r_hs_ = nan(size(hs));
for i = 1:size(hs_ind)
    geom_r_hs_(hs_ind(i,1), hs_ind(i,2)) = hs(hs_ind(i,1), hs_ind(i,2));
end
hs = geom_r_hs_(~isnan(geom_r_hs_));

% filter unrelated
unr = SM;
unr(~UNR | diag(true(1,length(UNR))) | isinf(unr)) = nan;
unr = unr(tril(triu(ones(size(unr)),1),1) == 1);
unr = unr(~isnan(unr));

end