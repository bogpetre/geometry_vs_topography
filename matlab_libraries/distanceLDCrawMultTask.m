function [d, Sig, names] = distanceLDCrawMultTask(Y,SPM,conditionVec,varargin)
% function d=rsa.spm.distanceLDCraw(Y,SPM,conditionVec,varargin);
% First, gets the regression coefficent from the SPM, and prewhitens them.
% Prewhiten can be controlled using different methods (run-wise or overall).
% The same code is used in rsa.spm.noiseNormalizeBeta.
% Secondly, it calcualtes LDC stances with a leave-one out crossvalidation,
% It uses optimal combination of the beta-coeeficients in the part that
% averages across partitions. By default, the different partitions are
% assumed to be the different imaging runs.
% Note that in calculating the distances, the structure of the first-level 
% design matrix is optimally taken into account. The command: 
%  RDM  = rsa.spm.distanceLDCraw(Y,SPM,condition); 
% is therefore equivalent to:  
%  beta = rsa.spm.noiseNormalizeBeta(Y,SPM); 
%  RDM  = rsa.distanceLDC(beta,partition,condition,SPM.xX.xKXs.X); 
% INPUT:
%    Y             raw timeseries, T by P
%    SPM:          SPM structure
%    conditionVec: Vector that indicates conditions for each column of design matrix 
%                  set to 0 for nuisance regressors
% OUTPUT:
%    d:            numCond*(numCond-1)/2 distances between experimental
%                  conditions
% OPTIONs:
%   'normmode':    'runwise': Does the multivariate noise normalisation by run
%                  'partwise': Does multivariate normalization jointly
%                      across all runs in a partition.
%                  'overall': Does the multivariate noise normalisation overall
%   'normmethod':  'multivariate': The is the default using ledoit-wolf reg.
%                  'univariate': Performing univariate noise normalisation (t-values)
%                  'none':    No noise normalisation
% (c) 2015 Joern Diedrichsen, Alex Walther
%
%
% Modified by Bogdan Petre
% edits made to take a cell array of SPM objects as input and compute RDMs
% jointly across all, to whitening matrix by rescaling residuals, not df (the latter severely 
% affects regularization of whitening matrix)
Opt.normmode = 'runwise';  % Either runwise or overall
Opt.normmethod = 'multivariate';  % Either runwise or overall
Opt = rsa.getUserOptions(varargin,Opt,{'normmode','normmethod'});

[T,numVox]=size(Y);                                             %%% number of time points and voxels

%%% Discard NaN voxels
test=isnan(sum(Y));
if (any(test))
    warning(sprintf('%d of %d voxels contained NaNs -discarding',sum(test),length(test)));
    Y=Y(:,test==0);
end;

% This script assumes that each "task" contains a single run per session,
% and then treats each session within a task as a 'run'. Lets verify. This
% script should fail if sessions contain concatenated runs.
if ismember(Opt.normmode, {'runwise'})
    for i = 1:length(SPM)
        NSess = length(SPM{i}.Sess);
        NRun = length(SPM{i}.xX.K);
        assert(length(SPM{i}.xX.K) == length(SPM{i}.Sess), ...
            sprintf('Expected %d runs for %d sessions, but found %d runs in task %d instead.',NSess,NSess,NRun,i));
    end
end

X = [];
for i = 1:length(SPM), X = blkdiag(X,SPM{i}.xX.xKXs.X); end

numReg = size(X,2);

% Check condition vector
numCond = max(conditionVec);
if (length(conditionVec)<numReg)
    conditionVec=[conditionVec;zeros(numReg-length(conditionVec),1)];
end;
Z = rsa.util.indicatorMatrix('identity_p',conditionVec);
nonInterest = all(Z==0,2);   % Regressors not in the conditions
numNonInterest = sum(nonInterest);
Z(nonInterest,end+1:end+sum(numNonInterest))=eye(numNonInterest);
C = rsa.util.indicatorMatrix('allpairs',[1:numCond]);

%%% Get partions: For each run (1:K), find the time points (T) and regressors (K+Q) that belong to the run
taskT = nan(T,1);
taskN = nan(numReg,1);
partT = nan(T,1);
partN = nan(numReg,1);
numPart=length(SPM{1}.Sess);                                     %%% number of runs
for i = 1:length(SPM)
    if numPart ~= length(SPM{i}.Sess)
        error('Task %d has %d sessions but task 1 has %d sessions. All tasks must have the same number of sessions.', i, length(SPM{i}.Sess, numPart));
    end
end
for i=1:numPart
    row_ind = 0;
    col_ind = 0;
    for j = 1:length(SPM)
        partT(row_ind + SPM{j}.Sess(i).row,1)=i;
        partN(col_ind + SPM{j}.Sess(i).col,1)=i;
        taskT(row_ind + SPM{j}.Sess(i).row,1)=j;
        taskN(col_ind + SPM{j}.Sess(i).col,1)=j;

        % modified for compatability with concatenated multisession data
        intercept_ind = any((partT(row_ind + (1:size(SPM{j}.xX.X,1))) == i).*SPM{j}.xX.X(:,SPM{j}.xX.iB));
        partN(col_ind + SPM{j}.xX.iB(intercept_ind),1)=i;                                %%% Add intercepts
        taskN(col_ind + SPM{j}.xX.iB(intercept_ind),1)=j;

        row_ind = row_ind + size(SPM{j}.xX.X,1);
        col_ind = col_ind + size(SPM{j}.xX.X,2);
    end
end;

KWY = [];
res = [];
names = {};
row0 = 0;
Bcov = cell(1,length(SPM));
for i = 1:length(SPM)
    rows = row0+(1:size(SPM{i}.xX.X,1));
    this_KWY=spm_filter(SPM{i}.xX.K, SPM{i}.xX.W*Y(rows,:));                        %%% filter out low-frequence trends in Y
    res=[res; spm_sp('r', SPM{i}.xX.xKXs, this_KWY)];                               %%% residuals: res  = Y - X*beta
    KWY = [KWY; this_KWY];
    row0 = row0 + size(SPM{i}.xX.X,1);
    
    Bcov{i} = SPM{i}.xX.Bcov;

    names = [names, SPM{i}.xX.name];
end
Bcov = blkdiag(Bcov{:});

%%% do run-wise noise normalization
switch (Opt.normmethod)
    case 'none'
        % Do nothing 
    case 'multivariate'
        switch (Opt.normmode)
            case 'runwise'
                for i=1:numPart
                    % BP: we compute different spatial covariances for each
                    % task (rather than "run") because different scans may have 
                    % different spatial statistics due to when they were.
                    % acquired (e.g. different days). This could be improved by 
                    % estimating the correlation structure jointly and only 
                    % adjusting scale independently, since the latter is primarily
                    % what we expect to differ day-by-day (e.g. if a participant
                    % is caffinated vs. not vasoreactivity will change, but the
                    % the spatial distribution of vasculature stays the same).
                    for j = 1:length(SPM)
                        beta_ind = (i-1)*length(SPM)+j;
                        idxT = partT==i & taskT == j;
                        idxN = partN==i & taskN == j;
                        numFilt = size(SPM{j}.xX.K(i).X0,2);
                        %[Sw_hat(:,:, beta_ind),shrink(i)]=rsa.stat.covdiag(res(idxT,:),SPM{j}.xX.trRV/(numPart*mean(diag(SPM{j}.xX.Bcov))));   %%% regularize Sw_hat through optimal shrinkage
                        % we scale residuals here so that variance across
                        % betas of interest is ~1. Note: we only care about
                        % variance of betas we ultimately use in RSA.
                        % scaleFactor = sqrt(mean(diag(Bcov(conditionVec(idxN)>0,conditionVec(idxN)>0)))); % run specific factor
                        scaleFactor = sqrt(mean(diag(Bcov(conditionVec>0,conditionVec>0))));
                        [Sw_hat(:,:, beta_ind),shrink(i)]=rsa.stat.covdiag(res(idxT,:)*scaleFactor,SPM{j}.xX.trRV/numPart);   %%% regularize Sw_hat through optimal shrinkage
                        [V,L]=eig(Sw_hat(:,:,beta_ind));       % This is overall faster and numerical more stable than Sw_hat.^-1/2
                        l=diag(L);
                        sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
                        KWY(idxT,:)=KWY(idxT,:)*sq;
                    end
                end;
            case 'partwise'
                % This is a novel invention for multi-scan protocols. Here
                % we keep partitions independent, but we estimate variance
                % jointly across multiple runs within a partition. This
                % assumes homoskedasticity.
                for i=1:numPart
                    dof = 0;
                    for j = 1:length(SPM)
                        dof = dof + SPM{j}.xX.trRV/numPart;
                    end

                    idxT = partT==i;
                    scaleFactor = sqrt(mean(diag(Bcov(conditionVec>0,conditionVec>0))));
                    [Sw_hat(:,:, i),shrink(i)]=rsa.stat.covdiag(res(idxT,:)*scaleFactor, dof);   %%% regularize Sw_hat through optimal shrinkage
                    [V,L]=eig(Sw_hat(:,:,i));       % This is overall faster and numerical more stable than Sw_hat.^-1/2
                    l=diag(L);
                    sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
                    KWY(idxT,:)=KWY(idxT,:)*sq;
                end
            case 'overall'
                dof = 0;
                for j = 1:length(SPM)
                    dof = dof + SPM{j}.xX.trRV;
                end

                %numFilt = size(SPM{j}.xX.K(1).X0,2);
                %[Sw_hat(:,:, beta_ind),shrink(i)]=rsa.stat.covdiag(res(idxT,:),SPM{j}.xX.trRV/(numPart*mean(diag(SPM{j}.xX.Bcov))));   %%% regularize Sw_hat through optimal shrinkage
                scaleFactor = sqrt(mean(diag(Bcov(conditionVec>0,conditionVec>0))));
                [Sw_hat, shrink]=rsa.stat.covdiag(res*scaleFactor, dof);   %%% regularize Sw_hat through optimal shrinkage
                [V,L]=eig(Sw_hat);       % This is overall faster and numerical more stable than Sw_hat.^-1/2
                l=diag(L);
                sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
                KWY=KWY*sq;
        end;
    case 'univariate'
        switch (Opt.normmode)
            case 'runwise'
                for i=1:numPart
                    % BP: we compute different spatial covariances for each
                    % task (rather than run) because different scans may have 
                    % different spatial statistics due to when they were 
                    % acquired (e.g. different days).
                    for j = 1:length(SPM)
                        idxT = partT==i & taskT == j;

                        dof = SPM{j}.xX.trRV/numPart;
                        sigma = sum(res(idxT,:).^2)/dof;

                        scaleFactor = mean(diag(Bcov(conditionVec>0,conditionVec>0)));
                        sq = 1./sqrt(sigma*scaleFactor);
                        KWY(idxT,:)=KWY(idxT,:).*sq;
                    end
                end;
            case 'partwise'
                for i=1:numPart
                    dof = 0;
                    for j = 1:length(SPM)
                        dof = dof + SPM{j}.xX.trRV/numPart;
                    end
                    
                    idxT = partT==i;

                    sigma = sum(res(idxT,:).^2)/dof;

                    scaleFactor = mean(diag(Bcov(conditionVec>0,conditionVec>0)));
                    sq = 1./sqrt(sigma*scaleFactor);
                    KWY(idxT,:)=KWY(idxT,:).*sq;
                end;
            case 'overall'
                dof = 0;
                for j = 1:length(SPM)
                    dof = dof + SPM{j}.xX.trRV;
                end

                sigma = sum(res.^2)/dof;

                sq = 1./sqrt(sigma);
                KWY=KWY.*sq;
        end;
    otherwise
        error('normmethod needs to be ''multivariate'', ''univariate'', or ''none''');
end;


% Estimate condition means within each
A = zeros(length(nonInterest(partN==partN(1))),numVox,numPart);
for i=1:numPart
    % Get the betas from the test run 
    indxN = partN==i;
    indxT = partT==i;
    Za = Z(indxN,:);
    Za = Za(:,any(Za,1));
    Xa = X(indxT,indxN);
    Ma  = Xa*Za;
    A(:,:,i)     = (Ma'*Ma)\(Ma'*KWY(indxT,:));
    
    % Get the betas based on the other runs 
    indxN = partN~=i;
    indxT = partT~=i;
    Zb    = Z(indxN,:);
    Zb    = Zb(:,any(Zb,1));
    Xb    = X(indxT,indxN);
    Mb    = Xb*Zb;
    B     = (Mb'*Mb)\Mb'*KWY(indxT,:);
    % line below handles 24 motion vectors (which are often highly colinear) better
    %B     = pinv(Mb)*KWY(indxT,:);
    
    % Pick condition of interest
    % interest = ~nonInterest(partN==i);
    interest = ~nonInterest(partN==i);

    % BP: multiscan concat interleaves confounds, but Z matrix appends 
    % confounds to the end, so let's resort
    interest = sort(interest,'descend');
    
    % Caluclate distances 
    %d(i,:)= sum((C*A(1:numCond,:,i)).*(C*B(1:numCond,:)),2)'/numVox;      % Note that this is normalised to the number of voxels
    d(i,:)= sum((C*A(interest,:,i)).*(C*B(interest,:)),2)'/numVox;      % Note that this is normalised to the number of voxels
end;
d = sum(d)./numPart;

names = names(partN==1)';
names = names(~nonInterest(partN==1));

% If requested, also calculate the estimated variance-covariance 
% matrix from the residual across folds. 
A = A(interest,:,:);
if (nargout>1) 
    R=bsxfun(@minus,A,sum(A,3)/numPart);
    for i=1:numPart
        Sig(:,:,i)=R(:,:,i)*R(:,:,i)'/numVox;
    end;
    Sig=sum(Sig,3)/(numPart-1);
end; 
end
