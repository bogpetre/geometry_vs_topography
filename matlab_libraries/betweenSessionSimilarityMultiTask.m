function [sim, names] = betweenSessionSimilarityMultiTask(Y,SPM,conditionVec,fun,varargin)
% function sim=betweenSessionSimilarity(Y,SPM,fun,varargin)
% Estimates beta coefficiencts and residuals from raw time series Y for
% each k-fold partitioning of the data and computes similarity between 
% training/test folds of the partitioning. All sessions must be have the
% same design columns.
%
% Estimates the true activity patterns by applying noise normalization to
% betas independently across training/test folds.
%
% INPUT:
%    Y        raw timeseries, T by P
%    SPM:     Cell array of SPM structures
%    conditionVec: Vector that indicates conditions for each column of design matrix 
%                  set to 0 for nuisance regressors
%    fun:     function handle specifying similarity function to use. Must
%               operate on 2 vectors.
% OPTIONS:
%   'shrinkage':  Shrinkage coefficient. 
%                 0: No regularisation 
%                 1: Using only the diagonal - i.e. univariate noise normalisation 
%                 By default the shrinkage coeffcient is determined using
%                 the Ledoit-Wolf method. 
%   'target':     Shrinkage target (prior)
%                 'diagonal': equivalent to a scaled t-stat map
%                 'scaledidentity': identity scaled to mean variance
%   'nonlinearshrink': 
%                 0 or 1. If specified, nonlinear shrinkage is used.
%                 Requires covShrinkage package on matlab path, and in
%                 particular the QIS function: 
%                 https://www.mathworks.com/matlabcentral/fileexchange/106240-covshrinkage
%                 If specified, shrinkage and target have no effect except
%                 for regions with n <= 50 or p <= 50, for which nonlinear 
%                 shrinkage doesn't work well (Ledoit & Wolf 2021 Journal 
%                 of Financial Econometrics) and we fall back to linear 
%                 shrinkage.
%   'normmode':   'runwise': Does the multivariate noise normalisation by
%                     run. This is how SPM does temporal whitening. Treats
%                     each session of each SPM.mat obj as a separate run.
%                 'partwise': Does multivariate normalization jointly
%                     across all runs in a partition. This is how SPM would
%                     do univariate residual variance estimation if
%                     sessions were concatenated across SPM.mat objects.
%                 'overall': Does the multivariate noise normalisation overall
%
% OUTPUT:
%   sim           matrix of similarity values, each column is a session,
%                 each row is a task
% Code adapted from noiseNormalizeBeta by Bogdan Petre. noiseNormalizeBeta
% and distanceLDCraw provided by the rsatoolbox_matlab and written by
% Alexander Walther, Joern Diedrichsen
% joern.diedrichsen@googlemail.com
Opt.shrinkage = []; 
Opt.target = [];
Opt.nonlinearshrink = [];
Opt.normmode = 'runwise';
Opt = rsa.getUserOptions(varargin,Opt);
if strcmp(Opt.normmethod, 'univariate')
    if strcmp(Opt.target,'diagonal')
        Opt.shrinkage = 1;
    else
        error('Shrinkage of univariate covariance to isotropic covarince is not yet supported');
    end
end
[T,numvox]=size(Y);                                             %%% number of time points and voxels

%%% Discard NaN voxels
test=isnan(sum(Y));
if (any(test))
    warning(sprintf('%d of %d voxels contained NaNs -discarding',sum(test),length(test)));
    Y=Y(:,test==0);
end;

X = [];
for i = 1:length(SPM), X = blkdiag(X,SPM{i}.xX.xKXs.X); end
numReg = size(X,2);

% Check condition vector
numCond = max(conditionVec);
if (length(conditionVec)<numReg)
    conditionVec=[conditionVec; zeros(numReg-length(conditionVec),1)];
end;
Z = rsa.util.indicatorMatrix('identity_p',conditionVec);
nonInterest = all(Z==0,2);   % Regressors not in the conditions
numNonInterest = sum(nonInterest);
Z(nonInterest,end+1:end+sum(numNonInterest))=eye(numNonInterest);

%%% Get partions: For each run (1:K), find the time points (T) and regressors (K+Q) that belong to the run
partT = nan(T,1);
partN = nan(numReg,1);
partT = nan(T,1);
partN = nan(numReg,1);
numPart=length(SPM{1}.Sess);                                     %%% number of runs
for i = 1:length(SPM)
    if numPart ~= length(SPM{i}.Sess)
        error('Task %d has %d sessions but task 1 has %d sessions. All tasks must have the same number of sessions.', i, length(SPM{i}.Sess, numPart));
    end
end                                     %%% number of runs
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

%%% redo the first-level GLM using matlab functions 
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

%noMotion = ~contains(names,'Realign')'; % filter these from rescaling procedure since they can be on wildly different scales if using quadratics

switch (Opt.normmode)
    case 'runwise'
        for i=1:numPart
            for j = 1:length(SPM)
                idxT = partT==i & taskT == j;
                df = SPM{j}.xX.trRV/numPart;
                scaling = sqrt(mean(diag(Bcov(conditionVec>0,conditionVec>0)))); % note scaling doesn't matter for many measures
                if ~isempty(Opt.nonlinearshrink) && Opt.nonlinearshrink == 1 && df > 50 && size(res,2) > 50
                    Sw_hat(:,:,i) = QIS(res(idxT,:)*scaling,round(df));
                else                        
                    [Sw_hat(:,:,i),shrink(i)]=rsa.stat.covdiag(res(idxT,:)*scaling,df, ...
                            'shrinkage', Opt.shrinkage, 'target', Opt.target);%%% regularize Sw_hat through optimal shrinkage
                end
                [V,L]=eig(Sw_hat(:,:,i));       % This is overall faster and numerical more stable than Sw_hat.^-1/2
                l=diag(L);
                sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
                KWY(idxT,:)=KWY(idxT,:)*sq;
            end
        end
    case 'partwise'
        for i=1:numPart
            idxT = partT==i;
        
            df = 0;
            for j = 1:length(SPM)
                df = df + SPM{j}.xX.trRV/numPart;
            end

            scaling = sqrt(mean(diag(Bcov(conditionVec>0,conditionVec>0)))); % note scaling doesn't matter for many measures
            if ~isempty(Opt.nonlinearshrink) && Opt.nonlinearshrink == 1 && df > 50 && size(res,2) > 50
                Sw_hat(:,:,i) = QIS(res(idxT,:)*scaling,round(df));
            else                        
                [Sw_hat(:,:,i),shrink(i)]=rsa.stat.covdiag(res(idxT,:)*scaling,df, ...
                        'shrinkage', Opt.shrinkage, 'target', Opt.target);%%% regularize Sw_hat through optimal shrinkage
            end
            [V,L]=eig(Sw_hat(:,:,i));       % This is overall faster and numerical more stable than Sw_hat.^-1/2
            l=diag(L);
            sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
            KWY(idxT,:)=KWY(idxT,:)*sq;
        end
    case 'overall'
        df = 0;
        for j = 1:length(SPM)
            df = df + SPM{j}.xX.trRV;
        end

        scaling = sqrt(mean(diag(Bcov(conditionVec>0,conditionVec>0)))); % note scaling doesn't matter for many measures
        if ~isempty(Opt.nonlinearshrink) && Opt.nonlinearshrink == 1 && df > 50 && size(res,2) > 50
            Sw_hat = QIS(res*scaling,round(df));
        else                        
            [Sw_hat,shrink]=rsa.stat.covdiag(res*scaling,df, ...
                    'shrinkage', Opt.shrinkage, 'target', Opt.target);%%% regularize Sw_hat through optimal shrinkage
        end
        [V,L]=eig(Sw_hat);       % This is overall faster and numerical more stable than Sw_hat.^-1/2
        l=diag(L);
        sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
        KWY=KWY*sq;
end


% Estimate condition means within each
for i=1:numPart
    % Get the betas from the test run 
    indxN = partN==i;
    indxT = partT==i;
    Za = Z(indxN,:);
    Za = Za(:,any(Za,1));
    Xa = X(indxT,indxN);
    Ma  = Xa*Za;
    A(:,:,i)     = (Ma'*Ma)\Ma'*KWY(indxT,:);
    
    % Get the betas based on the other runs 
    indxN = partN~=i;
    indxT = partT~=i;
    Zb    = Z(indxN,:);
    Zb    = Zb(:,any(Zb,1));
    Xb    = X(indxT,indxN);
    Mb    = Xb*Zb;
    B     = (Mb'*Mb)\Mb'*KWY(indxT,:);
    
    % Pick condition of interest
    interest = find(~nonInterest(partN==i));
    
    % Caluclate similarity
    for j = 1:length(interest)
        sim(j,i) = fun(A(j,:,i), B(j,:));
    end
end;

sim = mean(sim,2);

names = names(partN==1)';
names = names(interest)';
end