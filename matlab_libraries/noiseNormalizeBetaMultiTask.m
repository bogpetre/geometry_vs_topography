function [u_hat,names,resMS,Sw_hat,beta_hat,shrinkage,trRR]=noiseNormalizeBetaMultiTask(Y,SPM,varargin)
% function [u_hat,Sw_hat,resMS,beta_hat]=rsa_noiseNormalizeBeta(Y,SPM,varargin)
% Estimates beta coefficiencts beta_hat and residuals from raw time series Y
% Estimates the true activity patterns u_hat by applying noise normalization to beta_hat
% INPUT:
%    Y        raw timeseries, T by P
%    SPM:     SPM structure
% OPTIONS:
%   'normmode':   'overall': Does the multivariate noise normalisation overall (default)
%                 'runwise': Does the multivariate noise normalisation by 
%                     run. This is the same as 'partwise' unless you use
%                     scan concatenation in your SPM design, in which case
%                     we need to distinguish between 'runs' (in SPMs
%                     language) and "partitions"/"sessions" (Diedrichsen/SPM 
%                     terms respectively). If concatenating runs this does
%                     noise normalization for each run separately, the same
%                     as timeseries whitening in SPM.
%                  'partwise': Does multivariate noise normalization by
%                     partition. If run concatenation is used this
%                     does multivariate noise normalization based on the
%                     concatenated residuals, same as SPMs residual error
%                     variance calculation.
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
%
% OUTPUT:
%    u_hat:       estimated true activity patterns (beta_hat after multivariate noise normalization),
%    resMS:       residual mean-square - diagonal of the Var-cov matrix of the average beta weight, 1 by P  
%    Sw_hat:      overall voxel error variance-covariance matrix (PxP), before regularisation 
%    beta_hat:    estimated raw regression coefficients, K*R by P
%    shrinkage:   applied shrinkage facto 
%    trRR:        Trace of the squared residual spatial correlation matrix
%                 This number provides an estimate for the residual spatial
%                 correlation. For variance-estimation, the effective
%                 number of voxels are effVox = numVox^2/trRR 
% Alexander Walther, Joern Diedrichsen
% joern.diedrichsen@googlemail.com
% 10/2018: Updated scaling of prehwitened-betas, to take into account the
% variance of betas (bCov).
%
% Modified for multirun multisession compatibility by Bogdan Petre on
% 05/22/2025
Opt.normmode = 'overall';  % Either runwise or overall
Opt.shrinkage = []; 
Opt.target = [];
Opt.nonlinearshrink = [];
Opt = rsa.getUserOptions(varargin,Opt);
[T,P]=size(Y);                                             %%% number of time points and voxels

%%% Discard NaN voxels
test=isnan(sum(Y));
if (any(test))
    warning(sprintf('%d of %d voxels contained NaNs -discarding',sum(test),length(test)));
    Y=Y(:,test==0);
end;

X = [];
iB = [];
noMotion = [];
numReg = 0;
for i = 1:length(SPM)
    X = blkdiag(X,SPM{i}.xX.X);
    iB = [iB, SPM{i}.xX.iB + numReg];
    noMotion = [noMotion; ~contains(SPM{i}.xX.name,'Realign')']; % filter these from rescaling procedure since they can be on wildly different scales if using quadratics
    numReg = numReg + size(SPM{i}.xX.X,2);
end
Q = size(X,2);

%%% Get partions: For each session (1:K), find the time points (T) and
%%% regressors (K+Q) that belong to the session
partT = nan(T,1);
partQ = nan(Q,1);
runT = nan(T,1);
runQ = nan(Q,1);
NSess=length(SPM{1}.Sess);                                     %%% number of partitions
NRun=length(SPM)*NSess;
for i = 1:length(SPM)
    if NSess ~= length(SPM{i}.Sess)
        error('Task %d has %d sessions but task 1 has %d sessions. All tasks must have the same number of sessions.', i, length(SPM{i}.Sess, numPart));
    end
end
for i=1:NSess
    row_ind = 0;
    col_ind = 0;
    for j = 1:length(SPM)
        % the run indexing here is important. This script will function
        % just fine  as long as all runs within sessions/SPM.mat files are
        % unique, but downstream scripts may break if the run order isn't
        % maintained within SPM.mat file. Hence we nest run_ind within
        % session so first you get all runs from SPM.mat 1, then all runs
        % from SPM.mat 2, etc. This ensures the betas that are returned are
        % the same as the betas you'd get by stacking the outputs of each
        % SPM.mat file passed into this function, in the order that thyere passed in.
        run_ind = (j-1)*NSess + i;
        partT(row_ind + SPM{j}.Sess(i).row,1)=i;
        partQ(col_ind + SPM{j}.Sess(i).col,1)=i;
        runT(row_ind + SPM{j}.Sess(i).row,1)=run_ind;
        runQ(col_ind + SPM{j}.Sess(i).col,1)=run_ind;

        %partQ(SPM.xX.iB(i),1)=i;                                %%% Add intercepts
        % infer intercepts belonging to this session from presence of a column
        % of ones
        is_one = abs(X(runT==run_ind,:)-1) < eps;
        is_zero = abs(X(runT==run_ind,:)) < eps;
        run_intercepts = find(sum(is_one) > 1 & all(is_one | is_zero)); % binary columns (e.g. includes spikes)
        run_intercepts = run_intercepts(ismember(run_intercepts,iB)); % filter for intercepts only
        partQ(run_intercepts,1) = i;
        runQ(run_intercepts,1) = run_ind;

        row_ind = row_ind + size(SPM{j}.xX.X,1);
        col_ind = col_ind + size(SPM{j}.xX.X,2);
    end
end;

%%% redo the first-level GLM using matlab functions 
KWY = [];
beta_hat = [];
res = [];
allnames = {};
row0 = 0;
Bcov = cell(1,length(SPM));
totaldf = 0;
for i = 1:length(SPM)
    rows = row0+(1:size(SPM{i}.xX.X,1));
    this_KWY=spm_filter(SPM{i}.xX.K, SPM{i}.xX.W*Y(rows,:));                        %%% filter out low-frequence trends in Y
    res=[res; spm_sp('r', SPM{i}.xX.xKXs, this_KWY)];                               %%% residuals: res  = Y - X*beta
    KWY = [KWY; this_KWY];
    beta_hat = [beta_hat; SPM{i}.xX.pKX*this_KWY];
    row0 = row0 + size(SPM{i}.xX.X,1);
    
    Bcov{i} = SPM{i}.xX.Bcov;
    totaldf = totaldf + SPM{i}.xX.trRV;
    allnames = [allnames, SPM{i}.xX.name];
end
Bcov = blkdiag(Bcov{:});

names = {};
switch (Opt.normmode)
    case 'runwise'              % do run-wise noise normalization
        u_hat   = zeros(size(beta_hat));
        shrinkage=zeros(NSess,1);
        for i=1:NRun
            idxT    = runT==i;             % Time points for this partition
            idxQ    = runQ==i;
            taskIdx = ceil(i/NSess);
            
            % in the scaling of the noise, take into account mean beta-variance 
            %[Sw_reg(:,:,i),shrinkage(i),Sw_hat(:,:,i)]=rsa.stat.covdiag(res(idxT,:),SPM.xX.trRV/(NSess*mean(diag(SPM.xX.Bcov))),'shrinkage',Opt.shrinkage);                    %%% regularize Sw_hat through optimal shrinkage
            % rescale the residuals inestead of the dof to avoid affecting
            % shrinkage calculations
            scaleFactor = sqrt(mean(diag(Bcov(noMotion,noMotion))));
            assert(imag(scaleFactor) == 0, 'Imaginary Bcov matrix found. Please check for badly scaled design matrix columns.')
            df = SPM{taskIdx}.xX.trRV/NSess;
            if df > 50 && size(res,2) > 50 && ~isempty(Opt.nonlinearshrink) && Opt.nonlinearshrink == 1
                X = res(idxT,:)*sqrt(mean(diag(Bcov(noMotion,noMotion))));
                Sw_hat(:,:,i) = 1/df*(X'*X);
                Sw_reg(:,:,i) = QIS(X,round(df));
                shrinkage(i) = nan;
            else
                [Sw_reg(:,:,i),shrinkage(i),Sw_hat(:,:,i)]=rsa.stat.covdiag(res(idxT,:)*sqrt(mean(diag(Bcov(noMotion,noMotion)))),df,...
                    'shrinkage',Opt.shrinkage,'target',Opt.target);                    %%% regularize Sw_hat through optimal shrinkage
            end
            % Calculating sq over the eigenvalues is numerically more
            % stable than sq = Sw_reg^-1/2 
            [V,L]=eig(Sw_reg(:,:,i));   
            l=diag(L);
            sq(:,:,i) = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
            % Postmultiply by the inverse square root of the estimated matrix 
            u_hat(idxQ,:)=beta_hat(idxQ,:)*sq(:,:,i);
            names(idxQ) = allnames(idxQ);
        end;
        shrinkage=mean(shrinkage);
        Sw_hat = mean(Sw_hat,3); 
        Sw_reg = mean(Sw_reg,3); 
    case 'partwise'              % do run-wise noise normalization
        u_hat   = zeros(size(beta_hat));
        shrinkage=zeros(NSess,1);
        for i=1:NSess
            idxT    = partT==i;             % Time points for this partition 
            idxQ    = partQ==i;             % Regressors for this partition 

            % in the scaling of the noise, take into account mean beta-variance 
            %[Sw_reg(:,:,i),shrinkage(i),Sw_hat(:,:,i)]=rsa.stat.covdiag(res(idxT,:),SPM.xX.trRV/(NSess*mean(diag(SPM.xX.Bcov))),'shrinkage',Opt.shrinkage);                    %%% regularize Sw_hat through optimal shrinkage
            % rescale the residuals inestead of the dof to avoid affecting
            % shrinkage calculations
            scaleFactor = sqrt(mean(diag(Bcov(noMotion,noMotion))));
            assert(imag(scaleFactor) == 0, 'Imaginary Bcov matrix found. Please check for badly scaled design matrix columns.')
            df = totaldf/NSess;
            if df > 50 && size(res,2) > 50 && ~isempty(Opt.nonlinearshrink) && Opt.nonlinearshrink == 1
                X = res(idxT,:)*sqrt(mean(diag(Bcov(noMotion,noMotion))));
                Sw_hat(:,:,i) = 1/df*(X'*X);
                Sw_reg(:,:,i) = QIS(X,round(df));
                shrinkage(i) = nan;
            else
                [Sw_reg(:,:,i),shrinkage(i),Sw_hat(:,:,i)]=rsa.stat.covdiag(res(idxT,:)*sqrt(mean(diag(Bcov(noMotion,noMotion)))),df,...
                    'shrinkage',Opt.shrinkage,'target',Opt.target);                    %%% regularize Sw_hat through optimal shrinkage
            end
            % Calculating sq over the eigenvalues is numerically more
            % stable than sq = Sw_reg^-1/2 
            [V,L]=eig(Sw_reg(:,:,i));   
            l=diag(L);
            sq(:,:,i) = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
            % Postmultiply by the inverse square root of the estimated matrix 
            u_hat(idxQ,:)=beta_hat(idxQ,:)*sq(:,:,i);
            names(idxQ) = allnames(idxQ);
        end;
        shrinkage=mean(shrinkage);
        Sw_hat = mean(Sw_hat,3); 
        Sw_reg = mean(Sw_reg,3); 
    case 'overall'              %%% do overall noise normalization
        % in the scaling of the noise, take into account mean beta-variance 
        %[Sw_reg,shrinkage,Sw_hat]=rsa.stat.covdiag(res,SPM.xX.trRV/mean(diag(SPM.xX.Bcov)),'shrinkage',Opt.shrinkage);   %%% regularize Sw_hat through optimal shrinkage
        % rescale the residuals inestead of the dof to avoid affecting
        % shrinkage calculations
        df = totaldf;
        if df > 50 && size(res,2) > 50 && ~isempty(Opt.nonlinearshrink) && Opt.nonlinearshrink == 1
            X = res(idxT,:)*sqrt(mean(diag(Bcov(noMotion,noMotion))));
            Sw_hat = 1/df*(X'*X);
            Sw_reg = QIS(X,round(df));
            shrinkage = nan;
        else
            [Sw_reg,shrinkage,Sw_hat]=rsa.stat.covdiag(res*sqrt(mean(diag(Bcov(noMotion, noMotion)))),df,...
                'shrinkage',Opt.shrinkage,'target',Opt.target);   %%% regularize Sw_hat through optimal shrinkage            
        end
        % Postmultiply by the inverse square root of the estimated matrix 
        [V,L]=eig(Sw_reg);  % in the scaling of the noise, take into account mean beta-variance 
        l=diag(L);
        sq = V*bsxfun(@rdivide,V',sqrt(l)); % Slightly faster than sq = V*diag(1./sqrt(l))*V';
        u_hat=beta_hat*sq;
        names = all_names;
end;

names = cellfun(@(x1)(regexprep(x1,'Sn\([0-9]+\)\ (.*).*','$1')),names,'UniformOutput',false); % drop session indicator
names = cellfun(@(x1)(regexprep(x1,'(.*)\*bf.*','$1')),names,'UniformOutput',false); % drop any trailing bfs

% Return the diagonal of Sw_hat - also weighted by the mean variance of beta-hat 
if (nargout>2)
    resMS=sum(res.^2)./totaldf*mean(diag(Bcov));
end; 

% return the trace of the residual covariance matrix
% If prewhiting was perfect, then the trace(sig*sig) = P and the effective number of voxels
% would be P. However, because we need to regularise our estimate, the residual 
% spatial covariance matrix is somwhat correlated. For variance estimation on the distances, 
% the effective voxel number would be numVox^2/trRR
if (nargout>6)
    % Caluclate the predicted redidual covariance 
    sq     = mean(sq,3); 
    V_hat  = sq'*Sw_hat*sq; 
    % Calculate the correlation matrix from this 
    Vsigma = sqrt(diag(V_hat));     
    R_hat = bsxfun(@rdivide,V_hat,Vsigma); 
    R_hat = bsxfun(@rdivide,R_hat,Vsigma'); % R = C ./ sigma*sigma';
    % get the trace 
    trRR  =  sum(sum(R_hat.*R_hat)); % traceRR 
end; 
