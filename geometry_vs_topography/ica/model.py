import sys
import numpy as np
import scipy as sp
from warnings import warn

def dual_regression(timeseries0, groupMap0, 
                    weights=None, zscorets=False, zscoremap=False):
    '''
    Performs basic dual regression. Optionally normalizing independent
    variables (maps or timeseries) and/or applying spatial weights.
    
    Input:

        timeseries0   - run specific timeseries (t x p numpy array)
        groupMap0     - group ICA maps to regress on timeseries0 (n x p numpy array)
        weights       - regression weights (1 x p numpy array)
        zscoremap     - whether to perform standardized spatial regression (Default: False)
        zscorets      - whether to perform standardized temporal regression (Default: False)

    Output:

        nodets        - scan specific component timeseries (t x n numpy array)
        betaICA       - scan specific component spatial maps (n x p numpy array)
        tICA          - t-stat spatial maps (n x p numpy array)
        zICA          - z-stat spatial maps (n x p numpy array)
    '''

    # below we use the pseudoinverse instead of inv(X'*X)*X'
    # because it's better behaved. In particular with iterative
    # dual regression when the timeseries can become rank 
    # deficient.

    if timeseries0.shape[1] != groupMap0.shape[1]:
        raise ValueError(f'timeseries0 {timeseries0.shape} and groupMap0 {groupMap0.shape} must have the same number of features')

    if weights is None:
        weights = np.ones((1, timeseries0.shape[1]))

    if np.any(weights < 0):
        raise ValueError(f'Weights must be nonnegative, but {sum(weights<0)} negative weights were given.')

    # deal with nans
    nan_features = np.any(np.isnan(groupMap0), axis=0)
    mask = np.ones(nan_features.shape, dtype=bool)
    mask[nan_features] = False

    # spatial regression
    timeseries = timeseries0[:,mask]*np.sqrt(weights[:,mask])
    groupMap = groupMap0[:,mask]*np.sqrt(weights[:,mask])
    if zscoremap:
        groupMap = sp.stats.zscore(groupMap, axis=1)
        timeseries = sp.stats.zscore(timeseries, axis=1)
    else:
        groupMap = groupMap - np.mean(groupMap, axis=1, keepdims=True)
        timeseries = timeseries - np.mean(timeseries, axis=1, keepdims=True)

    pGM = np.linalg.pinv(groupMap.T)

    # Note: we don't need to reexpand here to compensate for the earlier masking
    # because nodets is time x components, doesn't care about spatial sources,
    # and we don't use pGM again.
    nodets0 = (pGM @ timeseries.T).T # (t x n)

    # temporal regression
    if zscorets:
        nodets = sp.stats.zscore(nodets0, axis=0)
        timeseries = sp.stats.zscore(timeseries0, axis=0)
    else:
        nodets = nodets0 - np.mean(nodets0, axis=0, keepdims=True)
        timeseries = timeseries0 - np.mean(timeseries0, axis=0, keepdims=True)
    
    betaICA = np.linalg.pinv(nodets) @ timeseries # (n x p)

    # get stats
    # ported from https://github.com/Washington-University/HCPpipelines/blob/e165988f1fee786d950469665c9b7c7d111c0fb9/MSMAll/scripts/MSMregression.m
    df = nodets.shape[0] - nodets.shape[1] - 1
    RSS = np.sum((timeseries - nodets @ betaICA)**2, axis=0, keepdims=True)
    pN = np.linalg.pinv(nodets)
    dpN = np.diag(pN @ pN.T).T

    tICA = betaICA / np.sqrt(RSS.T * dpN/df).T

    zICA = np.zeros(tICA.shape)
    zICA[tICA > 0] = -sp.stats.norm.ppf(sp.stats.t.cdf(-tICA[tICA > 0],df))
    zICA[tICA < 0] = sp.stats.norm.ppf(sp.stats.t.cdf(tICA[tICA < 0],df))

    return nodets, betaICA, tICA, zICA


def get_alignment_map(timeseries, groupICAs, 
    weights=None, zscorets=False, zscoremap=False):
    '''
    This function produces an alignment quality map by evaluating how correlated
    groupICA components and (e.g. subject) specific components are at each 
    voxel/vertex. The idea is that groupICA component scores and component scores
    obtained through dual regression will be more similar in some areas than
    others. Glasser et al. (2016) Nature use these as regression weights in a 
    subsequent dual regression invocation to bias obtained timeseries towards those
    areas where subject and group are most in agreement. Your mileage may vary.

    Note: they subsequently smooth these results with sigma=14mm kernel. Do it 
    using external tools if desired.

    Input:

        timeseries    - list of (t x p) numpy arrays
        groupICAs     - list of (n_i x p) numpy arrays
        weights       - regression weights (1 x p numpy array)
        zscoremap     - whether to perform standardized spatial regression (Default: False)
        zscorets      - whether to perform standardized temporal regression (Default: False)

    Output:

        alignmentMap  - map of mean correlation between group and run specific ICAs (1 x p numpy array)
    '''

    r = []
    for gm in groupICAs:
        tICAs = []
        for ts in timeseries:
            _, _, tICA, _ = dual_regression(
                ts, gm, weights=weights, zscorets=zscorets, zscoremap=zscoremap)

            tICAs.append(tICA)
        tICA = np.sum(tICAs, axis=0)

        rr = []
        for t,m in zip(tICA.T, gm.T):
            this_rr = np.corrcoef(t, m)
            if np.isnan(this_rr[0,1]):
                rr.append(0)
            else:
                rr.append(np.arctanh(this_rr[0,1]))

        r.append(rr)
        
    alignmentMap = np.mean(r, axis=0, keepdims=True)

    return alignmentMap


def iterative_dual_regression(timeseries, groupMap, 
        iterations=2, 
        distortionWeights=None, alignmentMap=None, 
        zscorets=False, zscoremap=False):
    '''
    A high level function for calling iterations of dual regression. The
    first iteration runs against the groupMap to produce a subject or run
    specific map, while subsequent iterations are run against the prior
    iterations specific maps.
    
    Input:

        timeseries0   - list of run specific timeseries (t x p numpy arrays)
        groupMap0     - group ICA maps to regress on timeseries0 (n x p numpy array)
        distortionWeights   
                      - weights for regressing run ICAs on run timeseries. Ideally this
                        is a normalized areal distortion map for surface data. e.g. 
                        ArealDistortion/mean(ArealDistortion) (1 x p numpy array)
        alignmentMap  - weights to multiply with distortion weights when regressing 
                        group maps on run timeseries. Ideally this is the output of 
                        get_alignment_maps().
        zscoremap     - whether to perform standardized spatial regression (Default: False)
        zscorets      - whether to perform standardized temporal regression (Default: False)

    Output:

        tsICA         - scan specific component timeseries (t x n numpy array)
        betaICA       - scan specific component spatial maps (n x p numpy array)
        tstatICA      - t-stat spatial maps (n x p numpy array)
        zstatICA      - z-stat spatial maps (n x p numpy array)
    '''

    if distortionWeights is None:
        distortionWeights = np.ones((1, timeseries[0].shape[1]))
    
    if alignmentMap is None:
        alignmentMap = np.ones((1, timeseries[0].shape[1]))

    tmaps = groupMap # this gets overwritten by each iteration
    for i in range(iterations):
        if i == 0:
            weights = distortionWeights*alignmentMap
        else:
            weights = distortionWeights

        tsICA = []
        betaICA = []
        tstatICA = []
        zstatICA = []
        for ts in timeseries:
            if np.any(np.shape(ts) != np.shape(timeseries[0])):
                warn('Timeseries'' lengths are mismatched. Hierarchical modeling across timeseries won''t be accurate.')

            nodets, beta, tstat, zstat = dual_regression(
                ts, tmaps, weights=weights, zscorets=zscorets, zscoremap=zscoremap)

            tsICA.append(nodets)
            betaICA.append(beta)
            tstatICA.append(tstat)
            zstatICA.append(zstat)
        
        # if there are multiple scans we implement a back of the envelope hierarchical
        # model by averaging t-stats
        tmaps = np.mean(tstatICA, axis=0)

    return tsICA, betaICA, tstatICA, zstatICA
