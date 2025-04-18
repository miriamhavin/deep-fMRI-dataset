import numpy as np
import scipy.stats
from ridge_utils.npp import mcorr
from tqdm import tqdm
import time

def model_pvalue(wts, stim, resp, nboot=1e4, randinds=None):
    """Computes a bootstrap p-value by resampling the [wts] of the model, which
    is [wts] * [stim] ~ [resp].
    """
    origcorr = np.corrcoef(resp, np.dot(stim, wts))[0,1]
    if randinds is None:
        #randinds = np.random.randint(0, len(wts), (len(wts), nboot))
        randinds = make_randinds(len(wts), nboot)
    pwts = wts[randinds]
    pred = np.dot(stim, pwts)
    
    ## Compute correlations using vectorized method and bootstrap p-value
    zpred = (pred-pred.mean(0))/pred.std(0)
    zresp = (resp-resp.mean())/resp.std()
    bootcorrs = np.dot(zpred.T, zresp).ravel()/resp.shape[0]
    #bootcorrs = np.array([np.corrcoef(resp, p.T)[0,1] for p in pred.T])
    bspval = np.mean(bootcorrs>origcorr)
    
    ## Compute parametric p-value based on transformed distribution
    zccs = ztransformccs(bootcorrs)
    zorig = ztransformccs(origcorr)
    ppval = 1-scipy.stats.norm.cdf(zorig, loc=zccs.mean(), scale=zccs.std())
    
    print("Boostrap p-value: %0.3f, parametric p-value: %0.03f"%(bspval, ppval))
    return bspval, ppval

def make_randinds(nwts, nboot, algo="randint", maxval=None):
    if maxval is None:
        maxval = nwts
    
    if algo=="randint":
        return np.random.randint(0, maxval, (nwts, nboot))
    
    elif algo=="bytes":
        N = nwts*nboot*2
        return np.mod(np.frombuffer(np.random.bytes(N), dtype=np.uint16), maxval).reshape((nwts, nboot))

    elif algo=="bytes8":
        N = nwts*nboot
        return np.mod(np.frombuffer(np.random.bytes(N), dtype=np.uint8), maxval).reshape((nwts, nboot))

def ztransformccs(ccs):
    """Transforms the given correlation coefficients to be vaguely Gaussian.
    """
    return ccs/np.sqrt((1-ccs**2))

def mrsq(a, b, corr_units=False):
    rsq = 1 - ((a - b).var(0) / a.var(0))
    if corr_units:
        return np.sqrt(np.abs(rsq)) * np.sign(rsq)
    else:
        return rsq

def exact_correlation_pvalue(corr, N, alt="greater"):
    """Returns the exact p-value for the correlation, [corr] between two vectors of length [N].
    The null hypothesis is that the correlation is zero. The distribution of
    correlation coefficients given that the true correlation is zero and both
    [a] and [b] are gaussian is given at 
    http://en.wikipedia.org/wiki/Pearson_correlation#Exact_distribution_for_Gaussian_data

    Parameters
    ----------
    corr : float
        Correlation value
    N : int
        Length of vectors that were correlated
    alt : string
        The alternative hypothesis, is the correlation 'greater' than zero,
        'less' than zero, or just 'nonzero'.
        
    Returns
    -------
    pval : float
        Probability of sample correlation between [a] and [b] if actual correlation
        is zero.
    """
    f = lambda r,n: (1-r**2)**((n-4.0)/2.0)/scipy.special.beta(0.5, (n-2)/2.0)
    pval = scipy.integrate.quad(lambda r: f(r, N), corr, 1)[0]
    if alt=="greater":
        return pval
    elif alt=="less":
        return 1-pval
    elif alt=="nonzero":
        return min(pval, 1-pval)

def correlation_pvalue(a, b, nboot=1e4, confinterval=0.95, method="pearson"):
    """Computes a bootstrap p-value for the correlation between [a] and [b].
    The alternative hypothesis for this test is that the correlation is zero or less.
    This function randomly resamples the timepoints in the [a] and [b] and computes
    the correlation for each sample.

    Parameters
    ----------
    a : array_like, shape (N,)
    b : array_like, shape (N,)
    nboot : int, optional
        Number of bootstrap samples to compute, default 1e4
    conflevel : float, optional
        Confidence interval size, default 0.95
    method : string, optional
        Type of correlation to use, can be "pearson" (default) or "robust"
        
    Returns
    -------
    bspval : float
        The fraction of bootstrap samples with correlation less than zero.
    bsconf : (float, float)
        The [confinterval]-percent confidence interval according to the bootstrap.
    ppval : float
        The probability that the correlation is zero or less according to parametric
        computation using Fisher transform.
    pconf : (float, float)
        The parametric [confinterval]-percent confidence interval according to
        parametric computation using Fisher transform.
    bootcorrs : array_like, shape(nboot,)
        The correlation for each bootstrap sample
    """
    ocorr = np.corrcoef(a, b)[0,1]
    conflims = ((1-confinterval)/2, confinterval/2+0.5)
    confinds = map(int, (conflims[0]*nboot, conflims[1]*nboot))

    N = len(a)
    inds = make_randinds(N, nboot, algo="bytes")
    rsa = a[inds] ## resampled a
    rsb = b[inds] ## resampled b

    if method=="pearson":
        za = (rsa-rsa.mean(0))/rsa.std(0)
        zb = (rsb-rsb.mean(0))/rsb.std(0)
        bootcorrs = np.sum(za*zb, 0)/(N-1) ## The correlation between each pair
    elif method=="robust":
        bootcorrs = np.array([robust_correlation(x,y)[0] for (x,y) in zip(rsa.T, rsb.T)])
    else:
        raise ValueError("Unknown method: %s"%method)

    ## Compute the bootstrap p-value
    #bspval = np.mean(bootcorrs<0) ## Fraction of correlations smaller than zero
    bspval = np.mean(bootcorrs>ocorr)
    bsconf = (np.sort(bootcorrs)[confinds[0]], np.sort(bootcorrs)[confinds[1]])

    ## Compute the parametric bootstrap p-value using Fisher transform
    zccs = np.arctanh(bootcorrs)
    ppval = scipy.stats.norm.cdf(0, loc=zccs.mean(), scale=zccs.std())
    pconf = tuple(map(lambda c: np.tanh(scipy.stats.norm.isf(1-c, loc=zccs.mean(), scale=zccs.std())), conflims))

    ## return things!
    return bspval, bsconf, ppval, pconf, bootcorrs


def permutation_test(true, pred, metric, blocklen=10, nperms=1000):
    # Calculate max number of complete blocks
    n_samples = true.shape[0]
    n_complete_blocks = n_samples // blocklen

    # Truncate data to fit complete blocks
    usable_samples = n_complete_blocks * blocklen
    true_trunc = true[:usable_samples]
    pred_trunc = pred[:usable_samples]

    print(f"Using {n_complete_blocks} blocks of length {blocklen} ({usable_samples} of {n_samples} samples)")

    # Split into blocks
    block_true = np.dstack(np.vsplit(true_trunc, n_complete_blocks)).transpose((2, 0, 1))
    block_pred = np.dstack(np.vsplit(pred_trunc, n_complete_blocks)).transpose((2, 0, 1))

    # Select random blocks, compute metric for each
    a_inds = make_randinds(n_complete_blocks, nperms)
    b_inds = make_randinds(n_complete_blocks, nperms)
    start_time = time.time()
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=8)
    perm_rsqs = []
    for i, (a, b) in enumerate(tqdm(zip(a_inds.T, b_inds.T), total=nperms,
                                    desc="Running permutations")):
        result = metric(np.vstack(block_true[a]), np.vstack(block_pred[b]))
        perm_rsqs.append(result)
    pool.close()

    # Calculate metric on original full data (not truncated)
    real_rsqs = metric(true, pred)

    # Calculate p-values
    pvals = (real_rsqs <= perm_rsqs).mean(0)

    return pvals, perm_rsqs, real_rsqs


def fdr_qvalues(pvalues, lam=0.5):
    m = float(pvalues.size)
    psinds = np.argsort(pvalues)
    psoriginds = np.argsort(psinds)
    sortpvals = pvalues[psinds]
    pi_0 = max((pvalues > lam).sum(),1) / ((1-lam) * m)
    print(pi_0)
    p_less = (np.arange(m)+1) / m
    pfdr = pi_0 * sortpvals / (p_less * (1 - (1-sortpvals)**m))
    print(pfdr)
    qvals = np.zeros((m,))
    qvals[m-1] = pfdr[m-1]
    for ii in range(2, int(m)+1):
        qvals[m-ii] = min(pfdr[m-ii], qvals[m-ii+1])

    return qvals[psoriginds]

def fdr_correct(pval, thres):
   """Find the fdr corrected p-value thresholds
   pval - vector of p-values
   thres - FDR level
   pID - p-value thres based on independence or positive dependence
   pN - Nonparametric p-val thres"""
   # remove NaNs
   p = pval[np.nonzero(np.isnan(pval)==False)[0]]
   p = np.sort(p)
   V = float(len(p))
   I = np.arange(V) + 1

   cVID = 1
   cVN = (1/I).sum()

   th1 = np.nonzero(p <= I/V*thres/cVID)[0]
   th2 = np.nonzero(p <= I/V*thres/cVN)[0]
   if len(th1)>0:
       pID = p[th1.max()]
   else:
       pID = -np.inf
   if len(th2)>0:
       pN =  p[th2.max()]
   else:
       pN = -np.inf

   return pID, pN

