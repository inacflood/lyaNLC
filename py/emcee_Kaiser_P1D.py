#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:11:57 2019

@author: iflood
"""
import cosmoCAMB_newParams as cCAMB
import theoryLya as tLyA
import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize as op
#import corner
import time
import emcee
import tqdm
import get_npd_p1d_woFitsio as npd
#from scipy.stats import norm

t = time.process_time()

headFile = "Kaiser"
z=2.4
err = 0.5 # width of the uniform parameter priors
pos_method = 2 # emcee starts 1:from a small ball [SB], 2:in full param space [FS]
multiT = False
convTest = False

# Choose the "true" parameters.
def betaConvert(b,bp):
    """
    Function to convert our modified beta fitting variable to beta
    """
    return bp/b-1

beta_f = 1.656
b_f = -0.121 
bp_f = -0.321

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLyaP3D(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z)

# Get actual data
data = npd.LyA_P1D(2.4)
k = data.k
k_res = k*dkMz
P = data.k/np.pi*data.Pk_emp()
Perr = data.Pk_stat*data.k/np.pi


# Maximum Likelihood Estimate fit to the synthetic data

def lnlike(theta, k_res, P, Perr):
    b, bp = theta
    model = th.makeP1D_P(k_res, b_lya=b, beta_lya=betaConvert(b,bp))*k_res/np.pi
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))


var_bp = np.abs(bp_f)*err
var_b = np.abs(b_f)*err
min_bp = bp_f-var_bp
max_bp = bp_f+var_bp
min_b = b_f-var_b
max_b = b_f+var_b

#nll = lambda *args: -lnlike(*args)
#result = op.minimize(nll, [bp_f, beta_f], args=(k_res, P, Perr),method='L-BFGS-B',bounds=[(min_bp,max_bp),(min_beta,max_beta)])
#bp_ml, beta_ml = result["x"]


# Set up MLE for emcee error evaluation

def lnprior(theta):
    b, bp = theta
    if min_bp < bp < max_bp and min_b < b < max_b:
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

ndim, nwalkers = 2, 250

if pos_method==1:
    pos = [[b_f,bp_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
else:
    pos_1 = np.random.uniform(min_b,max_b,nwalkers)
    pos_2 = np.random.uniform(min_bp,max_bp,nwalkers)
    pos = [[pos_1[i],pos_2[i]] for i in range(nwalkers)]

# Run emcee error evaluation
nsteps=0

if convTest:
    filename = "test2.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr), backend=backend)
    
    max_n = 10000
    
    #sampler.run_mcmc(pos, 500)
    # We'll track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)
    
    old_tau = np.inf
    
    # Now we'll sample for up to max_n steps
    for sample in sampler.sample(pos, iterations=max_n, progress=True):
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
    
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        index += 1
    
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau
        
    nsteps = sampler.iteration
else:
    nsteps = 2000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr))
    sampler.run_mcmc(pos, nsteps)
    
chain = sampler.chain

elapsed_time = time.process_time() - t

paramfile = open('../output/Walks_'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(elapsed_time)))
paramfile.close()
c=sampler.chain
for w in range(nwalkers):
    file=open('../output/Walks_'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} \n'.format(str(c[w][i][0]), str(c[w][i][1]))) #, str(c[w][i][2])))
    file.close()






