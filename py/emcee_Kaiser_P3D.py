#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:11:57 2019

@author: iflood
"""
import cosmoCAMB_newParams as cCAMB
import theoryLya as tP3D
import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize as op
#import corner
import time
import emcee
import tqdm
import ptemcee
from ptemcee.sampler import Sampler

#from scipy.stats import norm

t = time.process_time()

headFile = "Kaiser_overflow"
z=2.4
err = 0.5 # width of the uniform parameter priors
pos_method = 2 # emcee starts 1:from a small ball, 2:in full param space
multiT = True
convTest = False

#z_str=str(int(z*10)) # for use in file names
#err_str = str(int(err*100))

# Choose the "true" parameters.
def betaConvert(b,bp,mu):
    """
    Function to convert our modified beta fitting variable to beta
    """
    return (bp/b-1)/mu**2

beta_true = 1.656
b_true = -0.121 
mu = 0.5
bp_true = (1+beta_true*mu**2)*b_true


cosmo = cCAMB.Cosmology(z)
th = tP3D.TheoryLyaP3D(cosmo)

# Generate some synthetic data from the model.
N = 100
k = np.sort(np.logspace(-4,0.9,N))
Perr = np.random.rand(N)
P = th.FluxP3D_hMpc(z,k,mu,beta_lya = beta_true, b_lya=b_true)
P += Perr * np.random.randn(N)

# Maximum Likelihood Estimate fit to the synthetic data
def lnlike(theta, k, P, Perr):
    b, bp = theta
    model = th.FluxP3D_hMpc(z,k,mu,beta_lya = betaConvert(b,bp,mu), b_lya=b)
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

var_bp=np.abs(bp_true)*err
var_b=np.abs(b_true)*err
min_bp=bp_true-var_bp
max_bp=bp_true+var_bp
min_b=b_true-var_b
max_b=b_true+var_b

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

if multiT:
    if pos_method==1:
        pos_1 = [[b_true,bp_true] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        pos_2 = [[b_true,bp_true] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        pos_3 = [[b_true,bp_true] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        pos = [pos_1,pos_2,pos_3]
    else:
        pos_1_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_1_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos_1 = [[pos_1_1[i],pos_1_2[i]] for i in range(nwalkers)]
        pos_2_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_2_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos_2 = [[pos_2_1[i],pos_2_2[i]] for i in range(nwalkers)]
        pos_3_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_3_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos_3 = [[pos_3_1[i],pos_3_2[i]] for i in range(nwalkers)]
        pos = [pos_1,pos_2,pos_3]
else:
    if pos_method==1:
        pos = [[b_true,bp_true] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
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
    chain = sampler.chain
elif multiT:
    nsteps = 2000
    betas = np.asarray([0.01, 0.505, 1.0])
    sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, loglargs=(k, P, Perr), betas=betas,threads=3)
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain[2][:,:,:]
else:
    nsteps = 2000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr))
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain



elapsed_time = time.process_time() - t

paramfile = open('../output/Walks_'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(mu),str(elapsed_time)))
paramfile.close()
c=chain
for w in range(nwalkers):
    file=open('../output/Walks_'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} \n'.format(str(c[w][i][0]), str(c[w][i][1]))) #, str(c[w][i][2])))
    file.close()






