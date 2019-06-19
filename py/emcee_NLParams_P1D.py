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
import get_npd_p1d_woFitsio as npd
import FiducialValues_Arinyo2015 as fv
import time
import emcee
import tqdm
#from scipy.stats import norm

t = time.process_time()

# Setup initial parameters (you only need to modify z and err here)
headFile = "Walks_test4"
z=2.4
err = 0.2 # width of the uniform parameter priors
z_str=str(int(z*10)) # for use in file names
err_str = str(int(err*100))

q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = fv.getFiducialValues(z)
bp_f = -0.321
beta_f = 1.656

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLyaP3D(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z) # 

def bConvert(beta,bp):
    """
    Function to convert our modified beta fitting variable to beta
    """
    return bp/(1+beta)

# Get actual data
data = npd.LyA_P1D(z)
k = data.k
k_res = k*dkMz  # data rescaled b/c of difference in units
P = data.Pk_emp()
Perr = data.Pk_stat 

# Maximum Likelihood Estimate fit to the synthetic data

def lnlike(theta, k_res, P, Perr):
    q1,bp,kp = theta
    model = th.makeP1D_P(k_res, q1=q1, q2=q2_f, kvav=kvav_f, kp=kp, av=av_f, bv=bv_f, b_lya=bConvert(beta_f,bp))*dkMz
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

var_1=np.abs(q1_f)*err
var_2=np.abs(bp_f)*err
var_3=np.abs(kp_f)*err
min_1= q1_f - var_1
max_1= q1_f + var_1
min_2= bp_f - var_2
max_2= bp_f + var_2
min_3= kp_f - var_3
max_3= kp_f + var_3

#y_f = th24.makeP1D_P(k, b_lya=bConvert(bp_f,beta_f), beta_lya=beta_f)*k*dkM24z/np.pi
#ax.plot(k,y_f)

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [q1_f, bp_f, kp_f], args=(k_res, P, Perr),method='L-BFGS-B',bounds=[(min_1,max_1),(min_2,max_2),(min_3,max_3)])
q1_ml, bp_ml, kp_ml = result["x"]

#result_plot = th.makeP1D_P(k_res, q1=q1_ml, q2=q2_f, kp=kp_f, kvav=kvav_f, av=av_ml, bv=bv_f)*k_res/np.pi
#ax.plot(k,result_plot)
#fig.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/IntitalFit_err"+err_str+".pdf")
#print(b_ml, betap_ml)



# Set up MLE for emcee error evaluation

def lnprior(theta):
    q1, bp, kp = theta
    if min_1 < q1 < max_1 and min_2 < bp < max_2 and min_3 < kp < max_3:
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

ndim, nwalkers = 3, 600
pos_1_1 = np.random.uniform(min_1,max_1,nwalkers)
pos_1_2 = np.random.uniform(min_2,max_2,nwalkers)
pos_1_3 = np.random.uniform(min_3,max_3,nwalkers)
pos_1 = [[pos_1_1[i],pos_1_2[i],pos_1_3[i]] for i in range(nwalkers)]
pos_2_1 = np.random.uniform(min_1,max_1,nwalkers)
pos_2_2 = np.random.uniform(min_2,max_2,nwalkers)
pos_2_3 = np.random.uniform(min_3,max_3,nwalkers)
pos_2 = [[pos_2_1[i],pos_2_2[i],pos_2_3[i]] for i in range(nwalkers)]
pos_3_1 = np.random.uniform(min_1,max_1,nwalkers)
pos_3_2 = np.random.uniform(min_2,max_2,nwalkers)
pos_3_3 = np.random.uniform(min_3,max_3,nwalkers)
pos_3 = [[pos_3_1[i],pos_3_2[i],pos_3_3[i]] for i in range(nwalkers)]
pos = [pos_1,pos_2,pos_3]

betas = np.asarray([0.01, 0.505, 1.0])
ntemps = len(betas)

sampler = emcee.PTSampler(ntemps, nwalkers, ndim, lnprob, lnprior, loglargs=(k_res, P, Perr), betas=betas,threads=3) 

# Run emcee error evaluation

# Set up the backend
# Don't forget to clear it in case the file already exists
#filename = "test4.h5"
#backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
#
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k_res, P, Perr),threads=4, backend=backend)
#
#max_n = 10000
#
## We'll track how the average autocorrelation time estimate changes
#index = 0
#autocorr = np.empty(max_n)
#
#old_tau = np.inf
#
## Now we'll sample for up to max_n steps
#for sample in sampler.sample(pos, iterations=max_n, progress=True):
#    # Only check convergence every 100 steps
#    if sampler.iteration % 100:
#        continue
#
#    # Compute the autocorrelation time so far
#    # Using tol=0 means that we'll always get an estimate even
#    # if it isn't trustworthy
#    tau = sampler.get_autocorr_time(tol=0)
#    autocorr[index] = np.mean(tau)
#    index += 1
#
#    # Check convergence
#    converged = np.all(tau * 100 < sampler.iteration)
#    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
#    if converged:
#        break
#    old_tau = tau
#    
#nsteps = sampler.iteration

nsteps = 10000
sampler.run_mcmc(pos, nsteps)

temp_idx = 2
chain = sampler.chain[temp_idx][:,:,:]

elapsed_time = time.process_time() - t

paramfile = open('../'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(3.1),str(elapsed_time)))
paramfile.close()
c=chain
for w in range(nwalkers):
    file=open('../'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} {2} \n'.format(str(c[w][i][0]), str(c[w][i][1]), str(c[w][i][2])))
    file.close()





