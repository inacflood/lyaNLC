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
headFile = "Walks_test3"
z=2.2
err = 0.5 # width of the uniform parameter priors
z_str=str(int(z*10)) # for use in file names
err_str = str(int(err*100))

q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = fv.getFiducialValues(z)

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLyaP3D(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z) # 

# Get actual data
data = npd.LyA_P1D(z)
k = data.k
k_res = k*dkMz  # data rescaled b/c of difference in units
P = data.Pk_emp()
Perr = data.Pk_stat 

# Maximum Likelihood Estimate fit to the synthetic data

def lnlike(theta, k_res, P, Perr):
    q1,av = theta
    model = th.makeP1D_P(k_res, q1=q1, q2=q2_f, kvav=kvav_f, kp=kp_f, av=av, bv=bv_f)*dkMz
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

var_1=np.abs(q1_f)*err
var_2=np.abs(av_f)*err
#var_3=np.abs(kp_f)*err
min_1= q1_f - var_1
max_1= q1_f + var_1
min_2= av_f - var_2
max_2= av_f + var_2
#min_3= kp_f - var_3
#max_3= kp_f + var_3

#y_f = th24.makeP1D_P(k, b_lya=bConvert(bp_f,beta_f), beta_lya=beta_f)*k*dkM24z/np.pi
#ax.plot(k,y_f)

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [q1_f, av_f], args=(k_res, P, Perr),method='L-BFGS-B',bounds=[(min_1,max_1),(min_2,max_2)])
q1_ml, av_ml = result["x"]

result_plot = th.makeP1D_P(k_res, q1=q1_ml, q2=q2_f, kp=kp_f, kvav=kvav_f, av=av_ml, bv=bv_f)*k_res/np.pi
#ax.plot(k,result_plot)
#fig.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/IntitalFit_err"+err_str+".pdf")
#print(b_ml, betap_ml)



# Set up MLE for emcee error evaluation

def lnprior(theta):
    q1, av = theta
    if min_1 < q1 < max_1 and min_2 < av < max_2: # and min_3 < kp < max_3:
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

ndim, nwalkers = 2, 30
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k_res, P, Perr),threads=4)#, backend=backend)

# Run emcee error evaluation

## Set up the backend
## Don't forget to clear it in case the file already exists
#filename = "test2.h5"
#backend = emcee.backends.HDFBackend(filename)
#backend.reset(nwalkers, ndim)
#
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k_res, P, Perr),threads=4, backend=backend)
#
#max_n = 300
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
#nsteps = index
nsteps = 300
sampler.run_mcmc(pos, nsteps)
chain = sampler.chain

elapsed_time = time.process_time() - t

paramfile = open('../'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(3.1),str(elapsed_time)))
paramfile.close()
c=sampler.chain
for w in range(nwalkers):
    file=open('../'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} \n'.format(str(c[w][i][0]), str(c[w][i][1]))) #, str(c[w][i][2])))
    file.close()





