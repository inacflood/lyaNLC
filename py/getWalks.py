#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 10:10:59 2019

@author: iflood
"""

import numpy as np
import matplotlib.pyplot as plt
import corner
import FiducialValues_Arinyo2015 as fv
import cosmoCAMB_newParams as cCAMB
import theoryLya as tLyA
import get_npd_p1d_woFitsio as npd

headFile = "Walks_test2"
nwalkers, nsteps, ndim, z, err, param_code, runtime = np.loadtxt('../'+headFile+'/params.dat')

nwalkers = int(nwalkers)
nsteps = int(nsteps) 
ndim = int(ndim)
#nwalkers=300
#nsteps=3000
#ndim=3
#z=2.4
#err = 0.5 # width of the uniform parameter priors
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


data0=np.loadtxt('../'+headFile+'/walk0.dat')
data1=np.loadtxt('../'+headFile+'/walk1.dat')
chain=np.stack([data0,data1])
for w in range(nwalkers-2):
   data=np.loadtxt('../'+headFile+'/walk'+str(w+2)+'.dat')
   data=data.reshape((1,nsteps,ndim))
   chain=np.vstack([chain,data])

   
# Plots to visualize emcee walker paths parameter values
param1 = plt.figure(1)
plt.ylabel('q1')
for w in range(nwalkers):
    plt.plot([chain[w][s][0] for s in range(nsteps)])
    
param1.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/WalkerPathsq1_err"+err_str+".pdf")
param1.show()

param2 = plt.figure(2)
plt.ylabel('q2')
for w in range(nwalkers):
    plt.plot([chain[w][s][1] for s in range(nsteps)])
    
param2.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/WalkerPathsq2_err"+err_str+".pdf")
param2.show()

param3 = plt.figure(3)
plt.ylabel('kp')
for w in range(nwalkers):
    plt.plot([chain[w][s][2] for s in range(nsteps)])
    
param3.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/WalkerPathskp_err"+err_str+".pdf")
param3.show()


# Plot a few paths against data and fiducial fit
pathView = plt.figure(4)
samples = chain[:, 50:, :].reshape((-1, ndim))

for q1, q2, kp in samples[np.random.randint(len(samples), size=200)]:
    plt.plot(k, th.makeP1D_P(k_res, q1=q1, q2=q2, kvav=kvav_f, kp=kp, av=av_f, bv=bv_f)*k_res/np.pi, color="k", alpha=0.1)
plt.plot(k,th.makeP1D_P(k_res, q1=q1_f, q2=q2_f, kvav=kvav_f, kp=kp_f, av=av_f, bv=bv_f)*k_res/np.pi, color="r", lw=2, alpha=0.8)
plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")

plt.yscale('log')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)*k/pi')
plt.title('Parameter exploration for beta, bias')

pathView.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/SamplePaths_err"+err_str+".pdf")
pathView.show()

# Final results
cornerplt = corner.corner(samples, labels=["$q1$", "$q2$", "$kp$"],
                      truths=[q1_f, q2_f, kp_f],quantiles=[0.16, 0.5, 0.84],show_titles=True)
cornerplt.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/triangle_err"+err_str+".pdf")
cornerplt.show()


q1_mcmc, q2_mcmc, kp_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print("q1:", q1_mcmc, "q2:", q2_mcmc, "kp:",kp_mcmc)