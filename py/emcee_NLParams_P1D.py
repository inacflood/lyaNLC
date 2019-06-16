#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:11:57 2019

@author: iflood
"""
import cosmoCAMB_newParams as cCAMB
import theoryLya as tLyA
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import corner
import get_npd_p1d_woFitsio as npd
import FiducialValues_Arinyo2015 as fv
#from scipy.stats import norm

# Setup initial parameters (you only need to modify z and err here)
z=2.4
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

# Plot data 
#fig = plt.figure(1)
#ax = fig.add_subplot()
#plt.xscale('linear')
#plt.yscale('log')
#plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$')
#plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$')

#ax.errorbar(k,P*k/np.pi,yerr=Perr*k/np.pi,fmt='k.')


# Maximum Likelihood Estimate fit to the synthetic data

def lnlike(theta, k_res, P, Perr):
    q1,q2,kp = theta
    model = th.makeP1D_P(k_res, q1=q1, q2=q2, kvav=kvav_f, kp=kp, av=av_f, bv=bv_f)*dkMz
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

var_1=np.abs(q1_f)*err
var_2=np.abs(q2_f)*err
var_3=np.abs(kp_f)*err
min_1= q1_f - var_1
max_1= q1_f + var_1
min_2= q2_f - var_2
max_2= q2_f + var_2
min_3= kp_f - var_3
max_3= kp_f + var_3

#y_f = th24.makeP1D_P(k, b_lya=bConvert(bp_f,beta_f), beta_lya=beta_f)*k*dkM24z/np.pi
#ax.plot(k,y_f)

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [q1_f, q2_f,kp_f], args=(k_res, P, Perr),method='L-BFGS-B',bounds=[(min_1,max_1),(min_2,max_2),(min_3,max_3)])
q1_ml, q2_ml, kp_ml = result["x"]

result_plot = th.makeP1D_P(k_res, q1=q1_ml, q2=q2_ml, kp=kp_ml, kvav=kvav_f, av=av_f, bv=bv_f)*k_res/np.pi
#ax.plot(k,result_plot)
#fig.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/IntitalFit_err"+err_str+".pdf")
#print(b_ml, betap_ml)



# Set up MLE for emcee error evaluation

def lnprior(theta):
    q1, q2, kp = theta
    if min_1 < q1 < max_1 and min_2 < q2 < max_2 and min_3 < kp < max_3:
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

ndim, nwalkers = 3, 300
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Run emcee error evaluation
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k_res, P, Perr),threads=3)

nsteps = 3000
sampler.run_mcmc(pos, nsteps)
chain = sampler.chain

# Plots to visualize emcee walker paths parameter values
#param1 = plt.figure(2)
#plt.ylabel('q1')
#for w in range(nwalkers):
#    plt.plot([chain[w][s][0] for s in range(nsteps)])
#    
#param1.show()
#param1.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/WalkerPathsq1_err"+err_str+".pdf")
#
#param2 = plt.figure(3)
#plt.ylabel('q2')
#for w in range(nwalkers):
#    plt.plot([chain[w][s][1] for s in range(nsteps)])
#    
#param2.show()
#param2.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/WalkerPathsq2_err"+err_str+".pdf")
#
#param3 = plt.figure(3)
#plt.ylabel('kp')
#for w in range(nwalkers):
#    plt.plot([chain[w][s][2] for s in range(nsteps)])
#    
#param3.show()
#param3.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/WalkerPathskp_err"+err_str+".pdf")


wlks,itr,ndim = sampler.chain.shape
c=sampler.chain
for w in range(wlks):
    file=open('../Walks/walk'+str(w)+'.dat','w')
    for i in range(itr):
        file.write('{0} {1} {2} \n'.format(str(c[w][i][0]), str(c[w][i][1]), str(c[w][i][2])))
    file.close()
#samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Plot a few paths against data and fiducial fit
#pathView = plt.figure(4)
#
#for q1, q2, kp in samples[np.random.randint(len(samples), size=80)]:
#    plt.plot(k, th.makeP1D_P(k_res, q1=q1, q2=q2, kvav=kvav_f, kp=kp, av=av_f, bv=bv_f)*k_res/np.pi, color="k", alpha=0.1)
#plt.plot(k,th.makeP1D_P(k_res, q1=q1_f, q2=q2_f, kvav=kvav_f, kp=kp_f, av=av_f, bv=bv_f)*k_res/np.pi, color="r", lw=2, alpha=0.8)
#plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")
#
#plt.yscale('log')
#plt.xlabel('k [(Mpc/h)^-1]')
#plt.ylabel('P(k)*k/pi')
#plt.title('Parameter exploration for beta, bias')
#
#pathView.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/SamplePaths_err"+err_str+".pdf")
#pathView.show()
#
## Final results
#cornerplt = corner.corner(samples, labels=["$q1$", "$q2$", "$kp$"],
#                      truths=[q1_f, q2_f, kp_f],quantiles=[0.16, 0.5, 0.84],show_titles=True)
#cornerplt.savefig("../Figures/MCMC_NLParams_3/q1_q2_kp/z"+z_str+"/triangle_err"+err_str+".pdf")
#cornerplt.show()

#
#q1_mcmc, q2_mcmc, kp_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                            zip(*np.percentile(samples, [16, 50, 84],
#                                                axis=0)))
#file = open("../Figures/MCMC_NLParams_2/q1_q2/z"+z_str+"_q1_q2.txt","a+") 
# 
#file.write('Prior width {0}, q1 {1}, q2 {2}.\\'.format(str(err), str(q1_mcmc),
#                                                       str(q2_mcmc))) 
#file.close() 

#print("q1:", q1_mcmc, "q2:", q2_mcmc, "kp:",kp_mcmc)




