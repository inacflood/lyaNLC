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
#from scipy.stats import norm

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tLyA.TheoryLyaP3D(cosmo24)
dkM24z = th24.cosmo.dkms_dhMpc(z24)

# Get actual data
data = npd.LyA_P1D(2.4)
k = data.k
k_res = k*dkM24z
P = data.k/np.pi*data.Pk_emp()
Perr = data.Pk_stat*data.k/np.pi

# Plot our synthetic data along with fit from MLE
fig = plt.figure(1)
ax = fig.add_subplot()
plt.xscale('linear')
plt.yscale('log')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$')
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$')

ax.errorbar(k,P,yerr=Perr,fmt='k.')

def bConvert(bp,beta):
    """
    Function to convert our modified beta fitting variable to beta
    """
    b=bp/(1+beta)
    return b

# Maximum Likelihood Estimate fit to the synthetic data

def lnlike(theta, k_res, P, Perr):
    bp, beta = theta
    model = th24.makeP1D_P(k_res, b_lya=bConvert(bp,beta), beta_lya=beta)*k_res/np.pi
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

err = 0.9
bp_f = -0.321
beta_f = 1.656
var_bp=np.abs(bp_f)*err
var_beta=np.abs(beta_f)*err
min_bp=bp_f-var_bp
max_bp=bp_f+var_bp
min_beta=beta_f-var_beta
max_beta=beta_f+var_beta

#y_f = th24.makeP1D_P(k, b_lya=bConvert(bp_f,beta_f), beta_lya=beta_f)*k*dkM24z/np.pi
#ax.plot(k,y_f)

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [bp_f, beta_f], args=(k_res, P, Perr))
        #, method='L-BFGS-B',bounds=[(min_bp,max_bp),(min_beta,max_beta)])
bp_ml, beta_ml = result["x"]

result_plot = th24.makeP1D_P(k_res, b_lya=bConvert(bp_ml,beta_ml), beta_lya=beta_ml)*k_res/np.pi
ax.plot(k,result_plot)
fig.savefig("../Figures/P1D_IntitalFit_emcee.pdf")
#print(b_ml, betap_ml)



# Set up MLE for emcee error evaluation

def lnprior(theta):
    bp, beta = theta
    if min_bp < bp < max_bp and min_beta < beta < max_beta:
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

ndim, nwalkers = 2, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Run emcee error evaluation
import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k_res, P, Perr))

nsteps=300
sampler.run_mcmc(pos, nsteps)
chain = sampler.chain

# Plots to visualize emcee walker paths parameter values
param1 = plt.figure(2)
plt.ylabel('beta')
for w in range(nwalkers):
    plt.plot([chain[w][s][1] for s in range(nsteps)])
    
param1.show()
param1.savefig("../Figures/P1D_WalkerPathsBeta.pdf")

param2 = plt.figure(3)
plt.ylabel('b(1+beta)')
for w in range(nwalkers):
    plt.plot([chain[w][s][0] for s in range(nsteps)])
    
param2.show()
param2.savefig("../Figures/WalkerPathsBiasMod.pdf")



samples = sampler.chain[:, 100:, :].reshape((-1, ndim))

# Plot a few paths against data and intial fit
pathView = plt.figure(4)

for bp, beta in samples[np.random.randint(len(samples), size=80)]:
    plt.plot(k, th24.makeP1D_P(k_res, b_lya=bConvert(bp,beta), beta_lya=beta)*k_res/np.pi, color="k", alpha=0.1)
plt.plot(k,th24.makeP1D_P(k_res, beta_lya = beta_f, b_lya=bConvert(bp_f,beta_f))*k_res/np.pi, color="r", lw=2, alpha=0.8)
plt.errorbar(k, P, yerr=Perr, fmt=".k")

plt.yscale('log')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)*k/pi')
plt.title('Parameter exploration for beta, bias')

pathView.savefig("../Figures/P1D_SamplePaths.pdf")
pathView.show()

# Final results
cornerplt = corner.corner(samples, labels=["$bp$", "$beta$"],
                      truths=[bp_f, beta_f],quantiles=[0.16, 0.5, 0.84],show_titles=True)
cornerplt.savefig("../Figures/P1D_triangleBetaB.png")
cornerplt.show()


#samples[:, 1] = np.exp(samples[:, 1])
#b_mcmc, betap_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                             zip(*np.percentile(samples, [16, 50, 84],
#                                                axis=0)))
#print("b:", b_mcmc, "beta:", [betaConvert(betap,b_mcmc[1],mu) for betap in betap_mcmc])




