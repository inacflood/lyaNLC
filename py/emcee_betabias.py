#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 14:11:57 2019

@author: iflood
"""
import cosmoCAMB as cCAMB
import theoryLyaP3D as tP3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import corner
#from scipy.stats import norm

# Choose the "true" parameters.
bp_true = -0.321 
beta_true = 1.656
mu=1.0


z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tP3D.TheoryLyaP3D(cosmo24)

def bConvert(beta,bp,mu):
    """
    Function to convert our modified beta fitting variable to beta
    """
    return bp/(1+beta*mu**2)

# Generate some synthetic data from the model.
N = 100
k = np.sort(np.logspace(-4,0.9,N))
Perr = np.random.rand(N)
P = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=bConvert(beta_true,bp_true,mu))
P += Perr * np.random.randn(N)

# Plot our synthetic data along with fit from MLE
fig = plt.figure(1)
ax = fig.add_subplot()
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)')

ax.errorbar(k,P,yerr=Perr,fmt='k.')


# Maximum Likelihood Estimate fit to the synthetic data
def lnlike(theta, k, P, Perr):
    bp, beta = theta
    model = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta, b_lya=bConvert(beta,bp,mu))
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

err = 0.2
var_bp=np.abs(bp_true)*err
var_beta=np.abs(beta_true)*err
min_bp=bp_true-var_bp
max_bp=bp_true+var_bp
min_beta=beta_true-var_beta
max_beta=beta_true+var_beta


nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [bp_true, beta_true], args=(k, P, Perr))
                #, method='L-BFGS-B',bounds=[(min_b,max_b),(min_betap,max_betap)])
b_ml, betap_ml = result["x"]

result_plot = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = betaConvert(betap_ml,b_ml,mu), b_lya=b_ml)
ax.plot(k,result_plot)
fig.savefig("../Figures/IntitalFit_emcee.pdf")
#print(b_ml, betap_ml)



# Set up MLE for emcee error evaluation

def lnprior(theta):
    b, betap = theta
    if min_b < b < max_b and min_betap < betap < max_betap:
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
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr))

sampler.run_mcmc(pos, 500)
chain = sampler.chain

# Plots to visualize emcee walker paths parameter values
param1 = plt.figure(2)
plt.ylabel('bias')
for w in range(100):
    plt.plot([chain[w][s][0] for s in range(500)])
    
param1.show()
param1.savefig("../Figures/WalkerPathsBias.pdf")

param2 = plt.figure(3)
plt.ylabel('beta')
for w in range(100):
    plt.plot([betaConvert(chain[w][s][1],chain[w][s][0],mu) for s in range(500)])
    
param2.show()
param2.savefig("../Figures/WalkerPathsBeta.pdf")



samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Plot a few paths against data and intial fit
pathView = plt.figure(4)

for b, betap in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(k, th24.FluxP3D_hMpc(z24,k,mu,beta_lya = betaConvert(betap,b,mu), b_lya=b), color="k", alpha=0.1)
plt.plot(k,th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=b_true), color="r", lw=2, alpha=0.8)
plt.errorbar(k, P, yerr=Perr, fmt=".k")

plt.xscale('log')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)')
plt.title('Parameter exploration for beta, bias')

pathView.savefig("../Figures/SamplePaths.pdf")
pathView.show()

# Final results
cornerplt = corner.corner(samples, labels=["$b$", "$betap$"],
                      truths=[b_true, betap_true],quantiles=[0.16, 0.5, 0.84],show_titles=True)
cornerplt.savefig("../Figures/triangleBetaB.png")
cornerplt.show()


#samples[:, 1] = np.exp(samples[:, 1])
#b_mcmc, betap_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                             zip(*np.percentile(samples, [16, 50, 84],
#                                                axis=0)))
#print("b:", b_mcmc, "beta:", [betaConvert(betap,b_mcmc[1],mu) for betap in betap_mcmc])




