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
b_true = -0.134
beta_true = 1.650
mu=1.0
betap_true=b_true*(1+beta_true*mu)


z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tP3D.TheoryLyaP3D(cosmo24)

# Generate some synthetic data from the model.
N = 100
k = np.sort(np.logspace(-4,0.9,N))
Perr = np.random.rand(N)
P = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=b_true)
P += Perr * np.random.randn(N)

# Plot our synthetic data along with fit from MLE
fig = plt.figure(1)
ax = fig.add_subplot()
plt.xscale('log')
plt.yscale('linear')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)')

ax.errorbar(k,P,yerr=Perr,fmt='k.')

# Function to convert our modified beta fitting variable to beta

def betaConvert(betap,b,mu):
    beta_red=betap/b-1
    return beta_red/mu

#Maximum Likelihood Estimate fit to the synthetic data
def lnlike(theta, k, P, Perr):
    b = theta
    model = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=b)
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2 - np.log(inv_sigma2)))

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [b_true], args=(k, P, Perr))
[b_ml]= result["x"]

result_plot = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=b_ml)
ax.plot(k,result_plot)
fig.savefig("../Figures/IntitalFit_emcee_justbias.pdf")

#print(b_ml, betap_ml)



# Set up MLE for emcee error evaluation

def lnprior(theta):
    b = theta
    var_b=np.abs(b_ml)*.2
    #var_betap=np.abs(betap_ml)*.2
    if b_ml-var_b < b < b_ml+var_b: 
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

ndim, nwalkers = 1, 100
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
param1.savefig("../Figures/WalkerPathsBias_justbias.pdf")

#param2 = plt.figure(3)
#plt.ylabel('beta')
#for w in range(100):
#    plt.plot([betaConvert(chain[w][s][1],chain[w][s][0],mu) for s in range(500)])
#    
#param2.show()
#param2.savefig("../Figures/WalkerPathsBeta.pdf")

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=["$b$"],
                      truths=[b_true])
fig.savefig("../Figures/triangleB_justbias.png")


# Plot a few paths against data and intial fit
plt.xscale('log')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)')
plt.title('Parameter exploration for  bias')

#xl = np.array([0, 10])
for b in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(k, th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=b), color="k", alpha=0.1)
plt.plot(k,th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_true, b_lya=b_true), color="r", lw=2, alpha=0.8)
plt.errorbar(k, P, yerr=Perr, fmt=".k")

plt.savefig("../Figures/SamplePaths_justbias.pdf")

# Final results
samples[:, 0] = np.exp(samples[:, 0])
v=np.percentile(samples, [16, 50, 84],axis=0)
b_mcmc= (v[1][0], v[2][0]-v[1][0], v[1][0]-v[0][0])
print("b:", b_mcmc)



