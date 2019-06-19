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
#from scipy.stats import norm

t = time.process_time()

headFile = "BetaBiasPlay3D"
z=2.2
err = 0.5 # width of the uniform parameter priors
z_str=str(int(z*10)) # for use in file names
err_str = str(int(err*100))

# Choose the "true" parameters.
bp_true = -0.321 
beta_true = 1.656
mu=1.0

cosmo = cCAMB.Cosmology(z)
th = tP3D.TheoryLyaP3D(cosmo)

def bConvert(beta,bp,mu):
    """
    Function to convert our modified beta fitting variable to beta
    """
    return bp/(1+beta*mu**2)

# Generate some synthetic data from the model.
N = 100
k = np.sort(np.logspace(-4,0.9,N))
Perr = np.random.rand(N)
P = th.FluxP3D_hMpc(z,k,mu,beta_lya = beta_true, b_lya=bConvert(beta_true,bp_true,mu))
P += Perr * np.random.randn(N)

# Plot our synthetic data along with fit from MLE
#fig = plt.figure(1)
#ax = fig.add_subplot()
#plt.xscale('log')
#plt.yscale('linear')
#plt.xlabel('k [(Mpc/h)^-1]')
#plt.ylabel('P(k)')
#
#ax.errorbar(k,P,yerr=Perr,fmt='k.')


# Maximum Likelihood Estimate fit to the synthetic data
def lnlike(theta, k, P, Perr):
    bp, beta = theta
    model = th.FluxP3D_hMpc(z,k,mu,beta_lya = beta, b_lya=bConvert(beta,bp,mu))
    inv_sigma2 = 1.0/(Perr**2 + model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

err = 0.5
var_bp=np.abs(bp_true)*err
var_beta=np.abs(beta_true)*err
min_bp=bp_true-var_bp
max_bp=bp_true+var_bp
min_beta=beta_true-var_beta
max_beta=beta_true+var_beta


nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [bp_true, beta_true], args=(k, P, Perr))
                #, method='L-BFGS-B',bounds=[(min_b,max_b),(min_betap,max_betap)])
bp_ml, beta_ml = result["x"]
#
#result_plot = th24.FluxP3D_hMpc(z24,k,mu,beta_lya = beta_ml, b_lya=bConvert(beta_ml, bp_ml,mu))
#ax.plot(k,result_plot)
#fig.savefig("../Figures/IntitalFit_emcee.pdf")
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

ndim, nwalkers = 2, 300
#pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
pos_1 = np.random.uniform(min_bp,max_bp,nwalkers)
pos_2 = np.random.uniform(min_beta,max_beta,nwalkers)
pos = [[pos_1[i],pos_2[i]] for i in range(nwalkers)]

# Run emcee error evaluation

filename = "test2.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr),threads=4, backend=backend)

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

elapsed_time = time.process_time() - t

paramfile = open('../'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(0.0),str(elapsed_time)))
paramfile.close()
c=sampler.chain
for w in range(nwalkers):
    file=open('../'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} \n'.format(str(c[w][i][0]), str(c[w][i][1]))) #, str(c[w][i][2])))
    file.close()

# Plots to visualize emcee walker paths parameter values
#param1 = plt.figure(2)
#plt.ylabel('bias(1+beta*mu^2)')
#for w in range(nwalkers):
#    plt.plot([chain[w][s][0] for s in range(nsteps)])
#    
#param1.show()
##param1.savefig("../Figures/WalkerPathsBias.pdf")
#
#param2 = plt.figure(3)
#plt.ylabel('beta')
#for w in range(nwalkers):
#    plt.plot([chain[w][s][1] for s in range(nsteps)])
#    
#param2.show()
#param2.savefig("../Figures/WalkerPathsBeta.pdf")



#samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# Plot a few paths against data and intial fit
#pathView = plt.figure(4)
#
#for bp, beta in samples[np.random.randint(len(samples), size=100)]:
#    plt.plot(k, th.FluxP3D_hMpc(z,k,mu,beta_lya = beta, b_lya=bConvert(beta,bp,mu)), color="k", alpha=0.1)
#plt.plot(k,th.FluxP3D_hMpc(z,k,mu,beta_lya = beta_true, b_lya=bConvert(beta_true,bp_true,mu)), color="r", lw=2, alpha=0.8)
#plt.errorbar(k, P, yerr=Perr, fmt=".k")
#
#plt.xscale('log')
#plt.xlabel('k [(Mpc/h)^-1]')
#plt.ylabel('P(k)')
#plt.title('Parameter exploration for beta, bias')

#pathView.savefig("../Figures/SamplePaths.pdf")
#pathView.show()
#
## Final results
#cornerplt = corner.corner(samples, labels=["$bp$", "$beta$"],
#                      truths=[bp_true, beta_true],quantiles=[0.16, 0.5, 0.84],show_titles=True)
##cornerplt.savefig("../Figures/triangleBetaB.png")
#cornerplt.show()
#

#samples[:, 1] = np.exp(samples[:, 1])
#b_mcmc, betap_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                             zip(*np.percentile(samples, [16, 50, 84],
#                                                axis=0)))
#print("b:", b_mcmc, "beta:", [betaConvert(betap,b_mcmc[1],mu) for betap in betap_mcmc])




