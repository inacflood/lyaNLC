import cosmoCAMB as cCAMB
import theoryLya as tLyA
import numpy as np
import scipy.optimize as op
from arinyo2015 import getFiducialValues
import time
import emcee
import tqdm
import get_npd_p1d as npd
import ptemcee
from ptemcee.sampler import Sampler
import os
import argparse
from schwimmbad import MPIPool
import sys

t = time.process_time()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Fit the 1D power spectrum parametrized by beta and bias to NPD2013 data up to k = 0.6 h/Mpc')

    parser.add_argument('--out_dir',type=str,default=None,required=True,
        help='Output directory in output folder')
    
    parser.add_argument('--z',type=float,default=None,required=True,
        help='Redshift value')
    
    parser.add_argument('--err',type=float,default=None,required=True,
        help='Multiplicative half-width of the uniform parameter priors')
    
    parser.add_argument('--pos_method',type=int,choices=[1,2],default=2,required=True,
        help='Emcee starts 1:from a small ball, 2:in full param space')
    
    parser.add_argument('--multiT',default=False, action='store_true',required=False,  # will be True if included in call
        help='When True, MCMC will be run at 3 temperatures set in betas')            # False otherwise
    
    parser.add_argument('--CTSwitch',default=False, action='store_true',required=False, # will be True if included in call
        help='When True, and ONLY if multiT is False, emcee will run with convergence checking')  # False otherwise
    
    parser.add_argument('--linear',type=int,default=0,choices=[0,1],required=True,
        help='1D power spectrum is derived from 3D power spectrum (0) with (1) w/out NLC')
    
    parser.add_argument('--ndim',type=int,default=0,required=True,
        help='Number of parameters being fitted')
    
    parser.add_argument('--nwalkers',type=int,default=0,required=True,
        help='Number of walkers for emcee')
            
    parser.add_argument('--nsteps',type=int,default=0,required=False,
        help='Number of iterations of walkers in emcee')
    

    args = parser.parse_args()
    
    headFile = args.out_dir
    z = args.z
    err = args.err
    pos_method = args.pos_method
    multiT = args.multiT
    CTSwitch = args.CTSwitch
    linear = args.linear
    ndim = args.ndim
    nwalkers = args.nwalkers
    nsteps = args.nsteps

    if not os.path.exists('../output/'+headFile):
        os.makedirs('../output/'+headFile)
        
    convTest = (not multiT) and CTSwitch # convergence test cannot be run with multiTempering

    
    beta_f = 1.650
    b_f = -0.134
    
    cosmo = cCAMB.Cosmology(z)
    th = tLyA.TheoryLya(cosmo)
    dkMz = th.cosmo.dkms_dhMpc(z)
    
    # Get actual data
    data = npd.LyA_P1D(z)
    ii = np.where((data.k<=0.6/dkMz))[0] # Perform the cut on the data
    k = data.k[ii]
    P = data.Pk[ii]
    Perr = data.Pk_stat[ii]
    k_res = k*dkMz
    
    # Maximum Likelihood Estimate fit to the synthetic data
    
    def lnlike(theta):
        b, beta = theta
        model = th.FluxP1D_hMpc(z, k*dkMz, b_lya=b, beta_lya=beta, linear=linear)*dkMz
        inv_sigma2 = 1.0/(Perr**2)
        return -0.5*(np.sum((P-model)**2*inv_sigma2))
    
    
    var_beta = np.abs(beta_f)*err
    var_b = np.abs(b_f)*err
    min_beta = 0.4
    max_beta = 2.5
    min_b = b_f-var_b
    max_b = b_f+var_b
    
    # Set up MLE for emcee error evaluation
    
    def lnprior(theta):
        b, beta = theta
        if min_beta < beta < max_beta and min_b < b < max_b:
            return 0.0
        return -np.inf
    
    def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta)
    
    
    # Set up initial positions of walkers
    if multiT:
        if pos_method==1:
            pos_1 = [[b_f,beta_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            pos_2 = [[b_f,beta_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            pos_3 = [[b_f,beta_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            pos = [pos_1,pos_2,pos_3]
        else:
            pos_1_1 = np.random.uniform(min_b,max_b,nwalkers)
            pos_1_2 = np.random.uniform(min_beta,max_beta,nwalkers)
            pos_1 = [[pos_1_1[i],pos_1_2[i]] for i in range(nwalkers)]
            pos_2_1 = np.random.uniform(min_b,max_b,nwalkers)
            pos_2_2 = np.random.uniform(min_beta,max_beta,nwalkers)
            pos_2 = [[pos_2_1[i],pos_2_2[i]] for i in range(nwalkers)]
            pos_3_1 = np.random.uniform(min_b,max_b,nwalkers)
            pos_3_2 = np.random.uniform(min_beta,max_beta,nwalkers)
            pos_3 = [[pos_3_1[i],pos_3_2[i]] for i in range(nwalkers)]
            pos = [pos_1,pos_2,pos_3]
    else:
        if pos_method==1:
            pos = [[b_f,beta_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        else:
            pos_1 = np.random.uniform(min_b,max_b,nwalkers)
            pos_2 = np.random.uniform(min_beta,max_beta,nwalkers)
            pos = [[pos_1[i],pos_2[i]] for i in range(nwalkers)]
    
    # Run emcee error evaluation
    
    if convTest: # walker paths will be stored in backend and periodically checked for convergence
        filename = "test2.h5"
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)
    
        max_n = 10000
    
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
            # Using tol=0 means that we'll always get an estimate even if it isn't trustworthy
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
        probs = sampler.get_log_prob()
        maxprob=np.argmin(probs)
        hp_loc = np.unravel_index(maxprob, probs.shape)
        mle_soln = chain[(hp_loc[1],hp_loc[0])]
        print(mle_soln)
        
        quantiles = np.percentile(sampler.flatchain[:,0], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_1 = 0.5 * (quantiles[4] - quantiles[0])
        print("1 sigma spread", sigma1_1)
        print("2 sigma spread", sigma2_1)
        
        quantiles = np.percentile(sampler.flatchain[:,1], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_2 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_2 = 0.5 * (quantiles[4] - quantiles[0])
        print("1 sigma spread", sigma1_2)
        print("2 sigma spread", sigma2_2)
        
    elif multiT:
        betas = np.asarray([0.01, 0.505, 1.0]) #inverse temperatures for log-likelihood
        sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, betas=betas,threads=3)
        sampler.run_mcmc(pos, nsteps)
        chain = sampler.chain[2][:,:,:]
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
        sampler.run_mcmc(pos, nsteps)
        chain = sampler.chain
        probs = sampler.get_log_prob()
        
        maxprob=np.argmin(probs)
        hp_loc = np.unravel_index(maxprob, probs.shape)
        mle_soln = chain[(hp_loc[1],hp_loc[0])] #switching from order (nsteps,nwalkers) to (nwalkers,nsteps)
        print(mle_soln)
        
        quantiles = np.percentile(sampler.flatchain[:,0], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_1 = 0.5 * (quantiles[4] - quantiles[0])
        print("1 sigma spread", sigma1_1)
        print("2 sigma spread", sigma2_1)
        
        quantiles = np.percentile(sampler.flatchain[:,1], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_2 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_2 = 0.5 * (quantiles[4] - quantiles[0])
        print("1 sigma spread", sigma1_2)
        print("2 sigma spread", sigma2_2)
    
    elapsed_time = time.process_time() - t
    
    # Write walker paths to files, along with the fitting parameters
    paramfile = open('../output/'+headFile+'/params.dat','w')
    paramfile.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                    str(z),str(err),str(linear),str(elapsed_time)))
    paramfile.close()
    
    resultsfile = open('../output/'+headFile+'/results.dat','w')
    resultsfile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(mle_soln[0]),str(sigma1_1),str(sigma2_1),
                    str(mle_soln[1]),str(sigma1_2),str(sigma2_2)))
    resultsfile.close()
    
    c=chain
    for w in range(nwalkers):
        file=open('../output/'+headFile+'/walk'+str(w)+'.dat','w')
        for i in range(nsteps):
            file.write('{0} {1} \n'.format(str(c[w][i][0]), str(c[w][i][1]))) 
        file.close()
        
    file=open('../output/'+headFile+'/logprob.dat','w')   
    for s in range(nsteps):
        for w in range(nwalkers):
            file.write(str(probs[s][w])+' ')
        file.write('\n')
    file.close()
