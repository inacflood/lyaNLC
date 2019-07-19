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
    
    parser.add_argument('--err',type=float,default=0,required=False,
        help='Multiplicative half-width of the uniform parameter priors')
    
    parser.add_argument('--pos_method',type=int,choices=[1,2],default=2,required=True,
        help='Emcee starts 1:from a small ball, 2:in full param space')
    
    parser.add_argument('--multiT',default=False,action='store_true',required=False,  # will be True if included in call
        help='When True, MCMC will be run at 3 temperatures set in betas')            # False otherwise
    
    parser.add_argument('--CTSwitch',default=False,action='store_true',required=False, # will be True if included in call
        help='When True, and ONLY if multiT is False, emcee will run with convergence checking')  # False otherwise
    
    parser.add_argument('--ndim',type=int,default=0,required=True,
        help='Number of parameters being fitted')
    
    parser.add_argument('--nwalkers',type=int,default=0,required=True,
        help='Number of walkers for emcee')
            
    parser.add_argument('--nsteps',type=int,default=0,required=False,
        help='Number of iterations of walkers in emcee')
    

    args = parser.parse_args()
    
    headFile = args.out_dir
    z = args.z
    err = args.err                # width of the uniform parameter priors
    pos_method = args.pos_method  # emcee starts 1:from a small ball, 2:in full param space
    multiT = args.multiT          # when True, MCMC will be run at 3 temperatures set in 'betas'
    CTSwitch = args.CTSwitch
    ndim = args.ndim
    nwalkers = args.nwalkers
    nsteps = args.nsteps

    # Make a directory to store the sampling data and parameters
    if not os.path.exists('../output/'+headFile):
        os.makedirs('../output/'+headFile)
        
    convTest = (not multiT) and CTSwitch # convergence test cannot be run with multiTempering

    # Choose the "true" parameters.
    q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = getFiducialValues(z)
    fidList = [q1_f, q2_f, av_f]
    fids = len(fidList)
    
    #q1_e = 0.46008
    
    cosmo = cCAMB.Cosmology(z)
    th = tLyA.TheoryLya(cosmo)
    dkMz = th.cosmo.dkms_dhMpc(z)
    
    # Get actual data
    data = npd.LyA_P1D(z)
    k = data.k
    P = data.Pk
    Perr = data.Pk_stat
    k_res = k*dkMz
    
    # Maximum Likelihood Estimate fit to the synthetic data
    
    def lnlike(theta):
        q1,q2,av = theta
        model = th.FluxP1D_hMpc(z, k*dkMz, q1=q1, q2=q2, kp=kp_f, kvav=kvav_f, av=av, bv=bv_f)*dkMz
        inv_sigma2 = 1.0/(Perr**2)
        return -0.5*(np.sum((P-model)**2*inv_sigma2))
    
    #var_list = np.zeros(fids)
    #min_list = np.zeros(fids)
    #max_list = np.zeros(fids)
    #
    #for num in range(fids):
    #    fid_val = fidList[num]
    #    var = np.abs(fid_val)*err
    #    min_list[num] = fid_val - var
    #    max_list[num] = fid_val + var
    #    var_list[num] = var
    
    min_list = [0,0,0]
    max_list = [2,3,2]
    
    # Set up MLE for emcee error evaluation
    
    def lnprior(theta):
        q1,q2,av = theta
        if (min_list[0] < q1 < max_list[0] and min_list[1] < q2 < max_list[1] and min_list[2] < av < max_list[2]):
            return 0.0
        return -np.inf
    
    def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta)
    
    # Set up initial positions of walkers
    
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    
        if multiT:
            if pos_method==1:
                pos_1 = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
                pos_2 = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
                pos_3 = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
                pos = [pos_1,pos_2,pos_3]
            else:
                pos_11 = np.random.uniform(min_list[0],max_list[0],nwalkers)
                pos_12 = np.random.uniform(min_list[1],max_list[1],nwalkers)
                pos_13 = np.random.uniform(min_list[2],max_list[2],nwalkers)
                pos_1 = [[pos_11[i],pos_12[i],pos_13[i]] for i in range(nwalkers)]
                
                pos_21 = np.random.uniform(min_list[0],max_list[0],nwalkers)
                pos_22 = np.random.uniform(min_list[1],max_list[1],nwalkers)
                pos_23 = np.random.uniform(min_list[2],max_list[2],nwalkers)
                pos_2 = [[pos_21[i],pos_22[i],pos_23[i]] for i in range(nwalkers)]
                
                pos_31 = np.random.uniform(min_list[0],max_list[0],nwalkers)
                pos_32 = np.random.uniform(min_list[1],max_list[1],nwalkers)
                pos_33 = np.random.uniform(min_list[2],max_list[2],nwalkers)
                pos_3 = [[pos_31[i],pos_32[i],pos_33[i]] for i in range(nwalkers)]
                pos = [pos_1,pos_2,pos_3]
        else:
            if pos_method==1:
                pos = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
            else:
                pos_1 = np.random.uniform(min_list[0],max_list[0],nwalkers)
                pos_2 = np.random.uniform(min_list[1],max_list[1],nwalkers)
                pos_3 = np.random.uniform(min_list[2],max_list[2],nwalkers)
                pos = [[pos_1[i],pos_2[i],pos_3[i]] for i in range(nwalkers)]
        
        # Run emcee error evaluation
        sigma_arr = []
        
        if convTest: # walker paths will be stored in backend and periodically checked for convergence
            filename = headFile+".h5"
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)
        
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool)
        
            max_n = nsteps
        
            #sampler.run_mcmc(pos, 500)
            # We'll track how the average autocorrelation time estimate changes
            index = 0
            autocorr = np.empty(max_n)
        
            old_tau = np.inf
        
            # Now we'll sample for up to max_n steps
            for sample in sampler.sample(pos, store=True, iterations=max_n, progress=True):
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
            mle_soln = chain[(hp_loc[1],hp_loc[0])] #switching from order (nsteps,nwalkers) to (nwalkers,nsteps)
            print(mle_soln)
    
    
        elif multiT:
            betas = np.asarray([0.01, 0.505, 1.0]) #inverse temperatures for log-likelihood
            sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, betas=betas,pool=pool)
            sampler.run_mcmc(pos, nsteps)
            chain = sampler.chain[2][:,:,:]
            probs = sampler.logprobability[2]
            maxprob=np.argmin(probs)
            hp_loc = np.unravel_index(maxprob, probs.shape)
            mle_soln = chain[hp_loc] #already in order (nwalkers,nsteps)
            print(mle_soln)
            
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,pool=pool)
            sampler.run_mcmc(pos, nsteps, store=True)
            chain = sampler.chain
            probs = sampler.get_log_prob()
            maxprob=np.argmin(probs)
            hp_loc = np.unravel_index(maxprob, probs.shape)
            mle_soln = chain[(hp_loc[1],hp_loc[0])] #switching from order (nsteps,nwalkers) to (nwalkers,nsteps)
            print(mle_soln)
            
        if multiT:
            fc = sampler.flatchain[2]
        else:
            fc=sampler.flatchain
        
        quantiles = np.percentile(fc[:,0], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_1 = 0.5 * (quantiles[4] - quantiles[0])
        sigma_arr+=[sigma1_1, sigma2_1]
        
        quantiles = np.percentile(fc[:,1], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_2 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_2 = 0.5 * (quantiles[4] - quantiles[0])
        sigma_arr+=[sigma1_2, sigma2_2]
        
        quantiles = np.percentile(fc[:,2], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_3 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_3 = 0.5 * (quantiles[4] - quantiles[0])
        sigma_arr+=[sigma1_3, sigma2_3]
    
    elapsed_time = time.process_time() - t
    
    # Write walker paths to files, along with the fitting parameters
    paramfile = open('../output/'+headFile+'/params.dat','w')
    paramfile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                    str(z),str(err),str(elapsed_time)))
    paramfile.close()
    
    resultsfile = open('../output/'+headFile+'/results.dat','w')
    for d in range(ndim):
        resultsfile.write('{0} {1} {2} \n'.format(str(mle_soln[d]), str(sigma_arr[2*d]), str(sigma_arr[2*d+1])))
    resultsfile.close()
    
    file=open('../output/'+headFile+'/logprob.dat','w')   
    for s in range(nsteps):
        for w in range(nwalkers):
            if multiT:
                file.write(str(probs[w][s])+' ')
            else:
                file.write(str(probs[s][w])+' ')
        file.write('\n')
    file.close()
    
    c=chain
    for w in range(nwalkers):
        file=open('../output/'+headFile+'/walk'+str(w)+'.dat','w')
        for i in range(nsteps):
            file.write('{0} {1} {2} \n'.format(str(c[w][i][0]), str(c[w][i][1]), 
                       str(c[w][i][2]))) 
        file.close()       