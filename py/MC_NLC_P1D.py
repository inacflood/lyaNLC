import cosmoCAMB as cCAMB
import theoryLya as tLyA
import numpy as np
from arinyo2015 import getFiducialValues
import time
import emcee
import tqdm  # leave this even if your system tells you it is unused
import get_npd_p1d as npd
import ptemcee
from ptemcee.sampler import Sampler  # leave this even if your system tells you it is unused
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
    
    parser.add_argument('--params', nargs='+', type=int, 
                        default=None, required=True, help='Parameters being tested')
    
    parser.add_argument('--pos_method',type=int,choices=[1,2],default=2,required=True,
        help='Emcee starts 1:from a small ball, 2:in full param space')
    
    parser.add_argument('--ndim',type=int,default=0,required=True,
        help='Number of parameters being fitted')
    
    parser.add_argument('--nwalkers',type=int,default=0,required=True,
        help='Number of walkers for emcee')
            
    parser.add_argument('--nsteps',type=int,default=0,required=False,
        help='Number of iterations of walkers in emcee')
    
    parser.add_argument('--pooling',type=int,default=0,required=True,
        help='Run with MPI pooling?')
    
    parser.add_argument('--err',type=float,default=0,required=False,
        help='Multiplicative half-width of the uniform parameter priors')
    
    parser.add_argument('--multiT',default=False,action='store_true',required=False,  # will be True if included in call
        help='When True, MCMC will be run at 3 temperatures set in betas')            # False otherwise
    
    parser.add_argument('--CTSwitch',default=False,action='store_true',required=False, # will be True if included in call
        help='When True, and ONLY if multiT is False, emcee will run with convergence checking')  # False otherwise

    args = parser.parse_args()
    
    headFile = args.out_dir
    z = args.z
    pos_method = args.pos_method  # emcee starts 1:from a small ball, 2:in full param space
    ndim = args.ndim
    nwalkers = args.nwalkers
    nsteps = args.nsteps
    pooling = args.pooling
    param_opt = args.params
    err = args.err                # width of the uniform parameter priors
    multiT = args.multiT          # when True, MCMC will be run at 3 temperatures set in 'betas'
    CTSwitch = args.CTSwitch
    

    # Make a directory to store the sampling data and parameters
    if not os.path.exists('../output/'+headFile):
        os.makedirs('../output/'+headFile,exist_ok=True)
        
    convTest = (not multiT) and CTSwitch # convergence test cannot be run with multiTempering

    # Choose the "true" parameters.
    fiducials = getFiducialValues(z)
    q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = fiducials
    maxes = [2,3,20,1.55,3,3]
    
    fidList = [] 
    max_list = []
    params = []
    for f in range(len(param_opt)):
        if param_opt[f]:
            fidList.append(fiducials[f])
            max_list.append(maxes[f])
            params.append(0)
        else:
            params.append(fiducials[f])
            
    fids = len(fidList)
    min_list = [0 for i in range(ndim)]
    
    if err!=0:
        max_list = [fidList[k]*(1+err) for k in range(ndim)]
        min_list = [max(0,fidList[k]*(1-err)) for k in range(ndim)]
    
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
        for f in range(len(param_opt)):
            if param_opt[f]:
                params[f] = theta[0]
                np.delete(theta,0)
        model = th.FluxP1D_hMpc(z, k*dkMz, q1=params[0], q2=params[1], kp=params[2], 
                                kvav=params[3], av=params[4], bv=params[5])*dkMz 
        inv_sigma2 = 1.0/(Perr**2)
        return -0.5*(np.sum((P-model)**2*inv_sigma2))
    
    
    # Set up MLE for emcee error evaluation
    
    def lnprior(theta):
        bound_check = [min_list[i] < theta[i] < max_list[i] for i in range(ndim)]
        if sum(bound_check)==ndim:
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
            pos=[]
            for i in range(3):
                pos_temp = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
                pos+=[pos_temp]

        else:
            pos=[]
            for i in range(3):
                pos_res=[]
                for d in range(ndim):
                    pos_temp = np.random.uniform(min_list[d],max_list[d],nwalkers)
                    pos_res += [pos_temp]
                pos_alm = [[pos_res[d][w] for d in range(ndim)] for w in range(nwalkers)]
                pos += [pos_alm]

    else:
        if pos_method==1:
            pos = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        else:
            pos_res=[]
            for d in range(ndim):
                pos_temp = np.random.uniform(min_list[d],max_list[d],nwalkers)
                pos_res += [pos_temp]
            pos = [[pos_res[d][w] for d in range(ndim)] for w in range(nwalkers)]
                 

    def run_emcee(p,nwalkers,nsteps,ndim,multiT,convTest,pos,lnprob):
        
        '''
         Run emcee error evaluation
         
        '''
        
        if convTest: # walker paths will be stored in backend and periodically checked for convergence
            filename = headFile+".h5"
            backend = emcee.backends.HDFBackend(filename)
            backend.reset(nwalkers, ndim)
        
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=p)
        
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
            maxprob=np.argmax(probs)
            hp_loc = np.unravel_index(maxprob, probs.shape)
            mle_soln = chain[(hp_loc[1],hp_loc[0])] #switching from order (nsteps,nwalkers) to (nwalkers,nsteps)
            print(mle_soln)
            return nsteps, chain, mle_soln, probs, sampler
    
    
        elif multiT:
            betas = np.asarray([0.01, 0.505, 1.0]) #inverse temperatures for log-likelihood
            sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, betas=betas, pool=p)
            sampler.run_mcmc(pos, nsteps)
            chain = sampler.chain[2][:,:,:]
            probs = sampler.logprobability[2]
            maxprob = np.argmax(probs)
            hp_loc = np.unravel_index(maxprob, probs.shape)
            mle_soln = chain[hp_loc] #already in order (nwalkers,nsteps)
            print(mle_soln)
            return nsteps, chain, mle_soln, probs, sampler
            
        else:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=p)
            sampler.run_mcmc(pos, nsteps, store=True)
            chain = sampler.chain
            probs = sampler.get_log_prob()
            maxprob=np.argmax(probs)
            hp_loc = np.unravel_index(maxprob, probs.shape)
            mle_soln = chain[(hp_loc[1],hp_loc[0])] #switching from order (nsteps,nwalkers) to (nwalkers,nsteps)
            print(mle_soln)
            return nsteps, chain, mle_soln, probs, sampler
        
    if pooling:
        
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            nsteps, chain, mle_soln, probs, sampler = run_emcee(pool,nwalkers,nsteps,
                                                ndim,multiT,convTest,pos,lnprob)
        
    else:
        nsteps, chain, mle_soln, probs, sampler = run_emcee(None,nwalkers,nsteps,
                                                ndim,multiT,convTest,pos,lnprob)
        
    sigma_arr = []
    
    if multiT:
        fc = sampler.flatchain[2]
        
    else:
        fc=sampler.flatchain
        
    for d in range(ndim):
        quantiles = np.percentile(fc[:,d], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1_1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2_1 = 0.5 * (quantiles[4] - quantiles[0])
        sigma_arr+=[sigma1_1, sigma2_1]
    
        
    elapsed_time = time.process_time() - t
    
    # Write walker paths to files, along with the fitting parameters
    paramfile = open('../output/'+headFile+'/params.dat','w')
    paramfile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                    str(z),str(err),str(elapsed_time)))
    paramfile.close()
    
    fittofile = open('../output/'+headFile+'/fitto.dat','w')
    fittofile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(param_opt[0]),str(param_opt[1]),str(param_opt[2]),
                    str(param_opt[3]),str(param_opt[4]),str(param_opt[5])))
    fittofile.close()
    
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
            if ndim==3:
               file.write('{0} {1} {2} \n'.format(str(c[w][i][0]), str(c[w][i][1]), 
                       str(c[w][i][2])))
            elif ndim==4:
                file.write('{0} {1} {2} {3} \n'.format(str(c[w][i][0]), str(c[w][i][1]), 
                       str(c[w][i][2]),str(c[w][i][3])))
            elif ndim==5:
                file.write('{0} {1} {2} {3} {4} \n'.format(str(c[w][i][0]), str(c[w][i][1]), 
                       str(c[w][i][2]),str(c[w][i][3]),str(c[w][i][4])))
            else:
                if ndim!=6:
                    print("You are varying less than 3 parameters. Your walk files will be faulty.")
                file.write('{0} {1} {2} {3} {4} {5} \n'.format(str(c[w][i][0]), str(c[w][i][1]), 
                       str(c[w][i][2]),str(c[w][i][3]),str(c[w][i][4]),str(c[w][i][5])))
        file.close()
