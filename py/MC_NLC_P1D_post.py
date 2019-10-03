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
    
    parser.add_argument('--in_dir',type=str,default=None,required=True,
        help='Input directory to get starting position from')
    
    parser.add_argument('--pooling',type=int,default=0,required=True,
        help='Run with MPI pooling?')
    
    parser.add_argument('--CTSwitch',default=False,action='store_true',required=False, # will be True if included in call
        help='When True, and ONLY if multiT is False, emcee will run with convergence checking')  # False otherwise
    
    parser.add_argument('--multiT',default=False,action='store_true',required=False,  # will be True if included in call
        help='When True, MCMC will be run at 3 temperatures set in betas')   
            
    parser.add_argument('--nsteps',type=int,default=0,required=False,
        help='Number of iterations of walkers in emcee')
    

    args = parser.parse_args()
    
    ###################################################
    ## Set up parameters and model for emcee fitting ##
    ###################################################
    
    headFile = args.out_dir
    inFile = args.in_dir
    pooling = args.pooling        
    CTSwitch = args.CTSwitch
    multiT = args.multiT          
    nsteps = int(args.nsteps)
    
    # Retrieve parameters from previous emcee fitting to be extended
    nwalkers, nst, ndim, z, err, runtime = np.loadtxt('../output/'+inFile+'/params.dat')
    
    beta_f = 1.650
    b_f = -0.134
    
    nwalkers = int(nwalkers)
    ndim = int(ndim)
    z_str = str(int(z*10)) # for use in file names
    err_str = str(int(err*100))
    nst = int(nst)

    # Make a directory to store the sampling data and parameters
    if not os.path.exists('../output/'+headFile):
        os.makedirs('../output/'+headFile,exist_ok=True)
        
    convTest = (not multiT) and CTSwitch # convergence test cannot be run with multiTempering
    
    # Retrieve the parameters being fitted and copy file to this directory
    param_opt = np.loadtxt('../output/'+inFile+'/fitto.dat')
    param_opt = [int(param) for param in param_opt]
    os.system('cp ../output/'+inFile+'/fitto.dat ../output/'+headFile+'/fitto.dat')

    # Set the fiducial parameters and the uniform prior bounds (with min_list, max_list)
    fiducials = getFiducialValues(z)
    q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = fiducials
    
    fidList = [] #list of the fiducial values for the parameters being varied
    params = [] #list with the fiducial values for the fixed parameters and 0s 
                #in the positions of the parameters being varied
    for f in range(len(param_opt)):
        if param_opt[f]:
            fidList.append(fiducials[f])
            params.append(0)
        else:
            params.append(fiducials[f])
            
    fids = len(fidList)
    
    # Set up the 1D power spectrum model
    cosmo = cCAMB.Cosmology(z)
    th = tLyA.TheoryLya(cosmo)
    dkMz = th.cosmo.dkms_dhMpc(z)
    
    # Retrieve data for 1D power spectrum
    data = npd.LyA_P1D(z)
    k = data.k
    P = data.Pk
    Perr = data.Pk_stat
    k_res = k*dkMz
    
    # Get previous chain
    data0=np.loadtxt('../output/'+inFile+'/walk0.dat')
    data1=np.loadtxt('../output/'+inFile+'/walk1.dat')
    chain=np.stack([data0,data1])
    for w in range(nwalkers-2):
        data=np.loadtxt('../output/'+inFile+'/walk'+str(w+2)+'.dat')
        data=data.reshape((1,nst,ndim))
        chain=np.vstack([chain,data])
        
    # Get best fit values and uncertainties
    samples = chain[:, 50:, :].reshape((-1, ndim))
    v1, v2, v3 = map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(samples, [16, 50, 84],
                                            axis=0)))
    results = np.array([v1, v2, v3])
    min_list = [max(results[k][2],0) for k in range(ndim)]
    max_list = results[:,1]
    
    ########################################################
    ## Set up MLE and initial positions for emcee walkers ##
    ########################################################
        
    # Set up log likelihood function for emcee
    
    def lnlike(theta):
        for f in range(len(param_opt)):
            if param_opt[f]:
                params[f] = theta[0]
                np.delete(theta,0)
        model = th.FluxP1D_hMpc(z, k*dkMz, q1=params[0], q2=params[1], kp=params[2], 
                                kvav=params[3], av=params[4], bv=params[5])*dkMz 
        inv_sigma2 = 1.0/(Perr**2)
        return -0.5*(np.sum((P-model)**2*inv_sigma2))
    
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
    pos = chain[:,-1,:] 
    
    ##########################
    ## Function to run MCMC ##
    ##########################
    
    def run_emcee(p,nwalkers,nsteps,ndim,multiT,convTest,pos,lnprob):
        
        """
         Run MCMC with:
             Number of walkers = nwalkers
             Number of dimensions = ndim
             Number of steps = nsteps
             Log probability function = lnprob
             Pool = p
             Initial walker positions = pos
             
        If multiT is true, MCMC will be run at 3 different temperatures (inverses 
        given by betas). If convTest is true, MCMC will either run until 
        convergence or for nsteps steps, whichever happens first.
        
        """
        
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
                c = sample.coords
            
                # Write to file
                for w in range(nwalkers):
                    file=open('../output/'+headFile+'/walk'+str(w)+'.dat','a')
                    if ndim==3:
                       file.write('{0} {1} {2} \n'.format(str(c[w][0]), str(c[w][1]), 
                               str(c[w][2])))
                    elif ndim==4:
                        file.write('{0} {1} {2} {3} \n'.format(str(c[w][0]), str(c[w][1]), 
                               str(c[w][2]),str(c[w][3])))
                    elif ndim==5:
                        file.write('{0} {1} {2} {3} {4} \n'.format(str(c[w][0]), str(c[w][1]), 
                               str(c[w][2]),str(c[w][3]),str(c[w][4])))
                    else:
                        if ndim!=6:
                            print("You are varying less than 3 parameters. Your walk files will be faulty.")
                        file.write('{0} {1} {2} {3} {4} {5} \n'.format(str(c[w][0]), str(c[w][1]), 
                               str(c[w][2]),str(c[w][3]),str(c[w][4]),str(c[w][5]))) 
                    file.close()
                    
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
            # find mle_soln, the walker position with the maximum probability
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
            # find mle_soln, the walker position with the maximum probability
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
            # find mle_soln, the walker position with the maximum probability
            chain = sampler.chain
            probs = sampler.get_log_prob()
            maxprob=np.argmax(probs)
            hp_loc = np.unravel_index(maxprob, probs.shape)
            mle_soln = chain[(hp_loc[1],hp_loc[0])] #switching from order (nsteps,nwalkers) to (nwalkers,nsteps)
            print(mle_soln)
            return nsteps, chain, mle_soln, probs, sampler
        
    ##############
    ## Run MCMC ##
    ##############
        
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
        
    # Retrieve flatchain and sigma values from fitting
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
    
    ############################
    ## Write results to files ##
    ############################
    
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
            file.write(str(probs[s][w])+' ')
        file.write('\n')
    file.close()
    
    if not CTSwitch:
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
            

        
        
        
