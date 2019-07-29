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
            
    parser.add_argument('--nsteps',type=int,default=0,required=False,
        help='Number of iterations of walkers in emcee')
    

    args = parser.parse_args()
    
    headFile = args.out_dir
    inFile = args.in_dir        
    #CTSwitch = args.CTSwitch
    nsteps = int(args.nsteps)
    
    nwalkers, nst, ndim, z, err, runtime = np.loadtxt('../output/'+inFile+'/params.dat')
    beta_f=1.650
    b_f = -0.134
    
    nwalkers = int(nwalkers)
    ndim = int(ndim)
    z_str = str(int(z*10)) # for use in file names
    err_str = str(int(err*100))
    nst = int(nst)

    # Make a directory to store the sampling data and parameters
    if not os.path.exists('../output/'+headFile):
        os.makedirs('../output/'+headFile,exist_ok=True)
        
    
    # Retrieve the parameters being fitted and copy file to this directory
#    param_opt = np.loadtxt('../output/'+inFile+'/fitto.dat')
#    param_opt = [int(param) for param in param_opt]
#    os.system('cp ../output/'+inFile+'/fitto.dat ../output/'+headFile+'/fitto.dat')
    param_opt = [0,0,1,1,0,1]

    # Choose the "true" parameters.
    fiducials = getFiducialValues(z)
    q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = fiducials
    
    fidList = [] 
    params = []
    for f in range(len(param_opt)):
        if param_opt[f]:
            fidList.append(fiducials[f])
            params.append(0)
        else:
            params.append(fiducials[f])
            
    fids = len(fidList)
    
    cosmo = cCAMB.Cosmology(z)
    th = tLyA.TheoryLya(cosmo)
    dkMz = th.cosmo.dkms_dhMpc(z)
    
    # Get actual data
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
    results=np.loadtxt('../output/'+inFile+'/corner.dat')
    min_list = [max(results[k][2],0) for k in range(ndim)]
    max_list = [20,results[1,1],3]
        
    # Maximum Likelihood Estimate fit to the synthetic data
    
    def lnlike(theta):
        for f in range(len(param_opt)):
            if not param_opt[f]:
                params[f] = theta[0]
                np.delete(theta,0)
        model = th.FluxP1D_hMpc(z, k*dkMz, q1=params[0], q2=params[1], kp=params[2], 
                                kvav=params[3], av=params[4], bv=params[5])*dkMz 
        inv_sigma2 = 1.0/(Perr**2)
        return -0.5*(np.sum((P-model)**2*inv_sigma2))
    
    
    # Set up MLE for emcee error evaluation
    
    def lnprior(theta):
        var1,var2,var3 = theta
        if (min_list[0] < var1 < max_list[0] and min_list[1] < var2 < max_list[1] and min_list[2] < var3 < max_list[2]):
            return 0.0
        return -np.inf
    
    def lnprob(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta)
        
#    with MPIPool() as pool:
#        if not pool.is_master():
#            pool.wait()
#            sys.exit(0)
    
    # Set up initial positions of walkers
    pos = chain[:,-1,:]
        
    
    # Run emcee error evaluation
    sigma_arr = []
    
    betas = np.asarray([0.01, 0.505, 1.0]) #inverse temperatures for log-likelihood
    sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, betas=betas)#, pool=pool)
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain[2][:,:,:]
    probs = sampler.logprobability[2]
    maxprob=np.argmin(probs)
    hp_loc = np.unravel_index(maxprob, probs.shape)
    mle_soln = chain[hp_loc] #already in order (nwalkers,nsteps)
    print(mle_soln)
    
    for i in range(ndim): 
        quantiles = np.percentile(sampler.flatchain[:,i], [2.28, 15.9, 50, 84.2, 97.7])
        sigma1 = 0.5 * (quantiles[3] - quantiles[1])
        sigma2 = 0.5 * (quantiles[4] - quantiles[0])
        sigma_arr+=[sigma1, sigma2]

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
        
    fittofile = open('../output/'+headFile+'/fitto.dat','w')
    fittofile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(param_opt[0]),str(param_opt[1]),str(param_opt[2]),
                    str(param_opt[3]),str(param_opt[4]),str(param_opt[5])))
    fittofile.close()
        

        
        
        
