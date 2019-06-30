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

t = time.process_time()

# Make a directory to store the sampling data and parameters
headFile = "run18"
if not os.path.exists('../output/'+headFile):
    os.makedirs('../output/'+headFile)
    
z=2.4
err = 0.5 # width of the uniform parameter priors
pos_method = 2 # emcee starts 1:from a small ball, 2:in full param space
multiT = False # when True, MCMC will be run at 3 temperatures set in 'betas'
convTest = (not multiT) and True # convergence test cannot be run with multiTempering

# Choose the "true" parameters.
q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = getFiducialValues(z)
fidList = [q1_f, kp_f, kvav_f, av_f, bv_f]
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

# Maximum Likelihood Estimate fit to the synthetic data

def lnlike(theta, k, P, Perr):
    q1,kp,kvav,av,bv = theta
    model = th.FluxP1D_hMpc(z, k*dkMz, q1=q1, q2=0, kp=kp, kvav=kvav, av=av, bv=bv)*dkMz
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

min_list = [0,0,0,0,0]
max_list = [2,25,2,2,5]

# Set up MLE for emcee error evaluation

def lnprior(theta):
    q1,q2,kp,kvav,av,bv = theta
    if (min_list[0] < q1 < max_list[0] and min_list[1] < kp < max_list[1] and min_list[2] < kvav < max_list[2] 
            and min_list[3] < av < max_list[3]  and min_list[4] < bv < max_list[4]):
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

# Set up initial positions of walkers
ndim, nwalkers = 5, 500

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
        pos_14 = np.random.uniform(min_list[3],max_list[3],nwalkers)
        pos_15 = np.random.uniform(min_list[4],max_list[4],nwalkers)
        pos_1 = [[pos_11[i],pos_12[i],pos_13[i],pos_14[i],pos_15[i]] for i in range(nwalkers)]
        pos_21 = np.random.uniform(min_list[0],max_list[0],nwalkers)
        pos_22 = np.random.uniform(min_list[1],max_list[1],nwalkers)
        pos_23 = np.random.uniform(min_list[2],max_list[2],nwalkers)
        pos_24 = np.random.uniform(min_list[3],max_list[3],nwalkers)
        pos_25 = np.random.uniform(min_list[4],max_list[4],nwalkers)
        pos_2 = [[pos_21[i],pos_22[i],pos_23[i],pos_24[i],pos_25[i]] for i in range(nwalkers)]
        pos_31 = np.random.uniform(min_list[0],max_list[0],nwalkers)
        pos_32 = np.random.uniform(min_list[1],max_list[1],nwalkers)
        pos_33 = np.random.uniform(min_list[2],max_list[2],nwalkers)
        pos_34 = np.random.uniform(min_list[3],max_list[3],nwalkers)
        pos_35 = np.random.uniform(min_list[4],max_list[4],nwalkers)
        pos_3 = [[pos_31[i],pos_32[i],pos_33[i],pos_34[i],pos_35[i]] for i in range(nwalkers)]
        pos = [pos_1,pos_2,pos_3]
else:
    if pos_method==1:
        pos = [fidList + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        pos_1 = np.random.uniform(min_list[0],max_list[0],nwalkers)
        pos_2 = np.random.uniform(min_list[1],max_list[1],nwalkers)
        pos_3 = np.random.uniform(min_list[2],max_list[2],nwalkers)
        pos_4 = np.random.uniform(min_list[3],max_list[3],nwalkers)
        pos_5 = np.random.uniform(min_list[4],max_list[4],nwalkers)
        pos = [[pos_1[i],pos_2[i],pos_3[i],pos_4[i],pos_5[i]] for i in range(nwalkers)]

# Run emcee error evaluation
nsteps=0

if convTest: # walker paths will be stored in backend and periodically checked for convergence
    filename = headFile+".h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr), backend=backend)

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
    
elif multiT:
    nsteps = 1000
    betas = np.asarray([0.01, 0.505, 1.0]) #inverse temperatures for log-likelihood
    sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, loglargs=(k, P, Perr), betas=betas,threads=3)
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain[2][:,:,:]
    
else:
    nsteps = 1000
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr))
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain

elapsed_time = time.process_time() - t

# Write walker paths to files, along with the fitting parameters
paramfile = open('../output/'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(elapsed_time)))
paramfile.close()
c=chain
for w in range(nwalkers):
    file=open('../output/'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} {2} {3} {4} \n'.format(str(c[w][i][0]), str(c[w][i][1]), 
                   str(c[w][i][2]), str(c[w][i][3]), str(c[w][i][4]), str(c[w][i][5]))) 
    file.close()
