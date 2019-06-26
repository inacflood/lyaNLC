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

t = time.process_time()

headFile = "run4"
z=2.4
err = 0.5 # width of the uniform parameter priors
pos_method = 1 # emcee starts 1:from a small ball, 2:in full param space
multiT = False # when True, MCMC will be run at 3 temperatures set in 'betas'
convTest = (not multiT) and False # convergence test cannot be run with multiTempering

# Choose the "true" parameters.
#q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = getFiducialValues(z)

bp_f = 1.650
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

def lnlike(theta, k, P, Perr):
    b, bp = theta
    model = th.FluxP1D_hMpc(z, k*dkMz, b_lya=b, beta_lya=bp)*dkMz
#        q1=q1_f, q2=q2_f, kp=kp_f, kvav=kvav_f, av=av_f, bv=bv_f)
    inv_sigma2 = 1.0/(Perr**2)#+ model**2)
    return -0.5*(np.sum((P-model)**2*inv_sigma2))

#def lnlike(theta, k_res, P, Perr):
#    bp, beta = theta
#    model = th.FluxP1D_hMpc(k_res, b_lya=bConvert(bp,beta), beta_lya=betaConvert(b,bp),
#        q1=q1_f, q2=q2_f, kp=kp_f, kvav=kvav_f, av=av_f, bv=bv_f)*k_res/np.pi
#    inv_sigma2 = 1.0/(Perr**2 + model**2)
#    return -0.5*(np.sum((P-model)**2*inv_sigma2))


var_bp = np.abs(bp_f)*err
var_b = np.abs(b_f)*err
min_bp = bp_f-var_bp
max_bp = bp_f+var_bp
min_b = b_f-var_b
max_b = b_f+var_b

#nll = lambda *args: -lnlike(*args)
#result = op.minimize(nll, [bp_f, beta_f], args=(k_res, P, Perr),method='L-BFGS-B',bounds=[(min_bp,max_bp),(min_beta,max_beta)])
#bp_ml, beta_ml = result["x"]


# Set up MLE for emcee error evaluation

def lnprior(theta):
    b, bp = theta
    if min_bp < bp < max_bp and min_b < b < max_b:
        return 0.0
    return -np.inf

def lnprob(theta, k, P, Perr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, k, P, Perr)

# Set up initial positions of walkers
ndim, nwalkers = 2, 30

if multiT:
    if pos_method==1:
        pos_1 = [[b_f,bp_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        pos_2 = [[b_f,bp_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        pos_3 = [[b_f,bp_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
        pos = [pos_1,pos_2,pos_3]
    else:
        pos_1_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_1_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos_1 = [[pos_1_1[i],pos_1_2[i]] for i in range(nwalkers)]
        pos_2_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_2_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos_2 = [[pos_2_1[i],pos_2_2[i]] for i in range(nwalkers)]
        pos_3_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_3_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos_3 = [[pos_3_1[i],pos_3_2[i]] for i in range(nwalkers)]
        pos = [pos_1,pos_2,pos_3]
else:
    if pos_method==1:
        pos = [[b_f,bp_f] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    else:
        pos_1 = np.random.uniform(min_b,max_b,nwalkers)
        pos_2 = np.random.uniform(min_bp,max_bp,nwalkers)
        pos = [[pos_1[i],pos_2[i]] for i in range(nwalkers)]

# Run emcee error evaluation
nsteps=0

if convTest: # walker paths will be stored in backend and periodically checked for convergence
    filename = "test2.h5"
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
elif multiT:
    nsteps = 500
    betas = np.asarray([0.01, 0.505, 1.0]) #inverse temperatures for log-likelihood
    sampler = ptemcee.Sampler(nwalkers, ndim, lnprob, lnprior, loglargs=(k, P, Perr), betas=betas,threads=3)
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain[2][:,:,:]
else:
    nsteps = 500
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(k, P, Perr))
    sampler.run_mcmc(pos, nsteps)
    chain = sampler.chain

elapsed_time = time.process_time() - t

paramfile = open('../output/'+headFile+'/params.dat','w')
paramfile.write('{0} {1} {2} {3} {4} {5}\n'.format(str(nwalkers),str(nsteps),str(ndim),
                str(z),str(err),str(elapsed_time)))
paramfile.close()
c=chain
for w in range(nwalkers):
    file=open('../output/'+headFile+'/walk'+str(w)+'.dat','w')
    for i in range(nsteps):
        file.write('{0} {1} \n'.format(str(c[w][i][0]), str(c[w][i][1]))) #, str(c[w][i][2])))
    file.close()
