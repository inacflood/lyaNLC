import numpy as np
import matplotlib.pyplot as plt
import corner
from arinyo2015 import getFiducialValues
import cosmoCAMB as cCAMB
import theoryLya as tLyA
import get_npd_p1d as npd

headFile = "run2"
saveFigs = False
params3 = False
testingBB = True
P3D = False

if P3D:
    nwalkers, nsteps, ndim, z, err, mu, runtime = np.loadtxt('../output/'+headFile+'/params.dat')
    mu_str = str(int(mu*10))
    beta_f = 1.650
    b_f = -0.134
    bp_f = (1+beta_f*mu**2)*b_f
else:
    nwalkers, nsteps, ndim, z, err, lin, runtime = np.loadtxt('../output/'+headFile+'/params.dat')
    beta_f = 1.650
    b_f = -0.134
    bp_f = b_f*(1+beta_f)

nwalkers = int(nwalkers)
nsteps = int(nsteps)
ndim = int(ndim)
z_str = str(int(z*10)) # for use in file names
err_str = str(int(err*100))
linear = False
if lin==1:
    linear = True

#q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = getFiducialValues(z)

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLya(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z) #

# Get actual data
data = npd.LyA_P1D(z)
k = data.k
ii = np.where((data.k<=0.6/dkMz))[0] # Perform the cut on the data
k = data.k[ii]
P = data.Pk[ii]
Perr = data.Pk_stat[ii]
k_res = k*dkMz

data0=np.loadtxt('../output/'+headFile+'/walk0.dat')
data1=np.loadtxt('../output/'+headFile+'/walk1.dat')
chain=np.stack([data0,data1])
for w in range(nwalkers-2):
   data=np.loadtxt('../output/'+headFile+'/walk'+str(w+2)+'.dat')
   data=data.reshape((1,nsteps,ndim))
   chain=np.vstack([chain,data])

samples = chain[:, 50:, :].reshape((-1, ndim))
# Plots to visualize emcee walker paths parameter values

if testingBB:
    param1 = plt.figure(2)
    plt.ylabel('bias')
    for w in range(nwalkers):
        plt.plot([chain[w][s][0] for s in range(nsteps)])

    param1.show()
    #param1.savefig("../Figures/Kaiser/z"+z_str+"/WalkerPathsBias.pdf")

    param2 = plt.figure(3)
    plt.ylabel('bias(1+beta[*mu^2])')
    for w in range(nwalkers):
        plt.plot([chain[w][s][1] for s in range(nsteps)])

    param2.show()
    #param2.savefig("../Kaiser/z"+z_str+"/WalkerPathsBeta.pdf")

else:
    param1 = plt.figure(1)
    plt.ylabel('q1')
    for w in range(nwalkers):
        plt.plot([chain[w][s][0] for s in range(nsteps)])

    if saveFigs:
        param1.savefig("../Figures/MCMC_KaiserTests/"+headFile+"/z"+z_str+"/WalkerPathsq1_err"+err_str+".pdf")
    param1.show()

    param2 = plt.figure(2)
    plt.ylabel('bp')
    for w in range(nwalkers):
        plt.plot([chain[w][s][1] for s in range(nsteps)])

    if saveFigs:
        param2.savefig("../Figures/MCMC_KaiserTests/"+headFile+"/z"+z_str+"/WalkerPathsq2_err"+err_str+".pdf")
    param2.show()


    if params3:
        param3 = plt.figure(3)
        plt.ylabel('kp')
        for w in range(nwalkers):
            plt.plot([chain[w][s][2] for s in range(nsteps)])
        if saveFigs:
            param3.savefig("../Figures/MCMC_KaiserTests/q1_q2_kp/z"+z_str+"/WalkerPathskp_err"+err_str+"lin"+str(lin)+".pdf")
        param3.show()
if not P3D:
    pathView = plt.figure(4)

#    for b,bp in samples[np.random.randint(len(samples), size=200)]:
#        plt.plot(k, th.FluxP1D_hMpc(k_res, q1=q1_f, q2=q2_f, kvav=kvav_f, kp=kp_f, av=av_f, bv=bv_f, b_lya=b, beta_lya=(bp/b-1))*k_res/np.pi, color="k", alpha=0.1)
#    plt.plot(k,th.FluxP1D_hMpc(k_res, q1=q1_f, q2=q2_f, kvav=kvav_f, kp=kp_f, av=av_f, bv=bv_f, b_lya=b_f, beta_lya=beta_f)*k_res/np.pi, color="r", lw=2, alpha=0.8)
#    plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")

    for b,bp in samples[np.random.randint(len(samples), size=200)]:
        plt.plot(k, th.FluxP1D_hMpc(k*dkMz, z, b_lya=b, beta_lya=(bp/b-1))*k_res/np.pi, color="k", alpha=0.1)
    plt.plot(k,th.FluxP1D_hMpc(k*dkMz,z)*k_res/np.pi, color="r", lw=2, alpha=0.8)
    plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")

    plt.yscale('log')
    plt.xlabel('k [(Mpc/h)^-1]')
    plt.ylabel('P(k)*k/pi')
    plt.title('Parameter exploration for beta, bias')
    #pathView.savefig("../Figures/MCMC_KaiserTests/Kaiser/z"+z_str+"/SamplePaths_err"+err_str+"lin"+str(lin)+"posSMmtF.pdf")
    pathView.show()

# Final results
cornerplt = corner.corner(samples, labels=["$b$", "$bp$"],
                      truths=[b_f,bp_f],quantiles=[0.16, 0.5, 0.84],show_titles=True)
#if P3D:
#    cornerplt.savefig("../Figures/Kaiser/z"+z_str+"/triangle_err"+err_str+"lin"+str(lin)+"posFSmtTmu"+mu_str+".pdf")
#else:
#    cornerplt.savefig("../Figures/Kaiser/z"+z_str+"/triangle_err"+err_str+"lin"+str(lin)+"posSMmtF.pdf")
#cornerplt.show()


v1_mcmc, v2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
print("b:", v1_mcmc, "bp:", v2_mcmc)