import numpy as np
import matplotlib.pyplot as plt
import corner
from arinyo2015 import getFiducialValues
import cosmoCAMB as cCAMB
import theoryLya as tLyA
import get_npd_p1d as npd

headFile = "run5"
saveFigs = True
testingBB = False
P3D = False

if P3D:
    nwalkers, nsteps, ndim, z, err, mu, runtime = np.loadtxt('../output/'+headFile+'/params.dat')
    mu_str = str(int(mu*10))
    beta_f = 1.650
    b_f = -0.134
else:
    nwalkers, nsteps, ndim, z, err, runtime = np.loadtxt('../output/'+headFile+'/params.dat')
#    beta_f = 1.650
#    b_f = -0.134
#    bp_f = b_f*(1+beta_f)
    
    beta_f=1.650
    b_f = -0.134

nwalkers = int(nwalkers)
nsteps = int(nsteps)
ndim = int(ndim)
z_str = str(int(z*10)) # for use in file names
err_str = str(int(err*100))

q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = getFiducialValues(z)

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLya(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z) #

# Get actual data
if testingBB:
    
    data = npd.LyA_P1D(z)
    k = data.k
    ii = np.where((data.k<=0.6/dkMz))[0] # Perform the cut on the data
    k = data.k[ii]
    P = data.Pk[ii]
    Perr = data.Pk_stat[ii]
    k_res = k*dkMz
    
else:
    
    data = npd.LyA_P1D(z)
    k = data.k
    P = data.Pk
    Perr = data.Pk_stat
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
    #param1.savefig("../Figures/WalkerPathsBias.pdf")

    param2 = plt.figure(3)
    plt.ylabel('beta')
    for w in range(nwalkers):
        plt.plot([chain[w][s][1] for s in range(nsteps)])

    param2.show()
    #param2.savefig("../Figures/WalkerPathsBeta.pdf")

else:
    param1 = plt.figure(1)
    plt.ylabel('q1')
    for w in range(nwalkers):
        plt.plot([chain[w][s][0] for s in range(nsteps)])

    if saveFigs:
        param1.savefig("../output/"+headFile+"/z"+z_str+"WalkerPathsq1_err"+err_str+".pdf")
    param1.show()

    param2 = plt.figure(2)
    plt.ylabel('q2')
    for w in range(nwalkers):
        plt.plot([chain[w][s][1] for s in range(nsteps)])

    if saveFigs:
        param2.savefig("../output/"+headFile+"/z"+z_str+"WalkerPathsq2_err"+err_str+".pdf")
    param2.show()

    param3 = plt.figure(3)
    plt.ylabel('kp')
    for w in range(nwalkers):
        plt.plot([chain[w][s][2] for s in range(nsteps)])
    if saveFigs:
        param3.savefig("../output/"+headFile+"/z"+z_str+"WalkerPathskp_err"+err_str+".pdf")
    param3.show()
    
    param4 = plt.figure(4)
    plt.ylabel('kp')
    for w in range(nwalkers):
        plt.plot([chain[w][s][3] for s in range(nsteps)])
    if saveFigs:
        param3.savefig("../output/"+headFile+"/z"+z_str+"WalkerPathskvav_err"+err_str+".pdf")
    param3.show()
    
    param5 = plt.figure(5)
    plt.ylabel('kp')
    for w in range(nwalkers):
        plt.plot([chain[w][s][4] for s in range(nsteps)])
    if saveFigs:
        param3.savefig("../output/"+headFile+"/z"+z_str+"WalkerPathsav_err"+err_str+".pdf")
    param3.show()
    
    param6 = plt.figure(6)
    plt.ylabel('kp')
    for w in range(nwalkers):
        plt.plot([chain[w][s][5] for s in range(nsteps)])
    if saveFigs:
        param3.savefig("../output/"+headFile+"/z"+z_str+"WalkerPathsbv_err"+err_str+".pdf")
    param3.show()
    
if testingBB and (not P3D):
    pathView = plt.figure(4)
    for b,beta in samples[np.random.randint(len(samples), size=200)]:
        plt.plot(k, th.FluxP1D_hMpc(z, k*dkMz, b_lya=b, beta_lya=bp)*k_res/np.pi, color="b", alpha=0.1)
    plt.plot(k,th.FluxP1D_hMpc(z, k*dkMz)*k_res/np.pi, color="r", lw=2, alpha=0.8)
    plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")

    plt.yscale('log')
    plt.xlabel('k [(Mpc/h)^-1]')
    plt.ylabel('P(k)*k/pi')
    plt.title('Parameter exploration for beta, bias')
    pathView.savefig("../output/"+headFile+"/SamplePaths_err"+err_str+"posSMmtF.pdf")
    pathView.show()
    
if not testingBB:
    
    pathView = plt.figure(7)
    
    for q1,q2,kp,kvav,av,bv in samples[np.random.randint(len(samples), size=200)]:
        plt.plot(k, th.FluxP1D_hMpc(z, k*dkMz, q1=q1, q2=q2, kp=kp, kvav=kvav, av=av, bv=bv)*k_res/np.pi
                 , color="b", alpha=0.1)
    plt.plot(k,th.FluxP1D_hMpc(z, k*dkMz, q1=q1_f,q2=q2_f,kp=kp_f,kvav=kvav_f,av=av_f,bv=bv_f)*k_res/np.pi
             , color="r", lw=2, alpha=0.8)
    plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")

    plt.yscale('log')
    plt.xlabel('k [(Mpc/h)^-1]')
    plt.ylabel('P(k)*k/pi')
    plt.title('Parameter exploration for beta, bias')
    
    pathView.savefig("../output/"+headFile+"/SamplePaths_err"+err_str+"posSMmtF.pdf")
    pathView.show()
    

# Final results
if testingBB:
    
    cornerplt = corner.corner(samples, labels=["$b$", "$beta$"],
                          truths=[b_f,beta_f],quantiles=[0.16, 0.5, 0.84],show_titles=True)
    
    if P3D:
        cornerplt.savefig("../output"+headFile+"/triangle_err"+err_str+"posFSmtTmu"+mu_str+".pdf")
    else:
        cornerplt.savefig("../output/"+headFile+"/triangle_err"+err_str+"posFSmtT.pdf")
    cornerplt.show()
    
    v1_mcmc, v2_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                            zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
    print("b:", v1_mcmc, "beta:", v2_mcmc)
    
else:
    
   cornerplt = corner.corner(samples, labels=["$q1$", "$q2$", "$kp$", "$kvav$", "$av$", "$bv$"],
                truths=[q1_f,q2_f,kp_f,kvav_f,av_f,bv_f],quantiles=[0.16, 0.5, 0.84],show_titles=True)
   
   cornerplt.savefig("../output/"+headFile+"/triangle_err"+err_str+"posFSmtT.pdf")
   cornerplt.show()
   v1_mcmc, v2_mcmc, v3_mcmc, v4_mcmc, v5_mcmc, v6_mcmc = map(lambda v: 
               (v[1], v[2]-v[1], v[1]-v[0]),zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    
   print("q1:", v1_mcmc, "q2:", v2_mcmc, "kp:", v3_mcmc, "kvav:", v4_mcmc, 
         "av:", v5_mcmc, "bv:", v6_mcmc) 



