import numpy as np
import matplotlib.pyplot as plt
import corner
from arinyo2015 import getFiducialValues
import cosmoCAMB as cCAMB
import theoryLya as tLyA
import get_npd_p1d as npd

headFile = "run123"
chi_test = True
saveFigs = True

nwalkers, nsteps, ndim, z, err, runtime = np.loadtxt('../output/'+headFile+'/params.dat')
beta_f = 1.650
b_f = -0.134

nwalkers = int(nwalkers)
nsteps = int(nsteps)
ndim = int(ndim)
z_str = str(int(z*10)) # for use in file names
err_str = str(int(err*100))

# Retrieve the parameters that were fitted
param_opt = np.loadtxt('../output/'+headFile+'/fitto.dat')
param_opt = [int(param) for param in param_opt]
#param_opt = [1,1,0,0,1,0]

labels = ["q1","q2","kp","kvav","av","bv"]
pop_count = 0
for f in range(len(param_opt)):
    if not param_opt[f]:
        labels.pop(f-pop_count)
        pop_count+=1

fiducials = getFiducialValues(z)
q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = fiducials

params = []
fidList = [] 
for f in range(len(param_opt)):
    if param_opt[f]:
        params.append(0)
        fidList.append(fiducials[f])
    else:
        params.append(fiducials[f])

q1_e = 0.46008

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLya(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z) #

# Get actual data

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

param1 = plt.figure(1)
plt.ylabel(labels[0])
for w in range(nwalkers):
    plt.plot([chain[w][s][0] for s in range(nsteps)])

if saveFigs:
    param1.savefig("../output/"+headFile+"/WalkerPaths"+labels[0]+".pdf")
param1.show()

param2 = plt.figure(2)
plt.ylabel(labels[1])
for w in range(nwalkers):
    plt.plot([chain[w][s][1] for s in range(nsteps)])

if saveFigs:
    param2.savefig("../output/"+headFile+"/WalkerPaths"+labels[1]+".pdf")
param2.show()

param3 = plt.figure(3)
plt.ylabel(labels[2])
for w in range(nwalkers):
    plt.plot([chain[w][s][2] for s in range(nsteps)])
if saveFigs:
    param3.savefig("../output/"+headFile+"/WalkerPaths"+labels[2]+".pdf")
param3.show()
    
pathView = plt.figure(7)
plt.yscale('log')


for var1,var2,var3 in samples[np.random.randint(len(samples), size=200)]:
    theta = [var1,var2,var3]
    for f in range(len(param_opt)):
        if param_opt[f]:
            params[f] = theta[0]
            theta.pop(0) 
        
    plt.plot(k, th.FluxP1D_hMpc(z, k*dkMz, q1=params[0], q2=params[1], 
                                kp=params[2], kvav=params[3], av=params[4], 
                                bv=params[5])*k_res/np.pi, color="b", alpha=0.1)

plt.plot(k,th.FluxP1D_hMpc(z, k*dkMz, q1=q1_f,q2=q2_f,kp=kp_f,kvav=kvav_f,av=av_f,bv=bv_f)*k_res/np.pi
         , color="r", lw=2, alpha=0.8)
plt.errorbar(k, P*k/np.pi, yerr=Perr*k/np.pi, fmt=".k")

plt.xlabel('k [(km/s)^-1]')
plt.ylabel('P(k)*k/pi')
plt.title('Parameter exploration')

if saveFigs:
    pathView.savefig("../output/"+headFile+"/SamplePaths_err"+err_str+"posSMmtF.pdf")
pathView.show()
    

# Final results

cornerplt = corner.corner(samples, labels=[ "$"+labels[0]+"$", "$"+labels[1]+
                                           "$", "$"+labels[2]+"$"],truths=fidList,
                            quantiles=[0.16, 0.5, 0.84],show_titles=True)

if saveFigs:   
    cornerplt.savefig("../output/"+headFile+"/triangle_err"+err_str+"posFSmtT.pdf")
cornerplt.show()

v1, v2, v3 = map(lambda v: (v[1], v[2], v[0]),
                        zip(*np.percentile(samples, [16, 50, 84],
                                            axis=0)))
    
corner_res = [v1, v2, v3]
print(corner_res) 

resultsfile = open('../output/'+headFile+'/corner.dat','w')
for d in range(ndim):
    resultsfile.write('{0} {1} {2} \n'.format(str(corner_res[d][0]), str(corner_res[d][1]), 
                      str(corner_res[d][2])))
resultsfile.close()
    
   
if chi_test:
# Get chi-squared

    theta = np.array(corner_res)[:,0]
    for f in range(len(param_opt)):
            if param_opt[f]:
                params[f] = theta[0]
                theta=np.delete(theta,0)
    #print(params)
    chi_sum = 0
    
    for i in range(len(k)):
        kval = k[i]
        Pval = P[i]
        err = Perr[i]
        obs = th.FluxP1D_hMpc(z, kval*dkMz, q1=params[0], q2=params[1], 
                                kp=params[2], kvav=params[3], av=params[4], 
                                bv=params[5])*dkMz
        diff = Pval-obs
        #print(diff)
        #print(kval, (diff/err)**2)
        chi_sum += (diff/err)**2
    
    chi_sq = chi_sum/(len(k)-ndim)
        
    print("Chi-squared per dof:", chi_sq)
    



