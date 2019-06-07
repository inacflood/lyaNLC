import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import cosmoCAMB as cCAMB
import get_npd_p1d as npd
import theoryLyaP3D as tP3D


def log10(x):
    return np.log(x)/np.log(10.)


def P3D_to_P1D(z,theory_3D_hMpc,kpar_max_hMpc=10.,linear=False):

    """Compute 1D power spectrum P_1D(k) from 3D power spectrum P(k)
    
            P_1D(k) = \int dkpe ( kpe / (2pi) ) * P(kpe,kpa)
    
    where: 
        kpa = k*mu 
            and 
        kpe = k*sqrt(1-mu**2)

    Args:
        z = redshift
        theory_3D_hMpc = 3D flux power spectrum in hMpc
        linear : set to False to include Arinyo et al 2015 NLC

    Returns:
        params, P1D 
        params: k_parallel
        P1 (array): 1D velocity power spectrum

    [!] All units internally are in h/Mpc.
    """

    dlogk=0.01

    log_kpa=np.arange(-4.,log10(kpar_max_hMpc)-dlogk,dlogk)

    frac = 1.0/(2.0*np.pi)

    P1 = np.zeros(len(log_kpa))                                                                               

    for l in range(len(log_kpa)) : 
        kpa = 10**log_kpa[l] 
        kpe_max = np.sqrt(kpar_max_hMpc**2 - kpa**2)
        log_kpe= np.arange(-4.,log10(kpe_max),dlogk)
        kpe = 10**log_kpe
        k = np.sqrt(kpe**2 + kpa**2)
        pth = np.zeros(len(k))
        for i in range(len(k)):
            if linear == False :
                pth[i] = theory_3D_hMpc.FluxP3D_hMpc(z,k[i],(kpa/k[i]))
            else:
                pth[i] = theory_3D_hMpc.FluxP3D_hMpc(z,k[i],(kpa/k[i]),linear=True)

        P1[l] = dlogk * np.sum( kpe**2 * pth)
        return 10**log_kpa, P1

## get 3D data sets to play with

z22=2.2
cosmo22 = cCAMB.Cosmology(z22)
th22 = tP3D.TheoryLyaP3D(cosmo22)
dkM22=th22.cosmo.dkms_dhMpc

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tP3D.TheoryLyaP3D(cosmo24)
dkM24=th24.cosmo.dkms_dhMpc

z26=2.6
cosmo26 = cCAMB.Cosmology(z26)
th26 = tP3D.TheoryLyaP3D(cosmo26)
dkM26=th26.cosmo.dkms_dhMpc

z28=2.8
cosmo28 = cCAMB.Cosmology(z28)
th28 = tP3D.TheoryLyaP3D(cosmo28)
dkM28=th26.cosmo.dkms_dhMpc


z30=3.0
cosmo30 = cCAMB.Cosmology(z30)
th30 = tP3D.TheoryLyaP3D(cosmo30)
dkM30=th30.cosmo.dkms_dhMpc


## get 1D data sets to compare to

npd22 = npd.LyA_P1D(z22)
npd24 = npd.LyA_P1D(z24)
npd26 = npd.LyA_P1D(z26)
npd28 = npd.LyA_P1D(z28)
npd30 = npd.LyA_P1D(z30)


## generate our 1D power from the 3D test sets

p3t1_22 = P3D_to_P1D(z22,th22)#,linear=True)
p3t1_24 = P3D_to_P1D(z24,th24)#,linear=True)
p3t1_26 = P3D_to_P1D(z26,th26)#,linear=True)
p3t1_28 = P3D_to_P1D(z28,th28)#,linear=True)
p3t1_30 = P3D_to_P1D(z30,th30)#,linear=True)

## PLOTTING

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [
  r'\usepackage{siunitx}',   
  r'\sisetup{detect-all}',   
  r'\usepackage{helvet}',    
  r'\usepackage{sansmath}',  
  r'\sansmath'            
]
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1, 1, 1)
#ax.set_xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=16)
#ax.set_ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=16)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.yscale('log')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
plt.xlim(0.001,0.02)
plt.ylim(0.001,1)


plt.plot(p3t1_22[0]/dkM22(z22) , p3t1_22[0] * p3t1_22[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 2.2 $, th')
plt.plot(p3t1_24[0]/dkM24(z24) , p3t1_24[0] * p3t1_24[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot(p3t1_26[0]/dkM26(z26) , p3t1_26[0] * p3t1_26[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 2.6 $, th')
plt.plot(p3t1_28[0]/dkM28(z28) , p3t1_28[0] * p3t1_28[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.8 $, th')
plt.plot(p3t1_30[0]/dkM30(z30) , p3t1_30[0] * p3t1_30[1] / np.pi,'m',alpha=0.7, linewidth=2 ,label=r'$z = 3.0 $, th')


plt.plot( npd22.k , npd22.k/np.pi*(npd22.Pk_emp()) , 'ks', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')
plt.plot( npd26.k , npd26.k/np.pi*(npd26.Pk_emp()) , 'gH', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')
plt.plot( npd28.k , npd28.k/np.pi*(npd28.Pk_emp()) , 'b*', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')
plt.plot( npd30.k , npd30.k/np.pi*(npd30.Pk_emp()) , 'mo', alpha=0.4, linewidth=2, label=r'$z=3.0$, npd')


plt.legend(loc='best',fontsize=15)
plt.title(r"1D Flux power, theory (non linear) v Palanque-Delabrouille",fontsize=15)

plt.savefig("Figures/P3D_to_P1D_th_nonlinear.pdf")
plt.show()

