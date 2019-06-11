#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:08:41 2019

@author: iflood
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB as cCAMB
import theoryLya as tLyA
import get_npd_p1d_woFitsio as npd

# Create 1D power spectra from 3D spectra and get data to compare with

z22=2.2
cosmo22 = cCAMB.Cosmology(z22)
th22 = tLyA.TheoryLyaP3D(cosmo22)
dkM22 = th22.cosmo.dkms_dhMpc               # units correction factor
p3t1_22 = th22.makeP1D_I(z22)#,linear=True) # 1D power spectrum from manual Riemannian sum
p3t1_22_t = th22.makeP1D_T(z22)               # 1D power spectrum from np.trapz integration
npd22 = npd.LyA_P1D(z22)                    # Data for corresponding redshift

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tLyA.TheoryLyaP3D(cosmo24)
dkM24 = th24.cosmo.dkms_dhMpc
p3t1_24 = th24.makeP1D_I(z24)#,linear=True)
p3t1_24_t = th24.makeP1D_T(z24) 
npd24 = npd.LyA_P1D(z24)

z26=2.6
cosmo26 = cCAMB.Cosmology(z26)
th26 = tLyA.TheoryLyaP3D(cosmo26)
dkM26 = th26.cosmo.dkms_dhMpc
p3t1_26 = th26.makeP1D_I(z26)#,linear=True)
p3t1_26_t = th26.makeP1D_T(z26) 
npd26 = npd.LyA_P1D(z26)

z28=2.8
cosmo28 = cCAMB.Cosmology(z28)
th28 = tLyA.TheoryLyaP3D(cosmo28)
dkM28 = th28.cosmo.dkms_dhMpc
p3t1_28 = th28.makeP1D_I(z28)#,linear=True)
p3t1_28_t = th28.makeP1D_T(z28) 
npd28 = npd.LyA_P1D(z28)

z30=3.0
cosmo30 = cCAMB.Cosmology(z30)
th30 = tLyA.TheoryLyaP3D(cosmo30)
dkM30 = th30.cosmo.dkms_dhMpc
p3t1_30 = th30.makeP1D_I(z30)#,linear=True)
p3t1_30_t = th30.makeP1D_T(z30) 
npd30 = npd.LyA_P1D(z30)

# Plotting everything

manInt = plt.figure(1)
ax = plt.subplot(111)

plt.yscale('log')
plt.xscale('linear')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
plt.xlim(0.001,0.02)
plt.ylim(0.001,1)
plt.plot(p3t1_24[0]/dkM24(z24) , p3t1_24[0] * p3t1_24[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
#plt.plot(p3t1_24_S[0]/dkM24(z24) , p3t1_24_S[0] * p3t1_24_S[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')
plt.plot(p3t1_22[0]/dkM22(z22) , p3t1_22[0] * p3t1_22[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.2 $, th')
plt.plot( npd22.k , npd22.k/np.pi*(npd22.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')
plt.plot(p3t1_26[0]/dkM26(z24) , p3t1_26[0] * p3t1_26[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 2.6 $, th')
plt.plot( npd26.k , npd26.k/np.pi*(npd26.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=2.6$, npd')
plt.plot(p3t1_28[0]/dkM28(z28) , p3t1_28[0] * p3t1_28[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 2.8 $, th')
plt.plot( npd28.k , npd28.k/np.pi*(npd28.Pk_emp()) , 'k^', alpha=0.4, linewidth=2, label=r'$z=2.8$, npd')
plt.plot(p3t1_30[0]/dkM30(z30) , p3t1_30[0] * p3t1_30[1] / np.pi,'c',alpha=0.7, linewidth=2 ,label=r'$z = 3.0 $, th')
plt.plot( npd30.k , npd30.k/np.pi*(npd30.Pk_emp()) , 'c^', alpha=0.4, linewidth=2, label=r'$z=3.0$, npd')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"1D Flux power, theory (non linear) v P-D, Riem. Sum",fontsize=15)
manInt.savefig("../Figures/P3D_to_P1D_nonlinear_riemsum.pdf")

# Doing the same plot again, but this time using P1D from makeP1D_T, i.e. integral is done using
# np.trapz
trapzInt = plt.figure(2)
ax = plt.subplot(111)

plt.yscale('log')
plt.xscale('linear')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
plt.xlim(0.001,0.02)
plt.ylim(0.001,1)
plt.plot(p3t1_24_t[0]/dkM24(z24) , p3t1_24_t[0] * p3t1_24_t[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
#plt.plot(p3t1_24_S[0]/dkM24(z24) , p3t1_24_S[0] * p3t1_24_S[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')
plt.plot(p3t1_22_t[0]/dkM22(z22) , p3t1_22_t[0] * p3t1_22_t[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.2 $, th')
plt.plot( npd22.k , npd22.k/np.pi*(npd22.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')
plt.plot(p3t1_26_t[0]/dkM26(z24) , p3t1_26_t[0] * p3t1_26_t[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 2.6 $, th')
plt.plot( npd26.k , npd26.k/np.pi*(npd26.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=2.6$, npd')
plt.plot(p3t1_28_t[0]/dkM28(z28) , p3t1_28_t[0] * p3t1_28_t[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 2.8 $, th')
plt.plot( npd28.k , npd28.k/np.pi*(npd28.Pk_emp()) , 'k^', alpha=0.4, linewidth=2, label=r'$z=2.8$, npd')
plt.plot(p3t1_30_t[0]/dkM30(z30) , p3t1_30_t[0] * p3t1_30_t[1] / np.pi,'c',alpha=0.7, linewidth=2 ,label=r'$z = 3.0 $, th')
plt.plot( npd30.k , npd30.k/np.pi*(npd30.Pk_emp()) , 'c^', alpha=0.4, linewidth=2, label=r'$z=3.0$, npd')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"1D Flux power, theory (non linear) v P-De, Trapz. Int",fontsize=15)
trapzInt.savefig("../Figures/P3D_to_P1D_nonlinear_trapzint.pdf")