#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:08:41 2019

@author: iflood
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB_newParams as cCAMB
import theoryLya as tLyA
import get_npd_p1d_woFitsio as npd

#def makeP1D_I(z,theory_3D_hMpc, max_kpa=10.,linear=False):
#    prec=1000
#    kpa_start,kpa_stop=[-4,np.log10(max_kpa-0.01)]
#    kpa_list=np.logspace(kpa_start,kpa_stop,prec)
#    #print(kpa_list)
#    P1D = np.zeros(prec)  
#    for kpa_i in range(prec):   
#        kpa=kpa_list[kpa_i]
#        kpe_start,kpe_stop=[-4,np.log10(np.sqrt(max_kpa**2-kpa**2))-10**(-5)]
#        kpe_list=np.logspace(kpe_start,kpe_stop,prec)
#        k_list = np.sqrt(kpe_list**2 + kpa**2)
#        power_vals=[theory_3D_hMpc.FluxP3D_hMpc(z,k,(kpa/k),linear=linear) for k in k_list]
#        P1D[kpa_i]=np.trapz(power_vals,kpe_list)/2/np.pi
#    return kpa_list, P1D
#
#
#def makeP1D_S(z,theory_3D_hMpc, max_kpa=10.,linear=False):
#    dlogk=0.01
#
#    log_kpa=np.arange(-4.,np.log10(max_kpa)-dlogk,dlogk)
#
#    frac = 1.0/(2.0*np.pi)
#
#    P1 = np.zeros(len(log_kpa))                                                                               
#
#    for l in range(len(log_kpa)) : 
#        kpa = 10**log_kpa[l] 
#        kpe_max = np.sqrt(max_kpa**2 - kpa**2)
#        log_kpe= np.arange(-4.,np.log10(kpe_max),dlogk)
#        kpe = 10**log_kpe
#        k_list = np.sqrt(kpe**2 + kpa**2)
#        power_vals=[theory_3D_hMpc.FluxP3D_hMpc(z,k,(kpa/k),linear=linear) for k in k_list]
#        P1[l] = dlogk * np.sum( kpe**2 * power_vals)
#    return 10**log_kpa, P1*frac

#z22=2.2
#cosmo22 = cCAMB.Cosmology(z22)
#th22 = tP3D.TheoryLyaP3D(cosmo22)
#dkM22=th22.cosmo.dkms_dhMpc
#p3t1_22 = makeP1D(z22,th22)#,linear=True)
#npd22 = npd.LyA_P1D(z22)

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tLyA.TheoryLyaP3D(cosmo24)
dkM24=th24.cosmo.dkms_dhMpc
#print(dkM24)
p3t1_24_I = th24.makeP1D_I(2.4)
#print(p3t1_24_I[1])
p3t1_24_S = th24.makeP1D_S(2.4)
npd24 = npd.LyA_P1D(z24)

#z26=2.6
#cosmo26 = cCAMB.Cosmology(z26)
#th26 = tP3D.TheoryLyaP3D(cosmo26)
#dkM26=th26.cosmo.dkms_dhMpc
#p3t1_26 = makeP1D(z26,th26)#,linear=True)
#npd26 = npd.LyA_P1D(z26)



plt.yscale('log')
plt.xscale('linear')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
plt.xlim(0.001,0.02)
plt.ylim(0.001,1)
plt.plot(p3t1_24_I[0]/dkM24(z24) , p3t1_24_I[0] * p3t1_24_I[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot(p3t1_24_S[0]/dkM24(z24) , p3t1_24_S[0] * p3t1_24_S[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')
#plt.plot(p3t1_22[0]/dkM22(z22) , p3t1_22[0] * p3t1_22[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.2 $, th')
#plt.plot( npd22.k , npd22.k/np.pi*(npd22.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')
#plt.plot(p3t1_26[0]/dkM24(z24) , p3t1_26[0] * p3t1_26[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
#plt.plot( npd26.k , npd26.k/np.pi*(npd26.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')