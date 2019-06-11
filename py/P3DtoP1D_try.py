#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:08:41 2019

@author: iflood
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB as cCAMB
import theoryLyaP3D as tP3D
import get_npd_p1d_woFitsio as npd

def makeP1D(z,theory_3D_hMpc, max_kpa=10.,linear=False):
    prec=150
    kpa_start,kpa_stop=[-4,np.log10(max_kpa)-0.01]
    kpa_list=np.logspace(kpa_start,kpa_stop,prec)
    #print(kpa_list)
    P1D = np.zeros(prec)  
    for kpa_i in range(prec):   
        kpa=kpa_list[kpa_i]
        kpe_start,kpe_stop=[-4,np.log10(np.sqrt(max_kpa**2-kpa**2-.001))]
        kpe_list=np.logspace(kpe_start,kpe_stop,prec)
        k_list = np.sqrt(kpe_list**2 + kpa**2)
        power_vals=[theory_3D_hMpc.FluxP3D_hMpc(z,k,(kpa/k),linear=linear) for k in k_list]
       # Alternative calculation for the power array, better for debugging
#        power_vals=np.zeros(len(k_list))
#        for k_i in range(len(k_list)):
#            k=k_list[k_i]
#            power_vals[k_i]=theory_3D_hMpc.FluxP3D_hMpc(z,k,(kpa/k),linear=linear)
#            print(k,power_vals[k_i])
        P1D[kpa_i]=np.trapz(power_vals,kpe_list)
    return kpa_list, P1D

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tP3D.TheoryLyaP3D(cosmo24)
dkM24=th24.cosmo.dkms_dhMpc

p3t1_24 = makeP1D(z24,th24)#,linear=True)

npd24 = npd.LyA_P1D(z24)

plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
#plt.xlim(0.001,0.02)
#plt.ylim(0.001,1)
plt.plot(p3t1_24[0]/dkM24(z24) , p3t1_24[0] * p3t1_24[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')