#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:15:50 2019

@author: iflood
"""

import numpy as np
import cosmoCAMB_newParams as cCAMB
import theoryLyaP3D as tP3D
#import arinyo2015 as nlclya
import matplotlib.pyplot as plt

## Getting started : generate data to plot

k=np.logspace(-4,0.9,1000)	

## flux powers

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tP3D.TheoryLyaP3D(cosmo24)
l24 = th24.linPk(k)
p24 = th24.FluxP3D_hMpc(z24,k,0)

p24_beta10 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 1.0)
p24_beta12 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 1.2)
p24_beta14 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 1.4)
p24_beta16 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 1.6)
p24_beta18 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 1.8)
p24_beta20 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 2.0)
#p24_beta0 = th24.FluxP3D_hMpc(z24,k,0.5,beta_lya = 0)

p24_b10 = th24.FluxP3D_hMpc(z24,k,0,b_lya = -0.1)
p24_b11 = th24.FluxP3D_hMpc(z24,k,0,b_lya = -0.11)
p24_b12 = th24.FluxP3D_hMpc(z24,k,0,b_lya = -0.12)
p24_b13 = th24.FluxP3D_hMpc(z24,k,0,b_lya = -0.13)
p24_b14 = th24.FluxP3D_hMpc(z24,k,0,b_lya = -0.14)
p24_b15 = th24.FluxP3D_hMpc(z24,k,0,b_lya = -0.15)

#p24_025 = th24.FluxP3D_hMpc(z24,k,0.25)
#d_nl_24_000 = nlclya.D_hMpc_AiP2015(k,0.,th24.linPk(k))

z26=2.6
cosmo26 = cCAMB.Cosmology(z26)
th26 = tP3D.TheoryLyaP3D(cosmo26)
l26 = th26.linPk(k)
p26 = th26.FluxP3D_hMpc(z26,k,0)

z30=3.0
cosmo30 = cCAMB.Cosmology(z30)
th30 = tP3D.TheoryLyaP3D(cosmo30)
l30 = th30.linPk(k)
p30 = th30.FluxP3D_hMpc(z30,k,0)



## set the log log frame for a power plot

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xscale('log')
plt.yscale('log')

lyAplot = plt.figure(1)

plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k) * k / pi')
plt.plot(k,k * p24 / np.pi,'k',alpha=0.6, linewidth=2 ,label=r'$z = 2.4 $')
plt.plot(k,k * p26 / np.pi,'r',alpha=0.6, linewidth=2 ,label=r'$z = 2.6 $')
plt.plot(k,k * p30 / np.pi,'g',alpha=0.6, linewidth=2 ,label=r'$z = 3.0 $')
#plt.plot(k,k * p28_NLC_only / np.pi,'b',alpha=0.6, linewidth=2 ,label=r'$z = 2.8 $')
#plt.plot(k,k * p30_NLC_only / np.pi,'m',alpha=0.6, linewidth=2 ,label=r'$z = 3.0 $')

plt.title(r"Test LyA-P3D plots ",fontsize=15)
plt.legend(loc='best',fontsize=16)
#plt.savefig("Figures/P3D_nonlinear.pdf")
lyAplot.show()


linplot = plt.figure(2)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k) * k / pi')
plt.plot(k,k * l24 / np.pi,'k',alpha=0.6, linewidth=2 ,label=r'$z = 2.4 $')
plt.plot(k,k * l26 / np.pi ,'r',alpha=0.6, linewidth=2 ,label=r'$z = 2.6 $')
plt.plot(k, k * l30 / np.pi ,'g',alpha=0.6, linewidth=2 ,label=r'$z = 3.0 $')
#plt.plot(k,k * p28_NLC_only / np.pi,'b',alpha=0.6, linewidth=2 ,label=r'$z = 2.8 $')
#plt.plot(k,k * p30_NLC_only / np.pi,'m',alpha=0.6, linewidth=2 ,label=r'$z = 3.0 $')

plt.title(r"Test Linear power spectrum plots ",fontsize=15)
plt.legend(loc='best',fontsize=16)
#plt.savefig("Figures/P3D_nonlinear.pdf")



betacomplot = plt.figure(3)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k) * k / pi')
plt.plot(k,k * p24_beta10 / np.pi,'k',alpha=0.6, linewidth=1 ,label=r'$beta = 1.0 $')
#plt.plot(k,k * p24_beta12 / np.pi ,'r',alpha=0.6, linewidth=0.5 ,label=r'$beta = 1.2 $')
plt.plot(k, k * p24_beta14 / np.pi ,'g',alpha=0.6, linewidth=0.5 ,label=r'$beta = 1.4 $')
#plt.plot(k,k * p24_beta16 / np.pi,'b',alpha=0.6, linewidth=1 ,label=r'$beta = 1.6 $')
plt.plot(k,k * p24_beta18 / np.pi,'m',alpha=0.6, linewidth=0.5 ,label=r'$beta = 1.8 $')
#plt.plot(k,k * p24_beta20 / np.pi,'c',alpha=0.6, linewidth=0.5 ,label=r'$beta = 2.0 $')
#plt.plot(k,k * p24_beta0 / np.pi ,'c',alpha=0.6, linewidth=1 ,label=r'$beta = 0 $')


plt.title(r"Comparison of P_3D for different beta values, z=2.4",fontsize=15)
plt.legend(loc='best',fontsize=16)
#plt.savefig("Figures/P3D_nonlinear.pdf")

bcomplot = plt.figure(4)

plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xscale('log')
plt.yscale('log')

plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k) * k / pi')
plt.plot(k,k * p24_b10 / np.pi,'k',alpha=0.6, linewidth=1 ,label=r'$b = -0.1 $')
#plt.plot(k,k * p24_b11 / np.pi ,'r',alpha=0.6, linewidth=1 ,label=r'$b = -0.11 $')
plt.plot(k, k * p24_b12 / np.pi ,'g',alpha=0.6, linewidth=1 ,label=r'$b = -0.12 $')
#plt.plot(k,k * p24_b13 / np.pi,'b',alpha=0.6, linewidth=1 ,label=r'$b = -0.13 $')
plt.plot(k,k * p24_b14 / np.pi,'m',alpha=0.6, linewidth=1 ,label=r'$b = -0.14 $')
#plt.plot(k,k * p24_b15 / np.pi,'c',alpha=0.6, linewidth=1 ,label=r'$b = -0.15 $')


plt.title(r"Comparison of P_3D for different bias values, z=2.4",fontsize=15)
plt.legend(loc='best',fontsize=16)
#plt.savefig("Figures/P3D_nonlinear.pdf")
