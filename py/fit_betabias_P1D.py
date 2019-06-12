#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:05:25 2019

@author: iflood
"""
import numpy as np
import cosmoCAMB_newParams as cCAMB
import theoryLya as tLyA
#import arinyo2015 as nlclya
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.interpolate as interp

#exp_arr = np.linspace(-4.,0.9,10)
#k_arr = [10**i for i in exp_arr]

def make1DPower(beta=1.650,b=-0.134,z=2.4):
    
    """
    """
    
    cosmo = cCAMB.Cosmology(z)
    th = tLyA.TheoryLyaP3D(cosmo)
    power1D = th.makeP1D_I(z, beta_lya=beta, b_lya=b)
    dkM = th.cosmo.dkms_dhMpc
    f1D = interp.interp1d(power1D[0]/dkM(z),np.multiply(power1D[0],power1D[1])/np.pi)
    
    return f1D

# Test function make1DPower
f1D = make1DPower()
k_list = np.linspace(.0025,.02,500)

plt.xscale('linear')
plt.yscale('log')

plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)*k/pi')

plt.plot(k_list,f1D(k_list))

def make1Dpower_varyb(b_list,beta=1.650,z=2.4,prec=200):
    
    """
    """
    cosmo = cCAMB.Cosmology(z)
    th = tLyA.TheoryLyaP3D(cosmo)
    dkM = th.cosmo.dkms_dhMpc
    
    num_b = len(b_list)
    values = np.zeros(num_b,prec)
    grid_x = b_list
    power1D_start = th.makeP1D_I(z, beta_lya=beta, b_lya=b_list[0])
    grid_y = power1D_start[0]/dkM(z)
    values[0,:]=np.multiply(power1D_start[0],power1D_start[1])/np.pi
    
    for b_i in range(1, num_b):
        power1D = th.makeP1D_I(z, beta_lya=beta, b_lya=b_list[b_i])
        values[b_i,:]=np.multiply(power1D[0],power1D[1])/np.pi
    
    ??? f1D = interp.griddata() ???
    
        
        
    
    

# Test function integPower
#integ_Power_k3_beta10_b10=integPower(10**(-3),1.0,-0.1,10)
#print(integ_Power_k3_beta10_b10)


def makePofk(beta,b,kRefine=100,muRefine=100,z=2.4):
    """
    Using a table of k_Refine integrated power spectrum values (with dmu in integration 
    given by muRefine) create a function P(k) using interp1D, with values for beta 
    and bias given.
    The redshift value z can also be adjusted.
    """
    exp_list = np.linspace(-4.,0.9,kRefine)
    counter = 0
    k_list = []
    P_list = []
    for exp in exp_list:
        counter+=1
        k=10**exp
        print("Working on k=", k, "sample number", counter)
        k_list+=[k]
        P_list+=[integPower(k,beta,b,muRefine,z)]
    print("Created interpolation list, interpolating...")
    Pofk = interp.interp1d(k_list,P_list)
    return Pofk

# Test function makePofk
Pofk_beta10_b10=makePofk(1.0,-0.1,kRefine=200,muRefine=50)
print(Pofk_beta10_b10(0.12))
k_new = np.linspace(10**(-4),0.9,200)
P_funcTest = [Pofk_beta10_b10(k)*k/np.pi for k in k_new]

plt.xscale('log')
plt.yscale('log')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)*k/pi')
plt.title(r"Interpolated Power plot, z=2.4, beta=1.0, bias=-0.1",fontsize=15)
plt.plot(k_new,P_funcTest)

