#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:05:25 2019

@author: iflood
"""
import numpy as np
import cosmoCAMB as cCAMB
import theoryLyaP3D as tP3D
#import arinyo2015 as nlclya
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.interpolate as interp

exp_arr = np.linspace(-4.,0.9,10)
k_arr = [10**i for i in exp_arr]

def makePower(k,mu,beta,b,z=2.4):
    
    """
    Find the value of the 3D power spectrum at k for input values of mu, beta, and bias.
    The redshift value z can also be adjusted.
    """
    
    cosmo = cCAMB.Cosmology(z)
    th = tP3D.TheoryLyaP3D(cosmo)
    power = th.FluxP3D_hMpc(z,k,mu,beta_lya=beta,b_lya=b)
    
    return power

# Test function makePower

#power_beta10_b10_mu5=[makePower(k,0.5,1.0,-0.1) for k in k_arr]

#plt.xscale('log')
#plt.yscale('log')
#
#plt.xlabel('k [(Mpc/h)^-1]')
#plt.ylabel('P(k)')
#
#plt.plot(k_arr,power_beta10_b10_mu5)
    
# Make function P(k ; beta, bias)
    
def makePowerList(k,mu_list,beta,b,z=2.4):
     """
     Create a list of the 3D power values corresponding to the values in mu_list, 
    given values of beta and bias.
    The redshift value z can also be adjusted.
     """
     power_list = [makePower(k,m,beta,b) for m in mu_list]
     return power_list

# Test function makePowerList
#mu_list=np.linspace(0.,1.,10)
#powerList_k3_beta10_b10=makePowerList(10**(-3),mu_list,1.0,-0.1)
#
#plt.xscale('log')
#plt.yscale('log')
#
#plt.xlabel('k [(Mpc/h)^-1]')
#plt.ylabel('P(k)')
#
#plt.plot(mu_list,powerList_k3_beta10_b10)

def integPower(k,beta,b,refinement=100,z=2.4):
    """
    Do a Riemannian integral of the Power at k from a list of 3D power values for 
    _refinement_ mu points given values of beta and bias.
    The redshift value z can also be adjusted.
    """
    mu_list = np.linspace(0.,1.,refinement)
    power_list = makePowerList(k,mu_list,beta,b,z)
    #print(power_list)
    integ_power = intg.trapz(power_list,x=mu_list)
    return integ_power

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
    k_list = [10**i for i in exp_list]
    P_list = [integPower(k,beta,b,muRefine,z) for k in k_list]
    Pofk = interp.interp1d(k_list,P_list)
    return Pofk

# Test function makePofk
Pofk_beta10_b10=makePofk(1.0,-0.1,kRefine=1000,muRefine=1000)
print(Pofk_beta10_b10(0.12))
k_new = np.linspace(10**(-4),0.9,50)
P_funcTest = [Pofk_beta10_b10(k) for k in k_new]

plt.xscale('log')
plt.yscale('log')
plt.xlabel('k [(Mpc/h)^-1]')
plt.ylabel('P(k)')
plt.title(r"Interpolated Power plot, z=2.4, beta=1.0, bias=-0.1",fontsize=15)
plt.plot(k_new,P_funcTest)

