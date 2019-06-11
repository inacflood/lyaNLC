#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:26:46 2019

@author: iflood
"""


import numpy as np
import lya_kaiser_uv_zevol as kailya
import arinyo2015 as nlclya
import cosmoCAMB as cCAMB

class TheoryLyaP3D(object):
    """
    Make predictions for Lyman alpha 3D P(z,k,mu).
    Uses CAMB to generate linear power, and McDonald (2006), Gontcho A Gontcho (2014) and Arinyo-i-Prats (2015) for Lya specificities.
    All units internally are in h/Mpc.
    """

    def __init__(self, cosmo=None):
        """
        cosmo is an optional cosmoCAMB.Cosmology object
        since our LyA reference values are at redshift 2.4, we set z_ref = 2.4
        """
        if cosmo:
            self.cosmo=cosmo
            self.zref=cosmo.pk_zref
        else:
            self.zref=2.4
            self.cosmo=cCAMB.Cosmology(self.zref)

        # get linear power spectrum 
        self.kmin=1.e-4
        self.kmax=1.e1
        self.linPk = self.cosmo.LinPk_hMpc(self.kmin,self.kmax,1000)

#    def FluxP3D_hMpc(self,z,k_hMpc,mu,linear=False,uv=True,zevol=True, 
#        q1=0.057,q2=0.368,kp=9.2,kvav=0.48,av=0.156,bv=1.57,
#        beta_lya = 1.650, b_lya = -0.134, b_g = 0.13, b_sa = 1, b_a = -2./3, k0 = 300, a_lya=2.9):
    def FluxP3D_hMpc(self,z,k_hMpc,mu,linear=False,zevol=True,beta_lya = 1.650, b_lya = -0.134, a_lya=2.9,
        q1=0.057,q2=0.368,kp=9.2,kvav=0.48,av=0.156,bv=1.57):
        """3D LyA power spectrum P_F(z,k,mu). 

            If linear = True, it will ignore small scale correction.
            If uv = False, it will ignire uv fluctuations.
            If zevol = False, it will ignore redshift evolution of the bias.

            NB: we use an approximated redshift evolution for the nlc parameters : gamma=0.9 ; 
                this needs to be double checked.
        """

        # get linear power at zref
        k = np.fmax(k_hMpc,self.kmin)
        k = np.fmin(k,self.kmax)
        P = self.linPk(k)

        # get the LyA Kaiser term
#        if (uv==True) and (zevol==True) :
#            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = 1.650, b_lya = -0.134, b_g = 0.13, b_sa = 1, b_a = -2./3, k0 = 300, a_lya=2.9)
#        if (uv==False) and (zevol==True) :
#            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = 1.650, b_lya = -0.134, b_g = 0., b_sa = 1, b_a = -2./3, k0 = 300, a_lya=2.9)
#        if (uv==True) and (zevol==False) :
#            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = 1.650, b_lya = -0.134, b_g = 0.13, b_sa = 1, b_a = -2./3, k0 = 300, a_lya=0)
#        if (uv==False) and (zevol==False) :
#            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = 1.650, b_lya = -0.134, b_g = 0., b_sa = 1, b_a = -2./3, k0 = 300, a_lya=0.)

        if zevol==True:
            kaiser = kailya.Kaiser_LyA_hMpc(k_hMpc, mu, z, beta_lya, b_lya, a_lya=2.9)

        else:
            kaiser = kailya.Kaiser_LyA_hMpc(k_hMpc, mu, z, beta_lya, b_lya, a_lya=0.)

        # get the (*approximated*) redshift evolution term for nlc -- we took nlc fiducial values at z=2.4
        zevol_nlc = pow( (1+z)/(1+2.4), 0.9)

        # get the lya nlc correction term
        if linear == False :
            nlc = nlclya.D_hMpc_AiP2015(k,mu,P,q1=0.057,q2=0.368,kp=9.2,kvav=0.48,av=0.156,bv=1.57) * zevol_nlc
        if linear == True :
            nlc = 1

        return P * kaiser * nlc
    
    def makeP1D_I(self, z, max_kpa=10.,linear=False):
        """Compute 1D power spectrum P_1D(k) from 3D power spectrum P(k) in hMpc, ie self.
            Uses a manual Rimannian sum to integrate.
    
            P_1D(k) = \int dkpe ( kpe / (2pi) ) * P(kpe,kpa)
    
        where: 
            kpa = k*mu 
                and 
            kpe = k*sqrt(1-mu**2)
    
        Args:
            z = redshift
            linear : set to False to include Arinyo et al 2015 NLC
    
        Returns:
            kpa_list, P1D
            params: k_parallel
            P1D (array): 1D velocity power spectrum
    
        [!] All units internally are in h/Mpc.
        """
        prec=200
        
        kpa_start,kpa_stop=[-4,np.log10(max_kpa-0.01)]
        kpa_list=np.logspace(kpa_start,kpa_stop,prec)
    
        P1D = np.zeros(prec)  
        
        for kpa_i in range(prec):   
            kpa = kpa_list[kpa_i]
            
            kpe_start,kpe_stop = [-4,np.log10(np.sqrt(max_kpa**2-kpa**2))-10**(-5)] 
            # 10**(-5) above used to prevent hitting the edge of interpolation limit
            kpe_list = np.logspace(kpe_start,kpe_stop,prec)
            
            k_list = np.sqrt(kpe_list**2 + kpa**2)
            power_vals = [self.FluxP3D_hMpc(z,k,(kpa/k),linear=linear) for k in k_list]
            for i in range(len(k_list)-1):  # perform Riemannian sum
                P1D[kpa_i]+=(k_list[i+1]-k_list[i])*kpe_list[i]**2*(power_vals[i]+power_vals[i+1])/2/(2*np.pi)
        return kpa_list, P1D
    
    def makeP1D_T(self, z, max_kpa=10.,linear=False):
        """Compute 1D power spectrum P_1D(k) from 3D power spectrum P(k) in hMpc, ie self.
            Uses np.trapz to integrate, giving different results from makeP1D_I.
    
            P_1D(k) = \int dkpe ( kpe / (2pi) ) * P(kpe,kpa)
    
            where: 
                kpa = k*mu 
                    and 
                kpe = k*sqrt(1-mu**2)
        
            Args:
                z = redshift
                linear : set to False to include Arinyo et al 2015 NLC
        
            Returns:
                kpa_list, P1D
                params: k_parallel
                P1D (array): 1D velocity power spectrum
        
            [!] All units internally are in h/Mpc.
            """
        prec=200
        
        kpa_start,kpa_stop=[-4,np.log10(max_kpa-0.01)]
        kpa_list=np.logspace(kpa_start,kpa_stop,prec)
    
        P1D = np.zeros(prec)  
        
        for kpa_i in range(prec):   
            kpa = kpa_list[kpa_i]
            
            kpe_start,kpe_stop = [-4,np.log10(np.sqrt(max_kpa**2-kpa**2))-10**(-5)]
            kpe_list = np.logspace(kpe_start,kpe_stop,prec)
            
            k_list = np.sqrt(kpe_list**2 + kpa**2)
            power_vals = [self.FluxP3D_hMpc(z,k,(kpa/k),linear=linear) for k in k_list]
            integrand = np.multiply(power_vals,kpe_list**2)
            P1D[kpa_i] = np.trapz(integrand,kpe_list)/(2*np.pi)
            
        return kpa_list, P1D


    def makeP1D_S(self, z, max_kpa=10.,linear=False):
        """
        Older version of calculating P1D from P3D, for reference.
        Uses constant spacing in Riemannian sum vs. spacing between sequential values of kpe
        """
        dlogk=0.01
    
        log_kpa=np.arange(-4.,np.log10(max_kpa)-dlogk,dlogk)
    
        frac = 1.0/(2.0*np.pi)
    
        P1 = np.zeros(len(log_kpa))                                                                               
    
        for l in range(len(log_kpa)) : 
            kpa = 10**log_kpa[l] 
            kpe_max = np.sqrt(max_kpa**2 - kpa**2)
            log_kpe= np.arange(-4.,np.log10(kpe_max),dlogk)
            kpe = 10**log_kpe
            k_list = np.sqrt(kpe**2 + kpa**2)
            power_vals=[self.FluxP3D_hMpc(z,k,(kpa/k),linear=linear) for k in k_list]
            P1[l] = dlogk * np.sum( kpe**2 * power_vals)
        return 10**log_kpa, P1*frac