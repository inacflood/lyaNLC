#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:26:46 2019

@author: iflood
"""


import numpy as np
import ORG_lya_kaiser_uv_zevol as kailya
import ORG_arinyo2015 as nlclya
import ORG_cosmoCAMB as cCAMB

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
