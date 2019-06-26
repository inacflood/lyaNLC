import numpy as np
import lya_biases as kailya
import arinyo2015 as nlclya
import cosmoCAMB as cCAMB

class TheoryLya(object):
    """
    Predictions for Lyman alpha 3D power spectrum P(z,k,mu) and 1D power spectrum P(z,kpa).
    Uses CAMB to generate linear power, as well as Gontcho A Gontcho (2014) and Arinyo-i-Prats (2015) for Lya specificities.
    [!] All units internally are in h/Mpc.
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

    def FluxP3D_hMpc(self, z, k_hMpc, mu, linear=False, uv=False, zevol=True,
        q1=0.057, q2=0.368, kp=9.2, kvav=0.48, av=0.156, bv=1.57,
        beta_lya = 1.650, b_lya = -0.134):

        """
        3D LyA power spectrum P_F(z,k,mu).

        If linear = True, it will ignore small scale correction.
        If uv = False, it will ignore uv fluctuations.
        If zevol = False, it will ignore redshift evolution of the bias.

        NB: we use an approximated redshift evolution for the nlc parameters : gamma=0.9 ;
            this needs to be double checked.

        Returns:
            P3D (array): 3D power spectrum of the Lyman Alpha Forest
        """

        # get linear power at zref
        k = np.fmax(k_hMpc,self.kmin)
        k = np.fmin(k,self.kmax)
        P = self.linPk(k)

        # get the LyA Kaiser term
        if (uv==True) and (zevol==True) :
            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = beta_lya, b_lya = b_lya, b_g = 0.13, a_lya=2.9)
        if (uv==False) and (zevol==True) :
            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = beta_lya, b_lya = b_lya, b_g = 0., a_lya=2.9)
        if (uv==True) and (zevol==False) :
            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = beta_lya, b_lya = b_lya, b_g = 0.13, a_lya=0)
        if (uv==False) and (zevol==False) :
            kaiser = kailya.Kaiser_LyA_hMpc(k, mu, z, beta_lya = beta_lya, b_lya = b_lya, b_g = 0., a_lya=0.)

        # get the (*approximated*) redshift evolution term for nlc -- we took nlc fiducial values at z=2.4
        zevol_nlc = pow( (1+z)/(1+2.4), 0.9)

        # get the lya nlc correction term
        if linear == False :
            nlc = nlclya.D_hMpc_AiP2015(k,mu,P,q1=q1,q2=q2,kp=kp,kvav=kvav,av=av,bv=bv) * zevol_nlc
        if linear == True :
            nlc = 1

        return P * kaiser * nlc

    def FluxP1D_hMpc(self, z, kpa_hMpc, max_kpa = 10., prec = 200,
            linear=False, uv=False, zevol=True,
            q1=0.057, q2=0.368, kp=9.2, kvav=0.48, av=0.156, bv=1.57,
            beta_lya = 1.650, b_lya = -0.134):

        """Compute 1D power spectrum P_1D(k_parallel) at one value of k_parallel from 3D power spectrum P(k) in hMpc, ie self.
            Uses a manual Rimannian sum to integrate.

            P_1D(kpa) = \int dkpe ( kpe / (2pi) ) * P(kpe,kpa)

        where:
            kpa = k*mu
                and
            kpe = k*sqrt(1-mu**2)

        Note:
            If linear = True, it will ignore small scale correction.
            If uv = False, it will ignore uv fluctuations.
            If zevol = False, it will ignore redshift evolution of the bias.

            NB: we use an approximated redshift evolution for the nlc parameters : gamma=0.9 ;
                this needs to be double checked.

        Args:
            redshift : z
            k_parallel : kpa_hMpc

        Returns:
            P1D (array): 1D power spectrum of the Lyman Alpha Forest

        """

        P1D = 0
        kpe_start,kpe_stop = [-4,np.log10(np.sqrt(max_kpa**2-kpa_hMpc**2))-10**(-5)]
        # 10**(-5) above used to prevent hitting the edge of interpolation limit
        kpe_list = np.logspace(kpe_start,kpe_stop,prec)

        k_list = np.sqrt(kpe_list**2 + kpa_hMpc**2)
        power_vals = [self.FluxP3D_hMpc(z,k,(kpa_hMpc/k),linear=linear, uv=uv, zevol=zevol, beta_lya=beta_lya, b_lya=b_lya,
        q1=q1, q2=q2, kp=kp, kvav=kvav, av=av, bv=bv) for k in k_list]

        for i in range(len(k_list)-1):  # perform Riemannian sum
            P1D+=(k_list[i+1]-k_list[i])*kpe_list[i]**2*(power_vals[i]+power_vals[i+1])/2/(2*np.pi)

        return P1D
