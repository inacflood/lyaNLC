import numpy as np
import scipy.interpolate
import camb
from camb import model, initialpower

class Cosmology(object):
    """
        Computes cosmological functions using CAMB.

        Original version from Andreu Font-Ribera Github:igmhub/lyaforecast (2017)
    """

    def __init__(self,pk_zref=None):
        """
            Setup cosmological model. 

            If pk_zref is set, it will compute linear power at z=pk_zref.
        """
        self.pars = camb.CAMBparams()
        ob = 0.02242
        om = 0.14240 
        oc = 0.11933 
        h = 0.6766
        self.pars.set_cosmology(H0=100.0*h,ombh2=ob,omch2=oc)
        self.pars.InitPower.set_params(As=2.105e-9,ns=0.9665, r=0)
        self.pk_zref=pk_zref
        if self.pk_zref:
            self.pars.set_matter_power(redshifts=[self.pk_zref], kmax=10.0)
            # compute and store linear power spectrum (at zref)
            self.pars.NonLinear = model.NonLinear_none
            self.results = camb.get_results(self.pars)
        else:
            # compute only background expansion
            self.results = camb.get_results(self.pars)
        # not sure where to put this
        self.c_kms = 2.998e5
        self.lya_A = 1215.67 #Angstrom

    def LinPk_hMpc(self,kmin=1.e-4,kmax=1.e1,npoints=1000):
        """Return linear power interpolator in units of h/Mpc, at zref"""
        if self.pk_zref:
            kh,_,pk = self.results.get_matter_power_spectrum(minkh=kmin,
                                                    maxkh=kmax,npoints=npoints)
            return scipy.interpolate.interp1d(kh,pk[0,:])
        else:
            print('if you want LinPk_hMpc, initialize Cosmology with pk_zref')
            raise SystemExit

    def dkms_dhMpc(self,z):
        """Convertion factor from Mpc/h to km/s, at redshift z."""
        return self.results.hubble_parameter(z)/self.pars.H0/(1+z)*100.0

    def dkms_dlobs(self,z):
        """Convertion factor from lambda_obs to km/s, at redshift z."""
        return self.c_kms / self.lya_A / (1+z) 

    def dhMpc_dlobs(self,z):
        """Convertion factor from lambda_obs to Mpc/h, at redshift z."""
        return self.dkms_dlobs(z) / self.dkms_dhMpc(z)

    def dhMpc_ddeg(self,z):
        """Convertion factor from degrees to Mpc/h, at redshift z."""
        dMpc_drad=self.results.angular_diameter_distance(z)*(1+z)
        return dMpc_drad*np.pi/180.0*self.pars.H0/100.0

