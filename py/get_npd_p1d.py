import numpy as np
import fitsio

class LyA_P1D():
    """
        Returns the One-dimensional LyA forest power spectrum obtained with the
        Fourier transform method {"0"} -or- the likelihood method {"1"} from Palanque-Delabrouille et al (2013)
        for redshift bins: 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4
    """

    def __init__(self,z_bin,method=0):
        """
            k        =  scale value ( s/km ) , [0.001, 0.020]
            Pk       =  Power spectrum ( km/s )
            Pk_stat  =  Statistical 1-sigma uncertainty on Pk ( km/s )
            Pk_noise =  Noise power spectrum Pk_noise-/W2  ( km/s )
            Pk_metal =  Metals power spectrum ( km/s )
            Pk_syst  =  Systematic 1-sigma uncertainty on Pk  ( km/s )
        """

        n_zbin=int((z_bin-2.2+0.01)/0.2)

        if method == 0 :
            #path_to_fitsfile = "/Users/iflood/Documents/Summer2019_DESI/NLC_Ina/NPD2013_data/table4a.fits"
            path_to_fitsfile = "../NPD2013_data/table4a.fits"
            n_kbin=35

        if method == 1 :
            #path_to_fitsfile = "/Users/iflood/Documents/Summer2019_DESI/NLC_Ina/NPD2013_data/table5a.fits"
            path_to_fitsfile = "../NPD2013_data/table5a.fits"
            n_kbin=32
            
        z=fitsio.read(path_to_fitsfile,columns="z")
        k=fitsio.read(path_to_fitsfile,columns="k")
        Pk=fitsio.read(path_to_fitsfile,columns="Pk")
        Pk_stat=fitsio.read(path_to_fitsfile,columns="Pk_stat")
        Pk_noise=fitsio.read(path_to_fitsfile,columns="Pk_noise")
        Pk_metal=fitsio.read(path_to_fitsfile,columns="Pk_metal")
        Pk_syst=fitsio.read(path_to_fitsfile,columns="Pk_syst")

        self.method=method
        self.z=z_bin
        self.k=k[n_zbin*n_kbin:(n_zbin+1)*n_kbin]
        self.Pk=Pk[n_zbin*n_kbin:(n_zbin+1)*n_kbin]
        self.Pk_stat=Pk_stat[n_zbin*n_kbin:(n_zbin+1)*n_kbin]
        self.Pk_noise=Pk_noise[n_zbin*n_kbin:(n_zbin+1)*n_kbin]
        self.Pk_metal=Pk_metal[n_zbin*n_kbin:(n_zbin+1)*n_kbin]
        self.Pk_syst=Pk_syst[n_zbin*n_kbin:(n_zbin+1)*n_kbin]


    def Pk_emp(self):
        """
            Returns the empirical function Pk_emp( k ; z_bin ) which fits each power spectrum distribution.
            Neglects the wiggles introduced by SiIII cross-correlation.
        """

        z0 = 3.0 # Redshift of np.pivot point from NPD2013
        k0 = 0.009 # Scale value of np.pivot point from NPD2013, ( s/km )
        d_nu = 2271 # Bump due to SiIII absorption, ( km/s )

        #a = 0.008 / ( 1-0.006 ) # normalization factor for SiIII / ( 1 - Fbar(z) ) ; Fbar(z) TBD

        if self.method == 0 :
            Af =  0.067
            nf =  -2.50
            alpha_f = -0.08
            Bf = 3.36
            beta_f = -0.29

        if self.method == 1 :
            Af = 0.064
            nf = -2.55
            alpha_f = -0.10
            Bf = 3.55
            beta_f = -0.28

        exp = 3 + nf + alpha_f*np.log(self.k/k0)+beta_f*np.log((1+self.z)/(1+z0))

        kPk = np.pi * Af * (self.k/k0)**exp * ((1+self.z)/(1+z0))**Bf #* ( 1 + a**2 + 2*a*np.cos(self.k*d_nu) )

        return kPk / self.k
