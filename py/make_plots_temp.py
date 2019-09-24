import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB as cCAMB
import theoryLya as thLya
import get_npd_p1d as npd
import itertools


## Tableau Color Blind 10
#tableau10blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
#              (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
#              (255, 188, 121), (207, 207, 207)]
#
## Rescale to values between 0 and 1
#for i in range(len(tableau10blind)):
#    r, g, b = tableau10blind[i]
#    tableau10blind[i] = (r / 255., g / 255., b / 255.)
#
#colors = itertools.cycle(tableau10blind)

def plot_3D(z,mu=1.0,uv=False,zevol=True):

    """
    Plots the 3D power spectrum for redshift values in z_list, at a given mu.

    If linear = True, it will ignore small scale correction.
    If uv = False, it will ignore uv fluctuations.
    If zevol = False, it will ignore redshift evolution of the bias.


    """

    if uv == False:
        uvb=""
    else:
        uvb="UV"

    if zevol == False:
        ev=", no zevol for bias"
    else:
        ev=""

    spe=uvb+ev+"$\mu = "+str(mu)+"$"
    spe2=uvb+ev

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'(k\pi)$P_{F}(k)$', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
    #plt.title("LyA P3D model linear and nonlinear ("+spe+")")

    k=np.logspace(-4,0.9,1000)

    cosmo = cCAMB.Cosmology(z)
    th = thLya.TheoryLya(cosmo)
    model_lin = th.FluxP3D_hMpc(z,k,mu=mu,linear=True,uv=uv,zevol=zevol)
    model_nonlin = th.FluxP3D_hMpc(z,k,mu=mu,linear=False,uv=uv,zevol=zevol)

    plt.plot(k, k * model_lin / np.pi, color='b', linewidth=2) #,label=r'linear')
    #plt.plot(k, k * model_nonlin / np.pi, color='r', linewidth=2 ,label=r'nonlinear')

    plt.legend(loc='best')
    plt.savefig("../Figures/P3D_lin.pdf")
    
def plot_1D(z,uv=False,zevol=True):

    """
    Plots the 3D power spectrum for redshift values in z_list, at a given mu.

    If linear = True, it will ignore small scale correction.
    If uv = False, it will ignore uv fluctuations.
    If zevol = False, it will ignore redshift evolution of the bias.


    """

    if uv == False:
        uvb=""
    else:
        uvb="UV"

    if zevol == False:
        ev=", no zevol for bias"
    else:
        ev=""


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'(k\pi)$P_{F}(k)$', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
    plt.title("LyA P1D model linear and nonlinear")

    k=np.logspace(-4,.9,1000)

    cosmo = cCAMB.Cosmology(z)
    th = thLya.TheoryLya(cosmo)
    model_lin = th.FluxP1D_hMpc(z,k,linear=True,uv=uv,zevol=zevol)
    model_nonlin = th.FluxP1D_hMpc(z,k,linear=False,uv=uv,zevol=zevol)

    plt.plot(k, k * model_lin / np.pi, color='b', linewidth=2 ,label=r'linear')
    plt.plot(k, k * model_nonlin / np.pi, color='r', linewidth=2 ,label=r'nonlinear')

    plt.legend(loc='best')
    plt.savefig("../Figures/P1D_lin_and_nonlin.pdf")
    
def plot_1Dtemp(z,uv=False,zevol=True):

    """
    Plots the 3D power spectrum for redshift values in z_list, at a given mu.

    If linear = True, it will ignore small scale correction.
    If uv = False, it will ignore uv fluctuations.
    If zevol = False, it will ignore redshift evolution of the bias.


    """

    if uv == False:
        uvb=""
    else:
        uvb="UV"

    if zevol == False:
        ev=", no zevol for bias"
    else:
        ev=""


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'(k\pi)$P_{F}(k)$', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
   # plt.title("LyA P1D model linear and nonlinear")

    k=np.logspace(-4,.9,1000)

    cosmo = cCAMB.Cosmology(z)
    th = thLya.TheoryLya(cosmo)
    model_lin = th.FluxP1D_hMpc(z,k,linear=True,uv=uv,zevol=zevol)
    model_nonlin = th.FluxP1D_hMpc(z,k,linear=False,uv=uv,zevol=zevol)

    plt.plot(k, k * model_lin / np.pi, color='b', linewidth=2) #,label=r'linear')
    #plt.plot(k, k * model_nonlin / np.pi, color='r', linewidth=2 ,label=r'nonlinear')

    plt.legend(loc='best')
    plt.savefig("../Figures/P1D_lin.pdf")

