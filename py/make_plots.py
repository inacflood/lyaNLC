import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB as cCAMB
import theoryLya as thLya
import get_npd_p1d as npd
import itertools


# Tableau Color Blind 10
tableau10blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
              (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
              (255, 188, 121), (207, 207, 207)]

# Rescale to values between 0 and 1
for i in range(len(tableau10blind)):
    r, g, b = tableau10blind[i]
    tableau10blind[i] = (r / 255., g / 255., b / 255.)

colors = itertools.cycle(tableau10blind)

def plot_3D(z_list,mu=1.0,linear=False,uv=False,zevol=True):

    """
    Plots the 3D power spectrum for redshift values in z_list, at a given mu.

    If linear = True, it will ignore small scale correction.
    If uv = False, it will ignore uv fluctuations.
    If zevol = False, it will ignore redshift evolution of the bias.


    """

    if linear == False:
        lin="nlc"
    else:
        lin="linear"

    if uv == False:
        uvb=""
    else:
        uvb="UV"

    if zevol == False:
        ev=", no zevol for bias"
    else:
        ev=""

    spe=lin+uvb+ev+",$\mu = "+str(mu)+"$"
    spe2=lin+uvb+ev

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'k [h $\rm{Mpc}^{-1}$]', fontsize=16)
    plt.xscale('log')
    plt.yscale('log')
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
    plt.title("LyA P3D model ("+spe+")")

    k=np.logspace(-4,0.9,1000)

    c=0
    for z in z_list:
        cosmo = cCAMB.Cosmology(z)
        th = thLya.TheoryLya(cosmo)
        model = th.FluxP3D_hMpc(z,k,mu=mu,linear=linear,uv=uv,zevol=zevol)

        plt.plot(k, k * model / np.pi, color=tableau10blind[c], linewidth=2 ,label=r'$z = '+str(z)+ ' $, th')
        c=c+1

    plt.legend(loc='best')
    plt.savefig("../Figures/P3D_data_v_model_"+spe2+".pdf")

def plot_p1d_ft():

    """
    Plots the Fourier transform data from Palanque-Delabrouille et al (2013).
    """

    p22 = npd.LyA_P1D(2.2,0)
    p24 = npd.LyA_P1D(2.4,0)
    p26 = npd.LyA_P1D(2.6,0)
    p28 = npd.LyA_P1D(2.8,0)
    p30 = npd.LyA_P1D(3.0,0)
    p32 = npd.LyA_P1D(3.2,0)
    p34 = npd.LyA_P1D(3.4,0)
    p36 = npd.LyA_P1D(3.6,0)
    p38 = npd.LyA_P1D(3.8,0)
    p40 = npd.LyA_P1D(4.0,0)
    p42 = npd.LyA_P1D(4.2,0)
    p44 = npd.LyA_P1D(4.4,0)

    colors1 = itertools.cycle(["purple","red","blue","black","limegreen","orange","darkturquoise","plum","yellow","lightskyblue","silver","forestgreen"])
    colors2 = itertools.cycle(["purple","red","blue","black","limegreen","orange","darkturquoise","plum","yellow","lightskyblue","silver","forestgreen"])

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'$kP(k)/\pi$', fontsize=16)
    plt.errorbar(p22.k,p22.k/np.pi*(p22.Pk),yerr=p22.k/np.pi*p22.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p24.k,p24.k/np.pi*(p24.Pk),yerr=p24.k/np.pi*p24.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p26.k,p26.k/np.pi*(p26.Pk),yerr=p26.k/np.pi*p26.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p28.k,p28.k/np.pi*(p28.Pk),yerr=p28.k/np.pi*p28.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p30.k,p30.k/np.pi*(p30.Pk),yerr=p30.k/np.pi*p30.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p32.k,p32.k/np.pi*(p32.Pk),yerr=p32.k/np.pi*p32.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p34.k,p34.k/np.pi*(p34.Pk),yerr=p34.k/np.pi*p34.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p36.k,p36.k/np.pi*(p36.Pk),yerr=p36.k/np.pi*p36.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p38.k,p38.k/np.pi*(p38.Pk),yerr=p38.k/np.pi*p38.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p40.k,p40.k/np.pi*(p40.Pk),yerr=p40.k/np.pi*p40.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p42.k,p42.k/np.pi*(p42.Pk),yerr=p42.k/np.pi*p42.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p44.k,p44.k/np.pi*(p44.Pk),yerr=p44.k/np.pi*p44.Pk_stat,fmt='o',color=next(colors1))

    plt.plot(p22.k,p22.k/np.pi*(p22.Pk_emp()),color=next(colors2))
    plt.plot(p24.k,p24.k/np.pi*(p24.Pk_emp()),color=next(colors2))
    plt.plot(p26.k,p26.k/np.pi*(p26.Pk_emp()),color=next(colors2))
    plt.plot(p28.k,p28.k/np.pi*(p28.Pk_emp()),color=next(colors2))
    plt.plot(p30.k,p30.k/np.pi*(p30.Pk_emp()),color=next(colors2))
    plt.plot(p32.k,p32.k/np.pi*(p32.Pk_emp()),color=next(colors2))
    plt.plot(p34.k,p34.k/np.pi*(p34.Pk_emp()),color=next(colors2))
    plt.plot(p36.k,p36.k/np.pi*(p36.Pk_emp()),color=next(colors2))
    plt.plot(p38.k,p38.k/np.pi*(p38.Pk_emp()),color=next(colors2))
    plt.plot(p40.k,p40.k/np.pi*(p40.Pk_emp()),color=next(colors2))
    plt.plot(p42.k,p42.k/np.pi*(p42.Pk_emp()),color=next(colors2))
    plt.plot(p44.k,p44.k/np.pi*(p44.Pk_emp()),color=next(colors2))

    plt.yscale('log')
    plt.xlim(0.0015, 0.01975)
    plt.ylim(0.005,0.55)
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
    plt.title("LyA P1D Fourier Transform (NPD2013)")
    plt.savefig("../Figures/LyA_P1D_ft_allz_simple.pdf")

def plot_p1d_likelihood():
    """
    Plots the Fourier transform data from Palanque-Delabrouille et al (2013).
    """

    p24 = npd.LyA_P1D(2.4,0)
    p26 = npd.LyA_P1D(2.6,0)
    p28 = npd.LyA_P1D(2.8,0)
    p30 = npd.LyA_P1D(3.0,0)
    p32 = npd.LyA_P1D(3.2,0)
    p34 = npd.LyA_P1D(3.4,0)
    p36 = npd.LyA_P1D(3.6,0)
    p38 = npd.LyA_P1D(3.8,0)
    p40 = npd.LyA_P1D(4.0,0)
    p42 = npd.LyA_P1D(4.2,0)
    p44 = npd.LyA_P1D(4.4,0)

    colors1 = itertools.cycle(["purple","red","blue","black","limegreen","orange","darkturquoise","plum","yellow","lightskyblue","silver","forestgreen"])
    colors2 = itertools.cycle(["purple","red","blue","black","limegreen","orange","darkturquoise","plum","yellow","lightskyblue","silver","forestgreen"])

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'$kP(k)/\pi$', fontsize=16)
    plt.errorbar(p22.k,p22.k/np.pi*(p22.Pk),yerr=p22.k/np.pi*p22.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p24.k,p24.k/np.pi*(p24.Pk),yerr=p24.k/np.pi*p24.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p26.k,p26.k/np.pi*(p26.Pk),yerr=p26.k/np.pi*p26.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p28.k,p28.k/np.pi*(p28.Pk),yerr=p28.k/np.pi*p28.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p30.k,p30.k/np.pi*(p30.Pk),yerr=p30.k/np.pi*p30.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p32.k,p32.k/np.pi*(p32.Pk),yerr=p32.k/np.pi*p32.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p34.k,p34.k/np.pi*(p34.Pk),yerr=p34.k/np.pi*p34.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p36.k,p36.k/np.pi*(p36.Pk),yerr=p36.k/np.pi*p36.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p38.k,p38.k/np.pi*(p38.Pk),yerr=p38.k/np.pi*p38.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p40.k,p40.k/np.pi*(p40.Pk),yerr=p40.k/np.pi*p40.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p42.k,p42.k/np.pi*(p42.Pk),yerr=p42.k/np.pi*p42.Pk_stat,fmt='o',color=next(colors1))
    plt.errorbar(p44.k,p44.k/np.pi*(p44.Pk),yerr=p44.k/np.pi*p44.Pk_stat,fmt='o',color=next(colors1))

    plt.plot(p22.k,p22.k/np.pi*(p22.Pk_emp()),color=next(colors2))
    plt.plot(p24.k,p24.k/np.pi*(p24.Pk_emp()),color=next(colors2))
    plt.plot(p26.k,p26.k/np.pi*(p26.Pk_emp()),color=next(colors2))
    plt.plot(p28.k,p28.k/np.pi*(p28.Pk_emp()),color=next(colors2))
    plt.plot(p30.k,p30.k/np.pi*(p30.Pk_emp()),color=next(colors2))
    plt.plot(p32.k,p32.k/np.pi*(p32.Pk_emp()),color=next(colors2))
    plt.plot(p34.k,p34.k/np.pi*(p34.Pk_emp()),color=next(colors2))
    plt.plot(p36.k,p36.k/np.pi*(p36.Pk_emp()),color=next(colors2))
    plt.plot(p38.k,p38.k/np.pi*(p38.Pk_emp()),color=next(colors2))
    plt.plot(p40.k,p40.k/np.pi*(p40.Pk_emp()),color=next(colors2))
    plt.plot(p42.k,p42.k/np.pi*(p42.Pk_emp()),color=next(colors2))
    plt.plot(p44.k,p44.k/np.pi*(p44.Pk_emp()),color=next(colors2))

    plt.yscale('log')
    plt.xlim(0.000, 0.021)
    plt.ylim(0.007,1)
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
    plt.title("LyA P1D Likelihood (NPD2013)")
    plt.savefig("../Figures/LyA_P1D_likelihood_allz_simple.pdf")

def plot_dataVmodel(z_list,linear=False,uv=False,zevol=True,emp=False):

    """
    Plots the data from Palanque-Delabrouille et al (2013) against the model from theoryLya
    for the 1D power spectrum (obtained from the 3D power spectrum) for redshift values in z_list.

    If linear = True, it will ignore small scale correction.
    If uv = False, it will ignore uv fluctuations.
    If zevol = False, it will ignore redshift evolution of the bias.


    """

    if linear == False:
        lin="nlc"
    else:
        lin="linear"

    if uv == False:
        uvb=""
    else:
        uvb="UV"

    if zevol == False:
        ev=", no zevol for bias"
    else:
        ev=""

    if emp == True:
        e="_emp"
    else:
        e=""

    spe=lin+uvb+ev
    spe2=lin+uvb+ev+e

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'$k\,\left(\rm km/s\right)^{-1}$', fontsize=16)
    ax.set_ylabel(r'$kP(k)/\pi$', fontsize=16)
    plt.yscale('log')
    plt.xlim(0.0015, 0.01975)
    plt.ylim(0.005,0.55)
    plt.tick_params(length=10,width=1.2,which='major')
    plt.tick_params(length=5,width=.9,which='minor')
    plt.title("LyA P1D (NPD2013) v. P1D model ("+spe+")")

    kpa_start,kpa_stop=[-4,np.log10(10.0-0.01)]
    kpa_list=np.logspace(kpa_start,kpa_stop,200)

    c=0
    for z in z_list:
        cosmo = cCAMB.Cosmology(z)
        th = thLya.TheoryLya(cosmo)
        dkMz = th.cosmo.dkms_dhMpc(z)     # units correction factor
        model = th.FluxP1D_hMpc(z,kpa_list,linear=linear,uv=uv,zevol=zevol)
        data = npd.LyA_P1D(z)             # NPD data for corresponding redshift

        #color_pick=next(colors)
        plt.plot(kpa_list/dkMz , kpa_list * model / np.pi, color=tableau10blind[c], linewidth=2 ,label=r'$z = '+str(z)+ ' $, th')
        plt.errorbar(data.k , data.k * data.Pk / np.pi, yerr=data.k * data.Pk_stat / np.pi, marker='o',
            mfc=tableau10blind[c], ls='None', linewidth=2, label=r'$z='+str(z)+ '$, npd')
        if emp == True :
            plt.plot(data.k , data.k * data.Pk_emp() / np.pi, color=tableau10blind[c], alpha=0.3, linewidth=2, linestyle='-')
        c=c+1

#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=11)
    plt.legend(loc='best')
    plt.savefig("../Figures/P1D_data_v_model_"+spe2+".pdf")
