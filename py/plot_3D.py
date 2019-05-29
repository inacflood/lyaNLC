import numpy as np
import cosmoCAMB as cCAMB
import theoryLyaP3D as tP3D
import arinyo2015 as nlclya

## Getting started : generate data to plot

k=np.logspace(-4,0.9,1000)	

## flux powers

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tP3D.TheoryLyaP3D(cosmo24)
p24 = th24.FluxP3D_hMpc(z24,k,0)
p24_UV_only = th24.FluxP3D_hMpc(z24,k,0,linear=True) 
p24_NLC_only = th24.FluxP3D_hMpc(z24,k,0,uv=False)
p24_noUV_noNLC = th24.FluxP3D_hMpc(z24,k,0,linear=True,uv=False)

z26=2.6
cosmo26 = cCAMB.Cosmology(z26)
th26 = tP3D.TheoryLyaP3D(cosmo26)
p26 = th26.FluxP3D_hMpc(z26,k,0) 
p26_UV_only = th26.FluxP3D_hMpc(z26,k,0,linear=True) 
p26_NLC_only = th26.FluxP3D_hMpc(z26,k,0,uv=False)
p26_noUV_noNLC = th26.FluxP3D_hMpc(z26,k,0,linear=True,uv=False)

z28=2.8
cosmo28 = cCAMB.Cosmology(z28)
th28 = tP3D.TheoryLyaP3D(cosmo28)
p28 = th28.FluxP3D_hMpc(z28,k,0) 
p28_UV_only = th28.FluxP3D_hMpc(z28,k,0,linear=True) 
p28_NLC_only = th28.FluxP3D_hMpc(z28,k,0,uv=False)
p28_noUV_noNLC = th28.FluxP3D_hMpc(z28,k,0,linear=True,uv=False)

z30=3.0
cosmo30 = cCAMB.Cosmology(z30)
th30 = tP3D.TheoryLyaP3D(cosmo30)
p30 = th30.FluxP3D_hMpc(z30,k,0) 
p30_UV_only = th30.FluxP3D_hMpc(z30,k,0,linear=True) 
p30_NLC_only = th30.FluxP3D_hMpc(z30,k,0,uv=False)
p30_noUV_noNLC = th30.FluxP3D_hMpc(z30,k,0,linear=True,uv=False)

z22=2.2
cosmo22 = cCAMB.Cosmology(z22)
th22 = tP3D.TheoryLyaP3D(cosmo22)
p22 = th22.FluxP3D_hMpc(z22,k,0) 
p22_UV_only = th22.FluxP3D_hMpc(z22,k,0,linear=True) 
p22_NLC_only = th22.FluxP3D_hMpc(z22,k,0,uv=False)
p22_noUV_noNLC = th22.FluxP3D_hMpc(z22,k,0,linear=True,uv=False)


## D_NL function for different mu

d_nl_24_000 = nlclya.D_hMpc_AiP2015(k,0.,th24.linPk(k))
d_nl_24_025 = nlclya.D_hMpc_AiP2015(k,0.25,th24.linPk(k))
d_nl_24_050 = nlclya.D_hMpc_AiP2015(k,0.5,th24.linPk(k))
d_nl_24_075 = nlclya.D_hMpc_AiP2015(k,0.75,th24.linPk(k))
d_nl_24_100 = nlclya.D_hMpc_AiP2015(k,1,th24.linPk(k))


## Flux power, z=2.4 at different mu

p24_025 = th24.FluxP3D_hMpc(z24,k,0.25)
p24_050 = th24.FluxP3D_hMpc(z24,k,0.50)
p24_075 = th24.FluxP3D_hMpc(z24,k,0.75)
p24_100 = th24.FluxP3D_hMpc(z24,k,1.00)


## set the log log frame for a power plot

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [
  r'\usepackage{siunitx}',   
  r'\sisetup{detect-all}',   
  r'\usepackage{helvet}',    
  r'\usepackage{sansmath}',  
  r'\sansmath'            
]
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'k [h $\rm{Mpc}^{-1}$]',fontsize=15)

## plot a comparison between different redshifts

#plt.ylabel(r'$(k/$\pi$)P_{F}(k)$',fontsize=15)

#plt.plot(k,k * p22_NLC_only / np.pi,'k',alpha=0.6, linewidth=2 ,label=r'$z = 2.2 $')
#plt.plot(k,k * p24_NLC_only / np.pi,'r',alpha=0.6, linewidth=2 ,label=r'$z = 2.4 $')
#plt.plot(k,k * p26_NLC_only / np.pi,'g',alpha=0.6, linewidth=2 ,label=r'$z = 2.6 $')
#plt.plot(k,k * p28_NLC_only / np.pi,'b',alpha=0.6, linewidth=2 ,label=r'$z = 2.8 $')
#plt.plot(k,k * p30_NLC_only / np.pi,'m',alpha=0.6, linewidth=2 ,label=r'$z = 3.0 $')

#plt.title(r"Flux power, non linear, at $\mu$ = 0 ",fontsize=15)
#plt.legend(loc='best',fontsize=16)
#plt.savefig("Figures/P3D_nonlinear.pdf")
#plt.show()

## plot a comparison between different models at a given reddshift

#plt.ylabel(r'$(k/$\pi$)P_{F}(k)$',fontsize=15)

#plt.plot(k,k * p24_noUV_noNLC / np.pi,'k',alpha=0.6, linewidth=2 ,label=r'$linear$')
#plt.plot(k,k * p24_NLC_only / np.pi,'r--',alpha=0.6, linewidth=2 ,label=r'$non linear$')

#plt.title(r"Flux power, at $z = 2.4$ and $\mu$ = 0 ",fontsize=15)
#plt.legend(loc='best',fontsize=16)
#plt.savefig("Figures/P3D_24_lin_v_nlin.pdf")
#plt.show()

## plot figure 3 Arinyo 2015 (right)

plt.ylabel(r'$D_{NL}(k,\mu)$',fontsize=15)
plt.xlim(0.04,10)
plt.ylim(0.09,4)
plt.tick_params(width=1.5,length=4)

plt.plot(k,d_nl_24_000,'k',alpha=0.6, linewidth=2 ,label=r'$\mu = 0.00 $')
plt.plot(k,d_nl_24_025,'r',alpha=0.6, linewidth=2 ,label=r'$\mu = 0.25 $')
plt.plot(k,d_nl_24_050,'g',alpha=0.6, linewidth=2 ,label=r'$\mu = 0.50 $')
plt.plot(k,d_nl_24_075,'b',alpha=0.6, linewidth=2 ,label=r'$\mu = 0.75 $')
plt.plot(k,d_nl_24_100,'m',alpha=0.6, linewidth=2 ,label=r'$\mu = 1.0 $')

plt.title(r"Arinyo 2015 $D_{NL}$, at $z = 2.4$ ",fontsize=15)
plt.legend(loc='best',fontsize=16)
plt.savefig("Figures/Arinyo2015_fig3b.pdf")
plt.show()


#plt.close()