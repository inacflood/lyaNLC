#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:08:41 2019

@author: iflood
"""

import numpy as np
import matplotlib.pyplot as plt
import cosmoCAMB as cCAMB
import theoryLya as tLyA
import get_npd_p1d_woFitsio as npd

# Create 1D power spectra from 3D spectra and get data to compare with

z22=2.2
cosmo22 = cCAMB.Cosmology(z22)
th22 = tLyA.TheoryLyaP3D(cosmo22)
dkM22 = th22.cosmo.dkms_dhMpc               # units correction factor
p3t1_22 = th22.makeP1D_I(z22)#,linear=True) # 1D power spectrum from manual Riemannian sum
p3t1_22_t = th22.makeP1D_T(z22)               # 1D power spectrum from np.trapz integration
npd22 = npd.LyA_P1D(z22)                    # Data for corresponding redshift

z24=2.4
cosmo24 = cCAMB.Cosmology(z24)
th24 = tLyA.TheoryLyaP3D(cosmo24)
dkM24 = th24.cosmo.dkms_dhMpc
p3t1_24 = th24.makeP1D_I(z24)#,linear=True)
p3t1_24_t = th24.makeP1D_T(z24) 
npd24 = npd.LyA_P1D(z24)

z26=2.6
cosmo26 = cCAMB.Cosmology(z26)
th26 = tLyA.TheoryLyaP3D(cosmo26)
dkM26 = th26.cosmo.dkms_dhMpc
p3t1_26 = th26.makeP1D_I(z26)#,linear=True)
p3t1_26_t = th26.makeP1D_T(z26) 
npd26 = npd.LyA_P1D(z26)

z28=2.8
cosmo28 = cCAMB.Cosmology(z28)
th28 = tLyA.TheoryLyaP3D(cosmo28)
dkM28 = th28.cosmo.dkms_dhMpc
p3t1_28 = th28.makeP1D_I(z28)#,linear=True)
p3t1_28_t = th28.makeP1D_T(z28) 
npd28 = npd.LyA_P1D(z28)

z30=3.0
cosmo30 = cCAMB.Cosmology(z30)
th30 = tLyA.TheoryLyaP3D(cosmo30)
dkM30 = th30.cosmo.dkms_dhMpc
p3t1_30 = th30.makeP1D_I(z30)#,linear=True)
p3t1_30_t = th30.makeP1D_T(z30) 
npd30 = npd.LyA_P1D(z30)

z32=3.2
cosmo32 = cCAMB.Cosmology(z32)
th32 = tLyA.TheoryLyaP3D(cosmo32)
dkM32 = th32.cosmo.dkms_dhMpc
p3t1_32 = th32.makeP1D_I(z30)#,linear=True)
p3t1_32_t = th32.makeP1D_T(z32) 
npd32 = npd.LyA_P1D(z32)

z34=3.4
cosmo34 = cCAMB.Cosmology(z34)
th34 = tLyA.TheoryLyaP3D(cosmo34)
dkM34 = th34.cosmo.dkms_dhMpc
p3t1_34 = th34.makeP1D_I(z34)#,linear=True)
p3t1_34_t = th34.makeP1D_T(z34) 
npd34 = npd.LyA_P1D(z34)

z36=3.6
cosmo36 = cCAMB.Cosmology(z36)
th36 = tLyA.TheoryLyaP3D(cosmo36)
dkM36 = th36.cosmo.dkms_dhMpc
p3t1_36 = th36.makeP1D_I(z36)#,linear=True)
p3t1_36_t = th36.makeP1D_T(z36) 
npd36 = npd.LyA_P1D(z36)

z38=3.8
cosmo38 = cCAMB.Cosmology(z38)
th38 = tLyA.TheoryLyaP3D(cosmo38)
dkM38 = th38.cosmo.dkms_dhMpc
p3t1_38 = th38.makeP1D_I(z38)#,linear=True)
p3t1_38_t = th38.makeP1D_T(z38) 
npd38 = npd.LyA_P1D(z38)

z40=4.0
cosmo40 = cCAMB.Cosmology(z40)
th40 = tLyA.TheoryLyaP3D(cosmo40)
dkM40 = th40.cosmo.dkms_dhMpc
p3t1_40 = th40.makeP1D_I(z40)#,linear=True)
p3t1_40_t = th40.makeP1D_T(z40) 
npd40 = npd.LyA_P1D(z40)

z42=4.2
cosmo42 = cCAMB.Cosmology(z42)
th42 = tLyA.TheoryLyaP3D(cosmo42)
dkM42 = th42.cosmo.dkms_dhMpc
p3t1_42 = th42.makeP1D_I(z42)#,linear=True)
p3t1_42_t = th42.makeP1D_T(z42) 
npd42 = npd.LyA_P1D(z42)

z44=4.4
cosmo44 = cCAMB.Cosmology(z44)
th44 = tLyA.TheoryLyaP3D(cosmo44)
dkM44 = th44.cosmo.dkms_dhMpc
p3t1_44 = th44.makeP1D_I(z44)#,linear=True)
p3t1_44_t = th44.makeP1D_T(z44) 
npd44 = npd.LyA_P1D(z44)



# Plotting everything

manInt = plt.figure(1)
ax = plt.subplot(111)

plt.yscale('log')
plt.xscale('linear')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
plt.xlim(0.001,0.02)
plt.ylim(0.001,1)

plt.plot(p3t1_24[0]/dkM24(z24) , p3t1_24[0] * p3t1_24[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')

plt.plot(p3t1_22[0]/dkM22(z22) , p3t1_22[0] * p3t1_22[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.2 $, th')
plt.plot( npd22.k , npd22.k/np.pi*(npd22.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')

plt.plot(p3t1_26[0]/dkM26(z24) , p3t1_26[0] * p3t1_26[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 2.6 $, th')
plt.plot( npd26.k , npd26.k/np.pi*(npd26.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=2.6$, npd')

plt.plot(p3t1_28[0]/dkM28(z28) , p3t1_28[0] * p3t1_28[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 2.8 $, th')
plt.plot( npd28.k , npd28.k/np.pi*(npd28.Pk_emp()) , 'k^', alpha=0.4, linewidth=2, label=r'$z=2.8$, npd')

plt.plot(p3t1_30[0]/dkM30(z30) , p3t1_30[0] * p3t1_30[1] / np.pi,'c',alpha=0.7, linewidth=2 ,label=r'$z = 3.0 $, th')
plt.plot( npd30.k , npd30.k/np.pi*(npd30.Pk_emp()) , 'c^', alpha=0.4, linewidth=2, label=r'$z=3.0$, npd')

plt.plot(p3t1_32[0]/dkM32(z32) , p3t1_32[0] * p3t1_32[1] / np.pi,'m',alpha=0.7, linewidth=2 ,label=r'$z = 3.2 $, th')
plt.plot( npd32.k , npd32.k/np.pi*(npd32.Pk_emp()) , 'm^', alpha=0.4, linewidth=2, label=r'$z=3.2$, npd')

plt.plot(p3t1_34[0]/dkM30(z34) , p3t1_34[0] * p3t1_34[1] / np.pi,'y',alpha=0.7, linewidth=2 ,label=r'$z = 3.4 $, th')
plt.plot( npd34.k , npd34.k/np.pi*(npd34.Pk_emp()) , 'y^', alpha=0.4, linewidth=2, label=r'$z=3.4$, npd')

plt.plot(p3t1_36[0]/dkM36(z36) , p3t1_36[0] * p3t1_36[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 3.6 $, th')
plt.plot( npd36.k , npd36.k/np.pi*(npd36.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=3.6$, npd')

plt.plot(p3t1_38[0]/dkM30(z38) , p3t1_38[0] * p3t1_38[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 3.8 $, th')
plt.plot( npd38.k , npd38.k/np.pi*(npd38.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=3.8$, npd')

plt.plot(p3t1_40[0]/dkM40(z40) , p3t1_40[0] * p3t1_40[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 4.0 $, th')
plt.plot( npd40.k , npd40.k/np.pi*(npd40.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=4.0$, npd')

plt.plot(p3t1_42[0]/dkM42(z42) , p3t1_42[0] * p3t1_42[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 4.0 $, th')
plt.plot( npd42.k , npd42.k/np.pi*(npd42.Pk_emp()) , 'k^', alpha=0.4, linewidth=2, label=r'$z=4.0$, npd')

plt.plot(p3t1_44[0]/dkM44(z44) , p3t1_44[0] * p3t1_44[1] / np.pi,'c',alpha=0.7, linewidth=2 ,label=r'$z = 4.0 $, th')
plt.plot( npd44.k , npd44.k/np.pi*(npd44.Pk_emp()) , 'c^', alpha=0.4, linewidth=2, label=r'$z=4.0$, npd')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"1D Flux power, theory (non linear) v P-D, Riem. Sum",fontsize=15)
manInt.savefig("../Figures/P3D_to_P1D_nonlinear_riemsum.pdf")

# Doing the same plot again, but this time using P1D from makeP1D_T, i.e. integral is done using
# np.trapz
trapzInt = plt.figure(2)
ax = plt.subplot(111)

plt.yscale('log')
plt.xscale('linear')
plt.xlabel(r'$k_{\parallel}\,\left(\rm km/s\right)^{-1}$', fontsize=15)
plt.ylabel(r'$(k_{\parallel}/\pi)*P_{1D}(k_{\parallel})$', fontsize=15)
plt.xlim(0.001,0.02)
plt.ylim(0.001,1)

plt.plot(p3t1_24_t[0]/dkM24(z24) , p3t1_24_t[0] * p3t1_24_t[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 2.4 $, th')
plt.plot( npd24.k , npd24.k/np.pi*(npd24.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=2.4$, npd')

plt.plot(p3t1_22_t[0]/dkM22(z22) , p3t1_22_t[0] * p3t1_22_t[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 2.2 $, th')
plt.plot( npd22.k , npd22.k/np.pi*(npd22.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=2.2$, npd')

plt.plot(p3t1_26_t[0]/dkM26(z24) , p3t1_26_t[0] * p3t1_26_t[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 2.6 $, th')
plt.plot( npd26.k , npd26.k/np.pi*(npd26.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=2.6$, npd')

plt.plot(p3t1_28_t[0]/dkM28(z28) , p3t1_28_t[0] * p3t1_28_t[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 2.8 $, th')
plt.plot( npd28.k , npd28.k/np.pi*(npd28.Pk_emp()) , 'k^', alpha=0.4, linewidth=2, label=r'$z=2.8$, npd')

plt.plot(p3t1_30_t[0]/dkM30(z30) , p3t1_30_t[0] * p3t1_30_t[1] / np.pi,'c',alpha=0.7, linewidth=2 ,label=r'$z = 3.0 $, th')
plt.plot( npd30.k , npd30.k/np.pi*(npd30.Pk_emp()) , 'c^', alpha=0.4, linewidth=2, label=r'$z=3.0$, npd')

plt.plot(p3t1_32_t[0]/dkM32(z32) , p3t1_32_t[0] * p3t1_32_t[1] / np.pi,'m',alpha=0.7, linewidth=2 ,label=r'$z = 3.2 $, th')
plt.plot( npd32.k , npd32.k/np.pi*(npd32.Pk_emp()) , 'm^', alpha=0.4, linewidth=2, label=r'$z=3.2$, npd')

plt.plot(p3t1_34_t[0]/dkM30(z34) , p3t1_34_t[0] * p3t1_34_t[1] / np.pi,'y',alpha=0.7, linewidth=2 ,label=r'$z = 3.4 $, th')
plt.plot( npd34.k , npd34.k/np.pi*(npd34.Pk_emp()) , 'y^', alpha=0.4, linewidth=2, label=r'$z=3.4$, npd')

plt.plot(p3t1_36_t[0]/dkM36(z36) , p3t1_36_t[0] * p3t1_36_t[1] / np.pi,'r',alpha=0.7, linewidth=2 ,label=r'$z = 3.6 $, th')
plt.plot( npd36.k , npd36.k/np.pi*(npd36.Pk_emp()) , 'r^', alpha=0.4, linewidth=2, label=r'$z=3.6$, npd')

plt.plot(p3t1_38_t[0]/dkM30(z38) , p3t1_38_t[0] * p3t1_38_t[1] / np.pi,'b',alpha=0.7, linewidth=2 ,label=r'$z = 3.8 $, th')
plt.plot( npd38.k , npd38.k/np.pi*(npd38.Pk_emp()) , 'b^', alpha=0.4, linewidth=2, label=r'$z=3.8$, npd')

plt.plot(p3t1_40_t[0]/dkM40(z40) , p3t1_40_t[0] * p3t1_40_t[1] / np.pi,'g',alpha=0.7, linewidth=2 ,label=r'$z = 4.0 $, th')
plt.plot( npd40.k , npd40.k/np.pi*(npd40.Pk_emp()) , 'g^', alpha=0.4, linewidth=2, label=r'$z=4.0$, npd')

plt.plot(p3t1_42_t[0]/dkM42(z42) , p3t1_42_t[0] * p3t1_42_t[1] / np.pi,'k',alpha=0.7, linewidth=2 ,label=r'$z = 4.0 $, th')
plt.plot( npd42.k , npd42.k/np.pi*(npd42.Pk_emp()) , 'k^', alpha=0.4, linewidth=2, label=r'$z=4.0$, npd')

plt.plot(p3t1_44_t[0]/dkM44(z44) , p3t1_44_t[0] * p3t1_44_t[1] / np.pi,'c',alpha=0.7, linewidth=2 ,label=r'$z = 4.0 $, th')
plt.plot( npd44.k , npd44.k/np.pi*(npd44.Pk_emp()) , 'c^', alpha=0.4, linewidth=2, label=r'$z=4.0$, npd')

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title(r"1D Flux power, theory (non linear) v P-De, Trapz. Int",fontsize=15)
trapzInt.savefig("../Figures/P3D_to_P1D_nonlinear_trapzint.pdf")