#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:21:16 2019

@author: iflood
"""

import cosmoCAMB as cCAMB
import theoryLya as tLyA
import numpy as np
from arinyo2015 import getFiducialValues
import get_npd_p1d as npd
import scipy.optimize as op
import os
import matplotlib.pyplot as plt

z = 2.4 #redshift value 
headFile = "run89" #subfolder of "output" to write fitting results to

# make the output folder if it does not already exist
if not os.path.exists('../output/'+headFile):
    os.makedirs('../output/'+headFile)
    

######################################################################
## Define fiducial parameter values and the 1D power spectrum model ##
######################################################################

q1_f, q2_f, kp_f, kvav_f, av_f, bv_f = getFiducialValues(z)
p0 = [q1_f, q2_f, kp_f, kvav_f, av_f, bv_f]
ndims = len(p0)

cosmo = cCAMB.Cosmology(z)
th = tLyA.TheoryLya(cosmo)
dkMz = th.cosmo.dkms_dhMpc(z)

data = npd.LyA_P1D(z)
k_vals = data.k
P = data.Pk
Perr = data.Pk_stat
k_res = k_vals*dkMz

def model(k, q1, q2, kp, kvav, av, bv):
    m = th.FluxP1D_hMpc(z, k*dkMz, q1=q1, q2=q2, kp=kp, kvav=kvav, av=av, bv=bv)*dkMz
    return m

###############################
## Perform the curve fitting ##
###############################
    
min_list = [0,0,0,0,0,0]
max_list = [2,3,25,2,2,5]

popt, pcov = op.curve_fit(model, k_vals, P, p0=p0, bounds=(min_list,max_list))

print("Fit values:", popt)
print("Uncertainties:",np.sqrt(pcov[0][0]),np.sqrt(pcov[1][1]),
      np.sqrt(pcov[2][2]),np.sqrt(pcov[3][3]),np.sqrt(pcov[4][4]),np.sqrt(pcov[5][5]))
print("Covariance:", pcov)

###########################################
## Write fitting results to output files ##
###########################################

valuesfile = open('../output/'+headFile+'/values.dat','w')
valuesfile.write('{0} {1} {2} {3} {4} \n'.format(str(popt[0]),str(popt[1]),
                     str(popt[2]),str(popt[3]),str(popt[4]),str(popt[5])))
valuesfile.close()

covfile = open('../output/'+headFile+'/covariance.dat','w')
for d in range(ndims):
    covfile.write('{0} {1} {2} {3} {4} \n'.format(str(pcov[d][0]),str(pcov[d][1]),
                     str(pcov[d][2]),str(pcov[d][3]),str(pcov[d][3]),str(pcov[d][4])))
covfile.close()

resultView = plt.figure(1)
plt.yscale('log')

q1_p, q2_p, kp_p, kvav_p, av_p, bv_p = popt
plt.plot(k_vals,th.FluxP1D_hMpc(z, k_vals*dkMz, q1=q1_p, q2=q2_p, kp=kp_p,kvav=kvav_p, av=av_p, bv=bv_p)*k_res/np.pi
         , color="r", lw=2, alpha=0.8)
plt.errorbar(k_vals, P*k_vals/np.pi, yerr=Perr*k_vals/np.pi, fmt=".k")

plt.xlabel('k [(km/s)^-1]')
plt.ylabel('P(k)*k/pi')
plt.title('Fit of P1D to data')

resultView.savefig("../output/"+headFile+"/fitFigure.pdf")
resultView.show()
