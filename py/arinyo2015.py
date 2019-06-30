import numpy as np

def D_NL_hMpc(k_hMpc,Pk_lin,q1=0.057,q2=0.368):
	"""
	Isotropic enhancement of power due to non linear growth.
	"""
	D2 = k_hMpc**3*Pk_lin/(2*np.pi**2)
	return q1*D2 + q2*D2**2

def D_p_hMpc(k_hMpc,kp=9.2):
	"""
	Isotropic suppression of power due to gas pressure below the Jeans scale.
	"""
	return (k_hMpc/kp)**2

def D_v_hMpc(k_hMpc,mu,kvav=0.48,av=0.156,bv=1.57):
	"""
	Suppression of power due to line-of-sight non linear peculiar velocity and thermal broadening.

	Note that kvav = kv ^ av from eq(3.6) of Arinyo et al 2015 [arXiv:1506.04519]
	"""
	return 1-(k_hMpc**av/kvav) * mu**bv

def D_hMpc_AiP2015(k_hMpc,mu,Pk_lin,q1=0.057,q2=0.368,kp=9.2,kvav=0.48,av=0.156,bv=1.57):
	"""
	Returns the corrective term D_nl (also noted F_nl in Bautista et al 2017a and dMdB et al 2017) at redshift 2.4
	Corrected 3D power is P3D(k,mu;z) x Dnl(k,mu;z) .

	Analytical formula : eq(3.6) of Arinyo-i-Prats et al 2015 [arXiv:1506.04519]
	"""

	to_exp = D_NL_hMpc(k_hMpc,Pk_lin,q1,q2) *  D_v_hMpc(k_hMpc,mu,kvav=kvav,av=av) - D_p_hMpc(k_hMpc,kp=kp)

	return np.exp(to_exp)

#def getFiducialValues(z):
#    q1, q2, kvav, kp, av, bv = np.zeros(6)
#    if z==2.2:
#        q1, q2, kp, kvav, av, bv = [0.090, 0.316, 8.9, 0.493, 0.145, 1.54]
#    elif z==2.4:
#        q1, q2, kp, kvav, av, bv = [0.057, 0.368, 9.2, 0.480, 0.156, 1.57]
#    elif z==2.6:
#        q1, q2, kp, kvav, av, bv = [0.068, 0.390, 9.6, 0.483, 0.190, 1.61]
#    elif z==2.8:
#        q1, q2, kp, kvav, av, bv = [0.086, 0.417, 9.9, 0.493, 0.217, 1.63]
#    elif z==3.0:
#        q1, q2, kp, kvav, av, bv = [0.104, 0.444, 10.1, 0.516, 0.248, 1.66]
#    else:
#        print("Invalid z-value")
#        return
#    return q1, q2, kp, kvav, av, bv
    
def getFiducialValues(z):
    q1, q2, kvav, kp, av, bv = np.zeros(6)
    if z==2.2:
        q1, kp, kvav, av, bv = [0.677, 13.3, 0.961, 0.533, 1.54]
    elif z==2.4:
        q1, kp, kvav, av, bv = [0.666, 13.5, 0.963, 0.561, 1.58]
    elif z==2.6:
        q1, kp, kvav, av, bv = [0.652, 13.6, 0.970, 0.590, 1.61]
    elif z==2.8:
        q1, kp, kvav, av, bv =  [0.644, 13.4, 0.979, 0.610, 1.64]
    elif z==3.0:
        q1, kp, kvav, av, bv = [0.648, 13.1, 1.01, 0.627, 1.66] 
    else:
        print("Invalid z-value")
        return
    return q1, q2, kp, kvav, av, bv
