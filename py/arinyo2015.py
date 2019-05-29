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

	to_exp = D_NL_hMpc(k_hMpc,Pk_lin,q1,q2) *  D_v_hMpc(k_hMpc,mu,kvav=0.48,av=0.156) - D_p_hMpc(k_hMpc,kp=9.2)

	return np.exp(to_exp) 