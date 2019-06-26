import numpy as np

def W(s):
	return np.arctan(s)/s

def b_UV_hMpc(k_hMpc, b_lya = -0.134, b_g = 0.13, b_sa = 1, b_a = -2./3, k0 = 300):
	"""
	Returns the scale dependent Lya bias from Gontcho A Gontcho et al (2014).
	b_g, b_sa, b_a and k0 are parameters described in Gontcho A Gontcho et al (2014).

	Reference b_lya is computed from Table 4 of dMdB et al (2017).
	It is extrapolated for z = 2.4 from AUTO (Bautisa et al 2017a ) and CROSS (dMdB et al 2017).

	To turn off the UV fluctation effect, set b_g = 0.
	"""

	window = W(k_hMpc/k0)

	return b_lya + b_g * b_sa  * ( window / (1 + b_a * window) )

def b_evol(z,a_lya=2.9):
	"""
	Redshift evolution for the bias.

	a_lya is the value observed in measurements of the flux correlation, xi_(ff,1D),
	within individual forests, cf McDonald et al (2006).

	The reference redshift for LyA parameters is 2.4

	To turn off the redshift evolution, set a_lya = 0.
	"""

	z_ref_lya = 2.4

	return ( (1 + z) / ( 1 + z_ref_lya ) )**a_lya

def beta_UV_hMpc(k_hMpc, z, beta_lya = 1.650, b_lya = -0.134, b_g = 0.13, b_sa = 1, b_a = -2./3, k0 = 300, a_lya=2.9):
	"""
	Returns the scale dependent Lya beta from Gontcho A Gontcho et al (2014).

	"""
	return b_lya * beta_lya / b_UV_hMpc(k_hMpc,b_lya,b_g,b_sa,b_a,k0)

def Kaiser_LyA_hMpc(k_hMpc, mu, z, beta_lya = 1.650, b_lya = -0.134, b_g = 0.13, a_lya=2.9):
	"""
	Returns the standard Kaiser factor for LyA (see Kaiser 1989).

	UV fluctuation from Gontcho A Gontcho et al (2014) included.
	To turn off the UV fluctation effect, set b_g = 0.

	Redshift evolution of LyA bias included.
	To turn off the redshift evolution, set a_lya = 0.

	Redshift evolution of LyA redshift space distortions not included.

	"""

	bias = b_UV_hMpc(k_hMpc, b_lya=b_lya, b_g=b_g) * b_evol(z, a_lya=a_lya)

	beta = beta_UV_hMpc(k_hMpc, z, beta_lya=beta_lya, b_lya=b_lya, b_g=b_g, a_lya=a_lya)

	return bias**2 * (1 + beta * mu**2 )**2
