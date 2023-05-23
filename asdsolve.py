import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from numba import njit

@njit
def kappa(r, a):
    """The kappa(r,a) function as defined by equation 15 in Rafikov 2023
    """
    return np.abs(1. - r/a)

def e_a_kernel(a, r, e_a):
    """General kernel for calculating the ASD of a disk with a precise
    eccentricity to semi-major axis relationship, e_a(a).
    """
    k = kappa(r, a)
    if e_a(a) > k:
        result = 1./np.sqrt(e_a(a)**2 - k**2)
    else:
        result = 0.
    return result

def e_dist_integrand(e, a, r, psi_e):
    """The integrand for the eccentricity distribution integral (eq. 28, Rafikov 2023)
    """
    return psi_e(a,e) / np.sqrt(e**2 - kappa(r,a)**2)

def e_dist_kernel(a, r, psi_e, quad_kwargs):
    """The kernel for calculating the ASD of a disk with a distribution
    of eccentricities at each semi-major axis, psi_e(a,e).
    """
    k = kappa(r,a)
    if k < 1.:
        result = integrate.quad(e_dist_integrand, k, 1., args=(a, r, psi_e), **quad_kwargs)[0]
    else:
        result = 0.
    return result

def rafi_integrand(a, r, f, kernel):
    """The general integrand for solving the asd, kernel is variable (eq. 16, Rafikov 2023)
    """
    return f(a) * kernel(a, r) / (2.*np.pi**2 * a**2)

def rafi_integral(r_val, a_i, a_o, f, kernel, quad_kwargs):
    """Perform the actual integration of equation 16 in Rafikov 2023
    """
    if a_i > (r_val/2.):
        result = integrate.quad(rafi_integrand, a_i, a_o, args=(r_val, f, kernel), **quad_kwargs)
    else:
        result = integrate.quad(rafi_integrand, r_val/2., a_o, args=(r_val, f, kernel), **quad_kwargs)
    return result

def asdsolve(r, a_i, a_o, f, e_func, kernel_type, quad_kwargs={}):
    """The general function for solving for an ASD of the given phase space disk.
    """
    # define kernel
    if kernel_type =='e_a':
        kernel = lambda a, r: e_a_kernel(a, r, e_func)
    elif kernel_type == 'phi_e':
        kernel = e_func
    elif kernel_type == 'psi_e':
        kernel = lambda a, r: e_dist_kernel(a, r, e_func, quad_kwargs)
    else:
        raise ValueError("kernel type not available")
    # vectorize integral
    v_rafi_integral = np.vectorize(rafi_integral)
    return v_rafi_integral(r, a_i, a_o, f, kernel, quad_kwargs)
