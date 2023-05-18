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
    if e_a(a) > kappa(r, a):
        result = 1./np.sqrt(e_a(a)**2 - kappa(r,a)**2)
    else:
        result = 0.
    return result

def e_dist_integrand(e, a, r, psi_e):
    """
    """
    return psi_e(a,e) / np.sqrt(e**2 - kappa(r,a)**2)

def e_dist_kernel(a, r, psi_e, quad_kwargs):
    """The kernel for calculating the ASD of a disk with a distribution
    of eccentricities at each semi-major axis, psi_e(a,e).
    """
    # TODO add singularity avoidance
    if kappa(r,a) < 1.:
        result = integrate.quad(e_dist_integrand, kappa(r,a), 1., args=(a, r, psi_e), **quad_kwargs)[0]
    else:
        result = 0.
    return result

def rafi_integrand(a, r, f, kernel):
    """
    """
    return f(a) * kernel(a, r) / (2.*np.pi**2 * a**2)

def rafi_integral(r_val, a_i, a_o, f, kernel, quad_kwargs):
    """
    """
    if a_i > (r_val/2.):
        result = integrate.quad(rafi_integrand, a_i, a_o, args=(r_val, f, kernel), **quad_kwargs)
    else:
        result = integrate.quad(rafi_integrand, r_val/2., a_o, args=(r_val, f, kernel), **quad_kwargs)
    return result

def asdsolve(r, a_i, a_o, f, e_func, kernel_type, quad_kwargs={}):
    """
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

# old ################################################

def find_singularities(r, e_a):
    """Find inner and outer singularities in the kernel
    """
    sing_func = lambda a, r: e_a(a)**2 - kappa(r, a)**2
    a_s_in = fsolve(sing_func, r/2., args=(r))[0]
    a_s_out = fsolve(sing_func, 2.*r, args=(r))[0]
    return (a_s_in, a_s_out)

def e_a_asd(r, a_i, a_o, f, e_func, quad_kwargs):
    """Calculates the ASD at the radius, r, due to the disk of orbits defined by f(a) and e_a(a).
    r is a float
    a_i and a_o are the inner and outer radii of the defined disk (floats)
    f is a function of a
    e_a is a function of a
    """
    if a_i > (r/2.):
        a_s = find_singularities(r, e_a)
        result, err = integrate.quad(integrand, a_i, a_o, args=(r, f, e_a), points=a_s, **quad_kwargs)
    else:
        a_s = find_singularities(r, e_a)
        result, err = integrate.quad(integrand, r/2., a_o, points=a_s, **quad_kwargs)
    return result, err

def psi_e_asd(r, a_i, a_o, f, psi_e, quad_limit=50):
    """Calculates the ASD at the radius r, due tothe disk of orbits defined by f(a) and phi(a,e).
    r is a float
    a_i and a_o are the inner and outer radii of the defined disk (floats)
    f is a function of a
    phi_e is a function of a and e
    """
    integrand = lambda a: f(a) * e_dist_kernel(r, a, phi_e) / (2. * np.pi**2 * a**2)
    if a_i > (r/2.):
        result, err = integrate.quad(integrand, a_i, a_o, limit=quad_limit)
    else:
        result, err = integrate.quad(integrand, r/2., a_o, limit=quad_limit)
    return result, err

def phi_e_asd(r, a_i, a_o, f, phi_e, quad_limit=50):
    """Calculates the ASD at the radius r, due tothe disk of orbits defined by f(a) and phi(a,e).
    r is a float
    a_i and a_o are the inner and outer radii of the defined disk (floats)
    f is a function of a
    phi_e is a function of a and e
    """
    # TODO
    pass
