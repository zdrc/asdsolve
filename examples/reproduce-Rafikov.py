import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy.special import erf

sys.path.insert(0, '/Users/zdrc/work/asdsolve')
from asdsolve import asdsolve, e_dist_kernel, kappa

def e_a_rafi(a, e_0, a_min, a_max):
    """The power law distribution used for e_a(a) in Figure 3 of Rafikov 2023.
    """
    if isinstance(a, float):
        if a > a_min and a < a_max:
            result = e_0 * (a_min/a)**0.7
        else:
            result = 0.
    else:
        result = np.zeros(a.shape)
        good_inds = (a<a_max) & (a>a_min)
        result[good_inds] = e_0 * (a_min/a[good_inds])**0.7
    return result

def sigma_a_rafi_sqrt(a, sigma_0, a_min, a_max):
    """The square root distribution in semi-major axis used for the middle panel of 
    Figure 3 in Rafikov 2023.
    """
    if isinstance(a, float):
        if a > a_min and a < a_max:
            result = sigma_0 * np.sqrt(a_min/a)
        else:
            result = 0.
    else:
        result = np.zeros(a.shape)
        good_inds = (a<a_max) & (a>a_min)
        result[good_inds] = sigma_0 * np.sqrt(a_min/a[good_inds])
    return result

def sigma_a_rafi_uni(a, sigma_0, a_min, a_max):
    """The uniform distribution in semi-major axis used for the bottom panel of 
    Figure 3 in Rafikov 2023.
    """
    if isinstance(a, float):
        if a > a_min and a < a_max:
            result = sigma_0
        else:
            result = 0.
    else:
        result = np.zeros(a.shape)
        good_inds = (a<a_max) & (a>a_min)
        result[good_inds] = sigma_0 * np.ones(a[good_inds].shape)
    return result

@njit
def psi_e_rafi(a, e, a_min, a_max):
    """The eccentricity distribution used to create Figure 4 in Rafikov 2023.
    A truncated Rayleigh distribution, see equation 20 in Rafikov 2023.
    """
    if isinstance(a, float):
        if a > a_min and a < a_max:
            scale = 0.4 * np.sqrt(a_min/a)
            result = e * np.exp(-e**2/(2.*scale**2)) / (scale**2 * (1.-np.exp(-1./(2.*scale**2))))
        else:
            result = 0.
    else:
        result = np.zeros(a.shape)
        good_inds = (a<a_max) & (a>a_min)
        scale = 0.4*np.sqrt(a_min/a[good_inds])
        result[good_inds] = e * np.exp(-e**2/(2.*scale**2)) / (scale**2 * (1.-np.exp(-1./(2.*scale**2)))) \
                            * np.ones(a[good_inds].shape)
    return result

def phi_e_rafi(a, r, a_min, a_max):
    """The analytically integrated kernel for the Rayleigh distribution used to make
    figure 4 in Rafikob 2023.
    """
    if isinstance(a, float):
        if a > a_min and a < a_max:
            scale = 0.4 * np.sqrt(a_min/a)
            result = np.sqrt(np.pi/2.) * np.exp(-kappa(r, a)**2 / (2.*scale**2)) * erf(np.sqrt( (1.-kappa(r,a)**2) / (2.*scale**2) )) \
                        / (scale * (1.-np.exp(-1./(2.*scale**2))))
        else:
            result = 0.
    else:
        result = np.zeros(a.shape)
        good_inds = (a<a_max) & (a>a_min)
        scale = 0.4 * np.sqrt(a_min/a[good_inds])
        result[good_inds] = np.sqrt(np.pi/2.) * np.exp(-kappa(r, a[good_inds])**2 / (2.*scale**2)) * erf(np.sqrt( (1.-kappa(r,a[good_inds])**2) / (2.*scale**2) )) \
                                / (scale * (1.-np.exp(-1./(2.*scale**2))))
    return result

def Rafikov_2023_fig3():
    """Reproduce Fig3 from Rafikov 2023
    """
    # radial range
    r = np.linspace(1e-3, 5.5, num=1000)
    # define first disk
    sigma_0 = 1.
    e_0 = 0.5
    a_min_sqrt, a_max_sqrt = 1., 4.
    # define distributions for first disk
    e_a_sqrt = lambda a: e_a_rafi(a, e_0, a_min_sqrt, a_max_sqrt)
    sigma_a_sqrt = lambda a: sigma_a_rafi_sqrt(a, sigma_0, a_min_sqrt, a_max_sqrt)
    f_sqrt = lambda a: sigma_a_sqrt(a) * (2*np.pi*a)
    # get the asd using the Rafikov method
    asd_rafikov_sqrt, asd_err_sqrt = asdsolve(r, a_min_sqrt, a_max_sqrt, f_sqrt, e_a_sqrt, 'e_a', {'limit': 200})
    # repeat for different disk
    a_min_uni, a_max_uni = 2., 3.
    sigma_a_uni = lambda a: sigma_a_rafi_uni(a, sigma_0, a_min_uni, a_max_uni)
    f_uni = lambda a: sigma_a_uni(a) * (2*np.pi*a)
    asd_rafikov_uni, asd_err_uni = asdsolve(r, a_min_uni, a_max_uni, f_uni, e_a_sqrt, 'e_a', {'limit': 100})
    # show both
    fig, ax = plt.subplots(2, constrained_layout=True, sharex=True, figsize=(6,4))
    ax[0].plot(r, sigma_a_sqrt(r), label=r'$\Sigma_a(a)$')
    ax[0].plot(r, asd_rafikov_sqrt, label=r'$\overline{\Sigma}(r)$')
    #ax_err_sqrt = ax[0].twinx()
    #ax_err_sqrt.plot(r, asd_err_sqrt, ls='--')
    ax[1].plot(r, sigma_a_uni(r), label=r'$\Sigma_a(a)$')
    ax[1].plot(r, asd_rafikov_uni, label=r'$\overline{\Sigma}(r)$')
    #ax_err_uni = ax[1].twinx()
    #ax_err_uni.plot(r, asd_err_uni, ls='--')
    ax[0].set_ylabel(r'$\Sigma_a(a),\,\overline{\Sigma}(r)$')
    #ax_err_sqrt.set_ylabel('integration error')
    ax[1].set_ylabel(r'$\Sigma_a(a),\,\overline{\Sigma}(r)$')
    #ax_err_uni.set_ylabel('integration error')
    ax[1].set_xlabel(r'$r,a$')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    plt.savefig('Rafikov-fig3.pdf')
    plt.show()

def Rafikov_2023_fig4():
    """Reproduce Fig4 from Rafikov 2023
    """
    # radial range
    r = np.linspace(1e-3, 5.5, num=1000)
    # define first disk
    sigma_0 = 1.
    e_0 = 0.5
    a_min_sqrt, a_max_sqrt = 1., 4.
    # define distributions for first disk
    sigma_a_sqrt = lambda a: sigma_a_rafi_sqrt(a, sigma_0, a_min_sqrt, a_max_sqrt)
    f_sqrt = lambda a: sigma_a_sqrt(a) * (2*np.pi*a)
    psi_e = lambda a, e: psi_e_rafi(a, e, a_min_sqrt, a_max_sqrt)
    phi_e = lambda a, r: phi_e_rafi(a, r, a_min_sqrt, a_max_sqrt)
    # get the kernel and plot
    a_kernel = np.linspace(1e-3, a_max_sqrt+0.5, num=100)
    r_grid, a_grid = np.meshgrid(r, a_kernel)
    v_e_dist_kernel = np.vectorize(e_dist_kernel)
    kernel = v_e_dist_kernel(a_grid, r_grid, psi_e, {})
    cm = plt.pcolormesh(r_grid, a_grid, kernel, shading='nearest')
    plt.ylabel(r'$a$')
    plt.xlabel(r'$r$')
    plt.colorbar(cm, label=r'$\Phi_e(r,a)$')
    plt.savefig('Rafikov-fig4-kernel.pdf')
    plt.show()
    # get the asd using psi_e
    asd_rafikov_sqrt, asd_err_sqrt = asdsolve(r, a_min_sqrt, a_max_sqrt, f_sqrt, psi_e, 'psi_e')
    # repeat for different disk and using phi_e
    a_min_uni, a_max_uni = 2., 3.
    sigma_a_uni = lambda a: sigma_a_rafi_uni(a, sigma_0, a_min_uni, a_max_uni)
    f_uni = lambda a: sigma_a_uni(a) * (2*np.pi*a)
    asd_rafikov_uni, asd_err_uni = asdsolve(r, a_min_uni, a_max_uni, f_uni, phi_e, 'phi_e')
    # show both
    fig, ax = plt.subplots(2, constrained_layout=True, sharex=True, figsize=(6,4))
    ax[0].plot(r, sigma_a_sqrt(r), label=r'$\Sigma_a(a)$')
    ax[0].plot(r, asd_rafikov_sqrt, label=r'$\overline{\Sigma}(r)$')
    ax[1].plot(r, sigma_a_uni(r), label=r'$\Sigma_a(a)$')
    ax[1].plot(r, asd_rafikov_uni, label=r'$\overline{\Sigma}(r)$')
    ax[0].set_ylabel(r'$\Sigma_a(a),\,\overline{\Sigma}(r)$')
    ax[1].set_ylabel(r'$\Sigma_a(a),\,\overline{\Sigma}(r)$')
    ax[1].set_xlabel(r'$r,a$')
    ax[0].grid()
    ax[1].grid()
    ax[0].legend()
    ax[1].legend()
    plt.savefig('Rafikov-fig4.pdf')
    plt.show()

if __name__ == "__main__":
    Rafikov_2023_fig3()
    #Rafikov_2023_fig4()
