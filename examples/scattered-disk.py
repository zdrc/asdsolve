import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '/Users/zdrc/work/asdsolve')
from asdsolve import asdsolve, e_dist_kernel, kappa

def pwr_lw(a, alpha, a_i, a_o, a_off=0.):
    """A general power law distribution
    """
    # calculate normalization constant
    if alpha == -1.:
        c = 1./np.log((a_o-a_off)/(a_i-a_off))
    else:
        c = (1.+alpha) / ((a_o-a_off)**(1.+alpha) - (a_i-a_off)**(1.+alpha))
    if isinstance(a, float):
        if a >= a_i and a <= a_o:
            result = c * (a-a_off)**(alpha)
        else:
            result = 0.
    else:
        result = np.zeros(a.shape)
        inds = (a<=a_o) & (a>=a_i)
        result[inds] = c * (a[inds]-a_off)**(alpha)
    return result

def sd_density_neg1(r, q, Q_i, Q_o, Md):
    """Analytic ASD for a scattered disk with a power law semi-major axis distribution with 
    f(a) \propto a^{-1}.
    """
    c = 1/np.log((Q_o+q)/(Q_i+q))
    sigma = np.zeros(r.shape[0])
    for i in range(r.shape[0]):
        if r[i] > q:
            if r[i] < Q_i:
                sigma[i] = 1/(2*np.sqrt(r[i]-q)*(r[i]+q)**(3/2)) * \
                         ( np.arctan(np.sqrt((Q_o-r[i])/(r[i]+q))) + np.sqrt((r[i]+q)*(Q_o-r[i]))/(Q_o+q) -
                           np.arctan(np.sqrt((Q_i-r[i])/(r[i]+q))) - np.sqrt((r[i]+q)*(Q_i-r[i]))/(Q_i+q) )
            elif r[i] < Q_o:
                sigma[i] = 1/(2*np.sqrt(r[i]-q)*(r[i]+q)**(3/2)) * \
                         ( np.arctan(np.sqrt((Q_o-r[i])/(r[i]+q))) + np.sqrt((r[i]+q)*(Q_o-r[i]))/(Q_o+q) )
    return (2*c*Md / np.pi**2) * sigma

def test_power_law():
    """Use Rafikov's method to numerically find the surface density for a power law scattered disk
    to compare with the analytic expression I found.
    """
    # define disk
    alpha = -1.
    q = 0.3
    a_i, a_o = 1., 10.
    # define dists
    e_a = lambda a: 1 - q/a
    f = lambda a: pwr_lw(a, alpha, a_i, a_o)
    sigma_a = lambda a: f(a) / (2.*np.pi*a)
    # get the asd using the Rafikov method
    r = np.logspace(np.log10(q-1e-3), np.log10(2.*a_o), num=1000)
    asd_rafikov = asdsolve(r, a_i, a_o, f, e_a, 'e_a')[0]
    # get the analytic asd
    asd_analytic = sd_density_neg1(r, q, a_i*(1.+e_a(a_i)), a_o*(1.+e_a(a_o)), 1.)
    # show the numerical asd with the analytic
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(r, asd_rafikov, c='k', ls='-', label='Rafikov method')
    ax.plot(r, asd_analytic, c='r', ls='--', label='analytic')
    ax.plot(r, sigma_a(r), c='b', ls='--', label=r'$\Sigma_a(a)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'$\overline{\Sigma}(r)$')
    ax.set_xlabel(r'$r$ $(a_i)$')
    plt.legend()
    plt.savefig('pwr-law-scattered-disk-asd.pdf')
    plt.show()

if __name__ == "__main__":
    test_power_law()
