import numpy as np
import scipy as sp
import pandas as pd


def wavelength(anode):
    anodes = {'Cr': 2.29, 'Fe': 1.94, 'Co': 1.79, 'Cu': 1.54, 'Mo': 0.71, 'Ag': 0.56, '11BM-B': 0.413}
    return anodes[anode]


def theta_to_q(theta, lamb):
    """
    Calculate the Theta value into Q

    Parameters
    ----------
    theta: theta values
    lamb: wavelength of the XRD radiation

    Returns
    -------
    array: Q
    """

    if isinstance(lamb, str):
        lamb = wavelength(lamb)

    q = (4 * np.pi * np.sin(np.deg2rad(theta))) / lamb
    return q


def q_to_theta(q, lamb):
    """
    Calculates theta from the Q value

    Parameters
    ----------
    q: Q value
    lamb: wavelength of XRD ratiation

    Returns
    -------

    """
    if isinstance(lamb, str):
        lamb = wavelength(lamb)

    theta = np.arcsin((lamb * q) / (4 * np.pi))
    return np.rad2deg(theta)

def pdd_transpose_wavelength(pdd, lambda1, lambda2, column='index'):
    """
    transposes the wavelength of a pandas dataframe from one wavelength to a second one.

    Parameters
    ----------
    pdd
    lambda1
    lambda2

    Returns
    -------

    """

    pdd = pdd.copy()

    if column == 'index':
        theta = pdd.index
    else:
        theta = pdd[column]

    q = theta_to_q(theta=theta, lamb=lambda1)
    theta_new = q_to_theta(q, lamb=lambda2)

    if column == 'index':
        pdd.index = theta_new
    else:
        pdd[column] = theta_new

    return pdd