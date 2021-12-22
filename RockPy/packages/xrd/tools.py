import numpy as np
import scipy as sp
import pandas as pd


def wavelength(anode):
    """
    Args:
        anode:
    """
    anodes = {'Cr': 2.29, 'Fe': 1.94, 'Co': 1.79, 'Cu': 1.54, 'Mo': 0.71, 'Ag': 0.56, '11BM-B': 0.413}
    return anodes[anode]


def theta_to_q(theta, lamb):
    """Calculate the Theta (1 Theta!) value into Q

    Args:
        theta (theta values):
        lamb (wavelength of the xrd radiation):

    Returns:
        Q: **array**
    """

    if isinstance(lamb, str):
        lamb = wavelength(lamb)

    return (4 * np.pi * np.sin(np.deg2rad(theta))) / lamb


def q_to_theta(q, lamb):
    """Calculates theta from the Q value

    Args:
        q (Q value):
        lamb (wavelength of xrd ratiation):

    Returns:
        list new theta (1 Theta!) values:
    """
    if isinstance(lamb, str):
        lamb = wavelength(lamb)

    theta = np.arcsin((lamb * q) / (4 * np.pi))
    return np.rad2deg(theta)

def pdd_transpose_wavelength(pdd, lambda1, lambda2, column='index'):
    """transposes the wavelength of a pandas dataframe from one wavelength to a
    second one.

    Args:
        pdd:
        lambda1:
        lambda2:
        column:
    """

    pdd = pdd.copy()

    theta = pdd.index if column == 'index' else pdd[column]
    q = theta_to_q(theta=theta/2, lamb=lambda1)
    theta_new = q_to_theta(q, lamb=lambda2) *2

    if column == 'index':
        pdd.index = theta_new
    else:
        pdd[column] = theta_new

    return pdd

