import numpy as np


def pressure(force, diameter):
    """
    calculates the pressure for a given force (T) and a diameter (mm)

    Returns
    -------
        pressure in Pa
    """

    force = force * 1000 * 10  # translate T into kN
    diameter /= 1000  # translate mm in m

    area = ((diameter / 2) ** 2) * np.pi

    pressure = force / area  # N/m2 / Pa

    return pressure

def overburden_pressure(thickness, density=2600):
    """
    Calculates the approximate pressure of a layer (thickness) of rock with density (density).

    Parameters
    ----------
    density: float
        default: 3300 kg/m^3
        in kg/m^3
    thickness: float
        in meter

    Returns
    -------
        pressure in Pa
    """

    return density * 10 * thickness