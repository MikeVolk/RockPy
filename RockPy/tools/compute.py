import numpy as np


def rotate(xyz, axis='x', deg=0):  # todo make rotation axis arbitrary
    """
    Rotates a vector by 'a' degrees around 'x','y', or 'z' axis.

    Parameters
    ----------
    df
    colX
    colY
    colZ
    axis
    deg
    """

    a = np.radians(deg)

    RX = [[1, 0, 0],
          [0, np.cos(a), -np.sin(a)],
          [0, np.sin(a), np.cos(a)]]

    RY = [[np.cos(a), 0, np.sin(a)],
          [0, 1, 0],
          [-np.sin(a), 0, np.cos(a)]]

    RZ = [[np.cos(a), -np.sin(a), 0],
          [np.sin(a), np.cos(a), 0],
          [0, 0, 1]]

    if axis.lower() == 'x':
        out = np.dot(xyz, RX)
    if axis.lower() == 'y':
        out = np.dot(xyz, RY)
    if axis.lower() == 'z':
        out = np.dot(xyz, RZ)

    return out


def convert_to_XYZ(D, I, M):
    M = 1 if M is None else M
    x = np.cos(np.radians(I)) * np.cos(np.radians(D)) * M
    y = np.cos(np.radians(I)) * np.sin(np.radians(D)) * M
    z = np.cos(np.radians(I)) * np.tan(np.radians(I)) * M
    return np.array([x, y, z])


def convert_to_DIM(x, y, z):
    M = np.linalg.norm([x, y, z], axis=0)  # calculate total moment for all rows
    D = np.degrees(np.arctan2(y, x)) % 360  # calculate D and map to 0-360 degree range
    I = np.degrees(np.arcsin(z / M))  # calculate I
    return np.array([D, I, M])