import numpy as np


### ROTATIONS

def rotate_around_axis(xyz, axis_unit_vector, theta, axis_di=False, dim=False, reshape=False):
    """
    Rotates a vector [x,y,z] or array of vectors around an arbitrary axis.
     
    Parameters
    ----------
    xyz array like
        data that shall get rotated
    axis_unit_vector array like
        axis around which the rotation is supposed to happen
    theta float
        angle of rotation
    dim: bool
        default: False
        if True the xyz array contains declination and inclination values
    axis_di: bool
        default: False
        if True the axis_unit_vector array contains declination and inclination values
    reshape: bool
        default: False
        changes the used input and output array shape from (n,3) if False to (3,n) 
        
    Returns
    -------
    np.array
        if dim = True will return DIM values
        if reshape = False: [[x1,y1,z1], [x2,y2,z2]]
        if reshape = True: [[x1,x1], [y2,y2], [z1,z2]]
    """
    xyz = maintain_shape(xyz)

    if dim:
        xyz = convert_to_xyz(xyz, reshape=reshape)

    if axis_di:
        axis_unit_vector = [axis_unit_vector[0], axis_unit_vector[1], 1]
        axis_unit_vector = convert_to_xyz(axis_unit_vector)[0]

    if not reshape:
        xyz = xyz.T

    axis_unit_vector = axis_unit_vector / np.linalg.norm(axis_unit_vector)
    ux, uy, uz = axis_unit_vector

    theta = np.radians(theta)
    cost = np.cos(theta)
    sint = np.sin(theta)

    R = [[cost + ux ** 2 * (1 - cost), ux * uy * (1 - cost) - uz * (sint), ux * uz * (1 - cost) + uy * (sint)],
         [uy * ux * (1 - cost) + uz * sint, cost + uy ** 2 * (1 - cost), uy * uz * (1 - cost) - ux * sint],
         [uz * ux * (1 - cost) - uy * sint, uz * uy * (1 - cost) + ux * sint, cost + uz ** 2 * (1 - cost)]]

    out = np.dot(R, xyz).T

    if dim:
        out = convert_to_dim(out, reshape = reshape)

    out = handle_near_zero(out)

    if reshape:
        return out.T
    else:
        return out


def rotate_arbitrary(xyz, alpha, beta, gamma, dim = False, reshape=False):
    """

    Parameters
    ----------
    xyz
    alpha
    beta
    gamma
    reshape

    Returns
    -------
    np.array
        if dim = True will return DIM values

    """
    xyz = maintain_shape(xyz)

    if dim:
        xyz = convert_to_xyz(xyz, reshape=reshape)

    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    if not reshape:
        xyz = xyz.T

    R = np.dot(np.dot(RZ(alpha), RY(beta)), RX(gamma))

    out = np.dot(R, xyz).T

    if dim:
        out = convert_to_dim(out, reshape = reshape)

    out = handle_near_zero(out)

    if reshape:
        return out.T
    else:
        return out


def RX(angle):
    """
    Rotation matrix around X axis

    Parameters
    ----------
    angle: float
        angle in radians

    Returns
    -------
    np.array
        Rotationmatrix
    """
    RX = [[1, 0, 0],
          [0, np.cos(angle), -np.sin(angle)],
          [0, np.sin(angle), np.cos(angle)]]
    return RX


def RY(angle):
    """
    Rotation matrix around Y axis

    Parameters
    ----------
    angle: float
        angle in radians

    Returns
    -------
    np.array
        Rotationmatrix
    """
    RY = [[np.cos(angle), 0, np.sin(angle)],
          [0, 1, 0],
          [-np.sin(angle), 0, np.cos(angle)]]
    return RY


def RZ(angle):
    """
    Rotation matrix around Z axis

    Parameters
    ----------
    angle: float
        angle in radians

    Returns
    -------
    np.array
        Rotationmatrix
    """
    RZ = [[np.cos(angle), -np.sin(angle), 0],
          [np.sin(angle), np.cos(angle), 0],
          [0, 0, 1]]
    return RZ


def rotate(xyz, axis='x', theta=0, reshape=False):
    """
    Rotates the vector 'xyz' by 'theta' degrees around 'x','y', or 'z' axis.

    Parameters
    ----------
    """
    xyz = maintain_shape(np.array(xyz))

    if reshape:
        xyz = xyz.T

    theta = np.radians(-theta)

    if axis.lower() == 'x':
        out = np.dot(xyz, RX(theta))
    if axis.lower() == 'y':
        out = np.dot(xyz, RY(theta))
    if axis.lower() == 'z':
        out = np.dot(xyz, RZ(theta))

    out = handle_near_zero(out)

    if reshape:
        return out.T
    else:
        return out


### CONVERSIONS

def convert_to_xyz(dim, reshape=False):
    """
    Converts a numpy array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]]) into an numpy array with [[d1,i1,m1], [d2,i2,m2]].
    Reshape allows to pass an [[x1,x2],[y1,y2],[z1,z2]] array instead.

    Parameters
    ----------
    dim: np.array, list
        data either shape(n,3) of shape(3,n)
    reshape: bool
        default: False
        changes the used input and output array shape from (n,3) if False to (3,n) 

    Returns
    -------
        np.array
        if reshape = False: [[x1,y1,z1], [x2,y2,z2]]
        if reshape = True: [[x1,x1], [y2,y2], [z1,z2]]
    """
    dim = maintain_shape(np.array(dim))

    if reshape:
        dim = dim.T

    D = dim[:, 0]
    I = dim[:, 1]
    M = dim[:, 2]

    M = 1 if M is None else M
    x = np.cos(np.radians(I)) * np.cos(np.radians(D)) * M
    y = np.cos(np.radians(I)) * np.sin(np.radians(D)) * M
    z = np.cos(np.radians(I)) * np.tan(np.radians(I)) * M

    out = np.array([x, y, z]).T
    out = handle_near_zero(out)

    if reshape:
        return out.T
    else:
        return out


def convert_to_dim(xyz, reshape=False):
    """
    Converts a numpy array of [d,i,m] values (i.e. [[d1,i1,m1], [d2,i2,m2]]) into an numpy array with [[x1,y1,z1], [x2,y2,z2]]
    Reshape allows to pass an [[d1,d2],[i1,i2],[m1,m2]] array instead.

    Parameters
    ----------
    xyz: np.array, list
        data either shape(n,3) of shape(3,n)
    reshape: bool
        default: False
        changes the used input and output array shape from (n,3) if False to (3,n) 

    Returns
    -------
        np.array
        if reshape = False: [[d1,i1,m1], [d2,i2,m2]]
        if reshape = True: [[d1,d1], [i2,i2], [m1,m2]]
    """
    xyz = maintain_shape(xyz)

    if reshape:
        xyz = xyz.T

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    M = np.linalg.norm([x, y, z], axis=0)  # calculate total moment for all rows
    D = np.degrees(np.arctan2(y, x)) % 360  # calculate D and map to 0-360 degree range
    I = np.degrees(np.arcsin(z / M))  # calculate I

    out = np.array([D, I, M]).T
    out = handle_near_zero(out)

    if reshape:
        return out.T
    else:
        return out


def convert_to_stereographic(xyz, dim=False, reshape=False):
    """
    Transforms an array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]]) into an
    numpy array with [d,r,neg], where:
     d = declination,
     r = radius in equal area projection and
     neg = array of 0,1 values where 0 = i > 0 and 1 = i < 0

    r is calculated according to Collinson, 1983 by
    .. math:
        R = 1 - ( R_0 * [ 1 - \tan(\frac{\pi}{4} - \frac{I}{2})]

    Parameters
    ----------
    xyz array like
    dim bool, optional
        default False
        if True the xyz array does not contain [x,y,z] values but [d,i,m],
            where d = declination, i = inclination, and m = moment
    reshape: bool, optional
        default: False
        changes the used input and output array shape from (n,3) if False to (3,n)

    Returns
    -------
        np.array
        if reshape = False: [[d1,i1,neg], [d2,i2,neg2]]
        if reshape = True: [[d1,d1], [i2,i2], [neg1,neg2]]

    See Also
    --------
    convert_to_dim, convert_to_xyz, convert_to_equal_area
    """

    xyz = maintain_shape(np.array(xyz))

    if not dim:
        dim = convert_to_dim(xyz, reshape=reshape)

    if reshape:
        dim = dim.T

    d = dim[:, 0]
    i = np.radians(dim[:, 1])
    neg = i < 0

    r = 1 - (1 - np.tan((np.pi / 4) - (abs(i) / 2)))

    out =  np.array([d, r, neg]).T

    if reshape:
        return out.T
    else:
        return out


def convert_to_equal_area(xyz, dim=False, reshape=False):
    """
    Transforms an array of [x,y,z] values (i.e. [[x1,y1,z1], [x2,y2,z2]]) into an
    numpy array with [d,r,neg], where:
     d = declination,
     r = radius in equal area projection and
     neg = array of 0,1 values where 0 = i > 0 and 1 = i < 0

    r is calculated according to Collinson, 1983 by
    .. math:
        R = 1 - ( R_0 * [ 1 - \sqrt{1- \sin(I)}])

    Parameters
    ----------
    xyz array like
    dim bool, optional
        default False
        if True the xyz array does not contain [x,y,z] values but [d,i,m],
            where d = declination, i = inclination, and m = moment
    reshape: bool, optional
        default: False
        changes the used input and output array shape from (n,3) if False to (3,n)

    Returns
    -------
        np.array
        if reshape = False: [[d1,i1,neg1], [d2,i2,neg2]]
        if reshape = True: [[d1,d1], [i2,i2], [neg1,neg2]]

    See Also
    --------
    convert_to_dim, convert_to_xyz, convert_to_stereographic
    """

    xyz = maintain_shape(xyz)

    if not dim:
        dim = convert_to_dim(xyz, reshape=reshape)
    else:
        dim = xyz

    if reshape:
        dim = dim.T

    d = dim[:, 0]
    i = dim[:, 1]
    neg = i < 0

    r = 1 - np.abs(i) / 90
    # i = np.radians(np.abs(i))
    # r = np.sqrt((1 - np.sin(i)) ** 2 + np.cos(i) ** 2) / np.sqrt(2) ?? why is this wrong???
    out =  np.array([d, r, neg]).T

    if reshape:
        return out.T
    else:
        return out

def convert_to_hvl(data, dim=False): #todo make consistent with other convert
    if not dim:
        data = convert_to_dim(data)
    D = data[:, 0]
    I = data[:, 1]
    M = data[:, 2]

    h = M * np.cos(np.radians(I))
    v = M * np.sin(np.radians(I))

    return array([h, v, M]).T

def maintain_shape(d):
    if isinstance(d, (list, tuple)):
        d = np.array(d)

    if len(d.shape) == 1:
        return np.array([d])
    else:
        return d


def handle_near_zero(d):
    d[np.isclose(d, 0)] = 0
    return d
