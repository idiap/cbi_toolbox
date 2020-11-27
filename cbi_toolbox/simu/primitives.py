"""
The primitives module generates basic 3D objects

**Conventions:**

arrays follow the ZXY convention, with

    - Z : depth axis (axial, focus axis)
    - X : horizontal axis (lateral)
    - Y : vertical axis (lateral, rotation axis when relevant)
"""

import numpy as np
import skimage.transform


def quadrant_symmetry(quadrant):
    """
    Generate a binary quadrant by rotating itself 90° on all 3 axes and summing.

    Parameters
    ----------
    quadrant : numpy.ndarray(bool)
        The original 3D quadrant.

    Returns
    -------
    numpy.ndarray
        The symmetrized quadrant. Will have cubic shape equal to the largest
        dimension of the original quadrant.
    """

    size = np.max(quadrant.shape)

    full_quadrant = np.zeros((size, size, size), dtype=bool)
    full_quadrant[:quadrant.shape[0], :quadrant.shape[1],
                  :quadrant.shape[2]] = quadrant
    full_quadrant[:quadrant.shape[1], :quadrant.shape[2],
                  :quadrant.shape[0]] |= quadrant.transpose((1, 2, 0))
    full_quadrant[:quadrant.shape[2], :quadrant.shape[0],
                  :quadrant.shape[1]] |= quadrant.transpose((2, 0, 1))

    return full_quadrant


def quadrant_to_volume(quadrant, odd=(False, False, False)):
    """
    Generate a volume by mirroring a quadrant in all 8 corners.

    Parameters
    ----------
    quadrant : numpy.ndarray
        The quadrant corresponding to the end of the axes.
    odd : tuple, optional
        If the target dimensions are odd, by default (False, False, False).
        If even, the dimension will be ``2 * quandrant.shape``.
        If odd, the dimension will be ``2 * quadrant.shape - 1``.

    Returns
    -------
    numpy.ndarray
        The full volume.
    """

    volume = np.empty(
        (2 * quadrant.shape[0] - odd[0], 2 * quadrant.shape[1] - odd[1], 2 * quadrant.shape[2] - odd[2]), dtype=quadrant.dtype)
    volume[quadrant.shape[0] - odd[0]:, quadrant.shape[1] -
           odd[1]:, quadrant.shape[2] - odd[2]:] = quadrant
    volume[:quadrant.shape[0], quadrant.shape[1] - odd[1]:,
           quadrant.shape[2] - odd[2]:] = np.flip(quadrant, 0)
    volume[:, :quadrant.shape[1] - odd[1], quadrant.shape[2] - odd[2]:] = np.flip(
        volume[:, quadrant.shape[1]:, quadrant.shape[2] - odd[2]:], 1)
    volume[:, :, :quadrant.shape[2] - odd[2]] = np.flip(
        volume[:, :, quadrant.shape[2]:], 2)

    return volume


def boccia(size, radius=None, n_stripes=3, deg_space=15, deg_width=7.5, rad_thick=0.12,
           antialias=2, dtype=np.float64):
    """
    Create a boccia simulated sample: resolution stripes on a sphere

    Parameters
    ----------
    size : int
        side of the cube containing the volume (must be even)
    radius : float, optional
        radius of the boccia, by default None (will be size-1)
    n_stripes : int, optional
        number of stripes to generate, by default 3
    deg_space : int, optional
        spacing in degrees between the center of the stripes, by default 15
    deg_width : float, optional
        width in degrees of the stripes, by default 7.5
    rad_thick : float, optional
        thickness of the stripes, as a proportion of the radius, by default 0.12
    antialias : int, optional
        antialiasing scale factor, by default 2
    dtype : numpy.dtype or str, optional
        the datatype, by default numpy.float64

    Returns
    -------
    numpy.ndarray
        The volume containing the boccia.

    Raises
    ------
    ValueError
        if the size is odd
    ValueError
        if the antialias is negative
    ValueError
        if the angle surpasses 90°
    """

    if size % 2:
        raise ValueError('The size must be even to cut it in quadrants')

    if antialias < 1 or not isinstance(antialias, int):
        raise ValueError('Antialias must be a positive integer')

    size //= 2
    if radius is None:
        radius = size - 1  # leave one pixel out for antialiasing
    size *= antialias
    radius *= antialias

    half_width = np.deg2rad(deg_width) / 2

    # the angles at the center of the stripes
    c_angles = np.arange(n_stripes / 2)

    # if even number of stripes
    if not n_stripes % 2:
        c_angles += 0.5

    c_angles *= np.deg2rad(deg_space)

    max_angle = c_angles[-1] + half_width
    if max_angle > np.pi / 2:
        raise ValueError('max angle must be less than 90°')

    width = int(np.ceil(size * np.sin(max_angle)))

    x = (np.arange(size) + 0.5).reshape((size, 1, 1))
    y = (np.arange(size) + 0.5).reshape((1, size, 1))
    z = (np.arange(width) + 0.5).reshape((1, 1, width))

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / r)
    phi = np.pi / 2 - phi

    angle_crit = np.zeros_like(phi, dtype=bool)
    for angle in c_angles:
        angle_crit |= (phi > (angle - half_width)
                       ) & (phi < (angle + half_width))

    quadrant = (r < radius) & (r > (radius * (1 - rad_thick))) & angle_crit

    quadrant = quadrant_symmetry(quadrant).astype(dtype)

    if antialias > 1:
        quadrant = skimage.transform.rescale(
            quadrant, 1 / antialias, mode='symmetric')

    return quadrant_to_volume(quadrant)


def torus_boccia(size, radius=None, n_stripes=3, deg_space=15, torus_radius=0.075,
                 antialias=2, dtype=np.float64):
    """
    Generate a boccia with torus stripes

    Parameters
    ----------
    size : int
        side of the cube containing the volume (must be even)
    radius : [type], optional
        radius of the boccia, by default None
    n_stripes : int, optional
        number of torus stripes, by default 3
    deg_space : int, optional
        spacing of the torus on the sphere in degrees, by default 15
    torus_radius : float, optional
        radius of the torus, as a fraction of the sphere radius, by default 0.075
    antialias : int, optional
        antialiasing scale factor, by default 2
    dtype : numpy.dtype or str, optional
        the datatype, by default numpy.float64

    Returns
    -------
    numpy.ndarray
        The volume containing the torus boccia.

    Raises
    ------
    ValueError
        if the size is odd
    ValueError
        if the antialias is negative
    ValueError
        if the angle surpasses 90°
    """

    if size % 2:
        raise ValueError('The size must be even to cut it in quadrants')

    if antialias < 1 or not isinstance(antialias, int):
        raise ValueError('Antialias must be a positive integer')

    size //= 2
    if radius is None:
        radius = size - 1  # leave one pixel out for antialiasing
    size *= antialias
    radius *= antialias

    # the angles at the center of the stripes
    c_angles = np.arange(n_stripes / 2)

    # if even number of stripes
    if not n_stripes % 2:
        c_angles += 0.5

    c_angles *= np.deg2rad(deg_space)

    max_angle = c_angles[-1] + np.arcsin(torus_radius)
    if max_angle > np.pi / 2:
        raise ValueError('max angle must be less than 90°')

    c_pos = np.array([np.cos(c_angles), np.sin(c_angles)]) * \
        radius * (1 - torus_radius)
    width = int(np.ceil(c_pos[1, :].max() + size * torus_radius))

    x = (np.arange(size) + 0.5).reshape((size, 1, 1, 1))
    y = (np.arange(size) + 0.5).reshape((1, size, 1, 1))
    z = (np.arange(width) + 0.5).reshape((1, 1, width, 1))

    r = np.sqrt(x ** 2 + y ** 2)

    rad = (r - c_pos[0, :]) ** 2 + (z - c_pos[1, :]) ** 2
    rad = rad.min(-1)

    quadrant = rad < (torus_radius * radius) ** 2

    quadrant = quadrant_symmetry(quadrant).astype(dtype)

    if antialias > 1:
        quadrant = skimage.transform.rescale(
            quadrant, 1 / antialias, mode='symmetric')

    return quadrant_to_volume(quadrant)


def ball(size, radius=None, in_radius=0, antialias=2, dtype=np.float64):
    """
    Generate hollow ball.

    :param size: 
    :param radius: 
    :param in_radius: inner radius fraction (set to 0 for full ball)
    :param antialias: antialiasing scale factor

    Parameters
    ----------
    size : int
        side of the cube containing the volume
    radius : float, optional
        radius of the ball, by default None
    in_radius : float, optional
        radius of the inner (empty) sphere expressed as a fraction of the outer
        radius [0-1], by default 0 (plain ball)
    antialias : int, optional
        antialiasing scale factor, by default 2
    dtype : numpy.dtype or str, optional
        the datatype, by default numpy.float64

    Returns
    -------
    numpy.ndarray
        The volume containing the ball.

    Raises
    ------
    ValueError
        if the size is odd
    ValueError
        if the antialias is negative
    """

    if size % 2:
        raise ValueError('The size must be even to cut it in quadrants')

    if antialias < 1 or not isinstance(antialias, int):
        raise ValueError('Antialias must be a positive integer')

    size //= 2
    if radius is None:
        radius = size - 1  # leave one pixel out for antialiasing
    size *= antialias
    radius *= antialias

    x = (np.arange(size) + 0.5).reshape((size, 1, 1))
    y = (np.arange(size) + 0.5).reshape((1, size, 1))
    z = (np.arange(size) + 0.5).reshape((1, 1, size))

    r = x ** 2 + y ** 2 + z ** 2

    quadrant = r < radius ** 2

    if in_radius > 0:
        quadrant &= r > (in_radius * radius) ** 2

    quadrant = quadrant.astype(dtype)

    if antialias > 1:
        quadrant = skimage.transform.rescale(
            quadrant, 1 / antialias, mode='symmetric')

    return quadrant_to_volume(quadrant)


def phantom(size, scale=1, antialias=2):
    """
    Generate 3D modified Shepp-Logann phantoms

    Parameters
    ----------
    size : int
        size of the output
    scale : float
        scaling factor for the size of the phantoms
    antialias : int, optional
        antialiasing factor, by default 2

    Returns
    -------
    array [ZXY]
        the phantoms

    Raises
    ------
    ValueError
        if antialias is not a positive integer
    """

    if antialias < 1 or not isinstance(antialias, int):
        raise ValueError('Antialias must be a positive integer')

    #      A         a     b      c     x0      y0     z0   phi  theta   psi
    ellipses = [
        [1,      .6900, .920,  .810,     0,      0,     0,    0,     0,    0],
        [-.8,    .6624, .874,  .780,     0, -.0184,     0,    0,     0,    0],
        [-.2,    .1100, .310,  .220,   .22,      0,     0,  -18,     0,   10],
        [-.2,    .1600, .410,  .280,  -.22,      0,     0,   18,     0,   10],
        [.1,     .2100, .250,  .410,     0,    .35,  -.15,    0,     0,    0],
        [.1,     .0460, .046,  .050,     0,     .1,   .25,    0,     0,    0],
        [.1,     .0460, .046,  .050,     0,    -.1,   .25,    0,     0,    0],
        [.1,     .0460, .023,  .050,  -.08,  -.605,     0,    0,     0,    0],
        [.1,     .0230, .023,  .020,     0,  -.606,     0,    0,     0,    0],
        [.1,     .0230, .046,  .020,   .06,  -.605,     0,    0,     0,    0]
    ]

    size *= antialias

    mat = np.zeros((size, size, size), dtype=np.float64)
    coords = 2 * np.indices(mat.shape) / size - 1
    coords /= scale

    for ellipse in ellipses:
        A = ellipse[0]
        a2 = ellipse[1] ** 2
        b2 = ellipse[2] ** 2
        c2 = ellipse[3] ** 2
        x0 = ellipse[4]
        y0 = ellipse[5]
        z0 = ellipse[6]
        phi = ellipse[7] * np.pi / 180
        theta = ellipse[8] * np.pi / 180
        psi = ellipse[9] * np.pi / 180

        cphi = np.cos(phi)
        sphi = np.sin(phi)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        # Euler rotation matrix with ZXY convention
        rotmat = np.array([
            [
                ctheta,
                stheta * sphi,
                -stheta * cphi,
            ],
            [
                spsi * stheta,
                cpsi * cphi - ctheta * sphi * spsi,
                cpsi * sphi + ctheta * cphi * spsi,
            ],
            [
                cpsi * stheta,
                -spsi * cphi - ctheta * sphi * cpsi,
                -spsi * sphi + ctheta * cphi * cpsi,
            ],
        ])

        rcoords = np.tensordot(rotmat, coords, 1)
        idx = ((rcoords[1, :] - x0) ** 2.0 / a2 +
               (rcoords[2, :] - y0) ** 2.0 / b2 +
               (rcoords[0, :] - z0) ** 2.0 / c2 <= 1)

        mat[idx] += A

    if antialias > 1:
        mat = skimage.transform.rescale(
            mat, 1 / antialias, mode='constant')

    return mat


if __name__ == '__main__':
    import napari

    TEST_SIZE = 128

    s_ball = ball(TEST_SIZE)
    s_torus = torus_boccia(TEST_SIZE)
    s_boccia = boccia(TEST_SIZE)
    s_phantom = phantom(TEST_SIZE)

    with napari.gui_qt():
        viewer = napari.view_image(s_ball)
        viewer.add_image(s_torus)
        viewer.add_image(s_boccia)
        viewer.add_image(s_phantom)
