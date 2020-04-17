import numpy as np


def quadrant_symmetry(quadrant):
    '''
    Generate a quadrant by aligning itself on all 3 axes
    :param quadrant:
    :return:
    '''

    size = np.max(quadrant.shape)

    full_quadrant = np.zeros((size, size, size), dtype=bool)
    full_quadrant[:quadrant.shape[0], :quadrant.shape[1], :quadrant.shape[2]] = quadrant
    full_quadrant[:quadrant.shape[1], :quadrant.shape[2], :quadrant.shape[0]] |= quadrant.transpose((1, 2, 0))
    full_quadrant[:quadrant.shape[2], :quadrant.shape[0], :quadrant.shape[1]] |= quadrant.transpose((2, 0, 1))

    return full_quadrant


def quadrant_to_volume(quadrant, symmetry=True):
    '''
    Generate a volume by mirroring a quadrant in all 8 corners

    :param quadrant:
    :return:
    '''

    if symmetry:
        quadrant = quadrant_symmetry(quadrant)

    volume = np.empty((2 * quadrant.shape[0], 2 * quadrant.shape[1], 2 * quadrant.shape[2]), dtype=bool)
    volume[quadrant.shape[0]:, quadrant.shape[1]:, quadrant.shape[2]:] = quadrant
    volume[:quadrant.shape[0], quadrant.shape[1]:, quadrant.shape[2]:] = np.flip(quadrant, 0)
    volume[:, :quadrant.shape[1], quadrant.shape[2]:] = np.flip(volume[:, quadrant.shape[1]:, quadrant.shape[2]:], 1)
    volume[:, :, :quadrant.shape[2]] = np.flip(volume[:, :, quadrant.shape[2]:], 2)

    return volume


def boccia(radius, n_stripes=3, deg_space=15, deg_width=7.5, rad_thick=0.12):
    '''
    Create a boccia simulated sample: resolution stripes on a sphere

    :param radius: radius of the sphere, in pixels
    :param n_stripes: number of stripes to generate
    :param deg_space: spacing in degrees between the center of the stripes
    :param deg_width: width in degrees of the stripes
    :param rad_thick: thickness of the stripes, as a proportion of the radius
    :return:
    '''
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

    width = int(np.ceil(radius * np.sin(max_angle)))

    x = (np.arange(radius) + 0.5).reshape((radius, 1, 1))
    y = (np.arange(radius) + 0.5).reshape((1, radius, 1))
    z = (np.arange(width) + 0.5).reshape((1, 1, width))

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arccos(z / r)
    phi = np.pi / 2 - phi

    angle_crit = np.zeros_like(phi, dtype=bool)
    for angle in c_angles:
        angle_crit |= (phi > (angle - half_width)) & (phi < (angle + half_width))

    quadrant = (r < radius) & (r > (radius * (1 - rad_thick))) & angle_crit

    return quadrant_to_volume(quadrant)


def torus_boccia(radius, n_stripes=3, deg_space=15, torus_radius=0.075):
    '''
    Generate a boccia with torus stripes

    :param radius: radius of the main outer sphere
    :param n_stripes: number of torus stripes
    :param deg_space: spacing of the torus on the sphere in degrees
    :param torus_radius: radius of the torus, as a fraction of the sphere radius
    :return:
    '''

    # the angles at the center of the stripes
    c_angles = np.arange(n_stripes / 2)

    # if even number of stripes
    if not n_stripes % 2:
        c_angles += 0.5

    c_angles *= np.deg2rad(deg_space)

    max_angle = c_angles[-1] + np.arcsin(torus_radius)
    if max_angle > np.pi / 2:
        raise ValueError('max angle must be less than 90°')

    c_pos = np.array([np.cos(c_angles), np.sin(c_angles)]) * radius * (1 - torus_radius)

    width = int(np.ceil(c_pos[1, :].max() + radius * torus_radius))

    x = (np.arange(radius) + 0.5).reshape((radius, 1, 1, 1))
    y = (np.arange(radius) + 0.5).reshape((1, radius, 1, 1))
    z = (np.arange(width) + 0.5).reshape((1, 1, width, 1))

    r = np.sqrt(x ** 2 + y ** 2)

    rad = (r - c_pos[0, :]) ** 2 + (z - c_pos[1, :]) ** 2
    rad = rad.min(-1)

    quadrant = rad < (torus_radius * radius) ** 2

    return quadrant_to_volume(quadrant)


if __name__ == '__main__':
    sample = torus_boccia(64)

    from mayavi import mlab
    mlab.contour3d(sample*1.0)

    sample = boccia(64)
    mlab.figure()
    mlab.contour3d(sample*1.0)

    mlab.show()
