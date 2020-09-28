import numpy as np
import skimage.transform


def quadrant_symmetry(quadrant):
    '''
    Generate a quadrant by aligning itself on all 3 axes
    :param quadrant:
    :return:
    '''

    size = np.max(quadrant.shape)

    full_quadrant = np.zeros((size, size, size), dtype=bool)
    full_quadrant[:quadrant.shape[0], :quadrant.shape[1],
                  :quadrant.shape[2]] = quadrant
    full_quadrant[:quadrant.shape[1], :quadrant.shape[2],
                  :quadrant.shape[0]] |= quadrant.transpose((1, 2, 0))
    full_quadrant[:quadrant.shape[2], :quadrant.shape[0],
                  :quadrant.shape[1]] |= quadrant.transpose((2, 0, 1))

    return full_quadrant


def quadrant_to_volume(quadrant):
    '''
    Generate a volume by mirroring a quadrant in all 8 corners

    :param quadrant:
    :return:
    '''

    volume = np.empty(
        (2 * quadrant.shape[0], 2 * quadrant.shape[1], 2 * quadrant.shape[2]), dtype=quadrant.dtype)
    volume[quadrant.shape[0]:, quadrant.shape[1]:, quadrant.shape[2]:] = quadrant
    volume[:quadrant.shape[0], quadrant.shape[1]:,
           quadrant.shape[2]:] = np.flip(quadrant, 0)
    volume[:, :quadrant.shape[1], quadrant.shape[2]:] = np.flip(
        volume[:, quadrant.shape[1]:, quadrant.shape[2]:], 1)
    volume[:, :, :quadrant.shape[2]] = np.flip(
        volume[:, :, quadrant.shape[2]:], 2)

    return volume


def boccia(size, n_stripes=3, deg_space=15, deg_width=7.5, rad_thick=0.12, antialias=2, dtype=np.float64):
    '''
    Create a boccia simulated sample: resolution stripes on a sphere

    :param size: side of the cube containing the volume
    :param n_stripes: number of stripes to generate
    :param deg_space: spacing in degrees between the center of the stripes
    :param deg_width: width in degrees of the stripes
    :param rad_thick: thickness of the stripes, as a proportion of the radius
    :param antialias: antialiasing scale factor
    :return:
    '''

    if size % 2:
        raise ValueError('The size must be even to cut it in quadrants')

    if antialias < 1 or type(antialias) is not int:
        raise ValueError('Antialias must be a positive integer')

    size //= 2
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


def torus_boccia(size, n_stripes=3, deg_space=15, torus_radius=0.075, antialias=2, dtype=np.float64):
    '''
    Generate a boccia with torus stripes

    :param size: side of the cube containing the volume
    :param n_stripes: number of torus stripes
    :param deg_space: spacing of the torus on the sphere in degrees
    :param torus_radius: radius of the torus, as a fraction of the sphere radius
    :param antialias: antialiasing scale factor
    :return:
    '''

    if size % 2:
        raise ValueError('The size must be even to cut it in quadrants')

    if antialias < 1 or type(antialias) is not int:
        raise ValueError('Antialias must be a positive integer')

    size //= 2
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


def ball(size, in_radius=0, antialias=2, dtype=np.float64):
    '''
    Generate a boccia with torus stripes

    :param size: side of the cube containing the volume
    :param in_radius: inner radius fraction (set to 0 for full ball)
    :param antialias: antialiasing scale factor
    :return:
    '''

    if size % 2:
        raise ValueError('The size must be even to cut it in quadrants')

    if antialias < 1 or type(antialias) is not int:
        raise ValueError('Antialias must be a positive integer')

    size //= 2
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


if __name__ == '__main__':
    import napari

    test_size = 128

    s_ball = ball(test_size)
    with napari.gui_qt():
        napari.view_image(s_ball)

    s_torus = torus_boccia(test_size)
    with napari.gui_qt():
        napari.view_image(s_torus)

    s_boccia = boccia(test_size)
    with napari.gui_qt():
        napari.view_image(s_boccia)
