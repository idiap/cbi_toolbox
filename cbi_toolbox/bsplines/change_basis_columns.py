# B-spline interpolation function for degree up to 7
# Christian Jaques, june 2016, Computational Bioimaging Group, Idiap
# This code is a translation of Michael Liebling's matlab code, 
# which was already largely based on a C-library written by Philippe 
# Thevenaz, BIG, EPFL

from cbi_toolbox.bsplines.convert_to_interpolation_coefficients import *


def change_basis(c, from_basis, to_basis, degree, axis=1, boundary_condition='Mirror'):
    tolerance = 1e-8

    # avoids errors when calling function with 'b-spline' instead of 'B-spline'
    from_basis = from_basis.upper()
    to_basis = to_basis.upper()
    # switch/case
    if from_basis == 'CARDINAL':
        if to_basis == 'CARDINAL':
            return c
        elif to_basis == 'B-SPLINE':
            c = convert_to_interpolation_coefficients(c, degree,
                                                      tolerance, boundary_condition=boundary_condition)
        elif to_basis == 'DUAL':
            c = change_basis(c, 'cardinal', 'c-spline', degree, axis,
                                     boundary_condition=boundary_condition)
            c = change_basis(c, 'b-spline', 'dual', degree, axis,
                                     boundary_condition=boundary_condition)
        else:
            raise ValueError("Illegal to_basis : {0}".format(to_basis))

    elif from_basis == 'B-SPLINE':
        if to_basis == 'CARDINAL':
            c = convert_to_samples(c, degree,
                                   boundary_condition=boundary_condition)
        elif to_basis == 'B-SPLINE':
            return c
        elif to_basis == 'DUAL':
            c = change_basis(c, 'b-spline', 'cardinal', degree, axis,
                                     boundary_condition=boundary_condition)
        else:
            raise ValueError("Illegal to_basis : {0}".format(to_basis))

    elif from_basis == 'DUAL':
        if to_basis == 'CARDINAL':
            c = change_basis(c, 'dual', 'b-spline', degree, axis,
                                     boundary_condition=boundary_condition)
            c = change_basis(c, 'b-spline', 'cardinal', degree, axis,
                                     boundary_condition=boundary_condition)
        elif to_basis == 'B-SPLINE':
            c = change_basis(c, 'cardinal', 'b-spline', 2 * degree + 1, axis,
                                     boundary_condition=boundary_condition)
        elif to_basis == 'DUAL':
            return c
        else:
            raise ValueError("Illegal to_basis : {0}".format(to_basis))

    return c


if __name__ == "__main__":
    data = np.array([[1, 5, -3, 4, 2, 6],
                     [1, 5, -3, 4, 2, 6],
                     [1, 5, -3, 4, 2, 6]])
    data = np.array([1, 5, -3, 4, 2, 6])
    data = np.vstack(data)
    c = change_basis(data, 'Cardinal', 'b-spline', 3)
    ar = change_basis(c, 'b-spline', 'Cardinal', 3)
    print('Relative error is ', np.sum(np.abs(np.subtract(data, ar))) / np.sum(np.abs(data)))
    print('are samples back to signal? ', np.allclose(data, ar))
