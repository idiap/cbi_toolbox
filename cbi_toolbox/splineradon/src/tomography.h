#include <xtensor-python/pytensor.hpp>
// pyarray can be changed to xarray if not using python/numpy

extern void radontransform(
        xt::pytensor<double, 3> &image,      /* Image */
        double h,                            /* Sampling step on the image (pixel size) */
        long nI,                             /* Interpolation degree on the Image */
        double x0,                           /* Rotation center in image coordinates */
        double y0,
        xt::pytensor<double, 3> &sinogram,   /* Sinogram of size Nangles x Nc x Nlayers*/
        double s,                            /* Sampling step of the captors (sinogram "pixel size") */
        long nS,                             /* Interpolation degree on the sinogram */
        double t0,                           /* Projection of rotation center */
        xt::pytensor<double, 1> &theta,      /* Projection angles in radian */
        xt::pytensor<double, 2> &kernel,     /* Kernel table of size Nangles x Nt */
        double a,                            /* Maximal argument of the kernel table (0 to a) */
        bool backprojection                  /* Perform a back-projection */
);
