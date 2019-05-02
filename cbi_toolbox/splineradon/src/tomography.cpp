#include "tomography.h"

#include <math.h>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

#define MAX(A, B)  ((A) > (B) ? (A) : (B))
#define MIN(A, B)  ((A) < (B) ? (A) : (B))

extern void radontransform(
        xt::pytensor<double, 3> &image,      /* Input image */
        double h,                            /* Sampling step on the image (pixel size) */
        long nI,                             /* Interpolation degree on the Image */
        double x0,                           /* Rotation center in image coordinates */
        double y0,
        xt::pytensor<double, 3> &sinogram,   /* Output sinogram of size Nangles x Nc x Nlayers*/
        double s,                            /* Sampling step of the captors (sinogram "pixel size") */
        long nS,                             /* Interpolation degree on the sinogram */
        double t0,                           /* Projection of rotation center */
        xt::pytensor<double, 1> &theta,      /* Projection angles in radian */
        xt::pytensor<double, 2> &kernel,     /* Kernel table of size Nt x Nangles */
        double a,                            /* Maximal argument of the kernel table (0 to a) */
        bool backprojection                  /* Perform a back-projection */
) {

    long Nc = sinogram.shape()[1];
    long Nt = kernel.shape()[0];
    long Nangles = theta.size();

    long Nx = image.shape()[1];
    long Ny = image.shape()[0];

    double tmax = s * ((double) (Nc - 1L));

    double tabfact = (double) (Nt - 1L) / a;

    if (backprojection) {
        // initialize the image
        image.fill(0.0);
    } else {
        // initialize the sinogram
        sinogram.fill(0.0);
    }

    auto cosine = xt::cos(theta);
    auto sine = xt::sin(theta);

    // compute the half-width of the spline kernels with respect to the angle
    auto a_theta = (double) (nI + 1L) / 2.0 * (xt::fabs(sine) + xt::fabs(cosine)) * h + (double) (nS + 1L) / 2.0 * s;

    // iterate over the projection angles
    for (long i_angle = 0; i_angle < Nangles; i_angle++) {
        double co = cosine[i_angle];
        double si = sine[i_angle];
        double atheta = a_theta[i_angle];

        // iterate over the width of the image
        for (long i_x = 0; i_x < Nx; i_x++) {
            // compute the x coordinate of the point on the image using the pixel size
            // and its projection on the sinogram using only its x coordinate
            double x = i_x * h;
            double ttemp = (x - x0) * co + t0;

            // iterate over the height of the image
            for (long i_y = 0; i_y < Ny; i_y++) {
                // compute the y coordinate of the point on the image using the pixel size
                // and its projection on the sinogram
                double y = i_y * h;
                double t = ttemp + (y - y0) * si;

                // TODO ask ML about kernel width and border conditions
                // if the projection is in the sinogram (depends on the alignment of the centers {x0, y0} and t0
                if ((t > 0.0) && (t <= tmax)) {
                    // compute the range of sinogram elements impacted by this point and its spline kernel
                    long imin = MAX(0L, (long) (ceil((t - atheta) / s)));
                    long imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));

                    // iterate over the affected sinogram values
                    for (long i_sino = imin; i_sino <= imax; i_sino++) {
                        // compute the position of the point in its spline kernel
                        double xi = fabs((double) i_sino * s - t);
                        long idx = (long) (floor(xi * tabfact + 0.5));

                        auto sino_view = xt::view(sinogram, i_angle, i_sino);
                        auto image_view = xt::view(image, i_y, i_x);

                        if (backprojection) {
                            // update the image
                            image_view = image_view + kernel(idx, i_angle) * sino_view;
                        } else {
                            // update the sinogram
                            sino_view = sino_view + kernel(idx, i_angle) * image_view;
                        }
                    }
                }
            }
        }
    }
}
