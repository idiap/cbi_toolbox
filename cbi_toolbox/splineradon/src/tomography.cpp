#include "tomography.h"

#include <math.h>

#define MAX(A, B)  ((A) > (B) ? (A) : (B))
#define MIN(A, B)  ((A) < (B) ? (A) : (B))

extern void radontransform(
        double *Input,    /* Input image */
        long Nx,            /* Size of image */
        long Ny,
        double h,            /* Sampling step on the image */
        long nI,            /* Interpolation degree on the Image */
        double x0,            /* Rotation center */
        double y0,
        double theta[],    /* Projection angles in radian */
        long Nangles,    /* Number of projection angles */
        double *kernel,    /* Kernel table of size Nt x Nangles */
        long Nt,            /* Number of samples in the kernel table*/
        double a,            /* Maximal argument of the kernel table (0 to a) */
        double *Sinogram,    /* Output sinogram of size Nc x Nangles */
        long Nc,            /* Number of captors */
        double s,            /* Sampling step of the captors */
        long nS,            /* Interpolation degree on the Sinogram */
        double t0            /* projection of rotation center*/
) {
    long i, j, k, l;
    long imin, imax;
    long imgidx, sinoidx, idx;
    double tmax = s * ((double) (Nc - 1L));
    double tabfact;
    long kerstart;
    long sinostart;
    long imstart;
    double co, si;
    double atheta;
    double xi;
    double x, y, t;
    double ttemp;
//    double PIo180 = 0.01745329251994329;

    tabfact = (double) (Nt - 1L) / a;


    for (j = 0; j < Nc * Nangles; j++) Sinogram[j] = 0.0;

    for (j = 0; j < Nangles; j++) {
        co = cos(theta[j]);
        si = sin(theta[j]);
        atheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;

        kerstart = j * Nt;
        sinostart = j * Nc;
        for (k = 0; k < Nx; k++) {
            x = k * h;
            ttemp = (x - x0) * co + t0;
            imstart = Ny * k;
            for (l = 0; l < Ny; l++) {
                y = l * h;
                t = ttemp + (y - y0) * si;
                if ((t > 0.0) && (t <= tmax)) {
                    imgidx = imstart + l;
                    imin = MAX(0L, (long) (ceil((t - atheta) / s)));
                    imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));
                    sinoidx = sinostart + imin;
                    for (i = imin; i <= imax; i++) {
                        xi = fabs((double) i * s - t);
                        idx = (long) (floor(xi * tabfact + 0.5));

                        idx += kerstart;
                        Sinogram[sinoidx++] += kernel[idx] *
                                               Input[imgidx];
                    }
                }
            }
        }
    }
}

extern void backprojection(
        double *Sinogram,    /* Output sinogram of size Nc x Nangles */
        long Nc,            /* Number of captors */
        long Nangles,    /* Number of projection angles */
        double s,            /* Sampling step of the captors */
        long nS,            /* Interpolation degree on the Sinogram */
        double t0,            /* projection of rotation center*/
        double theta[],    /* Projection angles in radian */
        double h,            /* Sampling step on the image */
        long nI,            /* Interpolation degree on the Image */
        double x0,            /* Rotation center */
        double y0,
        double *kernel,    /* Kernel table of size Nt x Nangles */
        long Nt,            /* Number of samples in the kernel table*/
        double a,            /* Maximal argument of the kernel table (0 to a) */
        double *Image,    /* Input image */
        long Nx,            /* Size of image */
        long Ny
) {
    long i, j, k, l;
    long imin, imax;
    long imgidx, sinoidx, idx;
    double tmax = s * ((double) (Nc - 1L));
    double tabfact;
    long kerstart;
    long sinostart;
    long imstart;
    double co, si;
    double atheta;
    double xi;
    double x, y, t;
    double ttemp;
//    double PIo180 = 0.01745329251994329;

    tabfact = (double) (Nt - 1L) / a;


    for (j = 0; j < Nx * Ny; j++) Image[j] = 0.0;

    for (j = 0; j < Nangles; j++) {
        co = cos(theta[j]);
        si = sin(theta[j]);
        atheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;
        kerstart = j * Nt;
        sinostart = j * Nc;
        for (k = 0; k < Nx; k++) {
            x = k * h;
            ttemp = (x - x0) * co + t0;
            imstart = Ny * k;
            for (l = 0; l < Ny; l++) {
                y = l * h;
                t = ttemp + (y - y0) * si;
                if ((t > 0.0) && (t <= tmax)) {
                    imgidx = imstart + l;
                    imin = MAX(0L, (long) (ceil((t - atheta) / s)));
                    imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));
                    sinoidx = sinostart + imin;
                    for (i = imin; i <= imax; i++) {
                        xi = fabs((double) i * s - t);
                        idx = (long) (floor(xi * tabfact + 0.5));

                        idx += kerstart;
                        Image[imgidx] += kernel[idx] * Sinogram[sinoidx++];
                    }
                }
            }
        }
    }
}
