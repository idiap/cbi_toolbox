#include "tomography_omp.h"

#include <math.h>
#include <cstring>
#include <omp.h>

#define MAX(A, B)  ((A) > (B) ? (A) : (B))
#define MIN(A, B)  ((A) < (B) ? (A) : (B))

void fill(
	  py::buffer_info &array_info,
	  double value
	  ) {
  std::memset(array_info.ptr, value, array_info.size * sizeof(double));
  return;
}

extern void radontransform_fbp(
			   py::array_t<double, py::array::c_style> &image,
			   double h,
			   long nI,
			   double x0,
			   double y0,
			   py::array_t<double, py::array::c_style> &sinogram,
			   double s,
			   long nS,
			   double t0,
			   py::array_t<double, py::array::c_style> &theta,
			   py::array_t<double, py::array::c_style> &kernel,
			   double a
			   ) {
  
  auto sinogram_info = sinogram.request();
  auto kernel_info = kernel.request();
  auto theta_info = theta.request();
  auto image_info = image.request();


  double *theta_ptr = reinterpret_cast<double *>(theta_info.ptr);
  double *sinogram_ptr = reinterpret_cast<double *>(sinogram_info.ptr);
  double *kernel_ptr = reinterpret_cast<double *>(kernel_info.ptr);
  double *image_ptr = reinterpret_cast<double *>(image_info.ptr);

  const long Nc = sinogram_info.shape[1];
  const long Nt = kernel_info.shape[1];
  const long Nangles = theta_info.size;

  const long Nx = image_info.shape[1];
  const long Ny = image_info.shape[0];
  const long Nz = image_info.shape[2];

  const double tabfact = (double) (Nt - 1L) / a;

  // initialize the image
  fill(image_info, 0.0); 

#pragma omp parallel for
  // iterate over the projection angles
  for (long i_angle = 0; i_angle < Nangles; i_angle++) {
    double co = cos(theta_ptr[i_angle]);
    double si = sin(theta_ptr[i_angle]);

    // compute the half-width of the spline kernels with respect to the angle
    double atheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;

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

	// compute the range of sinogram elements impacted by this point and its spline kernel
	long imin = MAX(0L, (long) (ceil((t - atheta) / s)));
	long imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));

	// iterate over the affected sinogram values
	for (long i_sino = imin; i_sino <= imax; i_sino++) {
	  // compute the position of the point in its spline kernel
	  double xi = fabs((double) i_sino * s - t);
	  long idx = (long) (floor(xi * tabfact + 0.5));
	  
	  for (long i_z = 0; i_z < Nz; i_z++){

	    auto image_index = i_y * Nx * Nz + i_x * Nz + i_z;
	    auto kernel_index = i_angle * Nt + idx;
	    auto sinogram_index = i_angle * Nc * Nz + i_sino * Nz + i_z;

	    // update the image
#pragma omp atomic update
	    image_ptr[image_index] += kernel_ptr[kernel_index] * sinogram_ptr[sinogram_index];
	    
	  }
	}
      }
    }
  }
}

py::array_t<double, py::array::c_style> radon_omp(
			       py::array_t<double, py::array::c_style> &image,
			       double h,
			       long nI,
			       double x0,
			       double y0,
			       py::array_t<double, py::array::c_style> &theta,
			       py::array_t<double, py::array::c_style> &kernel,
			       double a,
			       const long Nc,
			       double s,
			       long nS,
			       double t0
			       ) {
  
  auto sinogram_info = sinogram.request();
  auto kernel_info = kernel.request();
  auto theta_info = theta.request();
  auto image_info = image.request();


  double *theta_ptr = reinterpret_cast<double *>(theta_info.ptr);
  double *sinogram_ptr = reinterpret_cast<double *>(sinogram_info.ptr);
  double *kernel_ptr = reinterpret_cast<double *>(kernel_info.ptr);
  double *image_ptr = reinterpret_cast<double *>(image_info.ptr);

  const long Nc = sinogram_info.shape[1];
  const long Nt = kernel_info.shape[1];
  const long Nangles = theta_info.size;

  const long Nx = image_info.shape[1];
  const long Ny = image_info.shape[0];
  const long Nz = image_info.shape[2];

  const double tabfact = (double) (Nt - 1L) / a;

  // initialize the sinogram
  fill(sinogram_info, 0.0);

  #pragma omp parallel for
  // iterate over the projection angles
  for (long i_angle = 0; i_angle < Nangles; i_angle++) {
    double co = cos(theta_ptr[i_angle]);
    double si = sin(theta_ptr[i_angle]);

    // compute the half-width of the spline kernels with respect to the angle
    double atheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;

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

	// compute the range of sinogram elements impacted by this point and its spline kernel
	long imin = MAX(0L, (long) (ceil((t - atheta) / s)));
	long imax = MIN(Nc - 1L, (long) (floor((t + atheta) / s)));

	// iterate over the affected sinogram values
	for (long i_sino = imin; i_sino <= imax; i_sino++) {
	  // compute the position of the point in its spline kernel
	  double xi = fabs((double) i_sino * s - t);
	  long idx = (long) (floor(xi * tabfact + 0.5));
	  
	  for (long i_z = 0; i_z < Nz; i_z++){

	    auto image_index = i_y * Nx * Nz + i_x * Nz + i_z;
	    auto kernel_index = i_angle * Nt + idx;
	    auto sinogram_index = i_angle * Nc * Nz + i_sino * Nz + i_z;

	    // update the sinogram
	    sinogram_ptr[sinogram_index] += kernel_ptr[kernel_index] * image_ptr[image_index];
	  }
	}
      }
    }
  }
}
