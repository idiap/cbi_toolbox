//
// Created by fmarelli on 05/07/19.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <cstring>
#include <omp.h>

namespace py = pybind11;

#define MAX(A, B)  ((A) > (B) ? (A) : (B))
#define MIN(A, B)  ((A) < (B) ? (A) : (B))

void fill(double* array, double value, int size){
  for (int i = 0; i < size; i++){
    array[i] = value;
  }
}

/*
 * Precompute cosine, sine and atheta for each angle.
 * Stored in trigo (cosine, sine, atheta).
 * Grid-stride loop: blocks and threads for angles.
 */
void precompute_trigo(
		      double h,                           /* Sampling step on the image (pixel size) */
		      long nI,                            /* Interpolation degree on the Image */
		      size_t Nangles,                     /* Number of angles in the sinogram (shape[0]) */
		      double s,                           /* Sampling step of the captors (sinogram "pixel size") */
		      long nS,                            /* Interpolation degree on the sinogram */
		      double *theta,                      /* Projection angles in radian */
		      double *trigo                       /* Array containing cosine, sine and atheta for each angle shape {Nangles, 3}*/
		      ) {

  // iterate over the projection angles using blocks and threads alike
  for (size_t i_angle = 0; i_angle < Nangles; i_angle += 1) {

    double co = cos(theta[i_angle]);
    double si = sin(theta[i_angle]);

    double atheta = (double) (nI + 1L) / 2.0 * (fabs(si) + fabs(co)) * h + (double) (nS + 1L) / 2.0 * s;

    long index = i_angle * 3;
    trigo[index] = co;
    trigo[index + 1] = si;
    trigo[index + 2] = atheta;
  }
}


/*
 * Precompute the minimum and maximum indexes of sinogram values impacted by each pixel in the image, and the projected coordinates.
 * Stored in sino_bounds (min, max) and t_coord.
 * Grid-stride loop: blocks for angles (Nangles) , threads for image pixels (Nx * Ny).
 */
void precompute_radon(
		      size_t Nx,                          /* Image X size (shape[1]) */
		      size_t Ny,                          /* Image Y size (shape[0]) */
		      double h,                           /* Sampling step on the image (pixel size) */
		      double x0,                          /* Rotation center X in image coordinates */
		      double y0,                          /* Rotation center Y in image coordinates */
		      size_t Nangles,                     /* Number of angles in the sinogram (shape[0]) */
		      size_t Nc,                          /* Number of captors in the sinogram (shape[1]) */
		      double s,                           /* Sampling step of the captors (sinogram "pixel size") */
		      double t0,                          /* Projection of rotation center */
		      double *trigo,                      /* Array containing cosine, sine and atheta for each angle (shape {Nangles, 3}) */
		      long *sino_bounds,                  /* Indexes of sinogram impact for all pixels in the image (shape {A, x, y, 2}) */
		      double *t_coords                   /* Projected coordinates on the sinogram (shape {A, x, y}) */
		      ) {

  // iterate over the projection angles using blocks
  for (size_t i_angle = 0; i_angle < Nangles; i_angle += 1) {

    long index_angle = i_angle * 3;

    double co = trigo[index_angle];
    double si = trigo[index_angle + 1];
    double atheta = trigo[index_angle + 2];

    // iterate over the image using threads
    for (size_t id = 0; id < Nx * Ny; id += 1) {
      long i_x = id / Ny;
      long i_y = id % Ny;

      double x = i_x * h;
      double y = i_y * h;

      // compute the projected coordinate on the sinogram
      double t = t0 + ((x - x0) * co) + ((y - y0) * si);

      // compute the range of sinogram elements impacted by this point and its spline kernel
      auto imin = MAX(0L, (long) (ceil((t - atheta) / s)));
      auto imax = MIN((long)Nc - 1L, (long) (floor((t + atheta) / s)));

      // store in the relevant matrices
      auto t_index = i_angle * Nx * Ny + i_x * Ny + i_y;
      t_coords[t_index] = t;

      auto bound_index = t_index * 2;
      sino_bounds[bound_index] = imin;
      sino_bounds[bound_index + 1] = imax;
    }
  }
}

/*
 * Compute the (inverse) radon transform.
 * Grid-stide loop: blocks for angles (Nangles), threads for depth (Nz)
 */
void radontransform(
		    double *image,                      /* Image (shape {Ny, Nx, Nz})*/
		    size_t Nx,
		    size_t Ny,
		    size_t Nz,
		    double *sinogram,                   /* Sinogram (shape (Nangles, Nc, Nz)*/
		    size_t Nangles,
		    size_t Nc,
		    double s,                           /* Sampling step of the captors (sinogram "pixel size") */
		    double *kernel,                     /* Kernel table (shape {Nangles, Nt}) */
		    size_t Nt,
		    double tabfact,                     /* Sampling step of the kernel */
		    long *sino_bounds,                  /* Indexes of sinogram impact for all pixels in the image (shape {A, x, y, 2}) */
		    double *t_coords                    /* Projected coordinates on the sinogram (shape {A, x, y}) */
		    ) {

  // iterate over the projection angles
  for (size_t i_angle = 0; i_angle < Nangles; i_angle += 1) {

    // iterate over the width of the image
    for (size_t i_x = 0; i_x < Nx; i_x++) {

      // iterate over the height of the image
      for (size_t i_y = 0; i_y < Ny; i_y++) {

	// fetch the projected coordinate
	auto t_index = i_angle * Nx * Ny + i_x * Ny + i_y;
	auto t = t_coords[t_index];

	// fetch the sinogram bounds
	auto bounds_index = t_index * 2;
	auto imin = sino_bounds[bounds_index];
	auto imax = sino_bounds[bounds_index + 1];

	// iterate over the affected sinogram values
	for (long i_sino = imin; i_sino <= imax; i_sino++) {
	  // compute the position of the point in its spline kernel
	  double xi = fabs((double) i_sino * s - t);
	  long idx = (long) (floor(xi * tabfact + 0.5));

	  for (size_t i_z = 0; i_z < Nz; i_z += 1) {

	    auto image_index = i_y * Nx * Nz + i_x * Nz + i_z;
	    auto kernel_index = i_angle * Nt + idx;
	    auto sinogram_index = i_angle * Nc * Nz + i_sino * Nz + i_z;

	    // update the sinogram
	    sinogram[sinogram_index] += (kernel[kernel_index] * image[image_index]);
	  }
	}
      }
    }
  }
}

/*
 * Compute the (inverse) radon transform.
 * Grid-stide loop: blocks for angles (Nangles), threads for depth (Nz)
 */
void iradontransform(
		     double *image,                      /* Image (shape {Ny, Nx, Nz})*/
		     size_t Nx,
		     size_t Ny,
		     size_t Nz,
		     double *sinogram,                   /* Sinogram (shape (Nangles, Nc, Nz)*/
		     size_t Nangles,
		     size_t Nc,
		     double s,                           /* Sampling step of the captors (sinogram "pixel size") */
		     double *kernel,                     /* Kernel table (shape {Nangles, Nt}) */
		     size_t Nt,
		     double tabfact,                     /* Sampling step of the kernel */
		     long *sino_bounds,                  /* Indexes of sinogram impact for all pixels in the image (shape {A, x, y, 2}) */
		     double *t_coords                    /* Projected coordinates on the sinogram (shape {A, x, y}) */
		     ) {

  // iterate over the projection angles
  for (size_t i_angle = 0; i_angle < Nangles; i_angle += 1) {

    // iterate over the width of the image
    for (size_t i_x = 0; i_x < Nx; i_x++) {

      // iterate over the height of the image
      for (size_t i_y = 0; i_y < Ny; i_y++) {

	// fetch the projected coordinate
	auto t_index = i_angle * Nx * Ny + i_x * Ny + i_y;
	auto t = t_coords[t_index];

	// fetch the sinogram bounds
	auto bounds_index = t_index * 2;
	auto imin = sino_bounds[bounds_index];
	auto imax = sino_bounds[bounds_index + 1];

	// iterate over the affected sinogram values
	for (long i_sino = imin; i_sino <= imax; i_sino++) {
	  // compute the position of the point in its spline kernel
	  double xi = fabs((double) i_sino * s - t);
	  long idx = (long) (floor(xi * tabfact + 0.5));

	  for (size_t i_z = 0; i_z < Nz; i_z += 1) {

	    auto image_index = i_y * Nx * Nz + i_x * Nz + i_z;
	    auto kernel_index = i_angle * Nt + idx;
	    auto sinogram_index = i_angle * Nc * Nz + i_sino * Nz + i_z;

	    // update the image
	    // CUDA atomic
	    image[image_index] += kernel[kernel_index] * sinogram[sinogram_index];
	  }
	}
      }
    }
  }
}


py::array_t<double, py::array::c_style> radon(
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

  auto image_info = image.request();
  auto kernel_info = kernel.request();
  auto theta_info = theta.request();

  if (image_info.ndim != 3) {
    throw py::value_error("image must be a 3D array.");
  }

  if (theta_info.ndim != 1) {
    throw py::value_error("theta must be a 1D array.");
  }

  if (kernel_info.ndim != 2) {
    throw py::value_error("kernel must be a 2D array.");
  }

  if (nI < 0L) {
    throw py::value_error("nI must be greater or equal to 0.");
  }

  if (a < 0.0) {
    throw py::value_error("a, the maximal argument of the lookup table must be a positive.");
  }

  if (Nc < 1L) {
    throw py::value_error("The number of captor must at least be 1.");
  }

  if (nS < -1L) {
    throw py::value_error("nS must be greater of equal to -1.");
  }

  const long Nangles = theta_info.shape[0];

  if (Nangles != kernel_info.shape[0]) {
    throw py::value_error("The kernel must have Nangle rows.");
  }

  double *acc_theta = reinterpret_cast<double *>(theta_info.ptr);

  double *acc_trigo = new double[Nangles * 3];
    
  precompute_trigo(
		   h,
		   nI,
		   Nangles,
		   s,
		   nS,
		   acc_theta,
		   acc_trigo
		   );

  int Nx = image_info.shape[1];
  int Ny = image_info.shape[0];
  int Nz = image_info.shape[2];

  //cuda  delete [] acc_theta;

  long *acc_sino_bounds = new long[Nangles * Nx * Ny * 2];
  double *acc_t_coords = new double[Nangles * Nx * Ny];
    
  precompute_radon(
		   Nx,
		   Ny,
		   h,
		   x0,
		   y0,
		   Nangles,
		   Nc,
		   s,
		   t0,
		   acc_trigo,
		   acc_sino_bounds,
		   acc_t_coords
		   );

  auto Nt = kernel_info.shape[1];
  double tabfact = (double) (Nt - 1L) / a;

  delete [] acc_trigo;
    
  double *acc_image = reinterpret_cast<double *>(image_info.ptr);
  double *acc_kernel = reinterpret_cast<double *>(kernel_info.ptr);

 
  py::array::ShapeContainer shape = {Nangles, Nc, image_info.shape[2]};

  auto sinogram = py::array_t<double, py::array::c_style>(shape);
  auto sinogram_info = sinogram.request();

  double *acc_sinogram = reinterpret_cast<double *>(sinogram_info.ptr);
  fill(acc_sinogram, 0.0, sinogram_info.size);
    
  radontransform(
		 acc_image,
		 Nx,
		 Ny,
		 Nz,
		 acc_sinogram,
		 Nangles,
		 Nc,
		 s,
		 acc_kernel,
		 kernel_info.shape[1],
		 tabfact,
		 acc_sino_bounds,
		 acc_t_coords
		 );

  
  //    checkCuda(cudaFree(cuda_image));

  //    checkCuda(cudaFree(cuda_kernel));
  delete [] acc_t_coords;
  delete [] acc_sino_bounds;

  //  auto sinogram = cuda_to_numpy(shape, cuda_sinogram);
  //checkCuda(cudaFree(cuda_sinogram));

  return sinogram;
}

py::array_t<double, py::array::c_style> iradon(
					       py::array_t<double, py::array::c_style> &sinogram,
					       double s,
					       long nS,
					       double t0,
					       py::array_t<double, py::array::c_style> &theta,
					       py::array_t<double, py::array::c_style> &kernel,
					       double a,
					       long Nx,
					       long Ny,
					       double h,
					       long nI,
					       double x0,
					       double y0
					       ) {

  auto sinogram_info = sinogram.request();
  auto kernel_info = kernel.request();
  auto theta_info = theta.request();

  if (sinogram_info.ndim != 3) {
    throw py::value_error("sinogram must be a 3D array.");
  }

  if (theta_info.ndim != 1) {
    throw py::value_error("theta must be a 1D array.");
  }

  if (kernel_info.ndim != 2) {
    throw py::value_error("kernel must be a 2D array.");
  }

  if (nS < 0L) {
    throw py::value_error("nS must be greater or equal to 0.");
  }

  const long Nangles = sinogram_info.shape[0];

  if (Nangles != theta_info.size) {
    throw py::value_error("The number of angles in theta in incompatible with the sinogram.");
  }

  if (nI < -1L) {
    throw py::value_error("nI must be greater or equal to -1.");
  }

  if (Nangles != kernel_info.shape[0]) {
    throw py::value_error("The kernel table must have Nangle rows.");
  }

  if (a < 0) {
    throw py::value_error("a, the max argument of the lookup table must be positive.");
  }

  if (Nx < 1L) {
    throw py::value_error("Nx must at least be 1.");
  }
  if (Ny < 1L) {
    throw py::value_error("Ny must at least be 1.");
  }

  double *acc_theta = reinterpret_cast<double *>(theta_info.ptr);

  double *acc_trigo = new double[Nangles * 3];

  precompute_trigo(
		   h,
		   nI,
		   Nangles,
		   s,
		   nS,
		   acc_theta,
		   acc_trigo
		   );


  int Nz = sinogram_info.shape[2];
  int Nc = sinogram_info.shape[1];
    
  long *acc_sino_bounds = new long[Nangles * Nx * Ny * 2];
  double *acc_t_coords = new double[Nangles * Nx * Ny];

  //    checkCuda(cudaFree(cuda_theta));

  precompute_radon(
		   Nx,
		   Ny,
		   h,
		   x0,
		   y0,
		   Nangles,
		   Nc,
		   s,
		   t0,
		   acc_trigo,
		   acc_sino_bounds,
		   acc_t_coords
		   );

  auto Nt = kernel_info.shape[1];
  double tabfact = (double) (Nt - 1L) / a;

  delete [] acc_trigo;
    
  double *acc_sinogram = reinterpret_cast<double *>(sinogram_info.ptr);
  double *acc_kernel = reinterpret_cast<double *>(kernel_info.ptr);

    
  py::array::ShapeContainer shape = {Ny, Nx, sinogram_info.shape[2]};

  auto image = py::array_t<double, py::array::c_style>(shape);
  auto image_info = image.request();

  double *acc_image = reinterpret_cast<double *>(image_info.ptr);
  fill(acc_image, 0.0, image_info.size);
    
  iradontransform(
		  acc_image,
		  Nx,
		  Ny,
		  Nz,
		  acc_sinogram,
		  Nangles,
		  Nc,
		  s,
		  acc_kernel,
		  kernel_info.shape[1],
		  tabfact,
		  acc_sino_bounds,
		  acc_t_coords
		  );

  //checkCuda(cudaFree(cuda_sinogram));

  //checkCuda(cudaFree(cuda_kernel));
  delete [] acc_t_coords;
  delete [] acc_sino_bounds;
    
  //auto image = cuda_to_numpy(shape, cuda_image);

  //checkCuda(cudaFree(cuda_image));

  return image;
}



PYBIND11_MODULE(accsplineradon, m) {

    m.doc() = "Radon transform (and inverse) using spline convolutions discretization with OMP GPU acceleration";

    m.def("radon", &radon, "Perform radon transform of an image with GPU acceleration");
    m.def("iradon", &iradon, "Perform inverse radon transform (back-projection) of a sinogram with GPU acceleration");

}
