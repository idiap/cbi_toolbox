extern void radontransform(
	double	*Input, 	/* Input image */
	long	Nx,			/* Size of image */
	long	Ny,
	double	h,			/* Sampling step on the image */
	long	nI,			/* Interpolation degree on the Image */
	double	x0,			/* Rotation center */
	double	y0,
	double	theta[],	/* Projection angles in radian */
	long	Nangles,	/* Number of projection angles */
	double	*kernel,	/* Kernel table of size Nt x Nangles */
	long	Nt,			/* Number of samples in the kernel table*/
	double	a,			/* Maximal argument of the kernel table (0 to a) */
	double	*Sinogram,	/* Output sinogram of size Nc x Nangles */
	long	Nc,			/* Number of captors */
	double	s,			/* Sampling step of the captors */
	long	nS,			/* Interpolation degree on the Sinogram */
	double	t0			/* projection of rotation center*/
);

extern void backprojection(
	double	*Sinogram,	/* Output sinogram of size Nc x Nangles */
	long	Nc,			/* Number of captors */
	long	Nangles,	/* Number of projection angles */
	double	s,			/* Sampling step of the captors */
	long	nS,			/* Interpolation degree on the Sinogram */
	double	t0,			/* projection of rotation center*/
	double	theta[],	/* Projection angles in radian */
	double	h,			/* Sampling step on the image */
	long	nI,			/* Interpolation degree on the Image */
	double	x0,			/* Rotation center */
	double	y0,
	double	*kernel,	/* Kernel table of size Nt x Nangles */
	long	Nt,			/* Number of samples in the kernel table*/
	double	a,			/* Maximal argument of the kernel table (0 to a) */
	double	*Image, 	/* Output image */
	long	Nx,			/* Size of image */
	long	Ny
);
