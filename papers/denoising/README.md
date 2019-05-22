

## Wiki

+ [white noise](https://en.wikipedia.org/wiki/White_noise)
    + white refers to the way signal power is distributed (independently) over time or along frequencies
    + white noise vector if each component has probability distribution with mean 0 and a finite variance, and are statistically independent
+ [gaussian white noise](https://en.wikipedia.org/wiki/Additive_white_Gaussian_noise)
    + white noise vector where each component assumes a gaussian distribution 
        + the vector is a multivariate gaussian distribution




## Paper


+ [1992_nonlinear_total_variantion_based_noise_removal_algorithms](1992_nonlinear_total_variantion_based_noise_removal_algorithms.pdf)
    + abstract
        + denoising by minimizing total variations, constraint using Lagrange multipliers
        + solved by gradient-projection
    + intro 
        + total variation norm
            + L1 norms of derivatives
            + nonlinear, computationall complex (compared to L2)
        + problem formulation 
            + constraint optimization as time dependent nonlinear parabolic pde
                + derive the euler-lagrange equation for the optimization
            + solved with with a time-steping algorithm
    + results   
        + denoise in flat regions AND preserves edge details
    + some follow-ups
        + primal-dual method for minimizing total variation
            + https://www.uni-muenster.de/AMM/num/Vorlesungen/MathemBV_SS16/literature/Chambolle2004.pdf

+ [2010_fast_image_recovery_using_variable_splitting_and_constraint_optimization](2010_fast_image_recovery_using_variable_splitting_and_constraint_optimization.pdf)
    + abstract
        + using ADMM for solving unconstrained problem where objective includes
            + L2 data-fidelity term 
            + non-smooth regularizer
        + variable splitting
            + equivalent to constrained optimization, addressed with augmented Lagrangian method
    + problem formulation 
        + synthesis approach (i.e. with _wavelets_)
            +  `x = W\beta` where 
                + `W` are elements of wavelet frame
                + `\beta` are parameter to be estimated
        + analysis approach
            + `x` sampled randomly 
            + based on regularizers that analyzes the image itself rather than representation in wavelet domain
                + i.e total variation regularizer
        + unified view
            + `min_x 1/2 || Ax - y ||_2^2 + \tau \phi(x)`
                + `A = BW`  where `A=B` for analysis approach
            + solvers
                + iterative shrinkage / thresholding (IST)
                    + relies on denoising function `\Psi(y)`
                    + iteration: `x_{k+1} = \Psi ( x_t - 1/\gamma A^T(Ax - y))`
                    + problem: slow
    + proposed algo: split augmented Lagrangian shrinkage algorithm (SALSA)
        + a variant of ADMM



#### Plug and Play P3 Prior

+ [2013_plug_and_play_priors_for_model_based_reconstruction](2013_plug_and_play_priors_for_model_based_reconstruction.pdf)
+ [2016_algorithm_induced_prior_for_image_restoration](2016_algorithm_induced_prior_for_image_restoration.pdf)
+ [2016_plug_and_play_admm_for_image_restoration_fixed_point_convergence_and_applications](2016_plug_and_play_admm_for_image_restoration_fixed_point_convergence_and_applications.pdf)



+ [2017_regularization_by_denoising](2017_regularization_by_denoising.pdf)
    + intro
        + plug-and-play (P3)
            + use implicit priors for regularizing general inverse problems
            + problems
                + no clear objective function
                + parameter tuning ADMM
        + use of denoising engine for regularization of inverse problems
            + `rho(x) = 1/2 x^T (x - f(x))`
                + is proved to be convex (guaranteed convergence)
            + the goodness
                + explicit objective
                + gradient manageable
                + any inverse problem handled by calling denoising engine iteratively
            + applications
                + single-image superresolution 
                + blurring