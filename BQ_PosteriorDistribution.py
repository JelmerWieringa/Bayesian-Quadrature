########################################################################################################################
###################################### BAYESIAN QUADRATURE POSTERIOR DISTRIBUTION ######################################
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

#=======================================================================================================================
#-------------------------------------------------- FUNCTION OF INTEREST -----------------------------------------------
#=======================================================================================================================
def Integrand_f(x):  # Integrand f(x)
    return np.exp( np.sin(x**2) * np.cos(x) )

N = 1000  # Number of points to evaluate sample functions from GP at
HorizontalAxis = np.linspace(start = 0, stop = 5, num = N)
X = np.array([0.7, 1.6, 2.3, 3, 3.4, 4, 4.5])
f_X = Integrand_f(X)

plt.figure(figsize = (7, 4))
# Plot Integrand f over [0,5]
plt.plot(HorizontalAxis, Integrand_f(HorizontalAxis), linewidth = 2, color = 'dodgerblue',
         label = r"$f(x)$ := $e^{\sin(x^2) \cdot \cos(x)}$")
plt.plot(X, f_X, linestyle = 'None', marker = 'o', markersize = 8, color = 'magenta',
         label = "Observations")  # Plot function evaluations
plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 5))
plt.legend()
plt.title(r"Integrand of Interest $f$")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()
plt.show()


#=======================================================================================================================
#------------------------------------------------ GAUSSIAN PROCESS PRIOR -----------------------------------------------
#=======================================================================================================================
#---------------------------------- Define Gaussian Kernel & Corresponding Gram matrix ---------------------------------
def Kernel_SE(x, y, l, sigma2):
    Norm_2 = (x - y)**2  # Squared Euclidean norm of x - y
    return (sigma2**2) * np.exp( -Norm_2 / (2 * l**2))  # Scalar

def KernelMatrix(KernelFunction, x, y, **parameters):
    x = x[:, None]  # Changes to a column vector, dim=nx1
    y = y[None, :]  # Changes to a row vector, dim=1xn
    return KernelFunction(x, y, **parameters)

#----------------------------------- Plot Observations & Sample Functions from GP Prior --------------------------------
plt.figure(figsize = (7, 4))
plt.plot(X, f_X, linestyle = 'None', marker = 'o', markersize = 8, color = 'magenta')  # Plot function evaluations

rng = np.random.default_rng(42)  # Reproducibility of randomness
N = 1000  # Number of points to evaluate sample functions from GP at
HorizontalAxis_GP = np.linspace(start = 0, stop = 5, num = N)

# Define mean function of GP prior
def m_Prior(x):
    return 0 * np.array(x) + 1  # Constant prior GP mean of 1

# Set hyperparameters, by trial-and-error
l_Hyperparameter = 0.5
sigma2_Hyperparameter = 1
KernelMatrix_SE = KernelMatrix(Kernel_SE, HorizontalAxis_GP, HorizontalAxis_GP,
                               l = l_Hyperparameter, sigma2 = sigma2_Hyperparameter)
KernelMatrix_SE += 1e-6 * np.eye(N)  # Add jitter

# Next, plot GP prior sample functions
n_samples = 4  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_N))^T from finite Gaussian distribution, for n_samples different samples
f_samples = rng.multivariate_normal(mean = m_Prior(HorizontalAxis_GP), cov = KernelMatrix_SE, size = n_samples,
                                    method = 'cholesky')
for f in f_samples:
    plt.plot(HorizontalAxis_GP, f, alpha = 0.6)
plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 5))
plt.title(r"Observations  &  $\mathcal{GP}(1, k_{SE})$ Prior Sample Functions")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()
plt.show()


#=======================================================================================================================
#---------------------------------------------- GAUSSIAN PROCESS POSTERIOR ---------------------------------------------
#=======================================================================================================================
#------------------------------------------ Posterior Mean & Covariance Kernel -----------------------------------------
def GP_Posterior(KernelFunction, X, f_X, **hyperparameters):
    X = np.asarray(X)
    f_X = np.asarray(f_X)

    m_X = m_Prior(X)  # Prior mean function of f_X
    K_XX = KernelMatrix(KernelFunction, X, X, **hyperparameters)  # Gram matrix
    K_XX += 1e-6 * np.eye(X.shape[0])  # Add jitter

    L = np.linalg.cholesky(K_XX)  # Cholesky decmoposition, K_{XX} = L L^T

    #--------------------------------- Posterior Mean ---------------------------------
    #m_fPosterior = m + k_Xx.T @ K_XX_Inverse @ (f_X - m_X)
    def m_fPosterior(X_Input):  # X_Input can be an array of N inputs, shape=(N,1)
        X_Input = np.asarray(X_Input).reshape(-1)  # dim=Nx1
        m_X_Input = m_Prior(X_Input).reshape(-1)  # dim=Nx1
        k_Xx = KernelMatrix(KernelFunction, X, X_Input, **hyperparameters)  # dim=nxN

        u = np.linalg.solve(L.T, np.linalg.solve(L, f_X - m_X))  # K_{XX}^{-1} [f_X - m_X], dim=Nx1
        MeanUpdateTerm =  (k_Xx.T @ u).reshape(-1)  # k_{xX} K_{XX}^{-1} [f_X - m_X], dim=Nx1
        return m_X_Input + MeanUpdateTerm  # k_{xX} K_{XX}^{-1} k_{X \tilde{x}}

    #-------------------------- Posterior Covariance Kernel --------------------------
    def k_fPosterior(X_Input, X_InputTilde):
        X_Input = np.asarray(X_Input).reshape(-1)  # dim=Nx1
        X_InputTilde = np.asarray(X_InputTilde).reshape(-1)  # dim=Nx1

        k_xxTilde = KernelMatrix(KernelFunction, X_Input, X_InputTilde, **hyperparameters)  # k(x, \tilde{x})
        k_Xx = KernelMatrix(KernelFunction, X, X_Input, **hyperparameters)  # k_{Xx}
        k_XxTilde = KernelMatrix(KernelFunction, X, X_InputTilde, **hyperparameters)  # k_{X \tilde{x}}

        v_1 = np.linalg.solve(L, k_Xx)  # L^{-1} k_{Xx}
        v_2 = np.linalg.solve(L, k_XxTilde)  # L^{-1} k_{X \tilde{x}}
        return k_xxTilde - v_1.T @ v_2  # k(x, \tilde{x}) - k_{xX} K_{XX}^{-1} k_{X \tilde{x}}

    return m_fPosterior, k_fPosterior


#------- Function to Plot Posterior Sample Functions, Posterior Mean and Shaded Regions & Compute Integral Value -------
def SampleFunctionsPlot_GPPosterior(ax, X_Grid, f_X_Grid, KernelFunction,
                                    NSamples=n_samples, NEvaluations=N, nSD=4, IntervalLeft=0, IntervalRight=5,
                                    **hyperparameters):
    HorizontalAxis_GPGrid = np.linspace(start = IntervalLeft, stop = IntervalRight, num = NEvaluations)
    m_fPosterior_, k_fPosterior_ = GP_Posterior(KernelFunction, X, f_X, **hyperparameters)

    MeanVector_PosteriorGrid = m_fPosterior_(HorizontalAxis_GPGrid)  # Posterior mean evaluated at grid
    # Posterior covariance kernel evaluated at grid
    KernelMatrix_PosteriorGrid = k_fPosterior_(HorizontalAxis_GPGrid, HorizontalAxis_GPGrid)
    KernelMatrix_PosteriorGrid += 1e-6 * np.eye(N)  # Add jitter

    #----------------------- Posterior Samples: Plot & Integral ------------------------
    SampleFunctionsPlot = []
    IntegralSampleFunctions = []
    ### Plot the data points and some Gaussian prior sample functions
    rng = np.random.default_rng(42)  # Reproducibility of randomness
    f_Samples = rng.multivariate_normal(mean = MeanVector_PosteriorGrid, cov = KernelMatrix_PosteriorGrid,
                                        size = NSamples, method = 'cholesky')
    for f_ in f_Samples:
        Path, = ax.plot(HorizontalAxis_GPGrid, f_, alpha = 0.6)  # Plot posterior sample functions
        SampleFunctionsPlot.append(Path)
        IntegralSampleFunctions.append( np.trapezoid(f_, HorizontalAxis_GPGrid) )  # Integral approximations

    #------------------------ Posterior Mean: Plot & Integral -------------------------
    PosteriorMeanPath = ax.plot(HorizontalAxis_GPGrid, MeanVector_PosteriorGrid,
                                color = 'C4', linestyle = '--', label = "Posterior mean")
    PosteriorMeanIntegral = np.trapezoid(MeanVector_PosteriorGrid, HorizontalAxis_GPGrid)

    #--------------------------------- Shaded Regions ---------------------------------
    VarianceGP_PosteriorGrid = np.diag(KernelMatrix_PosteriorGrid)
    ShadedRegions = ax.fill_between(HorizontalAxis_GPGrid,   # +/- 1*SD regions
                                    MeanVector_PosteriorGrid - np.sqrt(VarianceGP_PosteriorGrid),
                                    MeanVector_PosteriorGrid + np.sqrt(VarianceGP_PosteriorGrid),
                                    color = 'cyan', linestyle = '--', alpha = 0.21, label = r"$\pm \sigma$")
    ShadedRegions2 = []
    ShadedRegions2_Below = ax.fill_between(HorizontalAxis_GPGrid,  # +/- 2*SD regions (below +/- 1*SD region)
                                           MeanVector_PosteriorGrid - 1 * np.sqrt(VarianceGP_PosteriorGrid),
                                           MeanVector_PosteriorGrid - 2 * np.sqrt(VarianceGP_PosteriorGrid),
                                           color = 'lightgrey', linestyle = '--', alpha = 0.2, label = r"$\pm 2\sigma$")
    ShadedRegions2.append(ShadedRegions2_Below)
    ShadedRegions2_Above = ax.fill_between(HorizontalAxis_GPGrid,  # +/- 2*SD regions (above +/- 1*SD region)
                                           MeanVector_PosteriorGrid + 1 * np.sqrt(VarianceGP_PosteriorGrid),
                                           MeanVector_PosteriorGrid + 2 * np.sqrt(VarianceGP_PosteriorGrid),
                                           color = 'lightgrey', linestyle = '--', alpha = 0.2)
    ShadedRegions2.append(ShadedRegions2_Above)
    ShadedRegions2.append(ax.legend())

    ShadedRegions_nSD = {
    "Mean": HorizontalAxis_GPGrid,
    "Lower_2SD": MeanVector_PosteriorGrid - 2 * np.sqrt(VarianceGP_PosteriorGrid),
    "Lower_nSD": MeanVector_PosteriorGrid - nSD * np.sqrt(VarianceGP_PosteriorGrid),
    "Upper_2SD": MeanVector_PosteriorGrid + 2 * np.sqrt(VarianceGP_PosteriorGrid),
    "Upper_nSD": MeanVector_PosteriorGrid + nSD * np.sqrt(VarianceGP_PosteriorGrid),
    }

    return {"SampleFunctionsPlot": SampleFunctionsPlot,
            "PosteriorMeanPath": PosteriorMeanPath,
            "ShadedRegions": ShadedRegions, "ShadedRegions2": ShadedRegions2, "ShadedRegions_nSD": ShadedRegions_nSD,
            "IntegralSampleFunctions": IntegralSampleFunctions, "PosteriorMeanIntegral": PosteriorMeanIntegral}




#=======================================================================================================================
#--------------------------------- PLOT POSTERIOR SAMPLES & THEIR INTEGRAL DISTRIBUTION --------------------------------
#=======================================================================================================================
#--------------------------------------- Integral of 1 Posterior Sample Function ---------------------------------------
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))
N = 1000
n_samples1 = 1
PosteriorSamples1 = SampleFunctionsPlot_GPPosterior(ax = ax1, X_Grid = X, f_X_Grid = f_X,
                                                 KernelFunction = Kernel_SE,
                                                 NSamples = n_samples1, NEvaluations = N,
                                                 l = l_Hyperparameter, sigma2 = sigma2_Hyperparameter)

### Left Plot - Posterior Sample Function & Shaded Regions
# Note: PosteriorSamples1 automatically plots it because of the 'ax = ax1'
ax1.scatter(X, f_X, c = 'magenta', s = 60)  # Plot observations
ax1.legend()
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 5))
ax1.set_title(r"GP Posterior Sample Functions with $\pm \sigma$, $\pm 2\sigma$")
ax1.set_xlabel(r"$x_i$")
ax1.set_ylabel(r"$f(x_i)$")


### Right Plot: (Approximation) Integral of Sample Function
plt.hist(PosteriorSamples1["IntegralSampleFunctions"], rwidth = 0.1, density = True, color = 'darkorange')
PosteriorMeanIntegral = PosteriorSamples1["PosteriorMeanIntegral"]
ax2.axvline(PosteriorMeanIntegral, color = 'C4', linewidth = 2, label = "Integral of Posterior Mean")
ax2.legend()
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_title(r"Integral Value of GP Posterior Sample Function")
ax2.set_xlabel(r"(Approximation) Integral Value")
ax2.set_ylabel(r"Probability Density")

plt.tight_layout()
fig.subplots_adjust(wspace = 0.15)  # Spacing between plots
plt.show()


#----------------------------------- Integral of Multiple Posterior Sample Functions -----------------------------------
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))
N = 1000
n_samples2 = 42
PosteriorSamples2 = SampleFunctionsPlot_GPPosterior(ax = ax1, X_Grid = X, f_X_Grid = f_X,
                                                 KernelFunction = Kernel_SE,
                                                 NSamples = n_samples2, NEvaluations = N,
                                                 l = l_Hyperparameter, sigma2 = sigma2_Hyperparameter)

### Left Plot - Posterior Sample Functions & Shaded Regions
# Note: OnePriorSample automatically plots it because of the 'ax = ax1'
ax1.scatter(X, f_X, c = 'magenta', s = 60)  # Plot observations
ax1.legend()
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 5))
ax1.set_title(r"GP Posterior Sample Functions with $\pm \sigma$, $\pm 2\sigma$")
ax1.set_xlabel(r"$x_i$")
ax1.set_ylabel(r"$f(x_i)$")


### Right Plot: (Approximation) Integral of Sample Functions
plt.hist(PosteriorSamples2["IntegralSampleFunctions"], rwidth = 0.1, density = True, color = 'darkorange')
PosteriorMeanIntegral = PosteriorSamples2["PosteriorMeanIntegral"]
ax2.axvline(PosteriorMeanIntegral, color = 'C4', linewidth = 2, label = "Integral of Posterior Mean")
ax2.legend()
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_title(r"Integral Values of GP Posterior Sample Function")
ax2.set_xlabel(r"(Approximation) Integral Value")
ax2.set_ylabel(r"Probability Density")

plt.tight_layout()
fig.subplots_adjust(wspace = 0.15)  # Spacing between plots
plt.show()



#-------------------------------- Integral of Multiple Posterior Sample Functions [2.0] --------------------------------
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))
N = 1000
n_samples3 = 4221
PosteriorSamples3 = SampleFunctionsPlot_GPPosterior(ax = ax1, X_Grid = X, f_X_Grid = f_X,
                                                 KernelFunction = Kernel_SE,
                                                 NSamples = n_samples3, NEvaluations = N, nSD = 5,
                                                 l = l_Hyperparameter, sigma2 = sigma2_Hyperparameter)

### Left Plot - Posterior Sample Functions & Shaded Regions
# Note: OnePriorSample automatically plots it because of the 'ax = ax1'
ax1.scatter(X, f_X, c = 'magenta', s = 60)  # Plot observations
ShadedRegionExtra = PosteriorSamples3["ShadedRegions_nSD"]
ax1.fill_between(ShadedRegionExtra["Mean"], ShadedRegionExtra["Lower_2SD"], ShadedRegionExtra["Lower_nSD"],
                 color = 'aquamarine', linestyle = '--', alpha = 0.2, label = r"$\pm 5 \sigma$")
ax1.fill_between(ShadedRegionExtra["Mean"], ShadedRegionExtra["Upper_2SD"], ShadedRegionExtra["Upper_nSD"],
                 color = 'aquamarine', linestyle = '--', alpha = 0.2)
ax1.legend()
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 5))
ax1.set_title(r"GP Posterior Sample Functions with $\pm \sigma$, $\pm 2\sigma$, $\pm 5\sigma$")
ax1.set_xlabel(r"$x_i$")
ax1.set_ylabel(r"$f(x_i)$")


### Right Plot: (Approximation) Integral of Sample Functions
ax2.hist(PosteriorSamples3["IntegralSampleFunctions"], bins = 200, density = True, color = 'darkorange')
PosteriorMeanIntegral = PosteriorSamples3["PosteriorMeanIntegral"]
ax2.axvline(PosteriorMeanIntegral, color = 'C4', linewidth = 2, label = "Integral of Posterior Mean")
ax2.legend()
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_title(r"Integral Values of GP Posterior Sample Function")
ax2.set_xlabel(r"(Approximation) Integral Value")
ax2.set_ylabel(r"Probability Density")

plt.tight_layout()
fig.subplots_adjust(wspace = 0.15)  # Spacing between plots
plt.show()




