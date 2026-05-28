########################################################################################################################
########################################## BAYESIAN QUADRATURE KERNEL CHOICE ###########################################
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import time
from BQ_UniversalFunctions import (Kernel_SE, Kernel_RQ, Kernel_Matern_1_2, Kernel_Matern_3_2, Kernel_Matern_5_2,
                                   Kernel_Polynomial2, Kernel_Periodic,
                                   KernelMatrix, SumKernels, LogMarginalLikelihood, MLEHyperparameters)


#=======================================================================================================================
#----------------------------------- Maximum Marginal Likelihood - Family of Kernels -----------------------------------
#=======================================================================================================================
# We import kernel functions from BQ_UniversalFunctions




#=======================================================================================================================
#------------------------------------------------ Continue BHQ Example -------------------------------------------------
#=======================================================================================================================
def Integrand_f(x):  # Integrand f(x)
    return np.exp( np.sin(x**2) * np.cos(x) )

n = 3  # Number of observations
X = np.linspace(start = 0, stop = 5, num = n)
f_X = Integrand_f(X)

### Define Mean Function of GP prior
def m_Prior(x, c):
    return 0 * np.array(x) + c  # Constant prior GP mean of c
#c_PriorMean = np.mean(f_X)
c_PriorMean = 0
PriorMean = m_Prior(X, c = c_PriorMean)




#=======================================================================================================================
#--------------------------------------------------- Kernel Selection --------------------------------------------------
#=======================================================================================================================

#------------------------------------------------- Define Kernel Family ------------------------------------------------
UniversalInitialParameter = 1

KernelFamily = {"SE": dict(kernel = Kernel_SE, parameter_names = ["l_SE", "sigmaf2_SE"],
                           initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                "RQ": dict(kernel = Kernel_RQ, parameter_names = ["l_RQ", "alpha_RQ", "sigmaf2_RQ"],
                           initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                 UniversalInitialParameter]),
                "M12": dict(kernel = Kernel_Matern_1_2, parameter_names = ["l_M12", "sigmaf2_M12"],
                            initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                "M32": dict(kernel = Kernel_Matern_3_2, parameter_names = ["l_M32", "sigmaf2_M32"],
                            initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                "M52": dict(kernel = Kernel_Matern_5_2, parameter_names = ["l_M52", "sigmaf2_M52"],
                            initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                "Poly": dict(kernel = Kernel_Polynomial2, parameter_names = ["c_Poly", "sigmaf2_Poly"],
                             initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                "Per": dict(kernel = Kernel_Periodic, parameter_names = ["lmbd_P", "l_P", "sigmaf2_P"],
                            initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                  UniversalInitialParameter]),
                }
# {} creates a dictionary (= an ordered collection) that stores data values in "key: value" pairs
# dict() also creates a dictionary. Hence, every kernel (name) is associated with a dictionary containing kernel info.


#------------------------------- Collect Kernels & Sums of 2 Kernels into One Dictionary -------------------------------
KernelSimulations = []  # Store the MLE information for different kernels
KernelNames = list(KernelFamily.keys())
# .keys() returns (a view object containing) the keys of the dictionary,
#       i.e.: dict_keys(['SE', 'RQ', 'M12', 'M32', 'M52', 'Poly', 'Per'])
# list() makes it a list which results in ['SE', 'RQ', 'M12', 'M32', 'M52', 'Poly', 'Per']

### Add every (single) kernel, with information defined in KernelFamily, to KernelSimulations
for name in KernelNames:
    KernelSimulations.append(
        dict(
            label = name,
            kernel = KernelFamily[name]['kernel'],
            parameter_names = KernelFamily[name]['parameter_names'],
            initial_parameters = np.array(KernelFamily[name]['initial_parameters'], dtype = float),
        )
    )

### Add sums of two kernels, with corresponding information, to KernelSimulations
for i in range(len(KernelNames)):
    for j in range(i + 1, len(KernelNames)):
        Sum_2Kernels = SumKernels(KernelFamily[ KernelNames[i] ]['kernel'],
                                  KernelFamily[ KernelNames[j] ]['kernel'])
        Sum_parameter_names = (KernelFamily[ KernelNames[i] ]['parameter_names'] +
                               KernelFamily[ KernelNames[j] ]['parameter_names'])
        Sum_initial_parameters = np.array(KernelFamily[ KernelNames[i] ]['initial_parameters'] +
                                          KernelFamily[ KernelNames[j] ]['initial_parameters'], dtype = float)

        KernelSimulations.append(
            dict(
                label = f"{KernelNames[i]}+{KernelNames[j]}",
                kernel = Sum_2Kernels,
                parameter_names = Sum_parameter_names,
                initial_parameters = Sum_initial_parameters,
            )
        )
#print(KernelSimulations)  # Dictionary containing all kernel of interest (+ corresponding information)


#-------------- Estimate (Hyperparameters by) Maximum Marginal Likelihood for every Kernel in KernelFamily -------------
KernelChoiceMLE = []

for kernelinfo in KernelSimulations:
    label = kernelinfo['label']
    kernelfunction = kernelinfo['kernel']
    parameter_names = kernelinfo['parameter_names']
    initial_parameters = kernelinfo['initial_parameters']

    ### Estimate Hyperparameter by ML
    hyperparameters_MLE = MLEHyperparameters(
        initial_parameters = initial_parameters,
        X_Input = X,
        y = f_X,
        PriorMeanVector = PriorMean,
        KernelFunction = kernelfunction,
        ParameterNames = parameter_names,
    )

    ### Compute the corresponding Log Marginal Likelihood Value
    LMLValue = LogMarginalLikelihood(
        loghyperparameters = np.log(hyperparameters_MLE),
        X_Input = X,
        y = f_X,
        PriorMeanVector = PriorMean,
        KernelFunction = kernelfunction,
        ParameterNames = parameter_names,
    )

    KernelChoiceMLE.append(
        dict(
            label = label,
            kernel = kernelfunction,
            parameter_names = parameter_names,
            hyperparameters_MLE = hyperparameters_MLE,
            LML = float(LMLValue),
        )
    )

KernelChoiceMLE_Sorted = sorted(KernelChoiceMLE, key = lambda d: d['LML'], reverse = True)  # Sort by descending LML

for k in KernelChoiceMLE_Sorted:
    print(f"{k['label']:10s}    LML = {k['LML']: .5f}   &   MLE Hyperparameters = {k['hyperparameters_MLE']}")


### Select Best Kernel with corresponding Attributes
BestKernelChoice = KernelChoiceMLE_Sorted[0]  # The kernel with the highest LML

BestKernelChoice_label = BestKernelChoice['label']
BestKernelChoice_kernel = BestKernelChoice['kernel']
BestKernelChoice_parameternames = BestKernelChoice['parameter_names']
BestKernelChoice_hyperparametersValues = BestKernelChoice['hyperparameters_MLE']
BestKernelChoice_LML = BestKernelChoice['LML']

BestKernelChoice_hyperparameters = dict(zip(BestKernelChoice_parameternames, BestKernelChoice_hyperparametersValues))
#print(BestKernelChoice_hyperparameters)

print("_______________________________________________________________________________________________________________")
print("")
print("Best kernel choice:", BestKernelChoice_label)
print("Log Marginal Likelihood:", BestKernelChoice_LML)
print("MLE Hyperparameter:", BestKernelChoice_hyperparametersValues)


#----------------------------------------------------- Bar Plot --------------------------------------------------------
KernelChoiceMLE_ReasonableLMLValues = [k for k in KernelChoiceMLE_Sorted if k['LML'] > -50]
SelectedKernels = [k['label'] for k in KernelChoiceMLE_ReasonableLMLValues]
SelectedKernels_LML = [k['LML'] for k in KernelChoiceMLE_ReasonableLMLValues]

plt.figure(figsize = (10, 6))
plt.barh(SelectedKernels, SelectedKernels_LML, color = 'darkorange')
plt.gca().invert_yaxis()   # Put the kernel with highest LML at the top
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.title(rf"LML per Kernel (Sum), n = {n} Observations")
plt.xlabel(r"Log Marginal Likelihood (LML)")
plt.tight_layout()
plt.show()




#=======================================================================================================================
#---------------------------------- Posterior Sample Functions for Best Kernel Choice ----------------------------------
#=======================================================================================================================

#----------------------------------------- Posterior Mean & Covariance Kernel ------------------------------------------
def GP_Posterior(KernelFunction, X, f_X, c=0, **hyperparameters):
    X = np.asarray(X)
    f_X = np.asarray(f_X)

    m_X = m_Prior(X, c)  # Prior mean function of f_X
    K_XX = KernelMatrix(KernelFunction, X, X, **hyperparameters)  # Gram matrix
    K_XX += 1e-6 * np.eye(X.shape[0])  # Add jitter

    L = np.linalg.cholesky(K_XX)  # Cholesky decmoposition, K_{XX} = L L^T

    #--------------------------------- Posterior Mean ---------------------------------
    #m_fPosterior = m + k_Xx.T @ K_XX_Inverse @ (f_X - m_X)
    def m_fPosterior(X_Input, c = c):  # X_Input can be an array of N inputs, shape=(N,1)
        X_Input = np.asarray(X_Input).reshape(-1)  # dim=Nx1
        m_X_Input = m_Prior(X_Input, c).reshape(-1)  # dim=Nx1
        k_Xx = KernelMatrix(KernelFunction, X, X_Input, **hyperparameters)  # dim=nxN

        u = np.linalg.solve(L.T, np.linalg.solve(L, f_X - m_X))  # K_{XX}^{-1} [f_X - m_X], dim=Nx1
        MeanUpdateTerm = (k_Xx.T @ u)#.reshape(-1)  # k_{xX} K_{XX}^{-1} [f_X - m_X], dim=Nx1
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
def SampleFunctionsPlot_GPPosterior(ax, X_Grid, f_X_Grid, KernelFunction, c=0, NSamples=4, NEvaluations=1000, nSD=4,
                                    IntervalLeft=0, IntervalRight=5, **hyperparameters):
    HorizontalAxis_GPGrid = np.linspace(start = IntervalLeft, stop = IntervalRight, num = NEvaluations)
    m_fPosterior, k_fPosterior = GP_Posterior(KernelFunction, X_Grid, f_X_Grid, c, **hyperparameters)

    MeanVector_PosteriorGrid = m_fPosterior(HorizontalAxis_GPGrid)  # Posterior mean evaluated at grid
    # Posterior covariance kernel evaluated at grid
    KernelMatrix_PosteriorGrid = k_fPosterior(HorizontalAxis_GPGrid, HorizontalAxis_GPGrid)
    KernelMatrix_PosteriorGrid += 1e-6 * np.eye(NEvaluations)  # Add jitter

    #----------------------- Posterior Samples: Plot & Integral ------------------------
    SampleFunctionsPlot = []
    IntegralSampleFunctions = []

    # Plot the data points and some Gaussian prior sample functions
    rng = np.random.default_rng(42)  # Reproducibility of randomness
    f_Samples = rng.multivariate_normal(mean = MeanVector_PosteriorGrid, cov = KernelMatrix_PosteriorGrid,
                                        size = NSamples, method = 'cholesky')
    for f_ in f_Samples:
        Path = ax.plot(HorizontalAxis_GPGrid, f_, alpha = 0.6)  # Plot posterior sample functions
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
                                           color = 'gray', linestyle = '--', alpha = 0.2, label = r"$\pm 2\sigma$")
    ShadedRegions2.append(ShadedRegions2_Below)
    ShadedRegions2_Above = ax.fill_between(HorizontalAxis_GPGrid,  # +/- 2*SD regions (above +/- 1*SD region)
                                           MeanVector_PosteriorGrid + 1 * np.sqrt(VarianceGP_PosteriorGrid),
                                           MeanVector_PosteriorGrid + 2 * np.sqrt(VarianceGP_PosteriorGrid),
                                           color = 'gray', linestyle = '--', alpha = 0.2)
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


#---------------------------- Proposal Integrand & Posterior Sample Functions - Generalized ----------------------------
plt.figure(figsize = (7, 4))
N = 1000
HorizontalAxis = np.linspace(start = 0, stop = 5, num = N)
n_samples = 1

### Plot Observations
#f_X = Integrand_f(X)
plt.plot(X, f_X, linestyle = 'None', marker = 'o', markersize = 8, color = 'magenta',
         label = "Observations", zorder = 5)  # Plot function evaluations

### Plot Proposal Integrand
plt.plot(HorizontalAxis, Integrand_f(HorizontalAxis),
         color = 'dodgerblue', label = r"$f(x)$", zorder = 6)

### Gram Matrix for Proposal Integrand - MLE Hyperparameters
PosteriorSamples = SampleFunctionsPlot_GPPosterior(ax = plt, X_Grid = X, f_X_Grid = f_X,
                                                   KernelFunction = BestKernelChoice_kernel,  # The best kernel function
                                                   c = c_PriorMean, NSamples = n_samples, NEvaluations = N,
                                                   **BestKernelChoice_hyperparameters)

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 5))
plt.legend()
plt.title(r"$\mathcal{GP}$"rf"({c_PriorMean:.1f}, {BestKernelChoice_label})"
          rf" Posterior Sample Functions with $\pm \sigma$, $\pm 2\sigma$,  n = {n}")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()
plt.show()




#=======================================================================================================================
#------------------------------------- Bayesian Quadrature $-$ Best Kernel Choice --------------------------------------
#=======================================================================================================================

#---------------------------------- True Integral Value - Estimated by Trapezoid Rule ----------------------------------
N = 1000  # Number of points to evaluate sample functions from GP at
HorizontalAxis = np.linspace(start = 0, stop = 5, num = N)
TrueIntegralValue = np.trapezoid(Integrand_f(HorizontalAxis), HorizontalAxis)
print(f"The true integral value is = {TrueIntegralValue}")


#------------------------------------------ Bayesian Quadrature Approximation ------------------------------------------
def BQ_1D_Approximations(X_Input, f_X_Input, ConstantPriorGPMean, KernelFunction,
                         NEvaluations=N, IntervalLeft=0, IntervalRight=5, ComputeVariance=True,
                         **kernelhyperparameters):
    ApproximationGrid = np.linspace(start = IntervalLeft, stop = IntervalRight, num = NEvaluations)

    #---------------------------- Prior GP Mean & Gram Matrix GP -----------------------------
    m_X = m_Prior(X_Input, c = ConstantPriorGPMean)  # Prior mean function of f
    K_XX = KernelMatrix(KernelFunction, X_Input, X_Input, **kernelhyperparameters)  # Gram matrix
    K_XX += 1e-6 * np.eye(X_Input.shape[0])  # Add jitter
    L = np.linalg.cholesky(K_XX)  # Cholesky decmoposition, K_{XX} = L L^T


    #-------------------------------- Posterior Mean Integral --------------------------------
    m_I0 = ConstantPriorGPMean * (IntervalRight - IntervalLeft)  # m_{I,0}

    k_xX_Grid = KernelMatrix(KernelFunction, ApproximationGrid, X_Input, **kernelhyperparameters)
    kappa_X = np.trapezoid(k_xX_Grid, ApproximationGrid, axis = 0)  # Approximate kappa_X

    u = np.linalg.solve(L.T, np.linalg.solve(L, f_X_Input - m_X))  # K_{XX}^{-1} [f_X - m_X], dim=NEvaluationsx1
    MeanUpdateTerm = (kappa_X.T @ u)  # k_{xX} K_{XX}^{-1} [f_X - m_X], dim=NEvaluationsx1
    PosteriorMeanIntegral_BQ = m_I0 + MeanUpdateTerm

    if ComputeVariance == False:
        return PosteriorMeanIntegral_BQ, None
    # If we only need the mean, this reduces the computational cost by a lost (e.g., for the time plots)

    #----------------------------- Posterior Covariance Integral -----------------------------
    ### Approximate k_{I,0}
    # x_Grid, xTilde_Grid = np.meshgrid(ApproximationGrid, ApproximationGrid, indexing = 'ij')
    # # Note: x_Grid[i, j] = x_i, xTilde_Grid[i, j] = xTilde_j
    # KernelonGrid = KernelFunction(x_Grid, xTilde_Grid, **kernelhyperparameters)
    KernelonGrid = KernelMatrix(KernelFunction, ApproximationGrid, ApproximationGrid, **kernelhyperparameters)
    k_I0 = np.trapezoid(np.trapezoid(KernelonGrid, ApproximationGrid, axis = 0), ApproximationGrid, axis = 0)

    w = np.linalg.solve(L, kappa_X)  # L^{-1} kappa_X
    PosteriorCovarianceIntegral_BQ = k_I0 - w.T @ w

    return PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ


#--------------------------------------- Plot Posterior Distribution of Integral ---------------------------------------
plt.figure(figsize = (7, 4))

### Compute BHQ Estimate and Uncertainty
NIntegrationGrid = 1000
PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ = BQ_1D_Approximations(X_Input = X, f_X_Input = f_X,
                                                                                ConstantPriorGPMean = c_PriorMean,
                                                                                KernelFunction = BestKernelChoice_kernel,
                                                                                NEvaluations = NIntegrationGrid,
                                                                                **BestKernelChoice_hyperparameters)
#print(PosteriorMeanIntegral_BQ, np.sqrt(PosteriorCovarianceIntegral_BQ))

### Plot Posterior Distribution & Integral Values
def GaussianPDF_1D(x, mu, sigma2):
    Norm2 = (x - mu)**2  # Squared Euclidean norm
    Factor = 1 / np.sqrt(2 * np.pi * sigma2)
    return Factor * np.exp( -Norm2 / (2 * sigma2) )

HorizontalAxis_PosteriorDistribution = np.linspace(start = 4, stop = 8, num = NIntegrationGrid)
PosteriorDistribution_BQ = GaussianPDF_1D(HorizontalAxis_PosteriorDistribution,
                                          mu = PosteriorMeanIntegral_BQ, sigma2 = PosteriorCovarianceIntegral_BQ)
plt.plot(HorizontalAxis_PosteriorDistribution, PosteriorDistribution_BQ)
# Integral of posterior mean function
plt.axvline(PosteriorMeanIntegral_BQ, color = 'C4', linewidth = 2, label = "Integral of Posterior Mean")
plt.axvline(TrueIntegralValue, color = 'lightgreen', linewidth = 2, label = "True Integral Value") # True integral value

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((4, 8))
plt.legend()
AbsoluteError_BQ = np.abs(TrueIntegralValue - PosteriorMeanIntegral_BQ)
plt.title(rf"$|$Error$|$ = {AbsoluteError_BQ:.3f}  &  " r"$\sqrt{k_{I,\curvearrowright}}$ = " 
          rf"{np.sqrt(PosteriorCovarianceIntegral_BQ):.3f},  n = {n}")
plt.xlabel(r"Integral Value")
plt.ylabel(r"Density")
plt.tight_layout()
plt.show()




########################################################################################################################
###################################### ADAPTIVE SAMPLING $-$ BAYESIAN QUADRATURE #######################################
########################################################################################################################

#=======================================================================================================================
#------------------------------------------------ Uncertainty Sampling -------------------------------------------------
#=======================================================================================================================
def UncertaintySampling(X_Candidates, X_Original, ConstantPriorGPMean, KernelFunction,
                        function=Integrand_f, **hyperparameters):
    """
    Return point with the highest posterior function()~GP variance: k(x,x) - k_{xX}K^{-1}k_{Xx}
    """
    #X_Candidates = np.asarray(X_Candidates).ravel()
    #X_Original = np.asarray(X_Original).ravel()
    X_Candidates = X_Candidates[~np.isin(X_Candidates, X_Original)]  # Delete points that are already in X_Original

    if X_Candidates.size == 0:
        return None  # If there a no new points to choose from.

    K_XX = KernelMatrix(KernelFunction, X_Original, X_Original, **hyperparameters)  # K_{XX}
    K_XX += 1e-6 * np.eye(len(X_Original))  # Add jitter
    L = np.linalg.cholesky(K_XX)  # Cholesky decomposition

    K_Xx = KernelMatrix(KernelFunction, X_Original, X_Candidates, **hyperparameters)  # k_{Xx} for x in X_Candidates
    w = np.linalg.solve(L, K_Xx)  # L^{-1} k_{Xx}
    # Note: K_Xx and w have dim=n x NGridSize
    MinusTerm = np.sum(w * w, axis = 0)  # dim=NGridSize x 1
    #MinusTerm = np.asarray(MinusTerm).ravel()

    # Compute the k(x,x) elements for x in X_Candidates (without computing the entire matrix which can become very
    #   computationally expensive)
    k_xx = np.empty_like(X_Candidates, dtype = float)
    for i in range(len(X_Candidates)):
        k_xx[i] = float(KernelFunction(np.asarray(X_Candidates[i]), np.asarray(X_Candidates[i]),  **hyperparameters))

    IndexMostUncertainPoint = np.argmax(k_xx - MinusTerm)  # np.argmax() returns the index of the maximiser
    return X_Candidates[IndexMostUncertainPoint]  #, np.diag(K_fPosterior)[IndexMostUncertainPoint]


#---------------- Proposal Integrand & Posterior Sample Functions - Uncertainty Sampling: 1 Extra Point ----------------
plt.figure(figsize = (7, 4))
N = 1000
HorizontalAxis = np.linspace(start = 0, stop = 5, num = N)
n_samples = 1

### Update X-grid with 1 Extra Point - Uncertainty Sampling
X_UncertaintySampling1 = UncertaintySampling(X_Candidates = np.linspace(start = 0, stop = 5, num = N), X_Original = X,
                                             ConstantPriorGPMean = c_PriorMean, KernelFunction = BestKernelChoice_kernel,
                                             **BestKernelChoice_hyperparameters)
f_X_UncertaintySampling1 = Integrand_f(X_UncertaintySampling1)
X_NewUS1 = np.sort(np.append(X, X_UncertaintySampling1))  # Add new observation to original batch
f_X_NewUS1 = Integrand_f(X_NewUS1)

### Plot all Observations, Posterior Mean & GP Posterior Samples
plt.plot(X_NewUS1, f_X_NewUS1, linestyle = 'None', marker = 'o', markersize = 8, color = 'magenta',
         label = "Observations", zorder = n_samples + 2)  # Plot function evaluations
plt.plot(X_UncertaintySampling1, f_X_UncertaintySampling1, linestyle = 'None', marker = 'D', markersize = 8,
         color = 'lime', label = "New Observations", zorder = n_samples + 2)  # New observations
plt.plot(HorizontalAxis, Integrand_f(HorizontalAxis), color = 'dodgerblue',
         label = r"$f(x)$", zorder = n_samples + 2)  # Plot integrand of interest f

PosteriorSamples = SampleFunctionsPlot_GPPosterior(ax = plt, X_Grid = X_NewUS1, f_X_Grid = f_X_NewUS1,
                                                   KernelFunction = BestKernelChoice_kernel, c = c_PriorMean,
                                                   NSamples = n_samples, NEvaluations = N,
                                                   **BestKernelChoice_hyperparameters)

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 5))
plt.legend()
plt.title(r"$\mathcal{GP}$"rf"({c_PriorMean:.1f}, {BestKernelChoice_label})"
          rf" Posterior Samples with $\pm \sigma$, $\pm 2\sigma$,  $n$ = {n}  &  " r"$n_{\text{Extra}}$" rf"= 1")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()
plt.show()


#-------------------- Plot Posterior Distribution of Integral - Uncertainty Sampling: 1 Extra Point --------------------
plt.figure(figsize = (7, 4))

### Compute BQ Estimate and Uncertainty - Uncertainty Sampling
NIntegrationGrid = 1000
PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ = BQ_1D_Approximations(
    X_Input = X_NewUS1, f_X_Input = f_X_NewUS1, ConstantPriorGPMean = c_PriorMean,
    KernelFunction = BestKernelChoice_kernel, NEvaluations = NIntegrationGrid, **BestKernelChoice_hyperparameters)

### Plot Integral Estimate & Posterior Integral Distribution
HorizontalAxis_PosteriorDistribution = np.linspace(start = 4, stop = 8, num = NIntegrationGrid)
PosteriorDistribution_BQ = GaussianPDF_1D(HorizontalAxis_PosteriorDistribution,
                                          mu = PosteriorMeanIntegral_BQ, sigma2 = PosteriorCovarianceIntegral_BQ)
plt.plot(HorizontalAxis_PosteriorDistribution, PosteriorDistribution_BQ)
# Integral of posterior mean function
plt.axvline(PosteriorMeanIntegral_BQ, color = 'C4', linewidth = 2, label = "Integral of Posterior Mean")
plt.axvline(TrueIntegralValue, color = 'lightgreen', linewidth = 2, label = "True Integral Value") # True integral value

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((4, 8))
plt.legend()
AbsoluteError_BQ = np.abs(TrueIntegralValue - PosteriorMeanIntegral_BQ)
plt.title(rf"$|$Error$|$ = {AbsoluteError_BQ:.3f}  &  " r"$\sqrt{k_{I,\curvearrowright}}$ = " 
          rf"{np.sqrt(PosteriorCovarianceIntegral_BQ):.3f},  $n$ = {n}  &  " r"$n_{\text{Extra}}$" rf"= 1")
plt.xlabel(r"Integral Value")
plt.ylabel(r"Density")
plt.tight_layout()
plt.show()


#-------------- Proposal Integrand & Posterior Sample Functions - Uncertainty Sampling: ... Extra Points ---------------
plt.figure(figsize = (7, 4))
N = 1000
HorizontalAxis = np.linspace(start = 0, stop = 5, num = N)
n_samples = 1

### Update X-grid with ... Extra Points - Uncertainty Sampling
NExtraPoints = 15
X_NewUS_History = [X.copy()]  # Keep track of every grid when a points is added

for i in range(NExtraPoints):
    x_New = UncertaintySampling( X_Candidates = np.linspace(0, 5, num = N), X_Original = X_NewUS_History[i],
                                 ConstantPriorGPMean = c_PriorMean,
                                 KernelFunction = BestKernelChoice_kernel, **BestKernelChoice_hyperparameters)

    X_NewUS_History.append(np.append(X_NewUS_History[i], x_New))  # Add new points to track record history

X_NewUS = np.sort(X_NewUS_History[-1])  # The last X-grid from history track record
f_X_NewUS = Integrand_f(X_NewUS)
X_NewUS_WithoutOriginal = X_NewUS[~np.isin(X_NewUS, X)]  # Extract only the new points
f_X_NewUS_WithoutOriginal = Integrand_f(X_NewUS_WithoutOriginal)


### Plot all Observations, Posterior Mean & GP Posterior Samples
plt.plot(X, f_X, linestyle = 'None', marker = 'o', markersize = 8, color = 'magenta',
         label = "Observations", zorder = n_samples + 2)  # Plot function evaluations
plt.plot(X_NewUS_WithoutOriginal, f_X_NewUS_WithoutOriginal, linestyle = 'None', marker = 'D',
         markersize = 8, color = 'lime', label = "New Observations", zorder = n_samples + 2)  # New observations
plt.plot(HorizontalAxis, Integrand_f(HorizontalAxis), color = 'dodgerblue',
         label = r"$f(x)$", zorder = n_samples + 2)  # Plot integrand of interest f

PosteriorSamples = SampleFunctionsPlot_GPPosterior(ax = plt, X_Grid = X_NewUS, f_X_Grid = f_X_NewUS,
                                                   KernelFunction = BestKernelChoice_kernel, c = c_PriorMean,
                                                   NSamples = n_samples, NEvaluations = 2000,
                                                   **BestKernelChoice_hyperparameters)

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 5))
plt.legend()
plt.title(r"$\mathcal{GP}$"rf"({c_PriorMean:.1f}, {BestKernelChoice_label})"
          rf" Posterior Samples with $\pm \sigma$, $\pm 2\sigma$,  $n$ = {n}  &  " 
          r"$n_{\text{Extra}}$" rf"= {NExtraPoints}")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()
plt.show()


#------------------ Plot Posterior Distribution of Integral - Uncertainty Sampling: ... Extra Points -------------------
plt.figure(figsize = (7, 4))

### Compute BQ Estimate and Uncertainty - Uncertainty Sampling
NIntegrationGrid = 1000
PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ = BQ_1D_Approximations(
    X_Input = X_NewUS, f_X_Input = f_X_NewUS, ConstantPriorGPMean = c_PriorMean,
    KernelFunction = BestKernelChoice_kernel, NEvaluations = NIntegrationGrid, **BestKernelChoice_hyperparameters)

### Plot Integral Estimate & Posterior Integral Distribution
HorizontalAxis_PosteriorDistribution = np.linspace(start = 4, stop = 8, num = NIntegrationGrid)
PosteriorDistribution_BQ = GaussianPDF_1D(HorizontalAxis_PosteriorDistribution,
                                          mu = PosteriorMeanIntegral_BQ, sigma2 = PosteriorCovarianceIntegral_BQ)
plt.plot(HorizontalAxis_PosteriorDistribution, PosteriorDistribution_BQ)
plt.axvline(PosteriorMeanIntegral_BQ, color = 'C4', linewidth = 2,
            label = "Integral of Posterior Mean")  # Integral of posterior mean function
plt.axvline(TrueIntegralValue, color = 'lightgreen', linewidth = 2,
            label = "True Integral Value")  # True integral value

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((4, 8))
plt.legend()
AbsoluteError_BQ = np.abs(TrueIntegralValue - PosteriorMeanIntegral_BQ)
plt.title(rf"$|$Error$|$ = {AbsoluteError_BQ:.3f}  &  " 
          r"$\sqrt{k_{I,\curvearrowright}}$ = " rf"{np.sqrt(PosteriorCovarianceIntegral_BQ):.3f},  $n$ = {n}  &  " 
          r"$n_{\text{Extra}}$" rf"= {NExtraPoints}")
plt.xlabel(r"Integral Value")
plt.ylabel(r"Density")
plt.tight_layout()
plt.show()




#=======================================================================================================================
#---------------------------------------- Minimise Posterior Integral Variance -----------------------------------------
#=======================================================================================================================
def MinimiseIntegralVariance(X_Candidates, X_Original, ConstantPriorGPMean, KernelFunction, function=Integrand_f,
                             NEvaluations=1000, IntervalLeft=0, IntervalRight=5, **hyperparameters):
    """
    Return point with the highest posterior function()~GP variance
    """
    X_Candidates = X_Candidates[~np.isin(X_Candidates, X_Original)]  # Delete points that are already in X_Original
    f_X_Original = function(X_Original)

    PosteriorCovarianceIntegral_Values = []
    for i in range(len(X_Candidates)):
        X_Withxi = np.append(X_Original, X_Candidates[i])  # X union {x_i}
        f_X_Withxi = np.append(f_X_Original, function(X_Candidates[i]))

        _, PosteriorCovarianceIntegral = BQ_1D_Approximations(
            X_Input = X_Withxi, f_X_Input = f_X_Withxi, ConstantPriorGPMean = ConstantPriorGPMean,
            KernelFunction = KernelFunction, NEvaluations = NEvaluations,
            IntervalLeft = IntervalLeft, IntervalRight = IntervalRight, **hyperparameters)
        # Add posterior integral variance with observations points 'X union {x_i}'
        PosteriorCovarianceIntegral_Values.append(PosteriorCovarianceIntegral)

    # np.argmax() returns the index of the maximiser
    IndexHighestIntegralVariance = np.argmin(PosteriorCovarianceIntegral_Values)
    return X_Candidates[IndexHighestIntegralVariance]

# MinimiseIntegralVariance(X_Candidates = np.array([1, 2]), X_Original = X, ConstantPriorGPMean = c_PriorMean,
#   KernelFunction = BestKernelChoice_kernel, **BestKernelChoice_hyperparameters)


#--------------- Proposal Integrand & Posterior Sample Functions - Minimise Posterior Integral Variance ----------------
plt.figure(figsize = (7, 4))
N = 1000
HorizontalAxis = np.linspace(start = 0, stop = 5, num = N)
n_samples = 1

### Update X-grid with ... Extra Points - Uncertainty Sampling
NExtraPoints = 0
X_NewMPIV_History = [X.copy()]  # Keep track of every grid when a points is added

for i in range(NExtraPoints):
    x_New = MinimiseIntegralVariance(X_Candidates = np.linspace(start = 0, stop = 5, num = N),
                                     X_Original = X_NewMPIV_History[i], ConstantPriorGPMean = c_PriorMean,
                                     KernelFunction = BestKernelChoice_kernel, **BestKernelChoice_hyperparameters)

    X_NewMPIV_History.append(np.append(X_NewMPIV_History[i], x_New))  # Add new points to track record history

X_NewMPIV = np.sort(X_NewMPIV_History[-1])  # The last X-grid from history track record
f_X_NewMPIV = Integrand_f(X_NewMPIV)
X_NewMPIV_WithoutOriginal = X_NewMPIV[~np.isin(X_NewMPIV, X)]  # Extract only the new points
f_X_NewMPIV_WithoutOriginal = Integrand_f(X_NewMPIV_WithoutOriginal)
print(X)
print(X_NewMPIV)

### Plot all Observations, Posterior Mean & GP Posterior Samples
plt.plot(X, f_X, linestyle = 'None', marker = 'o', markersize = 8, color = 'magenta',
         label = "Observations", zorder = n_samples + 2)  # Plot function evaluations
plt.plot(X_NewMPIV_WithoutOriginal, f_X_NewMPIV_WithoutOriginal, linestyle = 'None', marker = 'D',
         markersize = 8, color = 'lime', label = "New Observations", zorder = n_samples + 2)  # New observations
plt.plot(HorizontalAxis, Integrand_f(HorizontalAxis), color = 'dodgerblue',
         label = r"$f(x)$", zorder = n_samples + 2)  # Plot integrand of interest f

PosteriorSamples = SampleFunctionsPlot_GPPosterior(ax = plt, X_Grid = X_NewMPIV, f_X_Grid = f_X_NewMPIV,
                                                   KernelFunction = BestKernelChoice_kernel,
                                                   c = c_PriorMean, NSamples = n_samples, NEvaluations = 2000,
                                                   **BestKernelChoice_hyperparameters)

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 5))
plt.legend()
plt.title(r"$\mathcal{GP}$"rf"({c_PriorMean:.1f}, {BestKernelChoice_label})"
          rf" Posterior Samples with $\pm \sigma$, $\pm 2\sigma$,  $n$ = {n}  &  " r"$n_{\text{Extra}}$" 
          rf"= {NExtraPoints}")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()
plt.show()


#------------------- Plot Posterior Distribution of Integral - Minimise Posterior Integral Variance --------------------
plt.figure(figsize = (7, 4))

### Compute BQ Estimate and Uncertainty - Uncertainty Sampling
NIntegrationGrid = 1000
PosteriorMeanIntegral_BQMPIV, PosteriorCovarianceIntegral_BQMPIV = BQ_1D_Approximations(
    X_Input = X_NewMPIV, f_X_Input = f_X_NewMPIV, ConstantPriorGPMean = c_PriorMean,
    KernelFunction = BestKernelChoice_kernel, NEvaluations = NIntegrationGrid, **BestKernelChoice_hyperparameters)

### Plot Integral Estimate & Posterior Integral Distribution
HorizontalAxis_PosteriorDistribution_MPIV = np.linspace(start = 4, stop = 8, num = NIntegrationGrid)
PosteriorDistribution_BQMPIV = GaussianPDF_1D(HorizontalAxis_PosteriorDistribution_MPIV,
                                              mu = PosteriorMeanIntegral_BQMPIV,
                                              sigma2 = PosteriorCovarianceIntegral_BQMPIV)
plt.plot(HorizontalAxis_PosteriorDistribution_MPIV, PosteriorDistribution_BQMPIV)
plt.axvline(PosteriorMeanIntegral_BQMPIV, color = 'C4', linewidth = 2,
            label = "Integral of Posterior Mean")  # Integral of posterior mean function
plt.axvline(TrueIntegralValue, color = 'lightgreen', linewidth = 2,
            label = "True Integral Value")  # True integral value

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((4, 8))
plt.legend()
AbsoluteError_BQMPIV = np.abs(TrueIntegralValue - PosteriorMeanIntegral_BQMPIV)
plt.title(rf"$|$Error$|$ = {AbsoluteError_BQMPIV:.3f}  &  " r"$\sqrt{k_{I,\curvearrowright}}$ = "
          rf"{np.sqrt(PosteriorCovarianceIntegral_BQMPIV):.3f},  $n$ = {n}  &  " r"$n_{\text{Extra}}$" 
          rf"= {NExtraPoints}")
plt.xlabel(r"Integral Value")
plt.ylabel(r"Density")
plt.tight_layout()
plt.show()




########################################################################################################################
#################################### COMPARISON WITH SIMPLE MONTE CARLO INTEGRATION ####################################
########################################################################################################################

#=======================================================================================================================
#---------------------------------------------- Generalise Some Functions ----------------------------------------------
#=======================================================================================================================

#--------------------------------- Combine Uncertainty Sampling & BQ into 1 Function -----------------------------------
def BQ_1D_Approximations_UncertaintySampling(X_Input, ConstantPriorGPMean, KernelFunction, NExtraPoints,
                                             nGridSize, NIntegrationGrid=5000, function=Integrand_f,
                                             ComputeVariance=True, **hyperparameters):
    #---------------- Update X-grid with NExtraPoints Extra Points - Uncertainty Sampling ----------------
    X_NewUS_History = [X_Input.copy()]  # Keep track of every grid when a points is added

    for i in range(NExtraPoints):
        x_New = UncertaintySampling(X_Candidates = np.linspace(0, 5, num = nGridSize),
                                    X_Original = X_NewUS_History[i], ConstantPriorGPMean = ConstantPriorGPMean,
                                    KernelFunction = KernelFunction, **hyperparameters)

        if x_New is None:
            break  # Break when there are no more new points to choose from

        X_NewUS_History.append(np.append(X_NewUS_History[i], x_New))  # Add new points to track record history

    #-------------------- Compute BQ Estimate and Uncertainty - Uncertainty Sampling ---------------------
    X_NewUS = np.sort(X_NewUS_History[-1])  # The last X-grid from history track record
    f_X_NewUS = function(X_NewUS)
    X_NewUS_WithoutOriginal = X_NewUS[~np.isin(X_NewUS, X_Input)]  # Extract only the new points
    f_X_NewUS_WithoutOriginal = function(X_NewUS_WithoutOriginal)


    PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ = BQ_1D_Approximations(
        X_Input = X_NewUS, f_X_Input = f_X_NewUS, ConstantPriorGPMean = ConstantPriorGPMean,
        KernelFunction = KernelFunction, NEvaluations = NIntegrationGrid, ComputeVariance = ComputeVariance,
        **hyperparameters)

    return{"Estimate": PosteriorMeanIntegral_BQ, "Variance": PosteriorCovarianceIntegral_BQ,
           "X_Grid": X_NewUS, "f_X_Grid": f_X_NewUS, "X_Grid_Added": X_NewUS_WithoutOriginal,
           "f_X_Grid_Added": f_X_NewUS_WithoutOriginal}


#-------------------------------------- Generalise Kernel Selection into Function --------------------------------------
def KernelSelectionML(X, f_X, ConstantPriorGPMean=0, UniversalInitialParameter=1):
    """
    This function picks the kernel in the KernelFamily (containing the 'SE', 'RQ', 'M12', 'M32', 'M52', 'Poly' and 'Per'
        kernels & every sum of two of these up to symmetry) that has the highest log marginal likelihood.
    It is assumed that the following functions are already defined: Kernel_SE, Kernel_RQ, Kernel_Matern_1_2,
        Kernel_Matern_3_2, Kernel_Matern_5_2, Kernel_Polynomial2, Kernel_Periodic, SumKernels, KernelMatrix,
        LogMarginalLikelihood, MLEHyperparameters, Integrand_f, m_Prior.
    ...
    """
    #---------------------------------- Define Collection Containing (Single) Kernels ----------------------------------
    KernelFamily = {"SE": dict(kernel = Kernel_SE, parameter_names = ["l_SE", "sigmaf2_SE"],
                               initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "RQ": dict(kernel = Kernel_RQ, parameter_names = ["l_RQ", "alpha_RQ", "sigmaf2_RQ"],
                               initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                     UniversalInitialParameter]),
                    "M12": dict(kernel = Kernel_Matern_1_2, parameter_names = ["l_M12", "sigmaf2_M12"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "M32": dict(kernel = Kernel_Matern_3_2, parameter_names = ["l_M32", "sigmaf2_M32"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "M52": dict(kernel = Kernel_Matern_5_2, parameter_names = ["l_M52", "sigmaf2_M52"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "Poly": dict(kernel = Kernel_Polynomial2, parameter_names = ["c_Poly", "sigmaf2_Poly"],
                                 initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "Per": dict(kernel = Kernel_Periodic, parameter_names = ["lmbd_P", "l_P", "sigmaf2_P"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                      UniversalInitialParameter]),
                    }
    # {} creates a dictionary (= an ordered collection) that stores data values in "key: value" pairs
    # dict() also creates a dictionary. Hence, every kernel (name) is associated with a dictionary containing kernel
    #   information.


    #----------------------------- Collect Kernels & Sums of 2 Kernels into One Dictionary -----------------------------
    KernelSimulations = []  # Store the MLE information for different kernels
    KernelNames = list(KernelFamily.keys())
    # .keys() returns (a view object containing) the keys of the dictionary, i.e.: dict_keys(['SE', 'RQ', 'M12', 'M32',
    #       'M52', 'Poly', 'Per'])
    # list() makes it a list which results in ['SE', 'RQ', 'M12', 'M32', 'M52', 'Poly', 'Per']

    ### Add every (single) kernel, with information defined in KernelFamily, to KernelSimulations
    for name in KernelNames:
        KernelSimulations.append(
            dict(
                label = name,
                kernel = KernelFamily[name]['kernel'],
                parameter_names = KernelFamily[name]['parameter_names'],
                initial_parameters = np.array(KernelFamily[name]['initial_parameters'], dtype = float),
            )
        )

    ### Add sums of two kernels, with corresponding information, to KernelSimulations
    for i in range(len(KernelNames)):
        for j in range(i + 1, len(KernelNames)):
            Sum_2Kernels = SumKernels(KernelFamily[ KernelNames[i] ]['kernel'], KernelFamily[ KernelNames[j] ]['kernel'])
            Sum_parameter_names = (KernelFamily[ KernelNames[i] ]['parameter_names'] +
                                   KernelFamily[ KernelNames[j] ]['parameter_names'])
            Sum_initial_parameters = np.array(KernelFamily[ KernelNames[i] ]['initial_parameters'] +
                                              KernelFamily[ KernelNames[j] ]['initial_parameters'], dtype = float)

            KernelSimulations.append(
                dict(
                    label = f"{KernelNames[i]}+{KernelNames[j]}",
                    kernel = Sum_2Kernels,
                    parameter_names = Sum_parameter_names,
                    initial_parameters = Sum_initial_parameters,
                )
            )

    #----------- Estimate (Hyperparameters by) Maximum Marginal Likelihood for Every Kernel in KernelFamily ------------
    PriorMean_ = ConstantPriorGPMean * np.ones_like(f_X, dtype = float)
    KernelChoiceMLE = []
    for kernelinfo in KernelSimulations:
        label = kernelinfo['label']
        kernelfunction = kernelinfo['kernel']
        parameter_names = kernelinfo['parameter_names']
        initial_parameters = kernelinfo['initial_parameters']

        ### Estimate Hyperparameter by ML
        hyperparameters_MLE = MLEHyperparameters(
            initial_parameters = initial_parameters,
            X_Input = X,
            y = f_X,
            PriorMeanVector = PriorMean_,
            KernelFunction = kernelfunction,
            ParameterNames = parameter_names,
        )

        ### Compute the corresponding Log Marginal Likelihood Value
        LMLValue = LogMarginalLikelihood(
            loghyperparameters = np.log(hyperparameters_MLE),
            X_Input = X,
            y = f_X,
            PriorMeanVector = PriorMean_,
            KernelFunction = kernelfunction,
            ParameterNames = parameter_names,
        )

        KernelChoiceMLE.append(
            dict(
                label = label,
                kernel = kernelfunction,
                parameter_names = parameter_names,
                hyperparameters_MLE = hyperparameters_MLE,
                LML = float(LMLValue),
            )
        )

    #--------------------------------- Select Best Kernel with Corresponding Attributes --------------------------------
    KernelChoiceMLE_Sorted = sorted(KernelChoiceMLE, key = lambda d: d['LML'], reverse = True)  # Sort by descending LML
    BestKernelChoice = KernelChoiceMLE_Sorted[0]  # The kernel with the highest LML
    BestKernelChoice_label = BestKernelChoice['label']
    BestKernelChoice_kernel = BestKernelChoice['kernel']
    BestKernelChoice_parameternames = BestKernelChoice['parameter_names']
    BestKernelChoice_hyperparametersValues = BestKernelChoice['hyperparameters_MLE']
    BestKernelChoice_hyperparameters = dict(zip(BestKernelChoice_parameternames, BestKernelChoice_hyperparametersValues))
    BestKernelChoice_LML = BestKernelChoice['LML']

    return {"Label": BestKernelChoice_label, "Kernel": BestKernelChoice_kernel,
            "Hyperparameters": BestKernelChoice_hyperparameters,
            "HyperparametersValues": BestKernelChoice_hyperparametersValues,
            "HyperparametersNames": BestKernelChoice_parameternames,
            "LML": BestKernelChoice_LML,
            "List": KernelChoiceMLE_Sorted, "BestKernelChoice": BestKernelChoice}




#=======================================================================================================================
#-------------------------------------------------- Comparison Plots ---------------------------------------------------
#=======================================================================================================================

#----------------------------------------------- Monte Carlo Integration -----------------------------------------------
def MC_ShiftedStandardNormalProposal_Transform(x):  # Shifted Standard Normal Proposal PDF
    BetweenIntervalCondition = (x >= 0) & (x <= 5)
    Output = np.zeros_like(x)  # Vector or zeros of length equal to size of x
    # vector[condition] only applies the assigned value, if the condition is satisfied
    Output[BetweenIntervalCondition] = (Integrand_f(x[BetweenIntervalCondition]) *
                                        np.exp(0.5 * (x[BetweenIntervalCondition] - 5 / 2)**2))
    return Output


#----------------------------------------------------- Sample Grid -----------------------------------------------------
nMaxPower = 3  # Some nice values: 1.3, 1.7, 2, 3, 4
nSample_Grid = np.unique(np.logspace(start = 0, stop = nMaxPower, num = 200, base = 10, dtype = int))
# Note: This creates a sequence of numbers evenly spaces on a log_10 scale, starting at 10^0=1 and ending at 10^nMaxPower
rng = np.random.default_rng(3)


#---------------------------------------| MC Estimate - N(5/2, 1) Proposal PDF |----------------------------------------
# MCEstimate_ShiftedStandardNormal_Grid = []
# for i in range(len(nSample_Grid)):
#     StandardNormal_Samples = rng.normal(loc = 5 / 2, scale = 1, size = nSample_Grid[i])
#     MCEstimate_ShiftedStandardNormal = (np.sqrt(2 * np.pi) *
#                                         np.mean(MC_ShiftedStandardNormalProposal_Transform(StandardNormal_Samples),
#                                                 dtype = float))
#     MCEstimate_ShiftedStandardNormal_Grid.append(MCEstimate_ShiftedStandardNormal)
# print("--------------------------------- [MC_SSN] over n: Completed")
R = 100  # Number of repetitions of MC runs
MC_Values_SSN = np.empty((R, len(nSample_Grid)))  # dim=Rxlen(nSample_Grid)
for r in range(R):
    for i in range(len(nSample_Grid)):
        StandardNormal_Samples = rng.normal(loc = 5 / 2, scale = 1, size = nSample_Grid[i])

        # Compute the MC estimate up to and including the nSample_Grid[i]^th data point
        MC_Values_SSN[r, i] = (np.sqrt(2 * np.pi) * np.mean(
            MC_ShiftedStandardNormalProposal_Transform(StandardNormal_Samples), dtype = float))

MC_Estimate_SSN = np.mean(MC_Values_SSN, axis = 0)   # Compute the MC estimate per sample size
MC_Variance_SSN = np.var(MC_Values_SSN, axis = 0, ddof = 1)   # Compute the MC variance per sample size
print("--------------------------------- [MC] over n: Completed")


#-------------------------------------| MC Estimate - Uniform[0,5] Proposal PDF |---------------------------------------
# MCEstimate_Uniform_Grid = []
# for i in range(len(nSample_Grid)):
#     Uniform_Samples = rng.uniform(low = 0, high = 5, size = nSample_Grid[i])
#     MCEstimate_Uniform_Grid.append(5 * np.mean(Integrand_f(Uniform_Samples)))
# print("--------------------------------- [MC_Unif] over n: Completed")
R = 100  # Number of repetitions of MC runs
MC_Values_Uniform = np.empty((R, len(nSample_Grid)))  # dim=Rxlen(nSample_Grid)
for r in range(R):
    for i in range(len(nSample_Grid)):
        Uniform_Samples = rng.uniform(low = 0, high = 5, size = nSample_Grid[i])

        # Compute the MC estimate up to and including the nSample_Grid[i]^th data point
        MC_Values_Uniform[r, i] = 5 * np.mean(Integrand_f(Uniform_Samples), dtype = float)

MC_Estimate_Uniform = np.mean(MC_Values_Uniform, axis=0)  # Compute the MC estimate per sample size
MC_Variance_Uniform = np.var(MC_Values_Uniform, axis=0, ddof=1)  # Compute the MC variance per sample size
print("--------------------------------- [MC] over n: Completed")


#-------------------------------| BQ Estimate - Fixed Grid & MLE Hyperparameters Once |---------------------------------
VanillaBQ_Estimate_FixedGrid = []
# VanillaBQ_Variance_FixedGrid = []
NIntegrationGrid = 1000

nInitial = 15  # The number of sample points we start with
X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
f_X_nInitial = Integrand_f(X_nInitial)
KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
Index_nInitial = np.where(nSample_Grid == nInitial)[0][0]
nSample_Grid_StartnInitial = nSample_Grid[Index_nInitial : ]

for i in range(len(nSample_Grid_StartnInitial)):
    X_n = np.linspace(start = 0, stop = 5, num = nSample_Grid_StartnInitial[i])
    f_X_n = Integrand_f(X_n)

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (estimated once beforehand)
    VanillaBQ_Estimate, _  = BQ_1D_Approximations(
        X_Input = X_n, f_X_Input = f_X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial,
        NEvaluations = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_nInitial)
    VanillaBQ_Estimate_FixedGrid.append(VanillaBQ_Estimate)
    #VanillaBQ_Variance_FixedGrid.append(VanillaBQ_Variance)
print("--------------------------------- [VanillaBQ_MLEOnce] over n: Completed")


#--------------------| BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters |-----------------------
VanillaBQreHyperparameters_Estimate_FixedGrid = []
# The following is similar to the next-to-last code above, hence it does not need to run again. Only when something
#   is changed:
# nInitial = 15  # The number of sample points we start with
# X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
# f_X_nInitial = Integrand_f(X_nInitial)
# KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
# MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
# MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
InitialParameters_nInitial = 0 * MLEHyperparametersValues_nInitial + np.ones_like(MLEHyperparametersValues_nInitial)
MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']
# Index_nInitial = np.where(nSample_Grid == nInitial)[0][0]
# nSample_Grid_StartnInitial = nSample_Grid[Index_nInitial : ]
for i in range(len(nSample_Grid_StartnInitial)):
    X_n = np.linspace(start = 0, stop = 5, num = nSample_Grid_StartnInitial[i])
    f_X_n = Integrand_f(X_n)

    # Re-estimate the hyperparameters
    MLEHyperparameters_n = MLEHyperparameters(
        initial_parameters = InitialParameters_nInitial, X_Input = X_n, y = f_X_n, PriorMeanVector = c_PriorMean,
        KernelFunction = MLEKernel_nInitial, ParameterNames = MLEHyperparametersNames_nInitial)

    MLEHyperparameters_n = dict(zip(MLEHyperparametersNames_nInitial, MLEHyperparameters_n))  # Add names

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    VanillaBQreHyperparameters_Estimate, _ = BQ_1D_Approximations(
        X_Input = X_n, f_X_Input = f_X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial,
        NEvaluations = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n)
    VanillaBQreHyperparameters_Estimate_FixedGrid.append(VanillaBQreHyperparameters_Estimate)
    #VanillaBQreKernel_Variance_FixedGrid.append(VanillaBQreKernel_Variance)
print("--------------------------------- [VanillaBQ_reHyperparameters] over n: Completed")


#----------------------| BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML |------------------------
VanillaBQreKernel_Estimate_FixedGrid = []
# VanillaBQreKernel_Variance_FixedGrid = []
# NIntegrationGrid = 5000
for i in range(len(nSample_Grid)):
    X_n = np.linspace(start = 0, stop = 5, num = nSample_Grid[i])
    f_X_n = Integrand_f(X_n)

    # Re-estimate the kernel function & corresponding hyperparameters
    KernelSelectionML_n = KernelSelectionML(X_n, f_X_n, UniversalInitialParameter = 1)
    MLEKernel_n = KernelSelectionML_n['Kernel']  # Kernel with highest log(ML)
    MLEHyperparameters_n = KernelSelectionML_n['Hyperparameters']  # Kernel hyperparameters with highest log(ML)

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (re-estimated each time)
    VanillaBQreKernel_Estimate, _ = BQ_1D_Approximations(
        X_Input = X_n, f_X_Input = f_X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_n,
        NEvaluations = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n)
    VanillaBQreKernel_Estimate_FixedGrid.append(VanillaBQreKernel_Estimate)
    #VanillaBQreKernel_Variance_FixedGrid.append(VanillaBQreKernel_Variance)
print("--------------------------------- [VanillaBQ_reKernel] over n: Completed")


#--------------------------| BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once |----------------------------
BQUncertaintySampling_Estimate_FixedGrid = []
# BQUncertaintySampling_Variance_FixedGrid = []
# NIntegrationGrid = 1000
nGridSize = 2100
# The following is similar to the next-to-last code above, hence it does not need to run again. Only when something
#   is changed:
# nInitial = 15  # The number of sample points we start with
# X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
# f_X_nInitial = Integrand_f(X_nInitial)
# KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
# MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
# MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
#
# Index_nInitial = np.where(nSample_Grid == nInitial)[0][0]
# nSample_Grid_StartnInitial = nSample_Grid[Index_nInitial : ]
nGrid_Differences = np.insert(nSample_Grid_StartnInitial, 0, nInitial)
# Note: nInitial is added at the beginning for one extra time. Hence, when we take the (ascending) differences between
#   the elements, the first entry is zero.
for i in range(len(nSample_Grid_StartnInitial)):
    X_n = np.linspace(start = 0, stop = 5, num = nSample_Grid_StartnInitial[i])
    f_X_n = Integrand_f(X_n)

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (estimated once beforehand)
    BQUncertaintySampling = BQ_1D_Approximations_UncertaintySampling(
        X_Input = X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial,
        NExtraPoints = nGrid_Differences[i + 1] - nGrid_Differences[i], nGridSize = nGridSize,
        NIntegrationGrid = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_nInitial)

    BQUncertaintySampling_Estimate = BQUncertaintySampling['Estimate']
    #BQUncertaintySampling_Variance = BQUncertaintySampling['Variance']
    BQUncertaintySampling_Estimate_FixedGrid.append(BQUncertaintySampling_Estimate)
    #BQUncertaintySampling_Variance_FixedGrid.append(BQUncertaintySampling_Variance)
print("--------------------------------- [BQUS_MLEOnce] over n: Completed")


#-----------------| BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Re-estimate Hyperparameters |----------------
BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid = []
# BQUncertaintySampling_Variance_FixedGrid = []
# The following is similar to the next-to-last code above, hence it does not need to run again. Only when something
#   is changed:
# nInitial = 15  # The number of sample points we start with
# X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
# f_X_nInitial = Integrand_f(X_nInitial)
# KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
# MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
# MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
# MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
# InitialParameters_nInitial = 0 * MLEHyperparametersValues_nInitial + np.ones_like(MLEHyperparametersValues_nInitial)
# MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']
#
# Index_nInitial = np.where(nSample_Grid == nInitial)[0][0]
# nSample_Grid_StartnInitial = nSample_Grid[Index_nInitial : ]
# nGrid_Differences = np.insert(nSample_Grid_StartnInitial, 0, nInitial)
# Note: nInitial is added at the beginning for one extra time. Hence, when we take the (ascending) differences between
#   the elements, the first entry is zero.

for i in range(len(nSample_Grid_StartnInitial)):
    X_n = np.linspace(start = 0, stop = 5, num = nSample_Grid_StartnInitial[i])
    f_X_n = Integrand_f(X_n)

    # Re-estimate the hyperparameters
    MLEHyperparameters_n_US = MLEHyperparameters(
        initial_parameters = InitialParameters_nInitial, X_Input = X_n, y = f_X_n, PriorMeanVector = c_PriorMean,
        KernelFunction = MLEKernel_nInitial, ParameterNames = MLEHyperparametersNames_nInitial)

    MLEHyperparameters_n_US = dict(zip(MLEHyperparametersNames_nInitial, MLEHyperparameters_n_US))  # Add names

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    BQUncertaintySampling_reHyperparameters = BQ_1D_Approximations_UncertaintySampling(
        X_Input = X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial,
        NExtraPoints = nGrid_Differences[i + 1] - nGrid_Differences[i], nGridSize = nGridSize,
        NIntegrationGrid = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n_US)

    BQUncertaintySampling_reHyperparameters_Estimate = BQUncertaintySampling_reHyperparameters['Estimate']
    #BQUncertaintySampling_reKernel_Variance = BQUncertaintySampling_reKernel['Variance']
    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid.append(BQUncertaintySampling_reHyperparameters_Estimate)
    #BQUncertaintySampling_reKernel_Variance_FixedGrid.append(BQUncertaintySampling_reKernel_Variance)
print("--------------------------------- [BQUS_reHyperparameter] over n: Completed")


#-----------------| BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML |-------------------
BQUncertaintySampling_reKernel_Estimate_FixedGrid = []
# BQUncertaintySampling_reKernel_Variance_FixedGrid = []
# The following is similar to the next-to-last code above, hence it does not need to run again. Only when something
#   is changed:
# nInitial = 15  # The number of sample points we start with
# X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
# f_X_nInitial = Integrand_f(X_nInitial)
# Index_nInitial = np.where(nSample_Grid == nInitial)[0][0]
# nSample_Grid_StartnInitial = nSample_Grid[Index_nInitial : ]
# nGrid_Differences = np.insert(nSample_Grid_StartnInitial, 0, nInitial)
# Note: nInitial is added at the beginning for one extra time. Hence, when we take the (ascending) differences between
#   the elements, the first entry is zero.
for i in range(len(nSample_Grid_StartnInitial)):
    X_n = np.linspace(start = 0, stop = 5, num = nSample_Grid_StartnInitial[i])
    f_X_n = Integrand_f(X_n)

    # Re-estimate the kernel function & corresponding hyperparameters
    KernelSelectionML_n_US = KernelSelectionML(X_n, f_X_n, UniversalInitialParameter = 1)
    MLEKernel_n_US = KernelSelectionML_n_US['Kernel']  # Kernel with highest log(ML)
    MLEHyperparameters_n_US = KernelSelectionML_n_US['Hyperparameters']  # Kernel hyperparameters with highest log(ML)

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (re-estimated each time)
    BQUncertaintySampling_reKernel = BQ_1D_Approximations_UncertaintySampling(
        X_Input = X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_n_US,
        NExtraPoints = nGrid_Differences[i + 1] - nGrid_Differences[i], nGridSize = nGridSize,
        NIntegrationGrid = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n_US)

    BQUncertaintySampling_reKernel_Estimate = BQUncertaintySampling_reKernel['Estimate']
    #BQUncertaintySampling_reKernel_Variance = BQUncertaintySampling_reKernel['Variance']
    BQUncertaintySampling_reKernel_Estimate_FixedGrid.append(BQUncertaintySampling_reKernel_Estimate)
    #BQUncertaintySampling_reKernel_Variance_FixedGrid.append(BQUncertaintySampling_reKernel_Variance)
print("--------------------------------- [BQUS_reMLE] over n: Completed")


#-------------------------------------------- Plot Convergence of Estimate ---------------------------------------------
plt.figure(figsize = (10, 6))

### Convergence Estimate: MC Estimate - N(5/2, 1) Proposal PDF
# plt.plot(nSample_Grid, MCEstimate_ShiftedStandardNormal_Grid,
#          color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.plot(nSample_Grid, MC_Estimate_SSN,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid,
                 MC_Estimate_SSN - np.sqrt(MC_Variance_SSN),
                 MC_Estimate_SSN + np.sqrt(MC_Variance_SSN),
                 color = 'lightgrey', linestyle = '--', alpha = 0.3)

### Convergence Estimate: MC Estimate - Unif[0, 5] Proposal PDF
# plt.plot(nSample_Grid, MCEstimate_Uniform_Grid,
#          color = 'skyblue', linewidth = 1.8,  label = r"MC Estimate - Unif$[0,5]$")
plt.plot(nSample_Grid, MC_Estimate_Uniform,
         color = 'skyblue', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - Unif$[0,5]$")
plt.fill_between(nSample_Grid,
                 MC_Estimate_Uniform - np.sqrt(MC_Variance_Uniform),
                 MC_Estimate_Uniform + np.sqrt(MC_Variance_Uniform),
                 color = 'lightgrey', linestyle = '--', alpha = 0.3)

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Hyperparameters Once
plt.plot(nSample_Grid_StartnInitial, VanillaBQ_Estimate_FixedGrid,
         color = 'forestgreen', linewidth = 1.8,  label = r"Vanilla BQ - MLE Once")

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
plt.plot(nSample_Grid_StartnInitial, VanillaBQreHyperparameters_Estimate_FixedGrid,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Convergence Estimate: BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
plt.plot(nSample_Grid, VanillaBQreKernel_Estimate_FixedGrid,
         color = 'red', linewidth = 1.8,  label = r"Vanilla BQ - re-MLE")

### Convergence Estimate: BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once
plt.plot(nSample_Grid_StartnInitial, BQUncertaintySampling_Estimate_FixedGrid,
         color = 'deeppink', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - MLE Once")

### Convergence Estimate: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
plt.plot(nSample_Grid_StartnInitial, BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid,
         color = 'darkturquoise', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Convergence Estimate: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
plt.plot(nSample_Grid_StartnInitial, BQUncertaintySampling_reKernel_Estimate_FixedGrid,
         color = 'mediumslateblue', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE")


### Plot True Integral Value
plt.axhline(TrueIntegralValue, color = 'lightgreen', linewidth = 2, alpha = 0.7, label = "True Integral Value")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((10**0, 10**2))
plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**2) + 1, step = 5)]) )
plt.legend()
plt.title(r"Convergence of Integral Estimate")
plt.xlabel(r"Number of Observations $n$")
plt.ylabel(r"Integral Estimate/Value")
plt.tight_layout()
#plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations.png', dpi = 300)
plt.savefig('Images/BQ_ConvergenceEstimate_NumberofObservations.png', dpi = 300)
plt.show()


#------------------------------------------ Plot Absolute Error of Estimates -------------------------------------------
plt.figure(figsize = (10, 6))

### Absolute Error: MC Estimate - N(5/2, 1) Proposal PDF
# AbsoluteError_MC_ShiftedStandardNormal = np.abs(TrueIntegralValue - MCEstimate_ShiftedStandardNormal_Grid)
# plt.plot(nSample_Grid, AbsoluteError_MC_ShiftedStandardNormal,
#          color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
AbsoluteError_MC_SSN = np.abs(TrueIntegralValue - MC_Values_SSN)
AbsoluteError_MC_SSN_Mean = np.mean(AbsoluteError_MC_SSN, axis = 0)
AbsoluteError_MC_SSN_Std = np.std(AbsoluteError_MC_SSN, axis = 0, ddof = 1)
plt.plot(nSample_Grid, AbsoluteError_MC_SSN_Mean,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid,  # +/- 2*SD regions (above +/- 1*SD region)
                 AbsoluteError_MC_SSN_Mean - 1 * AbsoluteError_MC_SSN_Std,
                 AbsoluteError_MC_SSN_Mean + 1 * AbsoluteError_MC_SSN_Std,
                 color = 'lightgrey', linestyle = '--', alpha = 0.3)

### Absolute Error: MC Estimate - Unif[0, 5] Proposal PDF
# AbsoluteError_MC_Uniform = np.abs(TrueIntegralValue - MCEstimate_Uniform_Grid)
# plt.plot(nSample_Grid, AbsoluteError_MC_Uniform,
#          color = 'skyblue', linewidth = 1.8,  label = r"MC Estimate - Unif$[0,5]$")
AbsoluteError_MC_Uniform = np.abs(TrueIntegralValue - MC_Values_Uniform)
AbsoluteError_MC_Uniform_Mean = np.mean(AbsoluteError_MC_Uniform, axis = 0)
AbsoluteError_MC_Uniform_Std = np.std(AbsoluteError_MC_Uniform, axis = 0, ddof = 1)
plt.plot(nSample_Grid, AbsoluteError_MC_Uniform_Mean,
         color = 'skyblue', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - Unif$[0,5]$")
plt.fill_between(nSample_Grid,  # +/- 2*SD regions (above +/- 1*SD region)
                 AbsoluteError_MC_Uniform_Mean - 1 * AbsoluteError_MC_Uniform_Std,
                 AbsoluteError_MC_Uniform_Mean + 1 * AbsoluteError_MC_Uniform_Std,
                 color = 'lightgrey', linestyle = '--', alpha = 0.3)

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
AbsoluteError_VanillaBQ= np.abs(TrueIntegralValue - VanillaBQ_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQ,
         color = 'forestgreen', linewidth = 1.8,  label = r"Vanilla BQ - MLE Once")

### Absolute Error: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
AbsoluteError_VanillaBQ_reKernel= np.abs(TrueIntegralValue - VanillaBQreHyperparameters_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQ_reKernel,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
AbsoluteError_VanillaBQ_reKernel= np.abs(TrueIntegralValue - VanillaBQreKernel_Estimate_FixedGrid)
plt.plot(nSample_Grid, AbsoluteError_VanillaBQ_reKernel,
         color = 'red', linewidth = 1.8, label = r"Vanilla BQ - re-MLE")

### Absolute Error: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
AbsoluteError_VanillaBQUncertaintySampling = np.abs(TrueIntegralValue - BQUncertaintySampling_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQUncertaintySampling,
         color = 'deeppink', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - MLE Once")

### Absolute Error: BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Hyperparameters by ML
AbsoluteError_VanillaBQUncertaintySampling_reHyperparameters = np.abs(
    TrueIntegralValue - BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQUncertaintySampling_reHyperparameters,
         color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
AbsoluteError_VanillaBQUncertaintySampling_reKernel = np.abs(
    TrueIntegralValue - BQUncertaintySampling_reKernel_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQUncertaintySampling_reKernel,
         color = 'mediumslateblue', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE")


### Plot BigO Lines
def BigO_Line(n, c, p):
    return c * n**p

# BigO(n^{-1/2})
plt.axline((1, BigO_Line(1, 1 / 2, -1 / 2)), (10, BigO_Line(10, 1 / 2, -1 / 2)),
           color = 'grey', alpha = 0.5, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 1, -1 / 2)), (10, BigO_Line(10, 1, -1 / 2)),
           color = 'grey', alpha = 0.5, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 2, -1 / 2)), (10, BigO_Line(10, 2, -1 / 2)),
           color = 'grey', alpha = 0.5, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 4, -1 / 2)), (10, BigO_Line(10, 4, -1 / 2)),
           color = 'grey', alpha = 0.5, linewidth = 1, linestyle = ':')
Rotation_1over2 = np.degrees(np.arctan( (BigO_Line(1, 1, -1 / 2) - BigO_Line(10, 1, -1 / 2)) / (1 - 10) ))
plt.text(1.1, 0.45, r"$\mathcal{O}(n^{-1/2})$", color = 'grey', alpha = 1, rotation = Rotation_1over2)

# BigO(n^{-2})
plt.axline((1, BigO_Line(1, 5, -2)), (10, BigO_Line(10, 5, -2)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 10, -2)), (10, BigO_Line(10, 10, -2)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 20, -2)), (10, BigO_Line(10, 20, -2)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 40, -2)), (10, BigO_Line(10, 40, -2)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
Rotation_2 = np.degrees(np.arctan( (BigO_Line(1, 2, -2) - BigO_Line(10, 2, -2)) / (1 - 10) ))
plt.text(1.1, 5.4, r"$\mathcal{O}(n^{-2})$", color = 'grey', alpha = 1, rotation = Rotation_2)

# BigO(n^{-4})
plt.axline((1, BigO_Line(1, 400, -4)), (10, BigO_Line(10, 400, -4)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 800, -4)), (10, BigO_Line(10, 800, -4)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 1600, -4)), (10, BigO_Line(10, 1600, -4)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
plt.axline((1, BigO_Line(1, 3200, -4)), (10, BigO_Line(10, 3200, -4)),
           color = 'grey', alpha = 0.7, linewidth = 1, linestyle = ':')
Rotation_4 = np.degrees(np.arctan( (BigO_Line(1, 5, -4) - BigO_Line(10, 5, -4)) / (1 - 10) ))
plt.text(3, 3.55, r"$\mathcal{O}(n^{-4})$", color = 'grey', alpha = 1, rotation = Rotation_4)


plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xscale('log')  # log with 10 base is default
plt.yscale('log')
plt.xlim((10**0, 10**nMaxPower))
#plt.ylim((10**(-8), 10**(2)))
#plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**nMaxPower) + 1, step = 5)]) )
plt.legend()
plt.title(r"$\log_{10}$(Absolute Error of Estimates over Sample Sizes)")
plt.xlabel(r"$\log_{10}$(Number of Observations $n$)")
plt.ylabel(r"$\log_{10}(|I_{True} - \hat{I}|)$")
plt.tight_layout()
# plt.savefig('results/BQ_AbsoluteError_NumberofObservations.png', dpi = 300)
plt.savefig('Images/BQ_AbsoluteError_NumberofObservations.png', dpi = 300)
plt.show()




#=======================================================================================================================
#----------------------------------------------------- Time Plots ------------------------------------------------------
#=======================================================================================================================
rng = np.random.default_rng(21)
T_Max = 1000  # In seconds
NIntegrationGrid = 1000  # For approximation of BQ non-analytical integrals
nGridSize = 2100  # The number of possible points to consider for uncertainty sampling
#nGridSize = 3000  # The number of possible points to consider for uncertainty sampling

#-------------------------------------| Plot MC Estimate - N(5/2, 1) Proposal PDF |-------------------------------------
StartTime_MC_SSN = time.perf_counter()  # Start time counter
MCEstimate_ShiftedStandardNormal_Grid_Time = []
Times_MC_SSN = []
TotalTime_MC_SSN = 0
Transformation_Sum_MC_SSN = 0
n_MC_SSN = 1

while TotalTime_MC_SSN < T_Max:
    StandardNormal_Samples = rng.normal(loc = 5 / 2, scale = 1, size = 1)  # Draw normal realization for new observation
    # Add to the sum of n observations
    Transformation_Sum_MC_SSN += np.sum( MC_ShiftedStandardNormalProposal_Transform(StandardNormal_Samples) )

    MCEstimate_ShiftedStandardNormal_Time = np.sqrt(2 * np.pi) * (Transformation_Sum_MC_SSN / n_MC_SSN)  # MC estimate
    MCEstimate_ShiftedStandardNormal_Grid_Time.append(MCEstimate_ShiftedStandardNormal_Time)

    TotalTime_MC_SSN = time.perf_counter() - StartTime_MC_SSN  # Update current time
    Times_MC_SSN.append(TotalTime_MC_SSN)  # Update total time history
    n_MC_SSN += 1  # Update to n+1 observations
print("--------------------------------- [MC_SSN] over time: Completed")


#------------------------------------| Plot MC Estimate - Uniform[0,5] Proposal PDF |-----------------------------------
StartTime_MC_Unif = time.perf_counter()  # Start time counter
MCEstimate_Uniform_Grid_Time = []
Times_MC_Unif = []
TotalTime_MC_Unif = 0
Transformation_Sum_MC_Unif = 0
n_MC_Unif = 1

while TotalTime_MC_Unif < T_Max:
    Uniform_Samples = rng.uniform(low = 0, high = 5, size = 1)  # Draw normal realization for new observation
    Transformation_Sum_MC_Unif += np.sum( Integrand_f(Uniform_Samples) )  # Add to the sum of n observations

    MCEstimate_Uniform_Time = 5 * (Transformation_Sum_MC_Unif / n_MC_Unif)  # MC estimate
    MCEstimate_Uniform_Grid_Time.append(MCEstimate_Uniform_Time)

    TotalTime_MC_Unif = time.perf_counter() - StartTime_MC_Unif  # Update current time
    Times_MC_Unif.append(TotalTime_MC_Unif)  # Update total time history
    n_MC_Unif += 1  # Update to n+1 observations
print("--------------------------------- [MC_Unif] over time: Completed")


#------------------------------| Plot BQ Estimate - Fixed Grid & MLE Hyperparameters Once |-----------------------------
StartTime_VBQ_MLEOnce = time.perf_counter()  # Start time counter
VanillaBQ_Estimate_FixedGrid_Time = []
# VanillaBQ_Variance_FixedGrid_Time = []

nInitial = 15  # The number of sample points we start with
X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
f_X_nInitial = Integrand_f(X_nInitial)
KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']

Times_VBQ_MLEOnce = []
TotalTime_VBQ_MLEOnce = 0
n_VBQ_MLEOnce = nInitial

while TotalTime_VBQ_MLEOnce < T_Max:
    X_n = np.linspace(start = 0, stop = 5, num = n_VBQ_MLEOnce)
    f_X_n = Integrand_f(X_n)

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (estimated once beforehand)
    VanillaBQ_Estimate_Time, _  = BQ_1D_Approximations(
        X_Input = X_n, f_X_Input = f_X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial,
        NEvaluations = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_nInitial)

    VanillaBQ_Estimate_FixedGrid_Time.append(VanillaBQ_Estimate_Time)  # BQ estimate

    TotalTime_VBQ_MLEOnce = time.perf_counter() - StartTime_VBQ_MLEOnce  # Update current time
    Times_VBQ_MLEOnce.append(TotalTime_VBQ_MLEOnce)  # Update total time history
    n_VBQ_MLEOnce += 1  # Update to n+1 observations
print("--------------------------------- [VBQ_MLEOnce] over time: Completed")


#--------------------| BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters |-----------------------
StartTime_VBQ_reHyperparameters = time.perf_counter()  # Start time counter
VanillaBQreHyperparameters_Estimate_FixedGrid_Time = []
# VanillaBQreHyperparameters_Variance_FixedGrid_Time = []

nInitial = 15  # The number of sample points we start with
X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
f_X_nInitial = Integrand_f(X_nInitial)
KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
InitialParameters_nInitial = 0 * MLEHyperparametersValues_nInitial + np.ones_like(MLEHyperparametersValues_nInitial)
MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']

Times_VBQ_reHyperparameters = []
TotalTime_VBQ_reHyperparameters = 0
n_VBQ_reHyperparameters = nInitial

while TotalTime_VBQ_reHyperparameters < T_Max:
    X_n = np.linspace(start = 0, stop = 5, num = n_VBQ_reHyperparameters)
    f_X_n = Integrand_f(X_n)

    # Re-estimate the hyperparameters
    MLEHyperparameters_n = MLEHyperparameters(
        initial_parameters = InitialParameters_nInitial, X_Input = X_n, y = f_X_n, PriorMeanVector = c_PriorMean,
        KernelFunction = MLEKernel_nInitial, ParameterNames = MLEHyperparametersNames_nInitial)

    MLEHyperparameters_n = dict(zip(MLEHyperparametersNames_nInitial, MLEHyperparameters_n))  # Add names

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    VanillaBQreHyperparameters_Estimate_Time, _ = BQ_1D_Approximations(
        X_Input = X_n, f_X_Input = f_X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial,
        NEvaluations = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n)
    VanillaBQreHyperparameters_Estimate_FixedGrid_Time.append(VanillaBQreHyperparameters_Estimate_Time)

    TotalTime_VBQ_reHyperparameters = time.perf_counter() - StartTime_VBQ_reHyperparameters # Update current time
    Times_VBQ_reHyperparameters.append(TotalTime_VBQ_reHyperparameters)  # Update total time history
    n_VBQ_reHyperparameters += 1  # Update to n+1 observations
print("--------------------------------- [VanillaBQ_reHyperparameter] over time: Completed")


#--------------------| Plot BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML |---------------------
StartTime_VBQ_reMLE = time.perf_counter()  # Start time counter
VanillaBQreKernel_Estimate_FixedGrid_Time = []
# VanillaBQreKernel_Variance_FixedGrid_Time = []

Times_VBQ_reMLE = []
TotalTime_VBQ_reMLE = 0
n_VBQ_reMLE = 1

while TotalTime_VBQ_reMLE < T_Max:
    X_n = np.linspace(start = 0, stop = 5, num = n_VBQ_reMLE)
    f_X_n = Integrand_f(X_n)

    # Re-estimate the kernel function & corresponding hyperparameters
    KernelSelectionML_n_VBQ_reMLE = KernelSelectionML(X_n, f_X_n, UniversalInitialParameter = 1)
    MLEKernel_n_VBQ_reMLE = KernelSelectionML_n_VBQ_reMLE['Kernel']  # Kernel with highest log(ML)
    # Kernel hyperparameters with highest log(ML)
    MLEHyperparameters_n_VBQ_reMLE = KernelSelectionML_n_VBQ_reMLE['Hyperparameters']

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (re-estimated each time)
    VanillaBQreKernel_Estimate_Time, _ = BQ_1D_Approximations(
        X_Input = X_n, f_X_Input = f_X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_n_VBQ_reMLE,
        NEvaluations = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n_VBQ_reMLE)
    VanillaBQreKernel_Estimate_FixedGrid_Time.append(VanillaBQreKernel_Estimate_Time)

    TotalTime_VBQ_reMLE = time.perf_counter() - StartTime_VBQ_reMLE  # Update current time
    Times_VBQ_reMLE.append(TotalTime_VBQ_reMLE)  # Update total time history
    n_VBQ_reMLE += 1  # Update to n+1 observations
print("--------------------------------- [VBQ_reMLE] over time: Completed")


#------------------------| Plot BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once |-------------------------
StartTime_BQUS_MLEOnce = time.perf_counter()  # Start time counter
BQUncertaintySampling_Estimate_FixedGrid_Time = []
# BQUncertaintySampling_Variance_FixedGrid_Time = []
# The following is similar to the next-to-last code above, but we do need to run it again since we record the time
nInitial = 15  # The number of sample points we start with
X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
f_X_nInitial = Integrand_f(X_nInitial)
KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']

Times_BQUS_MLEOnce = []
TotalTime_BQUS_MLEOnce = 0
n_BQUS_MLEOnce = nInitial

while TotalTime_BQUS_MLEOnce < T_Max:
    X_n = np.linspace(start = 0, stop = 5, num = n_BQUS_MLEOnce)
    f_X_n = Integrand_f(X_n)

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (estimated once beforehand)
    BQUncertaintySampling = BQ_1D_Approximations_UncertaintySampling(
        X_Input = X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial, NExtraPoints = 1,
        nGridSize = nGridSize, NIntegrationGrid = NIntegrationGrid, ComputeVariance = False,
        **MLEHyperparameters_nInitial)

    BQUncertaintySampling_Estimate_Time = BQUncertaintySampling['Estimate']  # BQ estimate
    BQUncertaintySampling_Estimate_FixedGrid_Time.append(BQUncertaintySampling_Estimate_Time)

    TotalTime_BQUS_MLEOnce = time.perf_counter() - StartTime_BQUS_MLEOnce  # Update current time
    Times_BQUS_MLEOnce.append(TotalTime_BQUS_MLEOnce)  # Update total time history
    n_BQUS_MLEOnce += 1  # Update to n+1 observations
print("--------------------------------- [BQUS_MLEOnce] over time: Completed")


#-----------------| BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Re-estimate Hyperparameters |----------------
StartTime_BQUS_reHyperparameters = time.perf_counter()  # Start time counter
BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid_Time = []
# BQUncertaintySampling_reKernel_Variance_FixedGrid_Time = []

nInitial = 15  # The number of sample points we start with
X_nInitial = np.linspace(start = 0, stop = 5, num = nInitial)
f_X_nInitial = Integrand_f(X_nInitial)
KernelSelectionML_nInitial = KernelSelectionML(X_nInitial, f_X_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
InitialParameters_nInitial = 0 * MLEHyperparametersValues_nInitial + np.ones_like(MLEHyperparametersValues_nInitial)
MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']

Times_BQUS_reHyperparameters = []
TotalTime_BQUS_reHyperparameters = 0
n_BQUS_reHyperparameters = nInitial

while TotalTime_BQUS_reHyperparameters < T_Max:
    X_n = np.linspace(start = 0, stop = 5, num = n_BQUS_reHyperparameters)
    f_X_n = Integrand_f(X_n)

    # Re-estimate the hyperparameters
    MLEHyperparameters_n_US = MLEHyperparameters(
        initial_parameters = InitialParameters_nInitial, X_Input = X_n, y = f_X_n, PriorMeanVector = c_PriorMean,
        KernelFunction = MLEKernel_nInitial, ParameterNames = MLEHyperparametersNames_nInitial)

    MLEHyperparameters_n_US = dict(zip(MLEHyperparametersNames_nInitial, MLEHyperparameters_n_US))  # Add names

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    BQUncertaintySampling_reHyperparameters = BQ_1D_Approximations_UncertaintySampling(
        X_Input = X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_nInitial, NExtraPoints = 1,
        nGridSize = nGridSize, NIntegrationGrid = NIntegrationGrid, ComputeVariance = False, **MLEHyperparameters_n_US)

    BQUncertaintySampling_reHyperparameters_Estimate_Time = BQUncertaintySampling_reHyperparameters['Estimate']
    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid_Time.append(
        BQUncertaintySampling_reHyperparameters_Estimate_Time)

    TotalTime_BQUS_reHyperparameters = time.perf_counter() - StartTime_BQUS_reHyperparameters   # Update current time
    Times_BQUS_reHyperparameters.append(TotalTime_BQUS_reHyperparameters)  # Update total time history
    n_BQUS_reHyperparameters += 1  # Update to n+1 observations
print("--------------------------------- [VanillaBQ_reHyperparameter] over time: Completed")


#----------------| Plot BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML |---------------
StartTime_BQUS_reMLE = time.perf_counter()  # Start time counter
BQUncertaintySampling_reKernel_Estimate_FixedGrid_Time = []
# BQUncertaintySampling_reKernel_Variance_FixedGrid_Time = []

Times_BQUS_reMLE = []
TotalTime_BQUS_reMLE = 0
n_BQUS_reMLE = nInitial

while TotalTime_BQUS_reMLE < T_Max:
    X_n = np.linspace(start = 0, stop = 5, num = n_BQUS_reMLE)
    f_X_n = Integrand_f(X_n)

    # Re-estimate the kernel function & corresponding hyperparameters
    KernelSelectionML_n_BQUS_reMLE = KernelSelectionML(X_n, f_X_n, UniversalInitialParameter = 1)
    MLEKernel_n_BQUS_reMLE = KernelSelectionML_n_BQUS_reMLE['Kernel']  # Kernel with highest log(ML)
    # Kernel hyperparameters with highest log(ML)
    MLEHyperparameters_n_BQUS_reMLE = KernelSelectionML_n_BQUS_reMLE['Hyperparameters']

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (re-estimated each time)
    BQUncertaintySampling_reKernel_Time = BQ_1D_Approximations_UncertaintySampling(
        X_Input = X_n, ConstantPriorGPMean = c_PriorMean, KernelFunction = MLEKernel_n_BQUS_reMLE, NExtraPoints = 1,
        nGridSize = nGridSize, NIntegrationGrid = NIntegrationGrid, ComputeVariance = False,
        **MLEHyperparameters_n_BQUS_reMLE)

    BQUncertaintySampling_reKernel_Estimate_Time = BQUncertaintySampling_reKernel_Time['Estimate']
    BQUncertaintySampling_reKernel_Estimate_FixedGrid_Time.append(BQUncertaintySampling_reKernel_Estimate_Time)

    TotalTime_BQUS_reMLE = time.perf_counter() - StartTime_BQUS_reMLE  # Update current time
    Times_BQUS_reMLE.append(TotalTime_BQUS_reMLE)  # Update total time history
    n_BQUS_reMLE += 1  # Update to n+1 observations
print("--------------------------------- [BQUS_reMLE] over time: Completed")


#-------------------------------------------- Plot Absolute Error over Time --------------------------------------------
plt.figure(figsize = (10, 6))

### Absolute Error: MC Estimate - N(5/2, 1) Proposal PDF
plt.loglog(Times_MC_SSN, np.abs(MCEstimate_ShiftedStandardNormal_Grid_Time - TrueIntegralValue),
           color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")

### Absolute Error: MC Estimate - Unif[0, 5] Proposal PDF
plt.loglog(Times_MC_Unif, np.abs(MCEstimate_Uniform_Grid_Time - TrueIntegralValue),
           color = 'skyblue', linewidth = 1.8,  label = r"MC Estimate - Unif$[0,5]$")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
plt.loglog(Times_VBQ_MLEOnce, np.abs(VanillaBQ_Estimate_FixedGrid_Time - TrueIntegralValue),
           color = 'forestgreen', linewidth = 1.8, alpha = 0.7, label = r"Vanilla BQ - MLE Once")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
plt.loglog(Times_VBQ_reHyperparameters, np.abs(
    VanillaBQreHyperparameters_Estimate_FixedGrid_Time - TrueIntegralValue),
           color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(Times_VBQ_reMLE, np.abs(VanillaBQreKernel_Estimate_FixedGrid_Time - TrueIntegralValue),
           color = 'red', linewidth = 1.8,  label = r"Vanilla BQ - re-MLE")

### Absolute Error: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
plt.loglog(Times_BQUS_MLEOnce, np.abs(BQUncertaintySampling_Estimate_FixedGrid_Time - TrueIntegralValue),
           color = 'deeppink', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - MLE Once")

### Absolute Error: BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Hyperparameters by ML
plt.loglog(Times_BQUS_reHyperparameters, np.abs(
    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid_Time - TrueIntegralValue),
           color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(Times_BQUS_reMLE, np.abs(BQUncertaintySampling_reKernel_Estimate_FixedGrid_Time - TrueIntegralValue),
           color = 'mediumslateblue', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim(right = T_Max)
plt.legend()
plt.title(r"$\log_{10}$(Absolute Error of Estimate over Time)")
plt.xlabel(r"$\log_{10}$(Time in Seconds)")
plt.ylabel(r"$\log_{10}(|I_{True} - \hat{I}|)$")
plt.tight_layout()
# plt.savefig('results/BQ_AbsoluteError_Time.png', dpi = 300)
plt.show()

### Number of Observations per Method
print(rf"n_MC_SSN:       {n_MC_SSN}")
print(rf"n_MC_Unif:      {n_MC_Unif}")
print(rf"n_VBQ_MLEOnce:  {n_VBQ_MLEOnce}")
print(rf"n_VBQ_reMLE:    {n_VBQ_reMLE}")
print(rf"n_BQUS_MLEOnce: {n_BQUS_MLEOnce}")
print(rf"n_BQUS_reMLE:   {n_BQUS_reMLE}")


#--------------------------------------- Plot Number of Observations n over Time ---------------------------------------
plt.figure(figsize = (10, 6))

### Sample Size over Time: MC Estimate - N(5/2, 1) Proposal PDF
n_MC_SSN_Grid = np.arange(start = 1, stop = len(Times_MC_SSN) + 1, step = 1)
plt.loglog(Times_MC_SSN, n_MC_SSN_Grid,
           color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")

### Sample Size over Time: MC Estimate - Unif[0, 5] Proposal PDF
n_MC_Unif_Grid = np.arange(start = 1, stop = len(Times_MC_Unif) + 1, step = 1)
plt.loglog(Times_MC_Unif, n_MC_Unif_Grid,
           color = 'skyblue', linewidth = 1.8,  label = r"MC Estimate - Unif$[0,5]$")

### Sample Size over Time: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
n_VBQ_MLEOnce_Grid = np.arange(start = 1, stop = len(Times_VBQ_MLEOnce) + 1, step = 1)
plt.loglog(Times_VBQ_MLEOnce, n_VBQ_MLEOnce_Grid,
           color = 'forestgreen', linewidth = 1.8, alpha = 0.7, label = r"Vanilla BQ - MLE Once")

### Sample Size over Time: Vanilla BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
n_VBQ_reHyperparameters_Grid = np.arange(start = 1, stop = len(Times_VBQ_reHyperparameters) + 1, step = 1)
plt.loglog(Times_VBQ_reHyperparameters, n_VBQ_reHyperparameters_Grid,
           color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Sample Size over Time: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
n_VBQ_reMLE_Grid = np.arange(start = 1, stop = len(Times_VBQ_reMLE) + 1, step = 1)
plt.loglog(Times_VBQ_reMLE, n_VBQ_reMLE_Grid,
           color = 'red', linewidth = 1.8,  label = r"Vanilla BQ - re-MLE")

### Sample Size over Time: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
n_BQUS_MLEOnce_Grid = np.arange(start = 1, stop = len(Times_BQUS_MLEOnce) + 1, step = 1)
plt.loglog(Times_BQUS_MLEOnce, n_BQUS_MLEOnce_Grid,
           color = 'deeppink', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - MLE Once")

### Sample Size over Time: BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Hyperparameters by ML
n_BQUS_reHyperparameters_Grid = np.arange(start = 1, stop = len(Times_BQUS_reHyperparameters) + 1, step = 1)
plt.loglog(Times_BQUS_reHyperparameters, n_BQUS_reHyperparameters_Grid,
           color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Sample Size over Time: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
n_BQUS_reMLE_Grid = np.arange(start = 1, stop = len(Times_BQUS_reMLE) + 1, step = 1)
plt.loglog(Times_BQUS_reMLE, n_BQUS_reMLE_Grid,
           color = 'mediumslateblue', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE")


plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim(right = T_Max)
plt.legend()
plt.title(r"$\log_{10}$(Time versus Number of Observations $n$)")
plt.xlabel(r"$\log_{10}$(Time in Seconds)")
plt.ylabel(r"$\log_{10}$(Number of Observations $n$)")
#plt.tight_layout()
# plt.savefig('results/BQ_NumberofObservations_Time.png', dpi = 300)
plt.show()