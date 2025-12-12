########################################################################################################################
###################################### BAYESIAN QUADRATURE - MARGINAL PDF EXAMPLE ######################################
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize  # Optimiser
import time
import pandas as pd
import statsmodels.api as sm

from BQ_KernelChoice_Importing import KernelSelectionML
from BQ_KernelChoice_Importing import MLEHyperparameters
# Note: BQ_KernelChoice_Importing is the same as BQ_KernelChoice but with every output and plot deleted, because this
#       statement runs that entire document again to find the function.








#=======================================================================================================================
#----------------------------------------- Define Setting & Relevant Functions -----------------------------------------
#=======================================================================================================================
d = 2  # Dimension of beta

### Prior Distribution
sigma = 3
muVectorBetaPrior = np.zeros(d)
BetaPriorMean = muVectorBetaPrior
CovarianceMatrixBetaPrior = (sigma**2) * np.eye(d)

### Gaussian Mixture Model
K = 3  # 3 modes
MixtureMeans = np.array([
    [-3, -3],
    [-3,  3],
    [3,  3],
], dtype=float)  # dim=Kxd

MixtureStd = np.array([0.5, 0.5, 0.5], dtype=float)  # Standard deviation per Gaussian
# MixtureStd = np.array([1, 1, 1], dtype=float)  # Standard deviation per Gaussian
MixtureVar = MixtureStd**2  # sigma_k^2

MixtureWeights = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
MixtureWeights = MixtureWeights / MixtureWeights.sum()   # Standardize to 1
log_MixtureWeights = np.log(MixtureWeights)  # log(weights)



def LogPrior(BetaVector):
    """log p(beta), for  prior beta ~ N(muVectorBetaPrior, BetaPriorCovarianceMatrix)."""
    BetaMatrix = np.atleast_2d(BetaVector)  # dim=(samplesize)xd
    NormalizationTerm = (-d / 2) * np.log(2 * np.pi)
    DeterminantTerm = (-d / 2) * np.log( sigma**2 )
    ModelTerm = (-1 / (2 * sigma ** 2)) * np.sum((BetaMatrix - muVectorBetaPrior)**2, axis = 1)                       # shape (n_draws,)
    return NormalizationTerm + DeterminantTerm + ModelTerm


def LogLikelihood_GMM(BetaVector):
    """log(L(beta)) = log of the GMM in beta"""
    BetaMatrix = np.atleast_2d(BetaVector)  # (n_draws, d)

    #------------------------------ Log Gaussian ------------------------------
    DifferenceVector = BetaMatrix[:, None, :] - MixtureMeans[None, :, :]  # beta - mu_k
    Norm_DifferenceVector = np.sum(DifferenceVector ** 2, axis=2)  # (beta - mu_k)^T (beta - mu_k)

    LogGaussian_NormalizationTerm = (-d / 2) * np.log(2 * np.pi)
    LogGaussian_DeterminantTerm = (-d / 2) * np.log(MixtureVar)
    LogGaussian_ModelTerm = (-0.5) * Norm_DifferenceVector / MixtureVar[None, :]

    LogGaussian = LogGaussian_NormalizationTerm + LogGaussian_DeterminantTerm[None, :] + LogGaussian_ModelTerm

    #---------------------- Transformation & Log-Scaling -----------------------
    LogLikelihoodBeta = log_MixtureWeights[None, :] + LogGaussian  # log(w_k) + log(N())

    # Log-sum-exp over the K components (axis=1)
    MBeta = np.max(LogLikelihoodBeta, axis = 1, keepdims = True)
    return MBeta.squeeze(1) + np.log(
        np.sum(np.exp(LogLikelihoodBeta - MBeta), axis = 1)
    )


def ScaledLikelihood_GMM(BetaVector, L0):
    """ exp{log(L(beta)) - L0} """
    return np.exp(LogLikelihood_GMM(BetaVector) - L0)


def LogIntegrand_MarginalPDF(BetaVector):
    """ log(f(beta)) = log(L(beta)) + log(p(beta)) """
    return LogLikelihood_GMM(BetaVector) + LogPrior(BetaVector)


def ScaledIntegrand_MarginalPDF(BetaVector, L0):
    """exp{log(f(beta)) - L_0}"""
    return np.exp(LogIntegrand_MarginalPDF(BetaVector) - L0)






#=======================================================================================================================
#------------------------------------------------ 3-Dimensional Plots --------------------------------------------------
#=======================================================================================================================

#---------------------------------------------- Log-Scaled Prior p(beta) -----------------------------------------------
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection = '3d')
Beta0Grid = np.linspace(start = -5, stop = 5, num = 1000, endpoint = True)
Beta1Grid = np.linspace(start = -5, stop = 5, num = 1000, endpoint = True)
Beta0Grid, Beta1Grid = np.meshgrid(Beta0Grid, Beta1Grid)
BetaVector_Grid = np.column_stack((Beta0Grid.ravel(), Beta1Grid.ravel()))

L0 = np.max(LogPrior(BetaVector = BetaVector_Grid))
LogPrior_SurfacePlot = np.exp(LogPrior(BetaVector = BetaVector_Grid) - L0).reshape(Beta0Grid.shape)
surface = ax.plot_surface(Beta0Grid, Beta1Grid, LogPrior_SurfacePlot, cmap = 'cool')

fig.colorbar(surface, shrink = 0.5, pad = 0.1)
ax.set_title(r"Scaled Prior $p(\mathbf{\beta})$")
ax.set_xlabel(r"$\beta_0$")
ax.set_ylabel(r"$\beta_1$")
ax.set_zlabel(r"$\exp \left( \log(p(\mathbf{\beta}) - L_0 \right)$")
ax.set_box_aspect([1, 1, 1])
ax.margins(0)
fig.tight_layout(pad = 0)
fig.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
# plt.savefig("Images/MLExample_GMM_Prior_3DPlot.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()



#------------------------------------------- Log-Scaled Likelihood p(beta) ---------------------------------------------
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection = '3d')
Beta0Grid = np.linspace(start = -5, stop = 5, num = 1000, endpoint = True)
Beta1Grid = np.linspace(start = -5, stop = 5, num = 1000, endpoint = True)
Beta0Grid, Beta1Grid = np.meshgrid(Beta0Grid, Beta1Grid)
BetaVector_Grid = np.column_stack((Beta0Grid.ravel(), Beta1Grid.ravel()))

L0 = np.max(LogLikelihood_GMM(BetaVector = BetaVector_Grid))
Likelihood_GMM_SurfacePlot = np.exp(LogLikelihood_GMM(BetaVector = BetaVector_Grid) - L0).reshape(Beta0Grid.shape)
surface = ax.plot_surface(Beta0Grid, Beta1Grid, Likelihood_GMM_SurfacePlot, cmap = 'cool')

fig.colorbar(surface, shrink = 0.5, pad = 0.1)
ax.set_title(r"Scaled Likelihood $L(\mathbf{\beta})$")
ax.set_xlabel(r"$\beta_0$")
ax.set_ylabel(r"$\beta_1$")
ax.set_zlabel(r"$\exp \left( \log(L(\mathbf{\beta}) - L_0 \right)$")
ax.set_box_aspect([1, 1, 1])
ax.margins(0)
fig.tight_layout(pad = 0)
fig.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
# plt.savefig("Images/MLExample_GMM_Likelihood_3DPlot.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()


#------------------------------------------ Log-Scaled Integrand f(beta) -----------------------------------------------
fig = plt.figure(figsize = (6, 6))
ax = fig.add_subplot(111, projection = '3d')
Beta0Grid = np.linspace(start = -5, stop = 5, num = 1000, endpoint = True)
Beta1Grid = np.linspace(start = -5, stop = 5, num = 1000, endpoint = True)
Beta0Grid, Beta1Grid = np.meshgrid(Beta0Grid, Beta1Grid)
BetaVector_Grid = np.column_stack((Beta0Grid.ravel(), Beta1Grid.ravel()))

L0 = np.max(LogIntegrand_MarginalPDF(BetaVector = BetaVector_Grid))
LogIntegrand_SurfacePlot = np.exp(LogIntegrand_MarginalPDF(BetaVector = BetaVector_Grid) - L0).reshape(Beta0Grid.shape)
surface = ax.plot_surface(Beta0Grid, Beta1Grid, LogIntegrand_SurfacePlot, cmap = 'cool')

fig.colorbar(surface, shrink = 0.5, pad = 0.1)
ax.set_title(r"Scaled Integrand $f(\mathbf{\beta}) = L(\mathbf{\beta}) \cdot p(\mathbf{\beta})$")
ax.set_xlabel(r"$\beta_0$")
ax.set_ylabel(r"$\beta_1$")
ax.set_zlabel(r"$\exp \left( \log(f(\mathbf{\beta}) - L_0 \right)$")
ax.set_box_aspect([1, 1, 1])
ax.margins(0)
fig.tight_layout(pad = 0)
fig.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
# plt.savefig("Images/MLExample_GMM_Integrand_3DPlot.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()















#=======================================================================================================================
#------------------------------------- Case-Specific Bayesian Quadrature Function --------------------------------------
#=======================================================================================================================
rng = np.random.default_rng(21)  # Reproducibility of randomness

def BQ_MarginalPDF(Beta_Input,  # dim=nxk  (A grid with n beta nodes)
                   f_Beta_Input,  # dim=nx1
                   KernelFunction,
                   BetaPriorMean,  # dim=kx1
                   BetaPriorCovarianceMatrix,  # dim=kxk
                   NKernelMeanEmbeddingApprox=100000,  # Number of samples for the approximation of KME by MC
                   ComputeVariance=False,  # Boolean whether to compute posterior covariance,
                   **kernelhyperparameters):
    """"
    Since we are integrating over probabilities, we assume a (constant) zero mean for the GP prior on the integrand.
    """
    n, k = Beta_Input.shape  # n refers to the number of beta nodes (not the number of data observations)

    #--------------------------------- Create Kernel Matrix K_BB --------------------------------
    K_BB = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            K_BB[i, j] = KernelFunction(Beta_Input[i], Beta_Input[j], **kernelhyperparameters)

    Jitter = 1e-5
    for _ in range(4):
        try:
            L = np.linalg.cholesky(K_BB + Jitter * np.eye(n))  # Cholesky decomposition K_BB = L L^T & Add jitter
            break
        except np.linalg.LinAlgError:
            Jitter *= 10
    else:
        return np.nan, None
    # --------------------------------- Create Kernel Matrix K_BB ---------------------------------z
    #--------------------- Approximate the Kernel Mean Embedding (KME) by MC ---------------------
    NormalPriorBeta_MCSamples = rng.multivariate_normal(mean = BetaPriorMean, cov = BetaPriorCovarianceMatrix,
                                                           size = NKernelMeanEmbeddingApprox, method = 'cholesky')
    KernelMeanEmbedding_MC = []  # dim=kx1, MC estimate for every kappa(beta_j)
    for j in range(n):
        KME_NormalPriorBeta_MCSamples_j = np.array(  # Evaluate k(beta^{(i)}, beta_j) for sampled beta^{(i)}
            [KernelFunction(Beta_i, Beta_Input[j], **kernelhyperparameters) for Beta_i in NormalPriorBeta_MCSamples]
        )
        KernelMeanEmbedding_MC.append(np.mean(KME_NormalPriorBeta_MCSamples_j))

    KernelMeanEmbedding_MC = np.array(KernelMeanEmbedding_MC, dtype = float)
    #---------------------------- Compute Posterior Mean = Estimate -----------------------------
    w = np.linalg.solve(L.T, np.linalg.solve(L, f_Beta_Input))  # K_{XX}^{-1} f(beta) = (L^T)^{-1} (L^{-1} f(beta))
    BQ_Estimate = KernelMeanEmbedding_MC @ w
    if ComputeVariance == False:
        return BQ_Estimate, None

    #------------------------------- Compute Posterior Variance ----------------------------------
    # (skip for now)

    return BQ_Estimate, print("The code for the posterior variance is still in progress...")



#------------------------------------------------ Uncertainty Sampling -------------------------------------------------
def UncertaintySampling_Beta(Beta_Candidates, Beta_Original, KernelFunction, **hyperparameters):
    """
    Return point with the highest posterior function()~GP variance: k(x,x) - k_{xX}K^{-1}k_{Xx}.
    """
    # Note: Since Beta_Candidates are drawn randomly from a normal distribution, we do not worry whether or not they
    #       are already in Beta_Original

    if Beta_Candidates.size == 0:
        return None  # If there are no new points to choose from.

    n = Beta_Original.shape[0]  # Our 'current' number of observations

    #---------------------------------- Create Kernel Matrix K_BB --------------------------------
    K_BB = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            k_BetaiBetaj = KernelFunction(Beta_Original[i],Beta_Original[j], **hyperparameters)
            K_BB[i, j] = k_BetaiBetaj
            K_BB[j, i] = k_BetaiBetaj  # Use the symmetry property of covariance matrices to save computing power/time

    Jitter = 1e-5
    for _ in range(4):
        try:
            L = np.linalg.cholesky(K_BB + Jitter * np.eye(n))  # Cholesky decomposition K_BB = L L^T & Add jitter
            break
        except np.linalg.LinAlgError:
            Jitter *= 10
    else:
        return None

    #------------------- Compute Posterior GP Variance for each Candidate Beta -------------------
    # Compute the k(beta, beta) elements for beta in Beta_Candidates (without computing the entire matrix because that
    #   can become very computationally expensive)
    # k_BetaBeta_Posterior = np.empty_like(Beta_Candidates, dtype = float)
    k_BetaBeta_Posterior = np.empty(Beta_Candidates.shape[0], dtype = float)
    for i in range( Beta_Candidates.shape[0] ):
        Beta = Beta_Candidates[i]
        k_BetaBeta = KernelFunction(Beta, Beta, **hyperparameters)  # k(beta, beta)

        # k_{beta B} = [k(beta, beta_1), ..., k(beta, beta_n)], where B = [beta_1 ... beta_n]
        k_BetaB = np.array(
            [KernelFunction(Beta, Beta_Original[j], **hyperparameters) for j in range(n)], dtype = float
        )

        w = np.linalg.solve(L, k_BetaB)  # L^{-1} k_{beta B}
        k_BetaBeta_Posterior[i] = k_BetaBeta - w @ w  # k(beta beta) - k_{beta B} K_{BB}^{-1} k_{beta B}^T

    #---------------------------------- Select the "Best" Point ----------------------------------
    # IndexMostUncertainPoint = np.argmax(k_BetaBeta_Posterior)  # np.argmax() returns the index of the maximiser
    IndexMostUncertainPoint = np.argsort(k_BetaBeta_Posterior)[::-1]  # Sort indices in decreasing posterior variance
    for index in IndexMostUncertainPoint[:50]:  # Inspect the 50 best candidate points
        Beta_Proposed = Beta_Candidates[index]
        Distances = np.linalg.norm(Beta_Original - Beta_Proposed, axis = 1)  # Check the distances at all points
        MinDist = np.min(Distances)

        if MinDist > 1e-4:  # If point is not too close to existing points, keep it. Else, delete.
            return np.asarray(Beta_Proposed, dtype=float).ravel()

    #return np.asarray(Beta_Candidates[IndexMostUncertainPoint], dtype = float).ravel()
    print("All top candidates were too close. Stop searching.")
    return None



#------------------------------------- Bayesian Quadrature + Uncertainty Sampling --------------------------------------
def BQ_MarginalPDF_UncertaintySampling(Beta_Input,  # dim=nxk  (A grid with n beta nodes)
                                       f_Beta_Input,  # dim=nx1
                                       L0,
                                       KernelFunction,
                                       BetaPriorMean,  # dim=kx1
                                       BetaPriorCovarianceMatrix,  # dim=kxk
                                       NExtraPoints,  # Number of extra beta (nodes) to add
                                       NGridSize,  # Number of candidate beta's to consider per iteration
                                       NKernelMeanEmbeddingApprox=1000000,  # Number of samples for approx of KME by MC
                                       ComputeVariance=False,  # Boolean whether to compute posterior covariance,
                                       **kernelhyperparameters):

    #--------------- Update Beta-grid with NExtraPoints Extra Points - Uncertainty Sampling --------------
    Beta_NewUS_History = [Beta_Input.copy()]  # Keep track of every grid when a points is added to that grid
    f_Beta_NewUS_History = [f_Beta_Input.copy()]

    for i in range(NExtraPoints):
        Beta_Candidates = rng.multivariate_normal(mean = BetaPriorMean, cov = BetaPriorCovarianceMatrix,
                                                  size = NGridSize, method = 'cholesky')

        Beta_New = UncertaintySampling_Beta(Beta_Candidates = Beta_Candidates, Beta_Original = Beta_NewUS_History[i],
                                            KernelFunction = KernelFunction, **kernelhyperparameters)

        if Beta_New is None:
            break  # Break when there are no more new points to choose from

        Beta_New = np.asarray(Beta_New, dtype = float).ravel()  # dim=1xk
        B_Next = np.vstack([Beta_NewUS_History[i], Beta_New[None, : ]])  # dim=n_{i+1}xk
        Beta_NewUS_History.append(B_Next)  # Add new points to track record history

        # f_Beta_New = LogIntegrand_MarginalPDF(BetaVector = Beta_New)   #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        f_Beta_New = ScaledLikelihood_GMM(BetaVector=Beta_New, L0=L0)
        f_Beta_Next = np.concatenate( [f_Beta_NewUS_History[i], np.array([f_Beta_New]).ravel()] )
        # Note: Since integrand evaluations are scalars, we use np.concatenate instead of np.vstack
        f_Beta_NewUS_History.append(f_Beta_Next)

    #-------------------- Compute BQ Estimate and Uncertainty - Uncertainty Sampling ---------------------
    Beta_NewUS = Beta_NewUS_History[-1]  # The last X-grid from history track record, dim=(n+nExtraPoints)xk
    f_Beta_NewUS = f_Beta_NewUS_History[-1]

    # ------------------------------------------ BQ with New Grid -----------------------------------------
    PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ = BQ_MarginalPDF(
        Beta_Input = Beta_NewUS,
        f_Beta_Input = f_Beta_NewUS,
        KernelFunction = KernelFunction,
        BetaPriorMean = BetaPriorMean,
        BetaPriorCovarianceMatrix = BetaPriorCovarianceMatrix,
        NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
        ComputeVariance = ComputeVariance,
        **kernelhyperparameters
    )

    return{"Estimate": PosteriorMeanIntegral_BQ, "Variance": PosteriorCovarianceIntegral_BQ,
           "X_Grid": Beta_NewUS, "f_X_Grid": f_Beta_NewUS}



#--------------------------------------------------- BQ Grid Design ----------------------------------------------------
def BQ_GridDesign(n, Fraction=0.3):
    """ Grid design of size n for BQ, with Fraction of points near modes """
    nModes = int(round(Fraction * n))  # Number of points to be placed at the nodes
    nPrior = n - nModes  # Number of (remaining) points to be drawn from prior

    Beta_prior = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                         size = nPrior, method = "cholesky")

    if nModes > 0:  # Sanity check
        reps = int(np.ceil(nModes / K))  # Ceiling of points per mode
        BetaModes = np.repeat(MixtureMeans, reps, axis=0)[:nModes]  # Repeat the points at/near the modes

        JitterGrid = rng.normal(loc = 0, scale = 1e-5, size = BetaModes.shape)
        BetaModes = BetaModes + JitterGrid  # Add jitter
        BetaGrid = np.vstack([Beta_prior, BetaModes])
    else:
        BetaGrid = Beta_prior

    return BetaGrid




#=======================================================================================================================
#---------------------------------------- Approximation by Different Techniques ----------------------------------------
#=======================================================================================================================
nMaxPower = 2  # Some nice values: 1.3, 1.7, 2, 3, 4
# Grid of evaluation points (independent of the number of data observations d)
nSample_Grid = np.unique(np.logspace(start = 0, stop = nMaxPower, num = 40, base = 10, dtype = int))
# nSample_Grid = np.unique(np.linspace(start = 1, stop = 10, num = 10, dtype = int))

NKernelMeanEmbeddingApprox = 1000000  # The MC sample size used to approximate the kernel mean embedding integral
nGridSize = 2100  # The number of possible points to consider for uncertainty sampling

#------------------------------------------------ Exact Integral Value -------------------------------------------------
def TrueIntegralValue_MarginalPDF_GMM():
    """ True/Analytical Integral Value: I = sum_{k=1}^K w_k N(mu_k; mu, (sigma^2 + sigma_k^2) I) """
    Approximation = 0.0
    for k in range(K):
        sigma2_k   = MixtureVar[k]  # sigma_k^2
        sigma2_sum = sigma**2 + sigma2_k  # Variance scalar

        NormalizationTerm = (-d / 2.0) * np.log(2.0 * np.pi)
        DeterminantTerm   = (-d / 2.0) * np.log(sigma2_sum)
        ModelTerm = (-1.0 / (2.0 * sigma2_sum)) * np.sum((MixtureMeans[k] - muVectorBetaPrior)**2)
        LogGaussianPDFTerm = NormalizationTerm + DeterminantTerm + ModelTerm  # log( N_d(mu_k; mu, sigma2_sum I_d) )

        Approximation += MixtureWeights[k] * np.exp(LogGaussianPDFTerm)  # term in the sum: w_k N(...)
    return Approximation

TrueIntegralValue_MarginalPDF = TrueIntegralValue_MarginalPDF_GMM()
print(rf"True Integral Value/Marginal Likelihood: {TrueIntegralValue_MarginalPDF}")


#----------------------------------------------| Monte Carlo Integration |----------------------------------------------
R = 100  # Number of repetitions of MC runs
MC_Values = np.empty((R, len(nSample_Grid)))  # dim=Rxlen(nSample_Grid)
for r in range(R):
    for i in range(len(nSample_Grid)):
        NormalPriorBeta_Samples = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                                          size=nSample_Grid[i], method='cholesky')
        # L0_MC = np.max(LogIntegrand_MarginalPDF( BetaVector = NormalPriorBeta_Samples))  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Transformed_NormalPriorBeta_Samples = ScaledIntegrand_MarginalPDF(BetaVector = NormalPriorBeta_Samples,
        #                                                                   L0 = L0_MC)
        L0_MC = np.max(LogLikelihood_GMM(BetaVector = NormalPriorBeta_Samples))
        Transformed_NormalPriorBeta_Samples = ScaledLikelihood_GMM(BetaVector = NormalPriorBeta_Samples, L0 = L0_MC)
        MC_Values[r, i] =  np.exp(L0_MC) * np.mean(Transformed_NormalPriorBeta_Samples)

MC_Estimate = np.mean(MC_Values, axis = 0)   # Compute the MC estimate per sample size
MC_Variance = np.var(MC_Values, axis = 0, ddof = 1)   # Compute the MC variance per sample size

print("--------------------------------- [MC] over n: Completed")


#-------------------------------| VBQ Estimate - Fixed Grid & MLE Hyperparameters Once |--------------------------------
VanillaBQ_Estimate_FixedGrid = []
nBetaInitial = 25  # The number of sample points we start with
# First, create a grid of beta nodes by drawing them from the beta prior distribution
Beta_nInitial = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                        size = nBetaInitial, method = 'cholesky')
# Second, evaluate the integrand f(beta) on the beta nodes
# L0_nInitial = np.max(LogIntegrand_MarginalPDF(BetaVector = Beta_nInitial))  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# f_Beta_nInitial = ScaledIntegrand_MarginalPDF(BetaVector = Beta_nInitial, L0 = L0_nInitial)
L0_nInitial = np.max(LogLikelihood_GMM(BetaVector = Beta_nInitial))
f_Beta_nInitial = ScaledLikelihood_GMM(BetaVector = Beta_nInitial, L0 = L0_nInitial)
# Third, estimate the kernel & hyperparameters for this sample by maximum likelihood
KernelSelectionML_nInitial = KernelSelectionML(Beta_nInitial, f_Beta_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']  # MLE hyperparameters
# Fourth, delete the samples of size up to nBetaInitial
nSample_Grid_StartnInitial = nSample_Grid[nSample_Grid >= nBetaInitial]

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                     size = nSample_Grid_StartnInitial[i], method = 'cholesky')
    # Beta_n = BQ_GridDesign(n = nSample_Grid_StartnInitial[i], Fraction = 0.3)
    L0_VBQ_MLEOnce = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_VBQ_MLEOnce)

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (estimated once beforehand)
    VanillaBQ_Estimate, _ = BQ_MarginalPDF(Beta_Input = Beta_n,  # dim=nxk
                                           f_Beta_Input = f_Beta_n,  # dim=nx1
                                           KernelFunction = MLEKernel_nInitial,
                                           BetaPriorMean = muVectorBetaPrior,  # dim=kx1
                                           BetaPriorCovarianceMatrix = CovarianceMatrixBetaPrior,  # dim=kxk
                                           NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
                                           ComputeVariance = False,
                                           **MLEHyperparameters_nInitial)

    VanillaBQ_Estimate_FixedGrid.append( np.exp(L0_VBQ_MLEOnce) * VanillaBQ_Estimate )
#print(VanillaBQ_Estimate_FixedGrid)
print("--------------------------------- [VanillaBQ_MLEOnce] over n: Completed")






#--------------------| VBQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters |----------------------
VanillaBQreHyperparameters_Estimate_FixedGrid = []
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# However, add the following: extracting the hyperparameter values and corresponding names
MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
InitialParameters_nInitial = 0 * MLEHyperparametersValues_nInitial + np.ones_like(MLEHyperparametersValues_nInitial)
MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']
MLEHyperparameters_VBQ_n = MLEHyperparameters_nInitial

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                     size = nSample_Grid_StartnInitial[i], method = 'cholesky')
    # Beta_n = BQ_GridDesign(n = nSample_Grid_StartnInitial[i], Fraction = 0.3)
    L0_VBQ_MLEHyperparameter = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector = Beta_n, L0 = L0_VBQ_MLEHyperparameter)

    # Re-estimate the hyperparameters
    try:
        NewMLEHyperparameters_VBQ_n = MLEHyperparameters(
            initial_parameters = InitialParameters_nInitial, X_Input = Beta_n, y = f_Beta_n,
            PriorMeanVector = np.zeros_like(f_Beta_n),  # Zero prior mean
            KernelFunction = MLEKernel_nInitial,  # Kernel estimated from grid of size nBetaInitial
            ParameterNames = MLEHyperparametersNames_nInitial)
        MLEHyperparameters_VBQ_n = dict(zip(MLEHyperparametersNames_nInitial, NewMLEHyperparameters_VBQ_n))  # Add names
        print(f"MLEHyperparameters_VBQ_n, iteration={i}, n={nSample_Grid_StartnInitial[i]}: {MLEHyperparameters_VBQ_n}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Optimisation failed at n = {Beta_n.shape[0]}. Keeping the old hyperparameters. Error: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    VanillaBQreHyperparameters_Estimate, _ = BQ_MarginalPDF(Beta_Input = Beta_n,  # dim=nxk
                                                            f_Beta_Input = f_Beta_n,  # dim=nx1
                                                            KernelFunction = MLEKernel_nInitial,
                                                            BetaPriorMean = muVectorBetaPrior,  # dim=kx1
                                                            BetaPriorCovarianceMatrix = CovarianceMatrixBetaPrior,
                                                            NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
                                                            ComputeVariance = False,
                                                            **MLEHyperparameters_VBQ_n)  # Update hyperparameters

    VanillaBQreHyperparameters_Estimate_FixedGrid.append( np.exp(L0_VBQ_MLEHyperparameter) *
                                                          VanillaBQreHyperparameters_Estimate )
print("--------------------------------- [VanillaBQ_reHyperparameters] over n: Completed")



#----------------------| VBQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML |-----------------------
VanillaBQreKernel_Estimate_FixedGrid = []
MLEKernel_n = MLEKernel_nInitial
MLEHyperparameters_n = MLEHyperparameters_nInitial

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                     size = nSample_Grid_StartnInitial[i], method = 'cholesky')
    # Beta_n = BQ_GridDesign(n = nSample_Grid_StartnInitial[i], Fraction=0.3)
    L0_VBQ_reMLE = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_VBQ_reMLE)

    # Re-estimate the kernel function & corresponding hyperparameters
    try:
        KernelSelectionML_n = KernelSelectionML(Beta_n, f_Beta_n, UniversalInitialParameter = 1)
        MLEKernel_n = KernelSelectionML_n['Kernel']  # Kernel with highest log(ML)
        MLEHyperparameters_n = KernelSelectionML_n['Hyperparameters']  # Kernel hyperparameters with highest log(ML)
        print(f"reMLE_VBQ_n, iteration={i}, n={nSample_Grid_StartnInitial[i]}: {MLEHyperparameters_n}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Kernel Selection failed at n = {Beta_n.shape[0]}. Keeping the previous kernel. Error: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (re-estimated each time)
    VanillaBQreKernel_Estimate, _ = BQ_MarginalPDF(Beta_Input = Beta_n,  # dim=nxk
                                                   f_Beta_Input = f_Beta_n,  # dim=nx1
                                                   KernelFunction = MLEKernel_n,
                                                   BetaPriorMean = muVectorBetaPrior,  # dim=kx1
                                                   BetaPriorCovarianceMatrix = CovarianceMatrixBetaPrior,  # dim=kxk
                                                   NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
                                                   ComputeVariance = False,
                                                   **MLEHyperparameters_n)

    VanillaBQreKernel_Estimate_FixedGrid.append( np.exp(L0_VBQ_reMLE) * VanillaBQreKernel_Estimate )
print("--------------------------------- [VanillaBQ_reKernel] over n: Completed")



#--------------------------| BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once |----------------------------
BQUncertaintySampling_Estimate_FixedGrid = []
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# However, add the following: differences in number of evaluations
nGrid_Differences = np.insert(nSample_Grid_StartnInitial, 0, nBetaInitial)
# Note: nInitial is added at the beginning for one extra time. Hence, when we take the (ascending) differences between
#   the elements, the first entry is zero.
for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                     size = nSample_Grid_StartnInitial[i], method = 'cholesky')
    # Beta_n = BQ_GridDesign(n = nSample_Grid_StartnInitial[i], Fraction = 0.3)
    L0_BQUS = np.max(LogLikelihood_GMM(BetaVector = Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector = Beta_n, L0 = L0_BQUS)

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (estimated once beforehand)
    BQUncertaintySampling = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input = Beta_n,  # dim=nxk
        f_Beta_Input = f_Beta_n,  # dim=nx1
        L0 = L0_BQUS,
        KernelFunction = MLEKernel_nInitial,
        BetaPriorMean = muVectorBetaPrior,  # dim=kx1
        BetaPriorCovarianceMatrix = CovarianceMatrixBetaPrior,  # dim=kxk
        NExtraPoints = nGrid_Differences[i + 1] - nGrid_Differences[i],  # Number of extra beta (nodes) to add
        NGridSize = nGridSize,
        NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
        ComputeVariance = False,
        **MLEHyperparameters_nInitial)

    BQUncertaintySampling_Estimate = BQUncertaintySampling['Estimate']
    BQUncertaintySampling_Estimate_FixedGrid.append( np.exp(L0_BQUS) * BQUncertaintySampling_Estimate)
print("--------------------------------- [BQUS_MLEOnce] over n: Completed")



#-----------------| BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Re-estimate Hyperparameters |----------------
BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid = []
MLEHyperparameters_BQUS_n = MLEHyperparameters_nInitial
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# Similarly, the nGrid_Differences in "BQUS + MLE Once" are used here again (but omitted)

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                     size = nSample_Grid_StartnInitial[i], method = 'cholesky')
    # Beta_n = BQ_GridDesign(n = nSample_Grid_StartnInitial[i], Fraction = 0.3)
    L0_BQUS_reHyperparameter = np.max(LogLikelihood_GMM(BetaVector = Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector = Beta_n, L0 = L0_BQUS_reHyperparameter)

    # Re-estimate the hyperparameters
    try:
        NewMLEHyperparameters_BQUS_n = MLEHyperparameters(
            initial_parameters = InitialParameters_nInitial, X_Input = Beta_n, y = f_Beta_n,
            PriorMeanVector = np.zeros_like(f_Beta_n),  # Zero prior mean
            KernelFunction = MLEKernel_nInitial,  # Kernel estimated from grid of size nBetaInitial
            ParameterNames = MLEHyperparametersNames_nInitial)
        MLEHyperparameters_BQUS_n = dict(zip(MLEHyperparametersNames_nInitial, NewMLEHyperparameters_BQUS_n)) #Add names
        print(f"MLEHyperparameters_BQUS_n, iteration={i}, "
              f"n={nSample_Grid_StartnInitial[i]}: {MLEHyperparameters_BQUS_n}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Optimisation failed at n={Beta_n.shape[0]}. Keeping the old hyperparameters. Error: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    BQUncertaintySampling_reHyperparameters = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input = Beta_n,  # dim=nxk
        f_Beta_Input = f_Beta_n,  # dim=nx1
        L0 = L0_BQUS_reHyperparameter,
        KernelFunction = MLEKernel_nInitial,
        BetaPriorMean = muVectorBetaPrior,  # dim=kx1
        BetaPriorCovarianceMatrix = CovarianceMatrixBetaPrior,  # dim=kxk
        NExtraPoints = nGrid_Differences[i + 1] - nGrid_Differences[i],  # Number of extra beta (nodes) to add
        NGridSize = nGridSize,
        NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
        ComputeVariance = False,
        **MLEHyperparameters_BQUS_n)

    BQUncertaintySampling_reHyperparameters_Estimate = BQUncertaintySampling_reHyperparameters['Estimate']
    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid.append( np.exp(L0_BQUS_reHyperparameter) *
                                                                       BQUncertaintySampling_reHyperparameters_Estimate)
print("--------------------------------- [BQUS_reHyperparameter] over n: Completed")



#-----------------| BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML |-------------------
BQUncertaintySampling_reKernel_Estimate_FixedGrid = []
MLEKernel_n = MLEKernel_nInitial
MLEHyperparameters_n_US = MLEHyperparameters_nInitial
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# Similarly, the nGrid_Differences in "BQUS + MLE Once" are used here again (but omitted)

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                     size = nSample_Grid_StartnInitial[i], method = 'cholesky')
    # Beta_n = BQ_GridDesign(n = nSample_Grid_StartnInitial[i], Fraction = 0.3)
    L0_BQUS_reMLE = np.max(LogLikelihood_GMM(BetaVector = Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector = Beta_n, L0 = L0_BQUS_reMLE)

    # Re-estimate the kernel function & corresponding hyperparameters
    try:
        KernelSelectionML_n = KernelSelectionML(Beta_n, f_Beta_n, UniversalInitialParameter = 1)
        MLEKernel_n = KernelSelectionML_n['Kernel']  # Kernel with highest log(ML)
        MLEHyperparameters_n_US = KernelSelectionML_n['Hyperparameters']  # Kernel hyperparameters with highest log(ML)
        print(f"reMLE_BQUS_n, iteration={i}, n={nSample_Grid_StartnInitial[i]}: {MLEHyperparameters_n_US}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Kernel Selection failed at n={Beta_n.shape[0]}. Keeping the previous kernel. Error: {e}")

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (re-estimated each time)
    BQUncertaintySampling_reKernel = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input = Beta_n,  # dim=nxk
        f_Beta_Input = f_Beta_n,  # dim=nx1
        L0 = L0_BQUS_reMLE,
        KernelFunction = MLEKernel_n,
        BetaPriorMean = muVectorBetaPrior,  # dim=kx1
        BetaPriorCovarianceMatrix = CovarianceMatrixBetaPrior,  # dim=kxk
        NExtraPoints = nGrid_Differences[i + 1] - nGrid_Differences[i],  # Number of extra beta (nodes) to add
        NGridSize = nGridSize,
        NKernelMeanEmbeddingApprox = NKernelMeanEmbeddingApprox,
        ComputeVariance = False,
        **MLEHyperparameters_n_US)

    BQUncertaintySampling_reKernel_Estimate = BQUncertaintySampling_reKernel['Estimate']
    BQUncertaintySampling_reKernel_Estimate_FixedGrid.append( np.exp(L0_BQUS_reMLE) *
                                                              BQUncertaintySampling_reKernel_Estimate)
print("--------------------------------- [BQUS_reMLE] over n: Completed")










#=======================================================================================================================
#-------------------------------------------------- Comparison Plots ---------------------------------------------------
#=======================================================================================================================

#-------------------------------------------- Plot Convergence of Estimate ---------------------------------------------
plt.figure(figsize = (10, 6))

### Convergence Estimate: MC Estimate - N(5/2, 1) Proposal PDF
plt.loglog(nSample_Grid, MC_Estimate,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid,
                 MC_Estimate - np.sqrt(MC_Variance),
                 MC_Estimate + np.sqrt(MC_Variance),
                 color = 'lightgrey', linestyle = '--', alpha = 0.3, label = r"MC Estimate$\pm \sigma$")

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Hyperparameters Once
plt.loglog(nSample_Grid_StartnInitial, VanillaBQ_Estimate_FixedGrid,
         color = 'forestgreen', linewidth = 1.8, label = r"Vanilla BQ - MLE Once")

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
plt.loglog(nSample_Grid_StartnInitial, VanillaBQreHyperparameters_Estimate_FixedGrid,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Convergence Estimate: BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(nSample_Grid_StartnInitial, VanillaBQreKernel_Estimate_FixedGrid,
         color = 'red', linewidth = 1.8,  label = r"Vanilla BQ - re-MLE")

### Convergence Estimate: BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once
plt.loglog(nSample_Grid_StartnInitial, BQUncertaintySampling_Estimate_FixedGrid,
         color = 'deeppink', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - MLE Once")

### Convergence Estimate: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(nSample_Grid_StartnInitial, BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid,
         color = 'darkturquoise', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Convergence Estimate: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(nSample_Grid_StartnInitial, BQUncertaintySampling_reKernel_Estimate_FixedGrid,
         color = 'mediumslateblue', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE")


### Plot True Integral Value
plt.axhline(TrueIntegralValue_MarginalPDF, color = 'lightgreen', linewidth = 2, alpha = 0.7,
            label = "True Integral Value")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((10**0, 10**nMaxPower))
#plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**2) + 1, step = 5)]) )
plt.legend()
plt.title(r"Convergence of Scaled Marginal Likelihood Integral Estimate")
plt.xlabel(r"Number of Observations $n$")
plt.ylabel(r"Integral Estimate/Value")
plt.tight_layout()
plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations_loglog_GMM.png', dpi = 300)
plt.show()



########################################################################################################################
plt.figure(figsize = (10, 6))

### Convergence Estimate: MC Estimate - N(5/2, 1) Proposal PDF
plt.plot(nSample_Grid, MC_Estimate,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid, MC_Estimate - np.sqrt(MC_Variance), MC_Estimate + np.sqrt(MC_Variance),
                 color = 'lightgrey', linestyle = '--', alpha = 0.3, label = r"$MC Estimate$\pm \sigma$")

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Hyperparameters Once
plt.plot(nSample_Grid_StartnInitial, VanillaBQ_Estimate_FixedGrid,
         color = 'forestgreen', linewidth = 1.8, label = r"Vanilla BQ - MLE Once")

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
plt.plot(nSample_Grid_StartnInitial, VanillaBQreHyperparameters_Estimate_FixedGrid,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Convergence Estimate: BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
plt.plot(nSample_Grid_StartnInitial, VanillaBQreKernel_Estimate_FixedGrid,
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
plt.axhline(TrueIntegralValue_MarginalPDF, color = 'lightgreen', linewidth = 2, alpha = 0.7,
            label = "True Integral Value")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xscale('log')
plt.xlim((10**0, 10**nMaxPower))
#plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**2) + 1, step = 5)]) )
plt.legend()
plt.title(r"Convergence of Scaled Marginal Likelihood Integral Estimate")
plt.xlabel(r"Number of Observations $n$")
plt.ylabel(r"Integral Estimate/Value")
plt.tight_layout()
plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations_semilogx_GMM.png', dpi = 300)
plt.show()



########################################################################################################################
plt.figure(figsize = (10, 6))

### Convergence Estimate: MC Estimate - N(5/2, 1) Proposal PDF
plt.plot(nSample_Grid, MC_Estimate,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid, MC_Estimate - np.sqrt(MC_Variance), MC_Estimate + np.sqrt(MC_Variance),
                 color = 'lightgrey', linestyle = '--', alpha = 0.3, label = r"MC Estimate$\pm \sigma$")  # +/- 1*SD regions

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Hyperparameters Once
plt.plot(nSample_Grid_StartnInitial, VanillaBQ_Estimate_FixedGrid,
         color = 'forestgreen', linewidth = 1.8, label = r"Vanilla BQ - MLE Once")

### Convergence Estimate: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
plt.plot(nSample_Grid_StartnInitial, VanillaBQreHyperparameters_Estimate_FixedGrid,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Convergence Estimate: BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
plt.plot(nSample_Grid_StartnInitial, VanillaBQreKernel_Estimate_FixedGrid,
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
plt.axhline(TrueIntegralValue_MarginalPDF, color = 'lightgreen', linewidth = 2, alpha = 0.7,
            label = "True Integral Value")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((10**0, 10**nMaxPower))
#plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**2) + 1, step = 5)]) )
plt.legend()
plt.title(r"Convergence of Scaled Marginal Likelihood Integral Estimate")
plt.xlabel(r"Number of Observations $n$")
plt.ylabel(r"Integral Estimate/Value")
plt.tight_layout()
plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations_plot_GMM.png', dpi = 300)
plt.show()







#------------------------------------------ Plot Absolute Error of Estimates -------------------------------------------
plt.figure(figsize = (10, 6))

### Absolute Error: MC Estimate - N(5/2, 1) Proposal PDF
AbsoluteError_MC = np.abs(TrueIntegralValue_MarginalPDF - MC_Values)
AbsoluteError_MC_Mean = np.mean(AbsoluteError_MC, axis = 0)
AbsoluteError_MC_Std = np.std(AbsoluteError_MC, axis = 0, ddof = 1)
plt.plot(nSample_Grid, AbsoluteError_MC_Mean,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid,  # +/- 2*SD regions (above +/- 1*SD region)
                 AbsoluteError_MC_Mean - 1 * AbsoluteError_MC_Std,
                 AbsoluteError_MC_Mean + 1 * AbsoluteError_MC_Std, color = 'lightgrey', linestyle = '--', alpha = 0.3,
                 label = r"MC $\pm \sigma$")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
AbsoluteError_VanillaBQ = np.abs(TrueIntegralValue_MarginalPDF - VanillaBQ_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQ,
         color = 'forestgreen', linewidth = 1.8,  label = r"Vanilla BQ - MLE Once")

### Absolute Error: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
AbsoluteError_VanillaBQ_reHyperparameters = np.abs(TrueIntegralValue_MarginalPDF -
                                                   VanillaBQreHyperparameters_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQ_reHyperparameters,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
AbsoluteError_VanillaBQ_reKernel = np.abs(TrueIntegralValue_MarginalPDF - VanillaBQreKernel_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_VanillaBQ_reKernel,
         color = 'red', linewidth = 1.8, label = r"Vanilla BQ - re-MLE")

### Absolute Error: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
AbsoluteError_BQUncertaintySampling = np.abs(
    TrueIntegralValue_MarginalPDF - BQUncertaintySampling_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_BQUncertaintySampling,
         color = 'deeppink', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - MLE Once")

### Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
AbsoluteError_BQUncertaintySampling_reHyperparameters = np.abs(
    TrueIntegralValue_MarginalPDF - BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_BQUncertaintySampling_reHyperparameters,
         color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
AbsoluteError_BQUncertaintySampling_reKernel = np.abs(
    TrueIntegralValue_MarginalPDF - BQUncertaintySampling_reKernel_Estimate_FixedGrid)
plt.plot(nSample_Grid_StartnInitial, AbsoluteError_BQUncertaintySampling_reKernel,
         color = 'mediumslateblue', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xscale('log')  # log with 10 base is default
plt.yscale('log')
plt.xlim((10**0, 10**nMaxPower))
plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**nMaxPower) + 1, step = 5)]) )
plt.legend()
plt.title(r"Absolute Error of Scaled Marginal Likelihood Integral Estimates over Sample Sizes")
plt.xlabel(r"Number of Observations $n$")
plt.ylabel(r"$|I_{True} - \hat{I}|$")
plt.tight_layout()
plt.savefig('results/BQ_AbsoluteError_NumberofObservations_GMM.png', dpi = 300)
plt.show()








plt.figure(figsize = (10, 6))

### Absolute Error: MC Estimate - N(5/2, 1) Proposal PDF
RelativeAbsoluteError_MC = (np.abs(TrueIntegralValue_MarginalPDF - MC_Values) /
                                                  np.abs(TrueIntegralValue_MarginalPDF))
RelativeAbsoluteError_MC_Mean = np.mean(RelativeAbsoluteError_MC, axis = 0)
RelativeAbsoluteError_MC_Std = np.std(RelativeAbsoluteError_MC, axis = 0, ddof = 1)
plt.plot(nSample_Grid, RelativeAbsoluteError_MC_Mean,
         color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate - $\mathcal{N}(2.5, 1)$")
plt.fill_between(nSample_Grid,  # +/- 1*SD regions (above +/- 1*SD region)
                 RelativeAbsoluteError_MC_Mean - 1 * RelativeAbsoluteError_MC_Std,
                 RelativeAbsoluteError_MC_Mean + 1 * RelativeAbsoluteError_MC_Std,
                 color = 'lightgrey', linestyle = '--', alpha = 0.3, label = r"MC $\pm \sigma$")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
RelativeAbsoluteErrorVanillaBQ = (np.abs(TrueIntegralValue_MarginalPDF - VanillaBQ_Estimate_FixedGrid) /
                                                  np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteErrorVanillaBQ,
         color = 'forestgreen', linewidth = 1.8,  label = r"Vanilla BQ - MLE Once")


### Absolute Error: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
RelativeAbsoluteError_VanillaBQ_reHyperparameters = (np.abs(
    TrueIntegralValue_MarginalPDF - VanillaBQreHyperparameters_Estimate_FixedGrid) /
                                                     np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteError_VanillaBQ_reHyperparameters,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
RelativeAbsoluteError_VanillaBQ_reKernel = (np.abs(
    TrueIntegralValue_MarginalPDF - VanillaBQreKernel_Estimate_FixedGrid) /
                                            np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteError_VanillaBQ_reKernel,
         color = 'red', linewidth = 1.8, label = r"Vanilla BQ - re-MLE")

### Relative Absolute Error: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
RelativeAbsoluteError_BQUncertaintySampling = (np.abs(
    TrueIntegralValue_MarginalPDF - BQUncertaintySampling_Estimate_FixedGrid) / np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteError_BQUncertaintySampling,
         color = 'deeppink', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - MLE Once")

### Relative Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
RelativeAbsoluteError_BQUncertaintySampling_reHyperparameters = (np.abs(
    TrueIntegralValue_MarginalPDF - BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid) /
                                                                 np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteError_BQUncertaintySampling_reHyperparameters,
         color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Relative Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
RelativeAbsoluteError_BQUncertaintySampling_reKernel = (np.abs(
    TrueIntegralValue_MarginalPDF - BQUncertaintySampling_reKernel_Estimate_FixedGrid) /
                                                        np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteError_BQUncertaintySampling_reKernel,
         color = 'mediumslateblue', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE")



plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xscale('log')  # log with 10 base is default
plt.yscale('log')
plt.xlim((10**0, 10**nMaxPower))
plt.xticks( np.concatenate([np.array([1]), np.arange(start = 5, stop = int(10**nMaxPower) + 1, step = 5)]) )
plt.legend()
plt.title(r"Relative Absolute Error of Scaled Marginal Likelihood Integral Estimates over Sample Sizes")
plt.xlabel(r"Number of Observations $n$")
plt.ylabel(r"$|I_{True} - \hat{I}| / |I_{True}|$")
plt.tight_layout()
plt.savefig('results/BQ_RelativeAbsoluteError_NumberofObservations_GMM.png', dpi = 300)
plt.show()








#=======================================================================================================================
#----------------------------------------------------- Time Plots ------------------------------------------------------
#=======================================================================================================================
rng = np.random.default_rng(21)
T_Max = 1000  # In seconds
NIntegrationGrid = 1000  # For approximation of BQ non-analytical integrals
nGridSize = 2100  # The number of possible points to consider for uncertainty sampling
R = 100  # Number of repetitions of MC runs

#-------------------------------------| Plot MC Estimate - N(5/2, 1) Proposal PDF |-------------------------------------
StartTime_MC = time.perf_counter()  # Start time counter
MCEstimate_Grid_Time = []
MCVariance_Grid_Time = []
Times_MC = []
TotalTime_MC = 0
Transformation_Sum_MC = 0
n_MC = 1

while TotalTime_MC < T_Max:
    MC_Values = np.empty(R)
    for r in range(R):
        NormalPriorBeta_Samples = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                                          size=n_MC, method='cholesky')
        L0_MC = np.max(LogLikelihood_GMM(BetaVector = NormalPriorBeta_Samples))
        Transformed_NormalPriorBeta_Samples = ScaledLikelihood_GMM(BetaVector=NormalPriorBeta_Samples, L0=L0_MC)
        MC_Values[r] = np.exp(L0_MC) * np.mean(Transformed_NormalPriorBeta_Samples)

    # Compute the MC estimate per sample size
    MCEstimate_Grid_Time.append( np.mean(MC_Values, axis=0) )
    # Compute the MC variance per sample size
    MCVariance_Grid_Time.append( np.var(MC_Values, axis=0, ddof=1) )

    TotalTime_MC = time.perf_counter() - StartTime_MC  # Update current time
    Times_MC.append(TotalTime_MC)  # Update total time history
    n_MC += 1  # Update to n+1 observations
print("--------------------------------- [MC] over time: Completed")



#------------------------------| Plot BQ Estimate - Fixed Grid & MLE Hyperparameters Once |-----------------------------
StartTime_VBQ_MLEOnce = time.perf_counter()  # Start time counter
VanillaBQ_Estimate_FixedGrid_Time = []
nBetaInitial = 20  # The number of sample points we start with
# First, create a grid of beta nodes by drawing them from the beta prior distribution
Beta_nInitial = rng.multivariate_normal(mean = muVectorBetaPrior, cov = CovarianceMatrixBetaPrior,
                                        size = nBetaInitial, method = 'cholesky')
# Second, evaluate the integrand f(beta) on the beta nodes
L0_nInitial = np.max(LogLikelihood_GMM(BetaVector = Beta_nInitial))
f_Beta_nInitial = ScaledLikelihood_GMM(BetaVector = Beta_nInitial, L0 = L0_nInitial)
# Third, estimate the kernel & hyperparameters for this sample by maximum likelihood
KernelSelectionML_nInitial = KernelSelectionML(Beta_nInitial, f_Beta_nInitial, UniversalInitialParameter = 1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']  # MLE hyperparameters
# Fourth, delete the samples of size up to nBetaInitial
nSample_Grid_StartnInitial = nSample_Grid[nSample_Grid >= nBetaInitial]

Times_VBQ_MLEOnce = []
TotalTime_VBQ_MLEOnce = 0
n_VBQ_MLEOnce = nBetaInitial

while TotalTime_VBQ_MLEOnce < T_Max:
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=n_VBQ_MLEOnce, method='cholesky')
    L0_VBQ_MLEOnce = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_VBQ_MLEOnce)

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (estimated once beforehand)
    VanillaBQ_Estimate_Time, _ = BQ_MarginalPDF(Beta_Input=Beta_n,  # dim=nxk
                                                f_Beta_Input=f_Beta_n,  # dim=nx1
                                                KernelFunction=MLEKernel_nInitial,
                                                BetaPriorMean=muVectorBetaPrior,  # dim=kx1
                                                BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,  # dim=kxk
                                                NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
                                                ComputeVariance=False,
                                                **MLEHyperparameters_nInitial)

    VanillaBQ_Estimate_FixedGrid_Time.append(np.exp(L0_VBQ_MLEOnce) * VanillaBQ_Estimate_Time )

    TotalTime_VBQ_MLEOnce = time.perf_counter() - StartTime_VBQ_MLEOnce  # Update current time
    Times_VBQ_MLEOnce.append(TotalTime_VBQ_MLEOnce)  # Update total time history
    n_VBQ_MLEOnce += 1  # Update to n+1 observations
print("--------------------------------- [VanillaBQ_MLEOnce] over time: Completed")



#--------------------| BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters |-----------------------
StartTime_VBQ_reHyperparameters = time.perf_counter()  # Start time counter
VanillaBQreHyperparameters_Estimate_FixedGrid_Time = []
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# However, add the following: extracting the hyperparameter values and corresponding names
MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
InitialParameters_nInitial = 0 * MLEHyperparametersValues_nInitial + np.ones_like(MLEHyperparametersValues_nInitial)
MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']
MLEHyperparameters_VBQ_n = MLEHyperparameters_nInitial

Times_VBQ_reHyperparameters = []
TotalTime_VBQ_reHyperparameters = 0
n_VBQ_reHyperparameters = nBetaInitial

while TotalTime_VBQ_reHyperparameters < T_Max:
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=n_VBQ_reHyperparameters, method='cholesky')
    L0_VBQ_MLEHyperparameter = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_VBQ_MLEHyperparameter)

    try:
        NewMLEHyperparameters_VBQ_n = MLEHyperparameters(
            initial_parameters=InitialParameters_nInitial, X_Input=Beta_n, y=f_Beta_n,
            PriorMeanVector=np.zeros_like(f_Beta_n),  # Zero prior mean
            KernelFunction=MLEKernel_nInitial,  # Kernel estimated from grid of size nBetaInitial
            ParameterNames=MLEHyperparametersNames_nInitial)
        MLEHyperparameters_VBQ_n = dict(zip(MLEHyperparametersNames_nInitial, NewMLEHyperparameters_VBQ_n))  # Add names
        print(f"MLEHyperparameters_VBQ_n, n={n_VBQ_reHyperparameters}: {MLEHyperparameters_VBQ_n}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Optimisation failed at n = {Beta_n.shape[0]}. Keeping the old hyperparameters. Error: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    VanillaBQreHyperparameters_Estimate_Time, _ = BQ_MarginalPDF(Beta_Input=Beta_n,  # dim=nxk
                                                                 f_Beta_Input=f_Beta_n,  # dim=nx1
                                                                 KernelFunction=MLEKernel_nInitial,
                                                                 BetaPriorMean=muVectorBetaPrior,  # dim=kx1
                                                                 BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
                                                                 NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
                                                                 ComputeVariance=False,
                                                                 **MLEHyperparameters_VBQ_n)  # Update hyperparameters

    VanillaBQreHyperparameters_Estimate_FixedGrid_Time.append(np.exp(L0_VBQ_MLEHyperparameter) *
                                                              VanillaBQreHyperparameters_Estimate_Time)

    TotalTime_VBQ_reHyperparameters = time.perf_counter() - StartTime_VBQ_reHyperparameters # Update current time
    Times_VBQ_reHyperparameters.append(TotalTime_VBQ_reHyperparameters)  # Update total time history
    n_VBQ_reHyperparameters += 1  # Update to n+1 observations
print("--------------------------------- [VanillaBQ_reHyperparameter] over time: Completed")



#--------------------| Plot BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML |---------------------
StartTime_VBQ_reMLE = time.perf_counter()  # Start time counter
VanillaBQreKernel_Estimate_FixedGrid_Time = []
MLEKernel_n = MLEKernel_nInitial
MLEHyperparameters_n = MLEHyperparameters_nInitial

Times_VBQ_reMLE = []
TotalTime_VBQ_reMLE = 0
n_VBQ_reMLE = nBetaInitial

while TotalTime_VBQ_reMLE < T_Max:
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=n_VBQ_reMLE, method='cholesky')
    L0_VBQ_reMLE = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_VBQ_reMLE)
    try:
        KernelSelectionML_n = KernelSelectionML(Beta_n, f_Beta_n, UniversalInitialParameter=1)
        MLEKernel_n = KernelSelectionML_n['Kernel']  # Kernel with highest log(ML)
        MLEHyperparameters_n = KernelSelectionML_n['Hyperparameters']  # Kernel hyperparameters with highest log(ML)
        print(f"reMLE_VBQ_n, n={n_VBQ_reMLE}: {MLEHyperparameters_n}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Kernel Selection failed at n = {Beta_n.shape[0]}. Keeping the previous kernel. Error: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (re-estimated each time)
    VanillaBQreKernel_Estimate_Time, _ = BQ_MarginalPDF(Beta_Input=Beta_n,  # dim=nxk
                                                        f_Beta_Input=f_Beta_n,  # dim=nx1
                                                        KernelFunction=MLEKernel_n,
                                                        BetaPriorMean=muVectorBetaPrior,  # dim=kx1
                                                        BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,  # dim=kxk
                                                        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
                                                        ComputeVariance=False,
                                                        **MLEHyperparameters_n)

    VanillaBQreKernel_Estimate_FixedGrid_Time.append(np.exp(L0_VBQ_reMLE) * VanillaBQreKernel_Estimate_Time)

    TotalTime_VBQ_reMLE = time.perf_counter() - StartTime_VBQ_reMLE  # Update current time
    Times_VBQ_reMLE.append(TotalTime_VBQ_reMLE)  # Update total time history
    n_VBQ_reMLE += 1  # Update to n+1 observations
print("--------------------------------- [VanillaBQ_reKernel] over time: Completed")



#------------------------| Plot BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once |-------------------------
StartTime_BQUS_MLEOnce = time.perf_counter()  # Start time counter
BQUncertaintySampling_Estimate_FixedGrid_Time = []
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# However, add the following: differences in number of evaluations

Times_BQUS_MLEOnce = []
TotalTime_BQUS_MLEOnce = 0
n_BQUS_MLEOnce = nBetaInitial

while TotalTime_BQUS_MLEOnce < T_Max:
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=n_BQUS_MLEOnce, method='cholesky')
    L0_BQUS = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_BQUS)

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (estimated once beforehand)
    BQUncertaintySampling_Time = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input=Beta_n,  # dim=nxk
        f_Beta_Input=f_Beta_n,  # dim=nx1
        KernelFunction=MLEKernel_nInitial,
        BetaPriorMean=muVectorBetaPrior,  # dim=kx1
        BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,  # dim=kxk
        NExtraPoints=1,  # Number of extra beta (nodes) to add
        NGridSize=nGridSize,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
        ComputeVariance=False,
        L0=L0_BQUS,
        **MLEHyperparameters_nInitial)

    BQUncertaintySampling_Estimate_Time = BQUncertaintySampling_Time['Estimate']
    BQUncertaintySampling_Estimate_FixedGrid_Time.append(np.exp(L0_BQUS) * BQUncertaintySampling_Estimate_Time)

    TotalTime_BQUS_MLEOnce = time.perf_counter() - StartTime_BQUS_MLEOnce  # Update current time
    Times_BQUS_MLEOnce.append(TotalTime_BQUS_MLEOnce)  # Update total time history
    n_BQUS_MLEOnce += 1  # Update to n+1 observations
print("--------------------------------- [BQUS_MLEOnce] over time: Completed")



#-----------------| BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Re-estimate Hyperparameters |----------------
StartTime_BQUS_reHyperparameters = time.perf_counter()  # Start time counter
BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid_Time = []
MLEHyperparameters_BQUS_n = MLEHyperparameters_nInitial

Times_BQUS_reHyperparameters = []
TotalTime_BQUS_reHyperparameters = 0
n_BQUS_reHyperparameters = nBetaInitial

while TotalTime_BQUS_reHyperparameters < T_Max:
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=n_BQUS_reHyperparameters, method='cholesky')
    L0_BQUS_reHyperparameter = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_BQUS_reHyperparameter)

    # MLEHyperparameters_BQUS_n = dict(zip(MLEHyperparametersNames_nInitial, MLEHyperparameters_BQUS_n))  # Add names
    try:
        NewMLEHyperparameters_BQUS_n = MLEHyperparameters(
            initial_parameters=InitialParameters_nInitial, X_Input=Beta_n, y=f_Beta_n,
            PriorMeanVector=np.zeros_like(f_Beta_n),  # Zero prior mean
            KernelFunction=MLEKernel_nInitial,  # Kernel estimated from grid of size nBetaInitial
            ParameterNames=MLEHyperparametersNames_nInitial)
        MLEHyperparameters_BQUS_n = dict(
            zip(MLEHyperparametersNames_nInitial, NewMLEHyperparameters_BQUS_n))  # Add names
        print(f"MLEHyperparameters_BQUS_n, n={n_BQUS_reHyperparameters}: {MLEHyperparameters_BQUS_n}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Optimisation failed at n={Beta_n.shape[0]}. Keeping the old hyperparameters. Error: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    BQUncertaintySampling_reHyperparameters = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input=Beta_n,  # dim=nxk
        f_Beta_Input=f_Beta_n,  # dim=nx1
        KernelFunction=MLEKernel_nInitial,
        BetaPriorMean=muVectorBetaPrior,  # dim=kx1
        BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,  # dim=kxk
        NExtraPoints=1,  # Number of extra beta (nodes) to add
        NGridSize=nGridSize,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
        ComputeVariance=False,
        L0=L0_BQUS_reHyperparameter,
        **MLEHyperparameters_BQUS_n)

    BQUncertaintySampling_reHyperparameters_Estimate = BQUncertaintySampling_reHyperparameters['Estimate']
    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid_Time.append(np.exp(L0_BQUS_reHyperparameter) *
                                                                      BQUncertaintySampling_reHyperparameters_Estimate)

    TotalTime_BQUS_reHyperparameters = time.perf_counter() - StartTime_BQUS_reHyperparameters   # Update current time
    Times_BQUS_reHyperparameters.append(TotalTime_BQUS_reHyperparameters)  # Update total time history
    n_BQUS_reHyperparameters += 1  # Update to n+1 observations
print("--------------------------------- [VanillaBQ_reHyperparameter] over time: Completed")



#----------------| Plot BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML |---------------
StartTime_BQUS_reMLE = time.perf_counter()  # Start time counter
BQUncertaintySampling_reKernel_Estimate_FixedGrid_Time = []
MLEKernel_n = MLEKernel_nInitial
MLEHyperparameters_n_US = MLEHyperparameters_nInitial
# The four points in "VBQ + MLE Once" are used here again (but omitted)
# Similarly, the nGrid_Differences in "BQUS + MLE Once" are used here again (but omitted)

Times_BQUS_reMLE = []
TotalTime_BQUS_reMLE = 0
n_BQUS_reMLE = nBetaInitial

while TotalTime_BQUS_reMLE < T_Max:
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=n_BQUS_reMLE, method='cholesky')
    L0_BQUS_reMLE = np.max(LogLikelihood_GMM(BetaVector=Beta_n))
    f_Beta_n = ScaledLikelihood_GMM(BetaVector=Beta_n, L0=L0_BQUS_reMLE)

    try:
        KernelSelectionML_n = KernelSelectionML(Beta_n, f_Beta_n, UniversalInitialParameter=1)
        MLEKernel_n = KernelSelectionML_n['Kernel']  # Kernel with highest log(ML)
        MLEHyperparameters_n_US = KernelSelectionML_n[
            'Hyperparameters']  # Kernel hyperparameters with highest log(ML)
        print(f"reMLE_BQUS_n, n={n_BQUS_reMLE}: {MLEHyperparameters_n_US}")
    except Exception as e:  # Return the error and does not add the new points (if optimsation fails)
        print(f"Kernel Selection failed at n={Beta_n.shape[0]}. Keeping the previous kernel. Error: {e}")

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (re-estimated each time)
    BQUncertaintySampling_reKernel_Time = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input=Beta_n,  # dim=nxk
        f_Beta_Input=f_Beta_n,  # dim=nx1
        KernelFunction=MLEKernel_n,
        BetaPriorMean=muVectorBetaPrior,  # dim=kx1
        BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,  # dim=kxk
        NExtraPoints=1,  # Number of extra beta (nodes) to add
        NGridSize=nGridSize,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
        ComputeVariance=False,
        L0=L0_BQUS_reMLE,
        **MLEHyperparameters_n_US)

    BQUncertaintySampling_reKernel_Estimate_Time = BQUncertaintySampling_reKernel_Time['Estimate']
    BQUncertaintySampling_reKernel_Estimate_FixedGrid_Time.append(np.exp(L0_BQUS_reMLE) *
                                                                  BQUncertaintySampling_reKernel_Estimate_Time)

    TotalTime_BQUS_reMLE = time.perf_counter() - StartTime_BQUS_reMLE  # Update current time
    Times_BQUS_reMLE.append(TotalTime_BQUS_reMLE)  # Update total time history
    n_BQUS_reMLE += 1  # Update to n+1 observations
print("--------------------------------- [BQUS_reMLE] over time: Completed")



#-------------------------------------------- Plot Absolute Error over Time --------------------------------------------
plt.figure(figsize = (10, 6))

### Absolute Error: MC Estimate
plt.loglog(Times_MC, np.abs(MCEstimate_Grid_Time - TrueIntegralValue_MarginalPDF),
           color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate")

#MCVariance_Grid_Time

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
plt.loglog(Times_VBQ_MLEOnce, np.abs(VanillaBQ_Estimate_FixedGrid_Time - TrueIntegralValue_MarginalPDF),
           color = 'forestgreen', linewidth = 1.8, alpha = 0.7, label = r"Vanilla BQ - MLE Once")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
plt.loglog(Times_VBQ_reHyperparameters, np.abs(
    VanillaBQreHyperparameters_Estimate_FixedGrid_Time - TrueIntegralValue_MarginalPDF),
           color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Absolute Error: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(Times_VBQ_reMLE, np.abs(VanillaBQreKernel_Estimate_FixedGrid_Time - TrueIntegralValue_MarginalPDF),
           color = 'red', linewidth = 1.8,  label = r"Vanilla BQ - re-MLE")

### Absolute Error: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
plt.loglog(Times_BQUS_MLEOnce, np.abs(BQUncertaintySampling_Estimate_FixedGrid_Time - TrueIntegralValue_MarginalPDF),
           color = 'deeppink', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - MLE Once")

### Absolute Error: BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Hyperparameters by ML
plt.loglog(Times_BQUS_reHyperparameters, np.abs(
    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid_Time - TrueIntegralValue_MarginalPDF),
           color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Absolute Error: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
plt.loglog(Times_BQUS_reMLE, np.abs(BQUncertaintySampling_reKernel_Estimate_FixedGrid_Time -
                                    TrueIntegralValue_MarginalPDF),
           color = 'mediumslateblue', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE")

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim(right = T_Max)
plt.legend()
plt.title(r"Absolute Error of Estimate over Time)")
plt.xlabel(r"Time in Seconds")
plt.ylabel(r"$|I_{True} - \hat{I}|$")
plt.tight_layout()
plt.savefig('results/BQ_AbsoluteError_Time.png', dpi = 300)
plt.show()

### Number of Observations per Method
print(rf"n_MC:           {n_MC}")
print(rf"n_VBQ_MLEOnce:  {n_VBQ_MLEOnce}")
print(rf"n_VBQ_reMLE:    {n_VBQ_reMLE}")
print(rf"n_BQUS_MLEOnce: {n_BQUS_MLEOnce}")
print(rf"n_BQUS_reMLE:   {n_BQUS_reMLE}")


#--------------------------------------- Plot Number of Observations n over Time ---------------------------------------
plt.figure(figsize = (10, 6))

### Sample Size over Time: MC Estimate - N(5/2, 1) Proposal PDF
n_MC_SSN_Grid = np.arange(start = 1, stop = len(Times_MC) + 1, step = 1)
plt.loglog(Times_MC, n_MC_SSN_Grid,
           color = 'lightsalmon', linewidth = 1.8, alpha = 0.7, label = r"MC Estimate")

### Sample Size over Time: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
n_VBQ_MLEOnce_Grid = np.arange(start = nBetaInitial, stop = len(Times_VBQ_MLEOnce) + nBetaInitial, step = 1)
plt.loglog(Times_VBQ_MLEOnce, n_VBQ_MLEOnce_Grid,
           color = 'forestgreen', linewidth = 1.8, alpha = 0.7, label = r"Vanilla BQ - MLE Once")

### Sample Size over Time: Vanilla BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
n_VBQ_reHyperparameters_Grid = np.arange(start = nBetaInitial,
                                         stop = len(Times_VBQ_reHyperparameters) + nBetaInitial, step = 1)
plt.loglog(Times_VBQ_reHyperparameters, n_VBQ_reHyperparameters_Grid,
           color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Sample Size over Time: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
n_VBQ_reMLE_Grid = np.arange(start = nBetaInitial, stop = len(Times_VBQ_reMLE) + nBetaInitial, step = 1)
plt.loglog(Times_VBQ_reMLE, n_VBQ_reMLE_Grid,
           color = 'red', linewidth = 1.8,  label = r"Vanilla BQ - re-MLE")

### Sample Size over Time: BQ Estimate - Uncertainty Sampling & 1-time MLE Hyperparameters Once
n_BQUS_MLEOnce_Grid = np.arange(start = nBetaInitial, stop = len(Times_BQUS_MLEOnce) + nBetaInitial, step = 1)
plt.loglog(Times_BQUS_MLEOnce, n_BQUS_MLEOnce_Grid,
           color = 'deeppink', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - MLE Once")

### Sample Size over Time: BQ Estimate - Uncertainty Sampling & MLE Kernel Once & Hyperparameters by ML
n_BQUS_reHyperparameters_Grid = np.arange(start = nBetaInitial,
                                          stop = len(Times_BQUS_reHyperparameters) + nBetaInitial, step = 1)
plt.loglog(Times_BQUS_reHyperparameters, n_BQUS_reHyperparameters_Grid,
           color = 'darkturquoise', linewidth = 1.8, label = r"BQ & Uncertainty Sampling - re-MLE Hyperparameters")

### Sample Size over Time: BQ Estimate - Uncertainty Sampling & Re-estimate Kernel & Hyperparameters by ML
n_BQUS_reMLE_Grid = np.arange(start = nBetaInitial, stop = len(Times_BQUS_reMLE) + nBetaInitial, step = 1)
plt.loglog(Times_BQUS_reMLE, n_BQUS_reMLE_Grid,
           color = 'mediumslateblue', linewidth = 1.8,  label = r"BQ & Uncertainty Sampling - re-MLE")


plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim(right = T_Max)
plt.legend()
plt.title(r"Time versus Number of Observations $n$")
plt.xlabel(r"Time in Seconds")
plt.ylabel(r"Number of Observations $n$")
plt.tight_layout()
plt.savefig('results/BQ_NumberofObservations_Time.png', dpi = 300)
plt.show()


