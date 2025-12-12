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

# Note: BQ_KernelChoice_Importing is the same as BQ_KernelChoice but with every output and plot deleted.

#=======================================================================================================================
#----------------------------------------- Import Data & Define Setting/Model ------------------------------------------
#=======================================================================================================================
# https://www.kaggle.com/datasets/root64shivansh/profit-in-startup-of-a-company

Data = pd.read_csv('ProfitStartups.csv')
# Regressor = Profit. Three covariates = R&D Spend, Administration Cost, Marketing Spend.
# Note: (State categorical variable is omitted)

# print(Data.head())  # Inspect the first 5 rows
# print(Data.info())  # Information about the size and type of values

Data = Data[0: 30]  # Only consider the first 30 datapoints

Profit = Data['Profit'].values
Profit = (Profit - np.mean(Profit)) / np.std(Profit)  # Z-score (standardization)
RDSpend = Data['R&D Spend'].values
RDSpend = (RDSpend - np.mean(RDSpend)) / np.std(RDSpend)  # Z-score (standardization)
Administration = Data['Administration'].values
Administration = (Administration - np.mean(Administration)) / np.std(Administration)  # Z-score (standardization)
MarketingSpend = Data['Marketing Spend'].values
MarketingSpend = (MarketingSpend - np.mean(MarketingSpend)) / np.std(MarketingSpend)  # Z-score (standardization)

ColumnOnes = np.ones((len(Data), 1))  # Column vector, dim=nx1
ColumnRDSpend = RDSpend.reshape(-1, 1)  # Column vector, dim=nx1
ColumnAdministration = Administration.reshape(-1, 1)  # Column vector, dim=nx1
ColumnMarketingSpend = MarketingSpend.reshape(-1, 1)  # Column vector, dim=nx1
X = np.hstack((ColumnOnes, ColumnRDSpend, ColumnAdministration, ColumnMarketingSpend))  # Design matrix, dim=nx4

### Define parameters
k_Data = X.shape[1]  # Dimension of beta
d_Data = len(Data)  # Number of observations

# OLSPrediction = sm.OLS(Profit, X).fit().predict()  # X beta_{OLS}
# OLSResiduals = Profit - OLSPrediction
# sigma2Model = np.sum( OLSResiduals**2 ) / (d_Data - k_Data)  # Variance of residuals from OLS estimate
sigma2Model = 1
CovarianceMatrixModel = sigma2Model * np.identity(d_Data)  # Sigma_1 := sigma^2 * identity matrix

muVectorBetaPrior = np.zeros(k_Data)  # Prior mean of beta
tau2Prior = 1  # Prior (homoskedastic) variance of beta
CovarianceMatrixBetaPrior = tau2Prior * np.eye(k_Data)  # Sigma_2 := tau^2 * identity matrix


### Define Gaussian Multivariate PDF (For True Value)
def MultivariateGaussianPDF(yVector, MeanVector, CovarianceMatrix):
    """"
    The multivariate Gaussian PDF is computed by computing the log(PDF) first & using the Cholesky decomposition
    """
    n = len(yVector)
    PiTerm = (-n / 2) * np.log(2 * np.pi)
    L = np.linalg.cholesky(CovarianceMatrix)  # Cholesky decomposition
    DeterminantTerm = np.sum(np.log(np.diag(L)))
    MainTerm = np.linalg.solve(L, yVector - MeanVector)  # L^{-1} (y - mu)
    return np.exp(PiTerm - DeterminantTerm - 0.5 * (MainTerm.T @ MainTerm))


### Define Log-Integrand (Log-Likelihood) for Stability
def LogIntegrand_MarginalPDF(yVector, XMatrix, BetaVector, sigma2):
    """ log(f(beta)) = log(L(beta)) (Prior is implicitly handled by BQ measure) """
    y = np.asarray(yVector).reshape(-1)
    d = len(y)
    NormalizationConstantTerm = -0.5 * d * np.log(2 * np.pi * sigma2)

    Beta = np.atleast_2d(BetaVector)  # dim=len(BetaVector)xk
    # Calculate Residuals: y - X*beta
    ResidualsTerm = y[:, None] - (XMatrix @ Beta.T)
    Norm2 = np.sum(ResidualsTerm ** 2, axis=0)  # Sum squared errors

    return NormalizationConstantTerm - 0.5 * Norm2 / sigma2


### Define Scaled Integrand
def ScaledIntegrand_MarginalPDF(yVector, XMatrix, BetaVector, sigma2, L0):
    """ exp{log(f(beta)) - L_0} """
    LogMarginalPDF = LogIntegrand_MarginalPDF(yVector, XMatrix, BetaVector, sigma2)
    return np.exp(LogMarginalPDF - L0)


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

    # --- Robust Jitter to prevent Cholesky crashes ---
    Jitter = 1e-5
    for _ in range(4):
        try:
            L = np.linalg.cholesky(K_BB + Jitter * np.eye(n))  # Cholesky decomposition K_BB = L L^T & Add jitter
            break
        except np.linalg.LinAlgError:
            Jitter *= 10
    else:
        return np.nan, None

    #--------------------- Approximate the Kernel Mean Embedding (KME) by MC ---------------------
    NormalPriorBeta_MCSamples = rng.multivariate_normal(mean=BetaPriorMean, cov=BetaPriorCovarianceMatrix,
                                                        size=NKernelMeanEmbeddingApprox, method='cholesky')
    KernelMeanEmbedding_MC = []  # dim=kx1, MC estimate for every kappa(beta_j)
    for j in range(n):
        KME_NormalPriorBeta_MCSamples_j = np.array(  # Evaluate k(beta^{(i)}, beta_j) for sampled beta^{(i)}
            [KernelFunction(Beta_i, Beta_Input[j], **kernelhyperparameters) for Beta_i in NormalPriorBeta_MCSamples]
        )
        KernelMeanEmbedding_MC.append(np.mean(KME_NormalPriorBeta_MCSamples_j))

    KernelMeanEmbedding_MC = np.array(KernelMeanEmbedding_MC, dtype=float)

    #---------------------------- Compute Posterior Mean = Estimate -----------------------------
    w = np.linalg.solve(L.T, np.linalg.solve(L, f_Beta_Input))  # K_{XX}^{-1} f(beta) = (L^T)^{-1} (L^{-1} f(beta))
    BQ_Estimate = KernelMeanEmbedding_MC @ w

    if ComputeVariance == False:
        return BQ_Estimate, None

    return BQ_Estimate, None  # Variance code omitted for brevity


#------------------------------------------------ Uncertainty Sampling -------------------------------------------------
def UncertaintySampling_Beta(Beta_Candidates, Beta_Original, KernelFunction, **hyperparameters):
    """
    Return point with the highest posterior function()~GP variance: k(x,x) - k_{xX}K^{-1}k_{Xx}.
    """
    if Beta_Candidates.size == 0: return None
    n = Beta_Original.shape[0]  # Our 'current' number of observations

    #---------------------------------- Create Kernel Matrix K_BB --------------------------------
    K_BB = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            val = KernelFunction(Beta_Original[i], Beta_Original[j], **hyperparameters)
            K_BB[i, j] = val
            K_BB[j, i] = val  # Use symmetry

    # --- Robust Jitter ---
    Jitter = 1e-5
    for _ in range(4):
        try:
            L = np.linalg.cholesky(K_BB + Jitter * np.eye(n))
            break
        except np.linalg.LinAlgError:
            Jitter *= 10
    else:
        return None

    #------------------- Compute Posterior GP Variance for each Candidate Beta -------------------
    # Compute k(beta, beta) elements for beta in Beta_Candidates
    k_BetaBeta_Posterior = np.empty(Beta_Candidates.shape[0], dtype=float)
    for i in range(Beta_Candidates.shape[0]):
        Beta = Beta_Candidates[i]
        k_BetaBeta = KernelFunction(Beta, Beta, **hyperparameters)  # k(beta, beta)
        k_BetaB = np.array([KernelFunction(Beta, Beta_Original[j], **hyperparameters) for j in range(n)], dtype=float)
        w = np.linalg.solve(L, k_BetaB)  # L^{-1} k_{beta B}
        k_BetaBeta_Posterior[i] = k_BetaBeta - w @ w  # k(beta beta) - k_{beta B} K_{BB}^{-1} k_{beta B}^T

    #------------------ Robust "Next Best Point" Selection (Distance Check) -------------------
    IndexMostUncertainPoint = np.argsort(k_BetaBeta_Posterior)[::-1]  # Sort indices desc
    for index in IndexMostUncertainPoint[:50]:  # Inspect top 50 candidates
        Beta_Proposed = Beta_Candidates[index]
        Distances = np.linalg.norm(Beta_Original - Beta_Proposed, axis=1)  # Check distances
        MinDist = np.min(Distances)

        if MinDist > 1e-4:  # If point is far enough, accept it
            return np.asarray(Beta_Proposed, dtype=float).ravel()

    print("All top candidates were too close. Stop searching.")
    return None


#------------------------------------- Bayesian Quadrature + Uncertainty Sampling --------------------------------------
def BQ_MarginalPDF_UncertaintySampling(Beta_Input,  # dim=nxk
                                       f_Beta_Input,  # dim=nx1
                                       KernelFunction,
                                       BetaPriorMean,  # dim=kx1
                                       BetaPriorCovarianceMatrix,  # dim=kxk
                                       NExtraPoints,  # Number of extra beta (nodes) to add
                                       NGridSize,  # Number of candidate beta's
                                       L0=None,
                                       NKernelMeanEmbeddingApprox=1000000,
                                       ComputeVariance=False,
                                       **kernelhyperparameters):
    #--------------- Update Beta-grid with NExtraPoints Extra Points - Uncertainty Sampling --------------
    Beta_NewUS_History = [Beta_Input.copy()]  # Keep track of grid
    f_Beta_NewUS_History = [f_Beta_Input.copy()]

    for i in range(NExtraPoints):
        Beta_Candidates = rng.multivariate_normal(mean=BetaPriorMean, cov=BetaPriorCovarianceMatrix,
                                                  size=NGridSize, method='cholesky')

        Beta_New = UncertaintySampling_Beta(Beta_Candidates=Beta_Candidates, Beta_Original=Beta_NewUS_History[i],
                                            KernelFunction=KernelFunction, **kernelhyperparameters)

        if Beta_New is None:
            break  # Break when there are no more new points to choose from

        Beta_New = np.asarray(Beta_New, dtype=float).ravel()  # dim=1xk
        B_Next = np.vstack([Beta_NewUS_History[i], Beta_New[None, :]])  # dim=n_{i+1}xk
        Beta_NewUS_History.append(B_Next)  # Add new points to track record history

        # Calculate Scaled Likelihood for new point
        f_Beta_New = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_New,
                                                 sigma2=sigma2Model, L0=L0)
        f_Beta_Next = np.concatenate([f_Beta_NewUS_History[i], np.array([f_Beta_New]).ravel()])
        f_Beta_NewUS_History.append(f_Beta_Next)

    #-------------------- Compute BQ Estimate and Uncertainty - Uncertainty Sampling ---------------------
    Beta_NewUS = Beta_NewUS_History[-1]  # The last X-grid
    f_Beta_NewUS = f_Beta_NewUS_History[-1]

    #------------------------------------------ BQ with New Grid -----------------------------------------
    PosteriorMeanIntegral_BQ, PosteriorCovarianceIntegral_BQ = BQ_MarginalPDF(
        Beta_Input=Beta_NewUS,
        f_Beta_Input=f_Beta_NewUS,
        KernelFunction=KernelFunction,
        BetaPriorMean=BetaPriorMean,
        BetaPriorCovarianceMatrix=BetaPriorCovarianceMatrix,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
        ComputeVariance=ComputeVariance,
        **kernelhyperparameters
    )

    return {"Estimate": PosteriorMeanIntegral_BQ, "Variance": PosteriorCovarianceIntegral_BQ,
            "X_Grid": Beta_NewUS, "f_X_Grid": f_Beta_NewUS}


#=======================================================================================================================
#---------------------------------------- Approximation by Different Techniques ----------------------------------------
#=======================================================================================================================
nMaxPower = 2
nSample_Grid = np.unique(np.logspace(start=0, stop=nMaxPower, num=20, base=10, dtype=int))
NKernelMeanEmbeddingApprox = 1000000
nGridSize = 2100

#------------------------------------------------ Exact Integral Value -------------------------------------------------
TrueCovarianceMatrix = CovarianceMatrixModel + X @ CovarianceMatrixBetaPrior @ X.T
TrueMeanVector = X @ muVectorBetaPrior
TrueIntegralValue_MarginalPDF = MultivariateGaussianPDF(yVector=Profit, MeanVector=TrueMeanVector,
                                                        CovarianceMatrix=TrueCovarianceMatrix)
print(rf"True Marginal Likelihood: {TrueIntegralValue_MarginalPDF}")

#----------------------------------------------| Monte Carlo Integration |----------------------------------------------
R = 100
MC_Values = np.empty((R, len(nSample_Grid)))  # dim=Rxlen(nSample_Grid)
for r in range(R):
    for i in range(len(nSample_Grid)):
        NormalPriorBeta_Samples = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                                          size=nSample_Grid[i], method='cholesky')
        # Use Log and Scaled functions for stability
        L0_MC = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X,
                                                BetaVector=NormalPriorBeta_Samples, sigma2=sigma2Model))
        Transformed_Samples = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X,
                                                          BetaVector=NormalPriorBeta_Samples,
                                                          sigma2=sigma2Model, L0=L0_MC)
        MC_Values[r, i] = np.exp(L0_MC) * np.mean(Transformed_Samples)

MC_Estimate = np.mean(MC_Values, axis=0)  # Compute the MC estimate per sample size
MC_Variance = np.var(MC_Values, axis = 0, ddof = 1)  # Compute the MC variance per sample size
print("--------------------------------- [MC] over n: Completed")

#-------------------------------| BQ Estimate - Fixed Grid & MLE Hyperparameters Once |---------------------------------
VanillaBQ_Estimate_FixedGrid = []
nBetaInitial = 25  # The number of sample points we start with

# First, create a grid of beta nodes by drawing them from the beta prior distribution
Beta_nInitial = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                        size=nBetaInitial, method='cholesky')
# Second, evaluate the integrand f(beta) on the beta nodes
L0_nInitial = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_nInitial, sigma2=sigma2Model))
f_Beta_nInitial = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_nInitial,
                                              sigma2=sigma2Model, L0=L0_nInitial)

# Third, estimate the kernel & hyperparameters for this sample by maximum likelihood
KernelSelectionML_nInitial = KernelSelectionML(Beta_nInitial, f_Beta_nInitial, UniversalInitialParameter=1)
MLEKernel_nInitial = KernelSelectionML_nInitial['Kernel']  # Kernel with highest log(ML)
MLEHyperparameters_nInitial = KernelSelectionML_nInitial['Hyperparameters']
# Fourth, delete the samples of size up to nBetaInitial
nSample_Grid_StartnInitial = nSample_Grid[nSample_Grid >= nBetaInitial]

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=nSample_Grid_StartnInitial[i], method='cholesky')
    L0_VBQ_MLEOnce = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model))
    f_Beta_n = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n,
                                           sigma2=sigma2Model, L0=L0_VBQ_MLEOnce)

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (estimated once beforehand)
    VanillaBQ_Estimate, _ = BQ_MarginalPDF(Beta_Input=Beta_n,
                                           f_Beta_Input=f_Beta_n,
                                           KernelFunction=MLEKernel_nInitial,
                                           BetaPriorMean=muVectorBetaPrior,
                                           BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
                                           NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
                                           ComputeVariance=False,
                                           **MLEHyperparameters_nInitial)

    VanillaBQ_Estimate_FixedGrid.append(np.exp(L0_VBQ_MLEOnce) * VanillaBQ_Estimate)
print("--------------------------------- [VanillaBQ_MLEOnce] over n: Completed")

#--------------------| BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters |-----------------------
VanillaBQreHyperparameters_Estimate_FixedGrid = []
MLEHyperparametersValues_nInitial = KernelSelectionML_nInitial['HyperparametersValues']
InitialParameters_nInitial = np.ones_like(MLEHyperparametersValues_nInitial)
MLEHyperparametersNames_nInitial = KernelSelectionML_nInitial['HyperparametersNames']

#--- Initialization before loop ---
MLEHyperparameters_VBQ_n = MLEHyperparameters_nInitial

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=nSample_Grid_StartnInitial[i], method='cholesky')
    L0_VBQ_MLEHyp = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model))
    f_Beta_n = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n,
                                           sigma2=sigma2Model, L0=L0_VBQ_MLEHyp)

    # Re-estimate the hyperparameters (With Safety Net)
    try:
        NewParams = MLEHyperparameters(
            initial_parameters=InitialParameters_nInitial, X_Input=Beta_n, y=f_Beta_n,
            PriorMeanVector=np.zeros_like(f_Beta_n),  # Zero prior mean
            KernelFunction=MLEKernel_nInitial, ParameterNames=MLEHyperparametersNames_nInitial)
        MLEHyperparameters_VBQ_n = dict(zip(MLEHyperparametersNames_nInitial, NewParams))
    except Exception as e:
        print(f"Optimization failed at n={Beta_n.shape[0]}: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    VanillaBQreHyperparameters_Estimate, _ = BQ_MarginalPDF(Beta_Input=Beta_n,
                                                            f_Beta_Input=f_Beta_n,
                                                            KernelFunction=MLEKernel_nInitial,
                                                            BetaPriorMean=muVectorBetaPrior,
                                                            BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
                                                            NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
                                                            ComputeVariance=False,
                                                            **MLEHyperparameters_VBQ_n)

    VanillaBQreHyperparameters_Estimate_FixedGrid.append(np.exp(L0_VBQ_MLEHyp) * VanillaBQreHyperparameters_Estimate)
print("--------------------------------- [VanillaBQ_reHyperparameters] over n: Completed")

#----------------------| BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML |------------------------
VanillaBQreKernel_Estimate_FixedGrid = []
#--- Initialization before loop ---
MLEKernel_n = MLEKernel_nInitial
MLEHyperparameters_n = MLEHyperparameters_nInitial

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=nSample_Grid_StartnInitial[i], method='cholesky')
    L0_VBQ_reMLE = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model))
    f_Beta_n = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n,
                                           sigma2=sigma2Model, L0=L0_VBQ_reMLE)

    # Re-estimate the kernel function & corresponding hyperparameters
    try:
        Selection = KernelSelectionML(Beta_n, f_Beta_n, UniversalInitialParameter=1)
        MLEKernel_n = Selection['Kernel']
        MLEHyperparameters_n = Selection['Hyperparameters']
    except Exception as e:
        print(f"Kernel Selection failed at n={Beta_n.shape[0]}: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel & hyperparameters (re-estimated each time)
    VanillaBQreKernel_Estimate, _ = BQ_MarginalPDF(Beta_Input=Beta_n,
                                                   f_Beta_Input=f_Beta_n,
                                                   KernelFunction=MLEKernel_n,
                                                   BetaPriorMean=muVectorBetaPrior,
                                                   BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
                                                   NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
                                                   ComputeVariance=False,
                                                   **MLEHyperparameters_n)

    VanillaBQreKernel_Estimate_FixedGrid.append(np.exp(L0_VBQ_reMLE) * VanillaBQreKernel_Estimate)
print("--------------------------------- [VanillaBQ_reKernel] over n: Completed")

#--------------------------| BQ Estimate - Uncertainty Sampling & MLE Hyperparameters Once |----------------------------
BQUncertaintySampling_Estimate_FixedGrid = []
# The four points in "VBQ + MLE Once" are used here again (but omitted)
nGrid_Differences = np.insert(nSample_Grid_StartnInitial, 0, nBetaInitial)

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=nSample_Grid_StartnInitial[i], method='cholesky')
    L0_BQUS = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model))
    f_Beta_n = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model, L0=L0_BQUS)

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (estimated once beforehand)
    BQUncertaintySampling = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input=Beta_n,
        f_Beta_Input=f_Beta_n,
        KernelFunction=MLEKernel_nInitial,
        BetaPriorMean=muVectorBetaPrior,
        BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
        NExtraPoints=nGrid_Differences[i + 1] - nGrid_Differences[i],
        NGridSize=nGridSize,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox,
        ComputeVariance=False,
        L0=L0_BQUS,
        **MLEHyperparameters_nInitial)

    BQUncertaintySampling_Estimate = BQUncertaintySampling['Estimate']
    BQUncertaintySampling_Estimate_FixedGrid.append(np.exp(L0_BQUS) * BQUncertaintySampling_Estimate)
print("--------------------------------- [BQUS_MLEOnce] over n: Completed")

#-----------------| BQ Estimate - Uncertainty Sampling & Re-estimate Hyperparameters |----------------
BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid = []
# --- Initialization before loop ---
MLEHyperparameters_BQUS_n = MLEHyperparameters_nInitial

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=nSample_Grid_StartnInitial[i], method='cholesky')
    L0_BQUS = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model))
    f_Beta_n = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model, L0=L0_BQUS)

    # Re-estimate the hyperparameters
    try:
        NewParams = MLEHyperparameters(
            initial_parameters=InitialParameters_nInitial, X_Input=Beta_n, y=f_Beta_n,
            PriorMeanVector=np.zeros_like(f_Beta_n),  # Zero prior mean
            KernelFunction=MLEKernel_nInitial, ParameterNames=MLEHyperparametersNames_nInitial)
        MLEHyperparameters_BQUS_n = dict(zip(MLEHyperparametersNames_nInitial, NewParams))
    except Exception as e:
        print(f"Optimization failed at n={Beta_n.shape[0]}: {e}")

    # Estimate integral by Vanilla BQ, with MLE kernel (once) & re-estimated hyperparameters
    BQUncertaintySampling_reHyperparameters = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input=Beta_n, f_Beta_Input=f_Beta_n, KernelFunction=MLEKernel_nInitial,
        BetaPriorMean=muVectorBetaPrior, BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
        NExtraPoints=nGrid_Differences[i + 1] - nGrid_Differences[i], NGridSize=nGridSize,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox, ComputeVariance=False,
        L0=L0_BQUS, **MLEHyperparameters_BQUS_n)

    BQUncertaintySampling_reHyperparameters_Estimate_FixedGrid.append(
        np.exp(L0_BQUS) * BQUncertaintySampling_reHyperparameters['Estimate'])
print("--------------------------------- [BQUS_reHyperparameter] over n: Completed")

#-----------------| BQ Estimate - Uncertainty Sampling & Re-estimate Kernel |-------------------
BQUncertaintySampling_reKernel_Estimate_FixedGrid = []
# --- Initialization before loop ---
MLEKernel_n = MLEKernel_nInitial
MLEHyperparameters_n_US = MLEHyperparameters_nInitial

for i in range(len(nSample_Grid_StartnInitial)):
    Beta_n = rng.multivariate_normal(mean=muVectorBetaPrior, cov=CovarianceMatrixBetaPrior,
                                     size=nSample_Grid_StartnInitial[i], method='cholesky')
    L0_BQUS = np.max(LogIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model))
    f_Beta_n = ScaledIntegrand_MarginalPDF(yVector=Profit, XMatrix=X, BetaVector=Beta_n, sigma2=sigma2Model, L0=L0_BQUS)

    # Re-estimate the kernel function & corresponding hyperparameters
    try:
        Selection = KernelSelectionML(Beta_n, f_Beta_n, UniversalInitialParameter=1)
        MLEKernel_n = Selection['Kernel']
        MLEHyperparameters_n_US = Selection['Hyperparameters']
    except Exception as e:
        print(f"Kernel Selection failed at n={Beta_n.shape[0]}: {e}")

    # Estimate integral by BQ & uncertainty sampling, with MLE kernel & hyperparameters (re-estimated each time)
    BQUncertaintySampling_reKernel = BQ_MarginalPDF_UncertaintySampling(
        Beta_Input=Beta_n, f_Beta_Input=f_Beta_n, KernelFunction=MLEKernel_n,
        BetaPriorMean=muVectorBetaPrior, BetaPriorCovarianceMatrix=CovarianceMatrixBetaPrior,
        NExtraPoints=nGrid_Differences[i + 1] - nGrid_Differences[i], NGridSize=nGridSize,
        NKernelMeanEmbeddingApprox=NKernelMeanEmbeddingApprox, ComputeVariance=False,
        L0=L0_BQUS, **MLEHyperparameters_n_US)

    BQUncertaintySampling_reKernel_Estimate_FixedGrid.append(
        np.exp(L0_BQUS) * BQUncertaintySampling_reKernel['Estimate'])
print("--------------------------------- [BQUS_reMLE] over n: Completed")




#=======================================================================================================================
#-------------------------------------------------- Comparison Plots ---------------------------------------------------
#=======================================================================================================================

#--------------------------------------- Plot Convergence of Estimate - loglog -----------------------------------------
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
plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations_loglog.png', dpi = 300)
plt.show()



#-------------------------------------- Plot Convergence of Estimate - semilogx ----------------------------------------
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
plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations_semilogx.png', dpi = 300)
plt.show()



#---------------------------------------- Plot Convergence of Estimate - plot ------------------------------------------
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
plt.savefig('results/BQ_ConvergenceEstimate_NumberofObservations_plot.png', dpi = 300)
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
plt.savefig('results/BQ_AbsoluteError_NumberofObservations.png', dpi = 300)
plt.show()


#------------------------------------- Plot Relative Absolute Error of Estimates ---------------------------------------
plt.figure(figsize = (10, 6))

### Relative Absolute Error: MC Estimate - N(5/2, 1) Proposal PDF
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

### Relative Absolute Error: Vanilla BQ Estimate - Fixed Grid & MLE Hyperparameters
RelativeAbsoluteErrorVanillaBQ = (np.abs(TrueIntegralValue_MarginalPDF - VanillaBQ_Estimate_FixedGrid) /
                                                  np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteErrorVanillaBQ,
         color = 'forestgreen', linewidth = 1.8,  label = r"Vanilla BQ - MLE Once")


### Relative Absolute Error: BQ Estimate - Fixed Grid & MLE Kernel Once & Re-estimate Hyperparameters
RelativeAbsoluteError_VanillaBQ_reHyperparameters = (np.abs(
    TrueIntegralValue_MarginalPDF - VanillaBQreHyperparameters_Estimate_FixedGrid) /
                                                     np.abs(TrueIntegralValue_MarginalPDF))
plt.plot(nSample_Grid_StartnInitial, RelativeAbsoluteError_VanillaBQ_reHyperparameters,
         color = 'yellow', linewidth = 1.8, label = r"Vanilla BQ - re-MLE Hyperparameters")

### Relative Absolute Error: Vanilla BQ Estimate - Fixed Grid & Re-estimate Kernel & Hyperparameters by ML
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
plt.savefig('results/BQ_RelativeAbsoluteError_NumberofObservations.png', dpi = 300)
plt.show()