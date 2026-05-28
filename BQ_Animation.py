########################################################################################################################
####################################### BAYESIAN QUADRATURE - ANIMATION (GIF) ########################################
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.stats
import imageio.v2 as imageio
import os
from BQ_UniversalFunctions import (Kernel_SE, KernelMatrix, SumKernels, LogMarginalLikelihood,
                                   MLEHyperparameters, KernelSelectionML, KernelSelectionML_1D)


#=======================================================================================================================
#----------------------------------------------- Animation Settings ----------------------------------------------------
#=======================================================================================================================
Mode           = 'reMLE'       # 'reMLE' | 'UncertaintySampling' | 'UncertaintySampling_reMLE'
N_Start        = 5             # Number of initial observations
N_End          = 25            # Final number of observations
N_Grid         = 500           # Resolution of GP evaluation grid
N_Samples      = 3             # Number of posterior sample functions to draw
Frame_Duration = 1200          # Seconds per frame in the GIF
Random_Seed    = 23            # Fixed random seed
Interval_Left  = 0             # Integration interval left bound
Interval_Right = 5             # Integration interval right bound
Gif_Filename   = f'BQ_Animation_{Mode}_IntervalRight={Interval_Right}_nGrid={N_Grid}_nStart={N_Start}_nEnd={N_End}.gif'




#=======================================================================================================================
#---------------------------------------------- Integrand & True Integral ----------------------------------------------
#=======================================================================================================================
def Integrand_f(x):  # Integrand f(x)
    return np.exp( np.sin(x**2) * np.cos(x) )

### Prior Mean Function
def m_Prior(x, c):
    return 0 * np.array(x) + c  # Constant prior GP mean of c

c_PriorMean = 0

### True Integral Value - Estimated by Trapezoid Rule on a 10000-point grid
TrueGrid = np.linspace(start = Interval_Left, stop = Interval_Right, num = 10000)
TrueIntegralValue = np.trapezoid(Integrand_f(TrueGrid), TrueGrid)




#=======================================================================================================================
#---------------------------------------------------- GP Functions -----------------------------------------------------
#=======================================================================================================================

#----------------------------------------- Posterior Mean & Covariance Kernel ------------------------------------------
def GP_Posterior(KernelFunction, X, f_X, c=0, **hyperparameters):
    X = np.asarray(X)
    f_X = np.asarray(f_X)

    m_X = m_Prior(X, c)  # Prior mean function of f_X
    K_XX = KernelMatrix(KernelFunction, X, X, **hyperparameters)  # Gram matrix

    jitter = 1e-8
    for _ in range(6):
        try:
            L = np.linalg.cholesky( K_XX + jitter * np.eye(len(X)) )
            break
        except np.linalg.LinAlgError:
            jitter *= 10

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


#------------------------------------------ Bayesian Quadrature Approximation ------------------------------------------
def BQ_1D_Approximations(X_Input, f_X_Input, ConstantPriorGPMean, KernelFunction,
                         NEvaluations = N_Grid, IntervalLeft = Interval_Left, IntervalRight = Interval_Right,
                         ComputeVariance = True, **kernelhyperparameters):
    ApproximationGrid = np.linspace(start = IntervalLeft, stop = IntervalRight, num = NEvaluations)

    #---------------------------- Prior GP Mean & Gram Matrix GP -----------------------------
    m_X = m_Prior(X_Input, c = ConstantPriorGPMean)  # Prior mean function of f
    K_XX = KernelMatrix(KernelFunction, X_Input, X_Input, **kernelhyperparameters)  # Gram matrix

    jitter = 1e-8
    for _ in range(6):
        try:
            L = np.linalg.cholesky( K_XX + jitter * np.eye(len(X_Input)) )
            break
        except np.linalg.LinAlgError:
            jitter *= 10


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


#------------------------------------------------ Uncertainty Sampling -------------------------------------------------
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


### Gaussian PDF 1D
def GaussianPDF_1D(x, mu, sigma2):
    Norm2 = (x - mu)**2  # Squared Euclidean norm
    Factor = 1 / np.sqrt(2 * np.pi * sigma2)
    return Factor * np.exp( -Norm2 / (2 * sigma2) )




#=======================================================================================================================
#-------------------------------------------- Initial Kernel Selection -------------------------------------------------
#=======================================================================================================================
X_Initial       = np.linspace(start = Interval_Left, stop = Interval_Right, num = N_Start)
f_X_Initial     = Integrand_f(X_Initial)
PriorMean_Initial = m_Prior(X_Initial, c = c_PriorMean)

### Run KernelSelectionML_1D on the initial N_Start observations
InitialSelection             = KernelSelectionML_1D(X_Initial, f_X_Initial)
BestKernelChoice_kernel      = InitialSelection['Kernel']
BestKernelChoice_hyperparameters = InitialSelection['Hyperparameters']
BestKernelChoice_label       = InitialSelection['Label']




#=======================================================================================================================
#------------------------------------- Pre-build Uncertainty Sampling Sequence -----------------------------------------
#=======================================================================================================================
if Mode == 'UncertaintySampling':
    X_US_History        = [X_Initial.copy()]  # X_US_History[i] = observation grid at frame i
    X_New_Point_History = [None]              # point added going into frame i (None for frame 0)

    for _i in range(N_End - N_Start):
        x_New = UncertaintySampling(
            X_Candidates        = np.linspace(start = Interval_Left, stop = Interval_Right, num = N_Grid),
            X_Original          = X_US_History[_i],
            ConstantPriorGPMean = c_PriorMean,
            KernelFunction      = BestKernelChoice_kernel,
            **BestKernelChoice_hyperparameters
        )
        if x_New is None:
            X_US_History.append(X_US_History[_i].copy())
            X_New_Point_History.append(None)
        else:
            X_US_History.append(np.sort(np.append(X_US_History[_i], x_New)))
            X_New_Point_History.append(x_New)

elif Mode == 'UncertaintySampling_reMLE':
    X_US_History        = [X_Initial.copy()]          # X_US_History[i] = observation grid at frame i
    X_New_Point_History = [None]                      # point added going into frame i (None for frame 0)
    Kernel_History      = [BestKernelChoice_kernel]   # kernel used for GP at frame i
    Hp_History          = [BestKernelChoice_hyperparameters]  # hyperparameters at frame i
    Label_History       = [BestKernelChoice_label]    # kernel label at frame i

    PreviousResult_US = None  # Warm start for KernelSelectionML_1D across iterations

    for _i in range(N_End - N_Start):
        x_New = UncertaintySampling(
            X_Candidates        = np.linspace(start = Interval_Left, stop = Interval_Right, num = N_Grid),
            X_Original          = X_US_History[_i],
            ConstantPriorGPMean = c_PriorMean,
            KernelFunction      = Kernel_History[_i],
            **Hp_History[_i]
        )
        if x_New is None:
            X_US_History.append(X_US_History[_i].copy())
            X_New_Point_History.append(None)
        else:
            X_US_History.append(np.sort(np.append(X_US_History[_i], x_New)))
            X_New_Point_History.append(x_New)

        ### Re-run KernelSelectionML_1D on the updated observation grid
        UpdatedSelection = KernelSelectionML_1D(X_US_History[_i + 1], Integrand_f(X_US_History[_i + 1]),
                                                PreviousResult = PreviousResult_US)
        Kernel_History.append(UpdatedSelection['Kernel'])
        Hp_History.append(UpdatedSelection['Hyperparameters'])
        Label_History.append(UpdatedSelection['Label'])
        PreviousResult_US = UpdatedSelection  # Update warm start for next iteration




#=======================================================================================================================
#-------------------------------------------------- Animation Loop -----------------------------------------------------
#=======================================================================================================================
### Create frames folder
Frames_Folder = 'Frames'
os.makedirs(Frames_Folder, exist_ok = True)

N_Frames          = N_End - N_Start + 1
HorizontalAxis_GP = np.linspace(start = Interval_Left, stop = Interval_Right, num = N_Grid)
Frame_Paths       = []

### Warm start for reMLE mode — carries the best kernel result from frame to frame
PreviousResult_reMLE = None

for i in range(N_Frames):
    n = N_Start + i

    #------------------------------ Build Current Observation Grid --------------------------------
    if Mode == 'reMLE':
        X_Curr = np.linspace(start = Interval_Left, stop = Interval_Right, num = n)
        f_Curr = Integrand_f(X_Curr)
    elif Mode in ('UncertaintySampling', 'UncertaintySampling_reMLE'):
        X_Curr = X_US_History[i]
        f_Curr = Integrand_f(X_Curr)

    #------------------------------- Update Kernel / Hyperparameters ------------------------------
    if Mode == 'reMLE':
        Selection_Curr       = KernelSelectionML_1D(X_Curr, f_Curr, PreviousResult = PreviousResult_reMLE)
        Kernel_Curr          = Selection_Curr['Kernel']
        Hp_Curr              = Selection_Curr['Hyperparameters']
        Label_Curr           = Selection_Curr['Label']
        PreviousResult_reMLE = Selection_Curr  # Update warm start for next frame

    elif Mode == 'UncertaintySampling':
        Kernel_Curr = BestKernelChoice_kernel
        Hp_Curr     = BestKernelChoice_hyperparameters
        Label_Curr  = BestKernelChoice_label

    elif Mode == 'UncertaintySampling_reMLE':
        Kernel_Curr = Kernel_History[i]
        Hp_Curr     = Hp_History[i]
        Label_Curr  = Label_History[i]

    #--------------------------------- GP Posterior on Evaluation Grid ----------------------------
    m_fPosterior, k_fPosterior = GP_Posterior(Kernel_Curr, X_Curr, f_Curr, c = c_PriorMean, **Hp_Curr)

    MeanVector_Grid     = m_fPosterior(HorizontalAxis_GP)
    CovMatrix_Grid      = k_fPosterior(HorizontalAxis_GP, HorizontalAxis_GP)
    CovMatrix_Grid      += 1e-6 * np.eye(N_Grid)  # Add jitter
    VarianceVector_Grid = np.diag(CovMatrix_Grid)

    ### Draw posterior sample functions
    rng       = np.random.default_rng(Random_Seed)
    f_Samples = rng.multivariate_normal(mean = MeanVector_Grid, cov = CovMatrix_Grid,
                                        size = N_Samples, method = 'cholesky')
    Integral_Samples = [np.trapezoid(f_, HorizontalAxis_GP) for f_ in f_Samples]

    #--------------------------------- BQ Posterior Distribution ----------------------------------
    PostMean, PostVar = BQ_1D_Approximations(
        X_Input             = X_Curr,
        f_X_Input           = f_Curr,
        ConstantPriorGPMean = c_PriorMean,
        KernelFunction      = Kernel_Curr,
        **Hp_Curr
    )
    PostStd  = np.sqrt(PostVar)
    AbsError = np.abs(TrueIntegralValue - PostMean)

    Pdf_Range  = np.linspace(PostMean - 4 * PostStd, PostMean + 4 * PostStd, num = N_Grid)
    Pdf_Values = GaussianPDF_1D(Pdf_Range, mu = PostMean, sigma2 = PostVar)

    #-------------------------------------- Two-Panel Figure --------------------------------------
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 5))

    ### Left panel — GP posterior sample functions
    ax1.plot(HorizontalAxis_GP, Integrand_f(HorizontalAxis_GP), color = 'dodgerblue', label = r"$f(x)$")
    Sample_Colors = ['dodgerblue', 'darkorange', 'mediumseagreen']
    for idx, f_ in enumerate(f_Samples):
        ax1.plot(HorizontalAxis_GP, f_, alpha = 0.5, color = Sample_Colors[idx])
    ax1.plot(HorizontalAxis_GP, MeanVector_Grid,
             color = 'C4', linestyle = '--', label = "Posterior mean")
    ax1.fill_between(HorizontalAxis_GP,
                     MeanVector_Grid - np.sqrt(VarianceVector_Grid),
                     MeanVector_Grid + np.sqrt(VarianceVector_Grid),
                     color = 'cyan', linestyle = '--', alpha = 0.21, label = r"$\pm \sigma$")
    ax1.fill_between(HorizontalAxis_GP,
                     MeanVector_Grid - 1 * np.sqrt(VarianceVector_Grid),
                     MeanVector_Grid - 2 * np.sqrt(VarianceVector_Grid),
                     color = 'gray', linestyle = '--', alpha = 0.2, label = r"$\pm 2\sigma$")
    ax1.fill_between(HorizontalAxis_GP,
                     MeanVector_Grid + 1 * np.sqrt(VarianceVector_Grid),
                     MeanVector_Grid + 2 * np.sqrt(VarianceVector_Grid),
                     color = 'gray', linestyle = '--', alpha = 0.2)
    ax1.plot(X_Curr, f_Curr, linestyle = 'None', marker = 'o', markersize = 8,
             color = 'magenta', label = "Observations", zorder = N_Samples + 2)
    if Mode in ('UncertaintySampling', 'UncertaintySampling_reMLE') and i > 0 and X_New_Point_History[i] is not None:
        X_New_This = X_New_Point_History[i]
        ax1.plot(X_New_This, Integrand_f(X_New_This), linestyle = 'None', marker = 'D',
                 markersize = 8, color = 'lime', label = "New Observation", zorder = N_Samples + 3)
    ax1.set_title(rf"$\mathcal{{GP}}$({c_PriorMean:.1f}, {Label_Curr}) Posterior Samples, n = {n}")
    ax1.set_xlabel(r"$x_i$")
    ax1.set_ylabel(r"$f(x_i)$")
    ax1.set_xlim((Interval_Left, Interval_Right))
    ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    ax1.legend(loc = 'upper right')  # loc='lower center' when domain [0,3] & 'upper right' when [0,5]

    ### Middle panel — BQ posterior distribution
    ax2.plot(Pdf_Range, Pdf_Values)
    ax2.axvline(PostMean, color = 'C4', linewidth = 2, label = "Integral of Posterior Mean")
    ax2.axvline(TrueIntegralValue, color = 'lightgreen', linewidth = 2, label = "True Integral Value")
    ax2.set_title(rf"|Error| = {AbsError:.3f} & $\sqrt{{k_{{I,\curvearrowright}}}}$ = {PostStd:.3f}, n = {n}")
    ax2.set_xlabel(r"Integral Value")
    ax2.set_ylabel(r"Density")
    ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
    ax2.legend(loc = 'upper left')

    plt.tight_layout()

    #--------------------------------------- Save Frame -------------------------------------------
    Frame_Path = os.path.join(Frames_Folder, f"frame_{Mode}_IntervalRight={Interval_Right}_nGrid={N_Grid}_nStart="
                                             f"{N_Start}_nEnd={N_End}_{i:03d}.png")
    plt.savefig(Frame_Path, dpi = 150)
    plt.close(fig)
    Frame_Paths.append(Frame_Path)
    print(f"Frame {i + 1}/{N_Frames} completed")




#=======================================================================================================================
#-------------------------------------------------- Assemble GIF -------------------------------------------------------
#=======================================================================================================================
Images = [imageio.imread(fp) for fp in Frame_Paths]
imageio.mimsave(Gif_Filename, Images, duration = Frame_Duration, loop = 0)
print(f"GIF saved to {Gif_Filename}")