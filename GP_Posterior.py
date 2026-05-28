########################################################################################################################
############################################## GAUSSIAN PROCESS POSTERIOR ##############################################
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt


#=======================================================================================================================
#------------------------------------------------------ DATA POINTS ----------------------------------------------------
#=======================================================================================================================
X = np.array([1, 4, 9, 11, 15, 19])   # Generate a dataset
f_X = np.array([-3, 7, 5, -11, 2, 15])

plt.figure(figsize = (8, 5))
plt.scatter(X, f_X, c = 'magenta', s = 100)
plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.title(r"GP Regression $-$ Data Points")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
plt.tight_layout()  # Automatically determines the necessary spacing
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


#----------------------------------- Plot Data Points & Sample Functions from GP Prior ---------------------------------
plt.figure(figsize = (8, 5))
plt.scatter(X, f_X, c = 'magenta', s = 100)  # Plot data points

# Next, plot GP prior sample functions
rng = np.random.default_rng(42)  # Reproducibility of randomness
N = 1000  # Number of points to evaluate sample functions from GP at
HorizontalAxis_GP = np.linspace(start = 0, stop = 20, num = N)

# Define (zero) mean function of GP prior
def m_Prior(x):
    return 0 * np.array(x)

# Set hyperparameters, by trial-and-error
l_Hyperparameter = 1
sigma2_Hyperparameter = 10
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
plt.title(r"GP Regression $-$ Data Points & GP Prior Sample Functions")
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

    L = np.linalg.cholesky(K_XX)  # Cholesky decomposition, K_{XX} = L L^T

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


#------------------------------------------ Sample Functions from GP Posterior -----------------------------------------
rng = np.random.default_rng(42)  # Reproducibility of randomness
N = 1000  # Number of points to evaluate sample functions from GP at
HorizontalAxis_GP = np.linspace(start = 0, stop = 20, num = N)
# Setting parameters by trial-and-error
m_fPosterior, k_fPosterior = GP_Posterior(Kernel_SE, X, f_X, l =  l_Hyperparameter, sigma2 = sigma2_Hyperparameter)

MeanVector_Posterior = m_fPosterior(HorizontalAxis_GP)  # Posterior mean evaluated at grid
KernelMatrix_Posterior = k_fPosterior(HorizontalAxis_GP, HorizontalAxis_GP)  # Posterior covariance evaluated at grid
KernelMatrix_Posterior += 1e-6 * np.eye(N)  # Add jitter

### Plot the data points and some Gaussian prior sample functions
plt.figure(figsize = (14, 8))
n_samples = 4  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_N))^T from finite Gaussian distribution, for n_samples different samples
f_samples = rng.multivariate_normal(mean = MeanVector_Posterior, cov = KernelMatrix_Posterior, size = n_samples,
                                    method = 'cholesky')
for f in f_samples:
    plt.plot(HorizontalAxis_GP, f, alpha = 0.6)

plt.plot(HorizontalAxis_GP, MeanVector_Posterior, linestyle = '--', label = "Posterior mean")

VarianceGP_Posterior = np.diag(KernelMatrix_Posterior)  # Variance per evaluation point of posterior sample
plt.fill_between(HorizontalAxis_GP,  # +/- 1*SD regions
                 MeanVector_Posterior - np.sqrt(VarianceGP_Posterior),
                 MeanVector_Posterior + np.sqrt(VarianceGP_Posterior),
                 color = 'cyan', linestyle = '--', alpha = 0.12, label = r"$\pm \sigma$")

plt.fill_between(HorizontalAxis_GP,  # +/- 2*SD regions (below +/- 1*SD region)
                 MeanVector_Posterior - 1 * np.sqrt(VarianceGP_Posterior),
                 MeanVector_Posterior - 2 * np.sqrt(VarianceGP_Posterior),
                 color = 'lightgrey', linestyle = '--', alpha = 0.2, label = r"$\pm 2\sigma$")
plt.fill_between(HorizontalAxis_GP,  # +/- 2*SD regions (above +/- 1*SD region)
                 MeanVector_Posterior + 1 * np.sqrt(VarianceGP_Posterior),
                 MeanVector_Posterior + 2 * np.sqrt(VarianceGP_Posterior),
                 color = 'lightgrey', linestyle = '--', alpha = 0.2)

plt.scatter(X, f_X, c = 'magenta', s = 100)  # Plot data points

plt.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
plt.xlim((0, 20))
plt.legend()
plt.title(r"GP Regression $-$ Data Points & GP Posterior Sample Functions with $\pm \sigma$, $\pm 2\sigma$")
plt.xlabel(r"$x_i$")
plt.ylabel(r"$f(x_i)$")
#plt.tight_layout()
plt.show()