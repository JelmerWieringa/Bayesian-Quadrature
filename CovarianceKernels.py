########################################################################################################################
################################################## COVARIANCE KERNELS ##################################################
########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt


#=======================================================================================================================
#---------------------------------------------- SQUARED-EXPONENTIAL KERNEL ---------------------------------------------
#=======================================================================================================================
def Kernel_SE(x, y, l, sigma2):  # SE/Gaussian/RBF Kernel function
    Norm_2 = (x - y)**2  # Squared Euclidean norm of x - y
    return (sigma2**2) * np.exp( -Norm_2 / (2 * l**2))  # Scalar

def KernelMatrix(KernelFunction, x, y, **parameters):
    """Return K_{ij} = [KernelFunction(x_i, y_j)]_{i,j=1}^n."""
    x = x[:, None]  # Changes to a column vector, dim=nx1
    y = y[None, :]  # Changes to a row vector, dim=1xn
    return KernelFunction(x, y, **parameters)
# Note: **parameters allows an arbitrary number of (keyword) arguments to functions


#------------------------------------------------ Heatmap & Slice Plot -------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 4))

### Left Plot
HeatMap_Axis = np.linspace(start = 0, stop = 10, num = 200)
HeatMapData_Left = KernelMatrix(Kernel_SE, HeatMap_Axis, HeatMap_Axis, l = 0.5, sigma2 = 1)
HeatMap_Left = ax1.imshow(HeatMapData_Left, cmap = 'cool', origin = 'lower')  # Colour image
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html
fig.colorbar(HeatMap_Left, ax = ax1)
ax1.set_title(r"$k_{SE}$ Heatmap, $l = 0.5$")
ax1.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax1.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Middle Plot
HeatMapData_Middle = KernelMatrix(Kernel_SE, HeatMap_Axis, HeatMap_Axis, l = 3, sigma2 = 1)
HeatMap_Middle = ax2.imshow(HeatMapData_Middle, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Middle, ax = ax2)
ax2.set_title(r"$k_{SE}$ Heatmap, $l = 3$")
ax2.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax2.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Right Plot
HorizontalAxis = np.linspace(start = -5, stop = 10, num = 1000)
ax3.plot(HorizontalAxis, Kernel_SE(HorizontalAxis, y = 0, l = 1, sigma2 = 1), label = r"$\tilde{x} = 0$, $l = 1$")
ax3.plot(HorizontalAxis, Kernel_SE(HorizontalAxis, y = 0, l = 2, sigma2 = 1), label = r"$\tilde{x} = 0$, $l = 2$")
ax3.plot(HorizontalAxis, Kernel_SE(HorizontalAxis, y = 4, l = 1, sigma2 = 1), label = r"$\tilde{x} = 4$, $l = 1$")
ax3.plot(HorizontalAxis, Kernel_SE(HorizontalAxis, y = 4, l = 4, sigma2 = 1), label = r"$\tilde{x} = 4$, $l = 3$")
ax3.legend()
ax3.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax3.set_xlim((-5, 10))
ax3.set_title(r"$k_{SE}$ Slice Plot")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$k_{SE}(x, \tilde{x} \mid 1, l)$")

plt.tight_layout()
plt.show()


#------------------------------------------- Sample Functions from GP Prior --------------------------------------------
rng = np.random.default_rng(42)  # Reproducibility of randomness
n = 1000  # Number of function evaluations
HorizontalAxis_GP = np.linspace(start = 0, stop = 10, num = n)
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
KernelMatrix_SE = KernelMatrix(Kernel_SE, HorizontalAxis_GP, HorizontalAxis_GP, l = 0.5, sigma2 = 1)
KernelMatrix_SE += 1e-6 * np.eye(n)  # np.eye() gives an identity matrix
# This adds a jitter to the kernel matrix to ensure numerical stability
m = np.zeros(n)  # Zero mean function of GP

n_samples = 5  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_n))^T from finite Gaussian distribution, for 5 different samples
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_SE, size = n_samples, method = 'cholesky')
for f in f_samples_Left:
    ax1.plot(HorizontalAxis_GP, f)
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 10))
ax1.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{SE})$, $l = 0.5$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")

### Right Plot
KernelMatrix_SE = KernelMatrix(Kernel_SE, HorizontalAxis_GP, HorizontalAxis_GP, l = 3, sigma2 = 1)
KernelMatrix_SE += 1e-6 * np.eye(n)
m = np.zeros(n)

n_samples = 5
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_SE, size = n_samples, method = 'cholesky')
for f in f_samples_Right:
    ax2.plot(HorizontalAxis_GP, f)
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 10))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{SE})$, $l = 3$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()




#=======================================================================================================================
#----------------------------------------------------- MATERN KERNEL ---------------------------------------------------
#=======================================================================================================================
def Kernel_Matern_1_2(x, y, l, sigma2):
    Norm = np.sqrt((x - y)**2)  # Euclidean norm of x - y
    return (sigma2**2) * np.exp(-Norm / l)

def Kernel_Matern_3_2(x, y, l, sigma2):
    Norm = np.sqrt((x - y)**2)  # Euclidean norm of x - y
    return (sigma2**2) * (1 + np.sqrt(3) * Norm / l) * np.exp(-np.sqrt(3) * Norm / l)

def Kernel_Matern_5_2(x, y, l, sigma2):
    Norm = np.sqrt((x - y)**2)  # Euclidean norm of x - y
    return (sigma2**2) * (1 + np.sqrt(5) * Norm / l + 5 * Norm**2 / (3 * l**2)) * np.exp(-np.sqrt(5) * Norm / l)


#------------------------------------------------ Heatmap & Slice Plot -------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 4))

### Left Plot
HeatMap_Axis = np.linspace(start = 0, stop = 10, num = 200)
HeatMapData_Left = KernelMatrix(Kernel_Matern_3_2, HeatMap_Axis, HeatMap_Axis, l = 0.5, sigma2 = 1)
HeatMap_Left = ax1.imshow(HeatMapData_Left, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Left, ax = ax1)
ax1.set_title(r"$k_M$ Heatmap, $\alpha = 3/2$, $l = 0.5$")
ax1.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax1.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Middle Plot
HeatMapData_Middle = KernelMatrix(Kernel_Matern_3_2, HeatMap_Axis, HeatMap_Axis, l = 3, sigma2 = 1)
HeatMap_Middle = ax2.imshow(HeatMapData_Middle, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Middle, ax = ax2)
ax2.set_title(r"$k_M$ Heatmap, $\alpha = 3/2$, $l = 3$")
ax2.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax2.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Right Plot
HorizontalAxis = np.linspace(start = -5, stop = 10, num = 1000)
ax3.plot(HorizontalAxis, Kernel_Matern_3_2(HorizontalAxis, y = 0, l = 1, sigma2 = 1), label = r"$\tilde{x} = 0$, $l = 1$")
ax3.plot(HorizontalAxis, Kernel_Matern_3_2(HorizontalAxis, y = 0, l = 2, sigma2 = 1), label = r"$\tilde{x} = 0$, $l = 2$")
ax3.plot(HorizontalAxis, Kernel_Matern_3_2(HorizontalAxis, y = 4, l = 1, sigma2 = 1), label = r"$\tilde{x} = 4$, $l = 1$")
ax3.plot(HorizontalAxis, Kernel_Matern_3_2(HorizontalAxis, y = 4, l = 3, sigma2 = 1), label = r"$\tilde{x} = 4$, $l = 3$")
ax3.legend()
ax3.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax3.set_xlim((-5, 10))
ax3.set_title(r"$k_M$ Slice Plot")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$k_M(x, \tilde{x} \mid 1, 3/2, l)$")

plt.tight_layout()
plt.show()


#------------------------------------------- Sample Functions from GP Prior --------------------------------------------
rng = np.random.default_rng(42)
n = 1000  # Number of function evaluations
n_samples = 2  # Number of sample functions (to draw) per kernel
m = np.zeros(n)  # Zero mean function of GP
HorizontalAxis_GP = np.linspace(start = 0, stop = 10, num = n)
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
KernelMatrix_Matern_1_2 = KernelMatrix(Kernel_Matern_1_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 0.5, sigma2 = 1)
KernelMatrix_Matern_1_2 += 1e-6 * np.eye(n)  # Add jitter
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_Matern_1_2, size = n_samples, method = 'cholesky')
for i, f in enumerate(f_samples_Left):  # Plot for 1/2
    ax1.plot(HorizontalAxis_GP, f, color = 'orange', label=(r"$\alpha = 1/2$" if i == 0 else "_nolegend_"))
# Only the first plot gets a label, otherwise too many labels are printed in the legend

KernelMatrix_Matern_3_2 = KernelMatrix(Kernel_Matern_3_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 0.5, sigma2 = 1)
KernelMatrix_Matern_3_2 += 1e-6 * np.eye(n)  # Add jitter
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_Matern_3_2, size = n_samples, method = 'cholesky')
for i, f in enumerate(f_samples_Left):  # Plot for 3/2
    ax1.plot(HorizontalAxis_GP, f, color = 'blue', label=(r"$\alpha = 3/2$" if i == 0 else "_nolegend_"))
KernelMatrix_Matern_5_2 = KernelMatrix(Kernel_Matern_5_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 0.5, sigma2 = 1)
KernelMatrix_Matern_5_2 += 1e-6 * np.eye(n)  # Add jitter
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_Matern_5_2, size = n_samples, method = 'cholesky')
for i, f in enumerate(f_samples_Left):  # Plot for 5/2
    ax1.plot(HorizontalAxis_GP, f, color = 'magenta', label=(r"$\alpha = 5/2$" if i == 0 else "_nolegend_"))
ax1.legend()
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 10))
ax1.set_title(r"Samples Functions from $\mathcal{GP}(0, k_M)$, $l = 0.5$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")

### Right Plot
KernelMatrix_Matern_1_2 = KernelMatrix(Kernel_Matern_1_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 3, sigma2 = 1)
KernelMatrix_Matern_1_2 += 1e-6 * np.eye(n)  # Add jitter
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_Matern_1_2, size = n_samples, method = 'cholesky')
for i, f in enumerate(f_samples_Right):  # Plot for 1/2
    ax2.plot(HorizontalAxis_GP, f, color = 'orange', label=(r"$\alpha = 1/2$" if i == 0 else "_nolegend_"))

KernelMatrix_Matern_3_2 = KernelMatrix(Kernel_Matern_3_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 3, sigma2 = 1)
KernelMatrix_Matern_3_2 += 1e-6 * np.eye(n)  # Add jitter
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_Matern_3_2, size = n_samples, method = 'cholesky')
for i, f in enumerate(f_samples_Right):  # Plot for 3/2
    ax2.plot(HorizontalAxis_GP, f, color = 'blue', label=(r"$\alpha = 3/2$" if i == 0 else "_nolegend_"))
KernelMatrix_Matern_5_2 = KernelMatrix(Kernel_Matern_5_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 3, sigma2 = 1)
KernelMatrix_Matern_5_2 += 1e-6 * np.eye(n)  # Add jitter
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_Matern_5_2, size = n_samples, method = 'cholesky')
for i, f in enumerate(f_samples_Right):  # Plot for 5/2
    ax2.plot(HorizontalAxis_GP, f, color = 'magenta', label=(r"$\alpha = 5/2$" if i == 0 else "_nolegend_"))
ax2.legend()
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 10))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_M)$, $l = 3$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()




#=======================================================================================================================
#----------------------------------------------- RATIONAL QUADRATIC KERNEL ---------------------------------------------
#=======================================================================================================================
def Kernel_RQ(x, y, l, alpha, sigmaf2, **_):  # Rational Quadratic (RQ) Kernel function
    Norm2 = (x - y)**2
    return (sigmaf2**2) * (1 + Norm2 / (2 * alpha * l**2))**(-alpha)


#------------------------------------------------ Heatmap & Slice Plot -------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 4))

### Left Plot
HeatMap_Axis = np.linspace(start = 0, stop = 10, num = 200)
HeatMapData_Left = KernelMatrix(Kernel_RQ, HeatMap_Axis, HeatMap_Axis, l = 1, alpha = 1, sigmaf2 = 1)
HeatMap_Left = ax1.imshow(HeatMapData_Left, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Left, ax = ax1)
ax1.set_title(r"$k_{RQ}$ Heatmap, $l = 1$, $\alpha = 1$")
ax1.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax1.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Middle Plot
HeatMapData_Middle = KernelMatrix(Kernel_RQ, HeatMap_Axis, HeatMap_Axis, l = 1, alpha = 10, sigmaf2 = 1)
HeatMap_Middle = ax2.imshow(HeatMapData_Middle, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Middle, ax = ax2)
ax2.set_title(r"$k_{RQ}$ Heatmap, $l = 1$, $\alpha = 10$")
ax2.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax2.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Right Plot
HorizontalAxis = np.linspace(start = -5, stop = 10, num = 1000)
ax3.plot(HorizontalAxis, Kernel_RQ(HorizontalAxis, y = 0, alpha = 1, l = 0.5, sigmaf2 = 1),
         label = r"$\tilde{x} = 0$, $l = 0.5$, $\alpha = 1$)")
ax3.plot(HorizontalAxis, Kernel_RQ(HorizontalAxis, y = 0, alpha = 3, l = 0.5, sigmaf2 = 1),
         label = r"$\tilde{x} = 0$, $l = 0.5$, $\alpha = 3$)")
ax3.plot(HorizontalAxis, Kernel_RQ(HorizontalAxis, y = 4, alpha = 1, l = 2, sigmaf2 = 1),
         label = r"$\tilde{x} = 4$, $l = 2$, $\alpha = 1$)")
ax3.plot(HorizontalAxis, Kernel_RQ(HorizontalAxis, y = 4, alpha = 3, l = 2, sigmaf2 = 1),
         label = r"$\tilde{x} = 4$, $l = 2$, $\alpha = 3$)")
ax3.legend()
ax3.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax3.set_xlim((-5, 10))
ax3.set_title(r"$k_{RQ}$ Slice Plot")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$k_M(x, \tilde{x} \mid 1, l, \alpha)$")

plt.tight_layout()
plt.show()


#------------------------------------------- Sample Functions from GP Prior --------------------------------------------
rng = np.random.default_rng(42)  # Reproducability of randomness
n = 1000  # Number of function evaluations
HorizontalAxis_GP = np.linspace(start = 0, stop = 10, num = n)

fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
KernelMatrix_RQ = KernelMatrix(Kernel_RQ, HorizontalAxis_GP, HorizontalAxis_GP, l = 1, alpha = 1, sigmaf2 = 1)
KernelMatrix_RQ += 1e-6 * np.eye(n)  # np.eye() gives an identity matrix
# This adds a jitter to the kernel matrix to ensure numerical stability
m = np.zeros(n)  # Zero mean function of GP

n_samples = 5  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_n))^T from finite Gaussian distribution, for 5 different samples
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_RQ, size = n_samples, method = 'cholesky')
for f in f_samples_Left:
    ax1.plot(HorizontalAxis_GP, f)
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 10))
ax1.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{RQ})$, $l = 1$, $\alpha = 1$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")


### Right Plot
KernelMatrix_RQ = KernelMatrix(Kernel_RQ, HorizontalAxis_GP, HorizontalAxis_GP, l = 1, alpha = 10, sigmaf2 = 1)
KernelMatrix_RQ += 1e-6 * np.eye(n)
m = np.zeros(n)

n_samples = 5
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_RQ, size = n_samples, method = 'cholesky')
for f in f_samples_Right:
    ax2.plot(HorizontalAxis_GP, f)
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 10))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{RQ})$, $l = 1$, $\alpha = 10$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()




#=======================================================================================================================
#--------------------------------------------------- POLYNOMIAL KERNEL -------------------------------------------------
#=======================================================================================================================
def Kernel_Polynomial(x, y, p, c, sigma2):  # Polynomial Kernel function
    x = np.asarray(x)
    y = np.asarray(y)
    #return sigma2 * (x @ y + c)**p, or, return sigma2 * (np.dot(x,y) + c)**p
    # Note: np.dot() and @ yields te same, but I want it only for scalars here. If the input are vectors, then the
    #       inner product is computed instead of the kernel componentwise.
    return sigma2 * (x * y + c)**p


#------------------------------------------------ Heatmap & Slice Plot -------------------------------------------------
fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(1, 3, 1)                     # Heatmap
ax2 = fig.add_subplot(1, 3, 2)                     # Heatmap
ax3 = fig.add_subplot(1, 3, 3, projection='3d')    # 3D surface

### Left Plot
HeatMap_Axis = np.linspace(start = 0, stop = 10, num = 200)
HeatMapData_Left = KernelMatrix(Kernel_Polynomial, HeatMap_Axis, HeatMap_Axis, p = 0.5, c = 0, sigma2 = 1)
HeatMap_Left = ax1.imshow(HeatMapData_Left, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Left, ax = ax1)
ax1.set_title(r"$k_P$ Heatmap, $p = 0.5$, $c = 0$")
ax1.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax1.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Middle Plot
HeatMapData_Middle = KernelMatrix(Kernel_Polynomial, HeatMap_Axis, HeatMap_Axis, p = 3, c = 0, sigma2 = 1)
HeatMap_Middle = ax2.imshow(HeatMapData_Middle, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Middle, ax = ax2)
ax2.set_title(r"$k_P$ Heatmap, $p = 3$, $c = 0$")
ax2.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax2.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Right Plot
HorizontalAxis = np.linspace(start = -100, stop = 100, num = 1000)
X, Y = np.meshgrid(HorizontalAxis, HorizontalAxis)
Z = Kernel_Polynomial(X, Y, p = 2, c = 0, sigma2 = 1)
surf = ax3.plot_surface(X, Y, Z, cmap = 'cool')
ax3.set_title(r"$k_P$ 3D Plot, $p=2$, $c = 0$")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$\tilde{x}$")
ax3.set_zlabel(r"k_P(x, \tilde{x} \mid 1, p, c)")

plt.tight_layout()
plt.show()


#------------------------------------------- Sample Functions from GP Prior --------------------------------------------
rng = np.random.default_rng(42)  # Reproducibility of randomness
n = 1000  # Number of function evaluations
HorizontalAxis_GP = np.linspace(start = 0, stop = 20, num = n)
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
KernelMatrix_Polynomial = KernelMatrix(Kernel_Polynomial, HorizontalAxis_GP, HorizontalAxis_GP,
                                       p = 0.5, c = 0, sigma2 = 1)
KernelMatrix_Polynomial += 1e-6 * np.eye(n)  # Add Jitter
m = np.zeros(n)  # Zero mean function of GP

n_samples = 5  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_n))^T from finite Gaussian distribution, for 5 different samples
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_Polynomial, size = n_samples, method = 'cholesky')
for f in f_samples_Left:
    ax1.plot(HorizontalAxis_GP, f)
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 20))
ax1.set_title(r"Samples Functions from $\mathcal{GP}(0, k_P)$, $p = 0.5$, $c = 0$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")


### Right Plot
KernelMatrix_Polynomial = KernelMatrix(Kernel_Polynomial, HorizontalAxis_GP, HorizontalAxis_GP, p = 3, c = 0, sigma2 = 1)
KernelMatrix_Polynomial += 1e-6 * np.eye(n)
m = np.zeros(n)

n_samples = 5
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_Polynomial, size = n_samples, method = 'cholesky')
for f in f_samples_Right:
    ax2.plot(HorizontalAxis_GP, f)
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 20))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_P)$, $p = 3$, $c = 0$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()




#=======================================================================================================================
#--------------------------------------------------- PERIODIC KERNEL ---------------------------------------------------
#=======================================================================================================================
def Kernel_Periodic(x, y, lmbd, l, sigma2):  # # Periodic Kernel function, with scalar inputs only
    PeriodicTerm = np.sin(np.pi * (x - y) / lmbd)
    return sigma2 * np.exp(-(PeriodicTerm**2) / (2 * l**2))


#------------------------------------------------ Heatmap & Slice Plot -------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (14, 4))

### Left Plot
HeatMap_Axis = np.linspace(start = 0, stop = 10, num = 200)
HeatMapData_Left = KernelMatrix(Kernel_Periodic, HeatMap_Axis, HeatMap_Axis, lmbd = 2, l = 0.5, sigma2 = 1)
HeatMap_Left = ax1.imshow(HeatMapData_Left, cmap = 'cool', origin = 'lower')  # Colour image
# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html
fig.colorbar(HeatMap_Left, ax = ax1)
ax1.set_title(r"$k_{\pi}$ Heatmap, $l = 0.5$")
ax1.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax1.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Middle Plot
HeatMapData_Middle = KernelMatrix(Kernel_Periodic, HeatMap_Axis, HeatMap_Axis, lmbd = 2, l = 3, sigma2 = 1)
HeatMap_Middle = ax2.imshow(HeatMapData_Middle, cmap = 'cool', origin = 'lower')  # Colour image
fig.colorbar(HeatMap_Middle, ax = ax2)
ax2.set_title(r"$k_{\pi}$ Heatmap, $l = 3$")
ax2.set_xlabel(r"$x \mapsto k(x, \tilde{x})$")
ax2.set_ylabel(r"$\tilde{x} \mapsto k(x, \tilde{x})$")

### Right Plot
HorizontalAxis = np.linspace(start = -5, stop = 10, num = 1000)
ax3.plot(HorizontalAxis, Kernel_Periodic(HorizontalAxis, y = 0, lmbd = 2, l = 1, sigma2 = 1),
         label = r"$\tilde{x} = 0$, $\lambda = 2$, $l = 1$")
ax3.plot(HorizontalAxis, Kernel_Periodic(HorizontalAxis, y = 0, lmbd = 2, l = 2, sigma2 = 1),
         label = r"$\tilde{x} = 0$, $\lambda = 2$, $l = 2$")
ax3.plot(HorizontalAxis, Kernel_Periodic(HorizontalAxis, y = 4, lmbd = 5, l = 1, sigma2 = 1),
         label = r"$\tilde{x} = 4$, $\lambda = 5$, $l = 1$")
ax3.plot(HorizontalAxis, Kernel_Periodic(HorizontalAxis, y = 4, lmbd = 5, l = 2, sigma2 = 1),
         label = r"$\tilde{x} = 4$, $\lambda = 5$, $l = 2$")
ax3.legend()
ax3.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax3.set_xlim((-5, 10))
ax3.set_title(r"$k_{\pi}$ Slice Plot")
ax3.set_xlabel(r"$x$")
ax3.set_ylabel(r"$k_{\pi}(x, \tilde{x} \mid 1, \lambda, l)$")

plt.tight_layout()
plt.show()


#------------------------------------------- Sample Functions from GP Prior --------------------------------------------
rng = np.random.default_rng(42)  # Reproducibility of randomness
n = 1000  # Number of function evaluations
HorizontalAxis_GP = np.linspace(start = 0, stop = 10, num = n)
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
KernelMatrix_Periodic = KernelMatrix(Kernel_Periodic, HorizontalAxis_GP, HorizontalAxis_GP, lmbd = 2, l = 0.5, sigma2 = 1)
KernelMatrix_Periodic += 1e-6 * np.eye(n)  # Add Jitter
m = np.zeros(n)  # Zero mean function of GP

n_samples = 5  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_n))^T from finite Gaussian distribution, for 5 different samples
f_samples_Left = rng.multivariate_normal(mean = m, cov = KernelMatrix_Periodic, size = n_samples, method = 'cholesky')
for f in f_samples_Left:
    ax1.plot(HorizontalAxis_GP, f)
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((0, 10))
ax1.set_title(r"Samples Functions from $\mathcal{GP}(0, k_P)$, $\lambda = 2$, $l = 0.5$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$f(x)$")


### Right Plot
KernelMatrix_Periodic = KernelMatrix(Kernel_Periodic, HorizontalAxis_GP, HorizontalAxis_GP, lmbd = 2, l = 3, sigma2 = 1)
KernelMatrix_Periodic += 1e-6 * np.eye(n)
m = np.zeros(n)

n_samples = 5
f_samples_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_Periodic, size = n_samples, method = 'cholesky')
for f in f_samples_Right:
    ax2.plot(HorizontalAxis_GP, f)
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 10))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_P)$, $\lambda = 2$, $l = 3$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()




#=======================================================================================================================
#--------------------------------------------------- COMBINING KERNELS -------------------------------------------------
#=======================================================================================================================

#--------------------------------------------------- Sum of Kernels ----------------------------------------------------
####### Slice Plot & Sample Functions from GP Prior
rng = np.random.default_rng(42)  # Reproducibility of randomness
n = 1000  # Number of function evaluations
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
HorizontalAxis = np.linspace(start = -10, stop = 10, num = 1000)
KernelSum_1 = (Kernel_Periodic(HorizontalAxis, y = 0, lmbd = 1, l = 0.5, sigma2 = 1) +
               Kernel_Matern_3_2(HorizontalAxis, y = 0, l = 0.5, sigma2 = 1))
ax1.plot(HorizontalAxis, KernelSum_1, label = r"$\tilde{x} = 0$, $\lambda = 1$, $l = 0.5$")
KernelSum_2 = (Kernel_Periodic(HorizontalAxis, y = 0, lmbd = 1, l = 3, sigma2 = 1) +
               Kernel_Matern_3_2(HorizontalAxis, y = 0, l = 3, sigma2 = 1))
ax1.plot(HorizontalAxis, KernelSum_2, label = r"$\tilde{x} = 0$, $\lambda = 1$, $l = 3$")
ax1.legend()
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((-10, 10))
ax1.set_title(r"$k_{\pi} + k_M$ Slice Plot")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$k_{\pi}(x, 0 \mid 1, \lambda, l) + k_M(x, 0 \mid 1, \alpha = 3/2, l)$")

### Right Plot
HorizontalAxis_GP = np.linspace(start = 0, stop = 10, num = n)
KernelMatrix_Periodic = KernelMatrix(Kernel_Periodic, HorizontalAxis_GP, HorizontalAxis_GP, lmbd = 1, l = 3, sigma2 = 1)
KernelMatrix_Matern_1_2 = KernelMatrix(Kernel_Matern_3_2, HorizontalAxis_GP, HorizontalAxis_GP, l = 3, sigma2 = 1)
KernelMatrix_PeriodicMatern = KernelMatrix_Periodic + KernelMatrix_Matern_1_2 + 1e-6 * np.eye(n)  # Add Jitter
m = np.zeros(n)  # Zero mean function of GP

n_samples = 3  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_n))^T from finite Gaussian distribution, for 5 different samples
f_samples = rng.multivariate_normal(mean = m, cov = KernelMatrix_PeriodicMatern, size = n_samples, method = 'svd')
for f in f_samples:
    ax2.plot(HorizontalAxis_GP, f)
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 10))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{\pi} + k_M)$, $\alpha = 3/2$, $\lambda = 1$, $l = 3$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()


#--------------------------------------------(Hadamard) Product of Kernels ---------------------------------------------
rng = np.random.default_rng(42)  # Reproducibility of randomness
n = 1000  # Number of function evaluations
fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (14, 4))

### Left Plot
HorizontalAxis = np.linspace(start = -10, stop = 10, num = 1000)
KernelSum_1 = Kernel_SE(HorizontalAxis, y = 0, l = 0.5, sigma2 = 1) * Kernel_SE(HorizontalAxis,
                                                                                y = 0, l = 0.5, sigma2 = 1)
ax1.plot(HorizontalAxis, KernelSum_1, label = r"$\tilde{x} = 0$, $l_1 = l_2 = 0.5$")
KernelSum_2 = Kernel_SE(HorizontalAxis, y = 0, l = 3, sigma2 = 1) * Kernel_SE(HorizontalAxis, y = 0, l = 3, sigma2 = 1)
ax1.plot(HorizontalAxis, KernelSum_2, label = r"$\tilde{x} = 0$, $l_1 = l_2 = 3$")
ax1.legend()
ax1.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax1.set_xlim((-10, 10))
ax1.set_title(r"$k_{SE} \odot k_{SE}$ Slice Plot")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$k_{SE}(x, 0 \mid 1, l_1) \odot k_{SE}(x, 0 \mid 1, l_2)$")

### Right Plot
HorizontalAxis_GP = np.linspace(start = 0, stop = 10, num = n)
KernelMatrix_SE = KernelMatrix(Kernel_SE, HorizontalAxis_GP, HorizontalAxis_GP, l = 0.5, sigma2 = 1)
KernelMatrix_SE = KernelMatrix(Kernel_SE, HorizontalAxis_GP, HorizontalAxis_GP, l = 3, sigma2 = 1)
# Compute the Hadamard product of the two Gram matrices
KernelMatrix_SESE = KernelMatrix_SE * KernelMatrix_SE + 1e-6 * np.eye(n)  # Add Jitter
m = np.zeros(n)  # Zero mean function of GP

n_samples = 3  # Number of sample functions (to draw)
# Next, draw f := (f(x_1),...,f(x_n))^T from finite Gaussian distribution, for 5 different samples
f_samples = rng.multivariate_normal(mean = m, cov = KernelMatrix_SESE, size = n_samples, method = 'svd')
for f in f_samples:
    ax2.plot(HorizontalAxis_GP, f)
ax2.grid(True, which = 'both', linestyle = '--', alpha = 0.5)
ax2.set_xlim((0, 10))
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{SE} \odot k_{SE})$, $l_1 = 0.5$, $l_2 = 3$")
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"$f(x)$")

plt.tight_layout()
plt.show()


#-------------------------------------------- Kernels with 2D-Vector Inputs --------------------------------------------
rng = np.random.default_rng(42)  # Reproducibility of randomness
fig = plt.figure(figsize=(14, 4))
ax1 = fig.add_subplot(1, 3, 1, projection = '3d')
ax2 = fig.add_subplot(1, 3, 2, projection = '3d')
ax3 = fig.add_subplot(1, 3, 3, projection = '3d')

### Left Plot
HorizontalAxis = np.linspace(start = -5, stop = 5, num = 1000)
X, Y = np.meshgrid(HorizontalAxis, HorizontalAxis)
Z = Kernel_SE(X, 0, l = 2, sigma2 = 1) * Kernel_SE(Y, 0, l = 2, sigma2 = 1)
surf = ax1.plot_surface(X, Y, Z, cmap = 'cool')
ax1.set_title(r"$k_{SE} \cdot k_{SE}$ 3D Slice Plot, $l = 2$")
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"$\tilde{x}$")
ax1.set_zlabel(r"$k_{SE}((x_1,0)^T, \mathbf{0}) \cdot k_{SE}(\mathbf{0}, (\tilde{x}_1, 0)^T)$")


### Middle Plot
n = 30  # Number of function evaluations (dim(X))
# First, compute the Gram matrix per axis
X_Inputs = np.linspace(start = -5, stop = 5, num = n)
Y_Inputs = np.linspace(start = -5, stop = 5, num = n)

KernelMatrix_SE_x1 = KernelMatrix(Kernel_SE, X_Inputs, Y_Inputs, l = 0.5, sigma2 = 1)  # dim=nxn
KernelMatrix_SE_x2 = KernelMatrix(Kernel_SE, X_Inputs, Y_Inputs, l = 0.5, sigma2 = 1)  # dim=nxn
# Multiply the two kernel/Gram matrices
KernelMatrix_Product = np.kron(KernelMatrix_SE_x1, KernelMatrix_SE_x2)  # dim=(n^2)x(n^2)
KernelMatrix_Product += 1e-6 * np.eye(KernelMatrix_Product.shape[0])  # Add jitter
m = np.zeros(KernelMatrix_Product.shape[0])

# Next, compute the Gaussian samples for all nxn points
f_flat = rng.multivariate_normal(mean = m, cov = KernelMatrix_Product)  # dim=(n^2)x1
f = f_flat.reshape(n, n)  # dim=nxn
# Finally, create a 2D grid for the plot and plot the surface
X_GPGrid, Y_GPGrid = np.meshgrid(X_Inputs, Y_Inputs)
ax2.plot_surface(X_GPGrid, Y_GPGrid, f, cmap = 'cool')
ax2.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{SE} \cdot k_{SE})$, $l = 0.5$")
ax2.set_xlabel(r"$x_1$")
ax2.set_ylabel(r"$\tilde{x}_1$")
ax2.set_zlabel(r"$f(\mathbf{x}, \tilde{\mathbf{x}})$")


### Right Plot
KernelMatrix_SE_x1_Right = KernelMatrix(Kernel_SE, X_Inputs, Y_Inputs, l = 2, sigma2 = 1)  # dim=nxn
KernelMatrix_SE_x2_Right = KernelMatrix(Kernel_SE, X_Inputs, Y_Inputs, l = 2, sigma2 = 1)  # dim=nxn
KernelMatrix_Product_Right = np.kron(KernelMatrix_SE_x1_Right, KernelMatrix_SE_x2_Right)  # dim=(n^2)x(n^2)
KernelMatrix_Product_Right += 1e-6 * np.eye(KernelMatrix_Product_Right.shape[0])
m = np.zeros(KernelMatrix_Product_Right.shape[0])

f_flat_Right = rng.multivariate_normal(mean = m, cov = KernelMatrix_Product_Right)  # dim=(n^2)x1
f_Right = f_flat_Right.reshape(n, n)  # dim=nxn
X_GPGrid_Right, Y_GPGrid_Right = np.meshgrid(X_Inputs, Y_Inputs)
ax3.plot_surface(X_GPGrid_Right, Y_GPGrid_Right, f_Right, cmap = 'cool')
ax3.set_title(r"Samples Functions from $\mathcal{GP}(0, k_{SE} \cdot k_{SE})$, $l = 2$")
ax3.set_xlabel(r"$x_1$")
ax3.set_ylabel(r"$\tilde{x}_1$")
ax3.set_zlabel(r"$f(\mathbf{x}, \tilde{\mathbf{x}})$", labelpad = -160)

plt.tight_layout()
plt.show()