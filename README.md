# Bayesian Quadrature: Theory and Comparative Simulation Study

A probabilistic numerical method to approximate integrals, called **Bayesian Quadrature (BQ)**. The central idea is to model the integrand as a **Gaussian Process (GP)** and push the integration operator through this probabilistic model analytically. This yields a full Gaussian
posterior distribution over the integral value, not just a point estimate.

<img width="2100" height="750" alt="BQ_Animation_UncertaintySampling_reMLE_IntervalRight=3_nGrid=500_nStart=3_nEnd=14" src="https://github.com/user-attachments/assets/3db8721f-167d-48aa-a8da-4b2b63c1dd9b" />


---

## Overview

To fit the hyperparameters of the Gaussian Process, **Maximum Likelihood Estimation (MLE)** is
applied to the log marginal likelihood. By the closedness of Gaussian Processes under bounded
linear operators, the GP is pushed through the integral operator analytically. This yields a
Gaussian posterior distribution over the integral value.

A proof of this closedness property is provided in the thesis, together with a self-contained
review of:

- Expectation operators and Gaussian measures
- Gaussian Processes and covariance kernels (SE, Matérn 1/2, 3/2, 5/2, Periodic, Rational Quadratic, Polynomial)
- Kernel hyperparameter estimation via MLE and automatic kernel selection over a kernel family

In the special case of a Gaussian measure and a squared-exponential (i.e., Gaussian) kernel,
Bayesian Quadrature reduces to **Bayes–Hermite Quadrature**.

The thesis is complemented by a **comparative simulation study**, in which Bayesian Quadrature
(with multiple point-selection strategies, and re-estimation of kernel and corresponding hyperparameters) is compared to the classical Monte Carlo integration technique.

---

## Animations

**Re-MLE** — starting from $n=5$ initial observations and incrementally growing to $n=25$, at each step $n$ is increased by one, the observations are placed on an updated equidistant grid over $[0,5]$, and both the kernel and its hyperparameters are re-estimated via MLE, keeping the model calibrated throughout. 
In most frames the absolute error and posterior variance decrease as $n$ grows, but occasional increases occur. This is due to several factors: MLE on small samples is sensitive to local optima, 
the equidistant grid shifts entirely when a point is added (so previously well-placed observations might move), and the integrand $f(x) := \exp(\sin(x^2) \cdot \cos(x))$ has rapidly increasing oscillation frequency,
making consistent kernel selection difficult at moderate $n$.

<img width="2100" height="750" alt="BQ_Animation_reMLE_IntervalRight=5_nGrid=500_nStart=5_nEnd=25" src="https://github.com/user-attachments/assets/27b67aa9-a8d8-492c-b9a4-3697f47742e7" />


<p align="center">
  
**Sequential uncertainty sampling** — starting from $n=5$ initial observations on an equidistant grid over $[0,3]$, each new observation is placed at the location of maximum posterior variance, sequentially narrowing the GP posterior around the integral value. 
The kernel and hyperparameters are estimated once on the initial $n=5$ points and kept fixed throughout. 
As a result, the integral posterior variance​​ decreases monotonically by construction, but the absolute error may still fluctuate, reflecting the known limitation that uncertainty sampling minimises posterior variance of the integral, not the error itself.

<img width="2100" height="750" alt="BQ_Animation_UncertaintySampling_IntervalRight=3_nGrid=500_nStart=5_nEnd=25" src="https://github.com/user-attachments/assets/2f2ca67c-70f3-490a-abf1-042a16cb30be" />



<p align="center">


**Sequential uncertainty sampling with Re-MLE** — starting from $n=3$ initial observations on an equidistant grid over $[0,3]$, each new observation is placed at the location of maximum posterior variance. 
After each addition, both the kernel and its hyperparameters are re-estimated via MLE on the updated dataset, allowing the model to adapt as more structure becomes visible. 
Both the posterior variance and the absolute error generally decrease as $n$ grows, though occasional increases reflect the same sensitivity to local optima and small-sample MLE as in the Re-MLE case.

<img width="2100" height="750" alt="BQ_Animation_UncertaintySampling_reMLE_IntervalRight=3_nGrid=500_nStart=3_nEnd=14" src="https://github.com/user-attachments/assets/44da7133-e6b2-4de5-9ff0-fe0099533a7f" />




---

## Repository Structure

```
Bayesian-Quadrature/
│
├── README.md
│
├── BQ_UniversalFunctions.py                          # Functions used in multiple files
│
├── CovarianceKernels.py                              # Kernel heatmaps & GP prior samples
├── GP_Posterior.py                                   # GP prior & posterior regression demo
├── BQ_PosteriorDistribution.py                       # BQ posterior distribution over the integral
├── BQ_KernelHyperparameterEstimation.py              # MLE kernel selection & BQ simulation study
│
├── BQ_Example_MarginalLikelihood_SyntheticTrimodalLikelihood.py
├── BQ_Example_MarginalLikelihood_2Dimensional_SalaryData.py
├── BQ_Example_MarginalLikelihood_4Dimensional_ProfitData.py
│
├── BQ_KernelHyperparameterEstimation_GIF.py          # GIF animations
│
└── Thesis/
    └── Bachelor's Thesis Mathematics_Bayesian Quadrature Theory and Comparative Simulation Study_Jelmer Wieringa.pdf
```

---

## File Descriptions

### `BQ_UniversalFunctions.py`
Shared module containing the complete kernel library (SE, Matérn 1/2, 3/2, 5/2, Periodic,
Rational Quadratic, Polynomial), the Gram matrix constructor `KernelMatrix`, kernel composition
via `SumKernels`, the log marginal likelihood, the MLE hyperparameter optimiser
`MLEHyperparameters`, and automatic kernel selection `KernelSelectionML` over all base kernels
and pairwise sums.


### `CovarianceKernels.py`
Standalone visualisation of the full kernel library: heatmaps, slice plots, and GP prior sample
functions. Also demonstrates kernel composition (sums and products) and 2D vector-input kernels.

### `GP_Posterior.py`
Implements Gaussian Process regression from scratch. Demonstrates the GP prior, a
Cholesky-based posterior update, and plots of posterior sample functions with $\pm \sigma$ and $\pm 2 \sigma$ regions.

### `BQ_PosteriorDistribution.py`
Constructs the BQ posterior distribution over the integral value. Shows how integrating GP
posterior sample functions yields a histogram approximation of the integral's distribution and
how this histogram concentrates around the BQ posterior mean as the number of samples increases.

### `BQ_KernelHyperparameterEstimation.py`
The core BQ script. Implements:
- Automatic kernel selection over a family of 7 base kernels and all pairwise sums via MLE
- **Vanilla BQ** — fixed grid, kernel fitted once
- **Uncertainty Sampling** — sequential point selection at locations of maximum posterior variance
- **Minimise Posterior Integral Variance** — greedy point selection to reduce the BQ posterior variance
- Convergence plots comparing all BQ strategies against Monte Carlo methods

### `BQ_KernelHyperparameterEstimation_GIF.py`
Generates GIF animations of three sequential Bayesian quadrature procedures: (1) re-estimation of both kernel and hyperparameters via MLE at each step on an equidistant grid over $[0,5]$, 
(2) sequential uncertainty sampling with a fixed kernel over $[0,3]$, and (3) sequential uncertainty sampling combined with full kernel and hyperparameter re-estimation via MLE at each step over $[0,3]$.

### `BQ_Example_MarginalLikelihood_*.py`
Three application examples demonstrating BQ on real and synthetic datasets:
- **Trimodal likelihood** — synthetic 1D example with a multimodal integrand
- **2D salary data** — marginal likelihood integration over a 2-dimensional parameter space
- **4D profit data** — marginal likelihood integration over a 4-dimensional parameter space


---

## Thesis

The full thesis *"Bayesian Quadrature: Theory and Comparative Simulation Study"* is included in
the `Thesis/` folder. It covers the mathematical foundations, proof of the closedness property,
kernel theory, MLE hyperparameter estimation, and the complete comparative simulation study.
It is also available at the thesis research portal of the University of Groningen: https://fse.studenttheses.ub.rug.nl/37255/.

