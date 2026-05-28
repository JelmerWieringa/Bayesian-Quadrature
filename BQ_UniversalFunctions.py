########################################################################################################################
####################################### BAYESIAN QUADRATURE UNIVERSAL FUNCTIONS ########################################
########################################################################################################################
import numpy as np
from scipy.optimize import minimize


#=======================================================================================================================
#------------------------------------------------- 1D Kernel Functions -------------------------------------------------
#=======================================================================================================================
def Kernel_SE(x, y, l_SE, sigmaf2_SE, **_):
    Norm_2 = (x - y)**2  # Squared Euclidean norm of x - y
    return (sigmaf2_SE**2) * np.exp( -Norm_2 / (2 * l_SE**2))

def Kernel_RQ(x, y, l_RQ, alpha_RQ, sigmaf2_RQ, **_):
    Norm_2 = (x - y)**2  # Squared Euclidean norm of x - y
    return (sigmaf2_RQ**2) * (1 + Norm_2 / (2 * alpha_RQ * l_RQ**2))**(-alpha_RQ)

def Kernel_Matern_1_2(x, y, l_M12, sigmaf2_M12, **_):
    Norm = np.sqrt((x - y)**2)  # Euclidean norm of x - y
    return (sigmaf2_M12**2) * np.exp(-Norm / l_M12)

def Kernel_Matern_3_2(x, y, l_M32, sigmaf2_M32, **_):
    Norm = np.sqrt((x - y)**2)  # Euclidean norm of x - y
    return (sigmaf2_M32**2) * (1 + np.sqrt(3) * Norm / l_M32) * np.exp(-np.sqrt(3) * Norm / l_M32)

def Kernel_Matern_5_2(x, y, l_M52, sigmaf2_M52, **_):
    Norm = np.sqrt((x - y)**2)  # Euclidean norm of x - y
    return ((sigmaf2_M52**2) * (1 + np.sqrt(5) * Norm / l_M52 + 5 * Norm**2 / (3 * l_M52**2)) *
            np.exp(-np.sqrt(5) * Norm / l_M52))

def Kernel_Polynomial2(x, y, c_Poly, sigmaf2_Poly, **_):
    #x = np.asarray(x)
    #y = np.asarray(y)
    return sigmaf2_Poly * (x * y + c_Poly)**2

def Kernel_Periodic(x, y, lmbd_P, l_P, sigmaf2_P, **_):  # Scalar inputs only
    #x = np.asarray(x)[:, None]   # shape (n,1)
    #y = np.asarray(y)[None, :]   # shape (1,m)
    PeriodicTerm = np.sin(np.pi * (x - y) / lmbd_P)
    return sigmaf2_P * np.exp(-(PeriodicTerm**2) / (2 * l_P**2))


### Sum of Kernels
def SumKernels(Kernel1, Kernel2):
    def K(x, y, **parameters):
        # Each kernel will pick up the parameters it needs
        return Kernel1(x, y, **parameters) + Kernel2(x, y, **parameters)
    return K

### Gram Matrix
def KernelMatrix(KernelFunction, x, y, **parameters):
    x = x[:, None]  # Changes to a column vector, dim=nx1
    y = y[None, :]  # Changes to a row vector, dim=1xn
    return KernelFunction(x, y, **parameters)




#=======================================================================================================================
#----------------------------------------- Multi-Dimensional Kernel Functions ------------------------------------------
#=======================================================================================================================
def Kernel_SE_MultiDim(x, y, l_SE, sigmaf2_SE, **_):
    x = np.asarray(x).ravel()  # .ravel() flattens the array to a 1D array
    y = np.asarray(y).ravel()
    Norm_2 = (x - y).T @ (x - y)  # Squared Euclidean norm of x - y
    return (sigmaf2_SE**2) * np.exp( -Norm_2 / (2 * l_SE**2))

def Kernel_RQ_MultiDim(x, y, l_RQ, alpha_RQ, sigmaf2_RQ, **_):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    Norm_2 = (x - y).T @ (x - y)  # Squared Euclidean norm of x - y
    return (sigmaf2_RQ**2) * (1 + Norm_2 / (2 * alpha_RQ * l_RQ**2))**(-alpha_RQ)

def Kernel_Matern_1_2_MultiDim(x, y, l_M12, sigmaf2_M12, **_):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    Norm = (x - y).T @ (x - y)  # Euclidean norm of x - y
    return (sigmaf2_M12**2) * np.exp(-Norm / l_M12)

def Kernel_Matern_3_2_MultiDim(x, y, l_M32, sigmaf2_M32, **_):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    Norm = (x - y).T @ (x - y)  # Euclidean norm of x - y
    return (sigmaf2_M32**2) * (1 + np.sqrt(3) * Norm / l_M32) * np.exp(-np.sqrt(3) * Norm / l_M32)

def Kernel_Matern_5_2_MultiDim(x, y, l_M52, sigmaf2_M52, **_):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    Norm = (x - y).T @ (x - y)  # Euclidean norm of x - y
    return ((sigmaf2_M52**2) * (1 + np.sqrt(5) * Norm / l_M52 + 5 * Norm**2 / (3 * l_M52**2)) *
            np.exp(-np.sqrt(5) * Norm / l_M52))

def Kernel_Polynomial2_MultiDim(x, y, c_Poly, sigmaf2_Poly, **_):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return sigmaf2_Poly * ((x @ y) + c_Poly)**2

def Kernel_Periodic_MultiDim(x, y, lmbd_P, l_P, sigmaf2_P, **_):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    lmbd_P = np.asarray(lmbd_P)
    l_P = np.asarray(l_P)
    PeriodicTerm = np.sin(np.pi * (x - y) / lmbd_P)
    SquaredPeriodicTerm = (PeriodicTerm / l_P) @ (PeriodicTerm / l_P)
    return sigmaf2_P * np.exp(-0.5 * SquaredPeriodicTerm)


### Gram Matrix (Multi-Dimensional)
def KernelMatrix_MultiDim(KernelFunction, x, y, **parameters):  # For general (nxd)-dim arrays
    x = np.asarray(x)
    y = np.asarray(y)
    n = x.shape[0]
    m = y.shape[0]

    K_xy = np.empty((n, m), dtype = float)
    for i in range(n):
        for j in range(m):
            K_xy[i, j] = KernelFunction(x[i], y[j], **parameters)
    return K_xy




#=======================================================================================================================
#-------------------------------------------- Log Marginal Likelihood Function -----------------------------------------
#=======================================================================================================================
def LogMarginalLikelihood(loghyperparameters, X_Input, y, PriorMeanVector, KernelFunction, ParameterNames):
    """"
    loghyperparameters = 1D array of log-transformed positive hyperparameters (hence unconstrained)
    X_Input = 1D array of input data points
    y = f_X = function evaluations at X
    PriorMeanVector = 1D array of prior GP means of f
    KernelFunction = Kernel_SE, Kernel_RQ, Kernel_Matern32, ...
    ParameterNames = list of parameter names of that kernel (order must match loghyperparameters)
    """

    #---------------------------- Retransform Hyperparameters ----------------------------
    loghyperparameters = np.asarray(loghyperparameters, dtype = float).ravel()
    parameters_positive = np.exp(loghyperparameters)
    kernel_parameters = {name: value for name, value in zip(ParameterNames, parameters_positive)}
    # zip() gives a tuple of pairwise combined elements, e.g.: [('parameter_1', parameter_value_1), ...]
    # {name: value for ...} makes a dictionary, e.g.: ['parameter_1': parameter_value_1, ...]

    #------------------------------ Gram Matrix -------------------------------
    K_XX = KernelMatrix_MultiDim(KernelFunction, X_Input, X_Input, **kernel_parameters)
    #K_XX += 1e-6 * np.eye(len(X_Input))  # Add jitter

    # Change the jitter value, because K_XX becomes not positive definite when len(X_Input) gets larger
    jitter = 1e-8
    for _ in range(5):
        try:
            L = np.linalg.cholesky( K_XX + jitter * np.eye(len(X_Input)) )
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        # even with large jitter it's not PD: treat these params as impossible
        return -np.inf
    #L = np.linalg.cholesky(K_XX)  # Cholesky decomposition K_XX = L L^T

    #------------------------- Terms of Log-Likelihood ------------------------
    # Data fit term: -1/2 * (y - m_X)^T K_XX^{-1} (y - m_X)
    a = np.linalg.solve(L, y - PriorMeanVector)  # L^{-1} (y - m_X)
    DataFitTerm = -0.5 * a.T @ a

    # Model Complexity term: -1/2 * log(det(K_XX))
    ModelComplexityTerm = -np.sum( np.log(np.diag(L)) )

    # Normalization constant: -n/2 * log(2pi)
    n_Observations = len(y)
    NormalizationConstantTerm = -0.5 * n_Observations * np.log(2 * np.pi)

    return DataFitTerm + ModelComplexityTerm + NormalizationConstantTerm




#=======================================================================================================================
#-------------------------------------------- Maximum Likelihood Optimisation ------------------------------------------
#=======================================================================================================================
def MLEHyperparameters(initial_parameters, X_Input, y, PriorMeanVector, KernelFunction, ParameterNames,
                       N_Restarts=3):
    #--------------------- Define Objective for Minimisation Problem ---------------------
    def NegativeLogMarginalLikelihood(loghyperparameters, X_Input, y, PriorMeanVector):
        LML = LogMarginalLikelihood(loghyperparameters, X_Input, y, PriorMeanVector, KernelFunction,
                                          ParameterNames)
        if np.isfinite(LML) == False or abs(LML) > 1e12:
            return 1e21  # Penalty for infinintely big LML so that the optimisation algorithm does not keep searching
        return -LML

    #--------------------- Transform to Unconstraint Hyperparameters ---------------------
    initial_parameters = np.asarray(initial_parameters, dtype = float).ravel()
    initial_logparameters = np.log(initial_parameters)

    Bounds = [(np.log(0.05), np.log(100))] * len(initial_parameters)
    # bounds ensures that the hyperparameters cannot become 0.
    # (E.g.: in k_{RQ}, alpha can become so small that the Gram matrix becomes not PSD)
    # Lower bound np.log(0.05) prevents degenerate near-zero length scale solutions.

    #------------------------------ Optimisation Algorithm -------------------------------
    ### First run — original starting point
    FindingOptimizer = minimize(
        fun = NegativeLogMarginalLikelihood,  # fun(x, *args)
        x0 = initial_logparameters,
        args = (X_Input, y, PriorMeanVector),
        bounds = Bounds
    )
    # print(label, FindingOptimizer.success, FindingOptimizer.message)

    #---------------------------- Random Restarts ------------------------------------
    AllRuns = [FindingOptimizer]  # Collect all optimisation results

    for r in range(1, N_Restarts):
        rng = np.random.default_rng(23 + r)
        Perturbation = rng.uniform(-1.0, 1.0, size = len(initial_parameters))
        x0_restart = np.clip(
            initial_logparameters + Perturbation,
            a_min = [b[0] for b in Bounds],
            a_max = [b[1] for b in Bounds]
        )
        FindingOptimizer_r = minimize(
            fun = NegativeLogMarginalLikelihood,
            x0 = x0_restart,
            args = (X_Input, y, PriorMeanVector),
            bounds = Bounds
        )
        AllRuns.append(FindingOptimizer_r)

    #---------------------------- Select Best Result ---------------------------------
    ValidRuns = [run for run in AllRuns if run.success or run.fun < 1e20]

    if len(ValidRuns) == 0:
        BestRun = AllRuns[0]  # Fallback: return result of original single run
    else:
        BestRun = min(ValidRuns, key = lambda run: run.fun)

    #---------------------------- Retransform Hyperparameters ----------------------------
    OptimalLogHyperparameters = BestRun.x
    OptimalHyperparameters_Positive = np.exp(OptimalLogHyperparameters)
    return OptimalHyperparameters_Positive




#=======================================================================================================================
#--------------------------------------- Kernel Selection (Multi-Dimensional) ------------------------------------------
#=======================================================================================================================
def KernelSelectionML(X, f_X, ConstantPriorGPMean=0, UniversalInitialParameter=1):
    """
    This function picks the kernel in the KernelFamily (containing the 'SE', 'RQ', 'M12', 'M32', 'M52', 'Poly' and 'Per'
        kernels & every sum of two of these up to symmetry) that has the highest log marginal likelihood.
    It is assumed that the following functions are already defined: Kernel_SE_MultiDim, Kernel_RQ_MultiDim,
        Kernel_Matern_1_2_MultiDim, Kernel_Matern_3_2_MultiDim, Kernel_Matern_5_2_MultiDim,
        Kernel_Polynomial2_MultiDim, Kernel_Periodic_MultiDim, SumKernels, KernelMatrix_MultiDim,
        LogMarginalLikelihood, MLEHyperparameters.
    ...
    """
    #---------------------------------- Define Collection Containing (Single) Kernels ----------------------------------
    KernelFamily = {"SE": dict(kernel = Kernel_SE_MultiDim, parameter_names = ["l_SE", "sigmaf2_SE"],
                               initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "RQ": dict(kernel = Kernel_RQ_MultiDim, parameter_names = ["l_RQ", "alpha_RQ", "sigmaf2_RQ"],
                               initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                     UniversalInitialParameter]),
                    "M12": dict(kernel = Kernel_Matern_1_2_MultiDim, parameter_names = ["l_M12", "sigmaf2_M12"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "M32": dict(kernel = Kernel_Matern_3_2_MultiDim, parameter_names = ["l_M32", "sigmaf2_M32"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "M52": dict(kernel = Kernel_Matern_5_2_MultiDim, parameter_names = ["l_M52", "sigmaf2_M52"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "Poly": dict(kernel = Kernel_Polynomial2_MultiDim, parameter_names = ["c_Poly", "sigmaf2_Poly"],
                                 initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                    "Per": dict(kernel = Kernel_Periodic_MultiDim, parameter_names = ["lmbd_P", "l_P", "sigmaf2_P"],
                                initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                      UniversalInitialParameter]),
                    }
    # {} creates} a dictionary (= an ordered collection) that stores data values in "key: value" pairs
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
                    initial_parameters =Sum_initial_parameters,
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


def KernelSelectionML_1D(X_Input, y, UniversalInitialParameter=1, PreviousResult=None):
    """
    This function picks the kernel in the KernelFamily_1D (containing the 'SE', 'RQ', 'M12', 'M32', 'M52', 'Poly'
        and 'Per' kernels & every sum of two of these up to symmetry) that has the highest log marginal likelihood.
    It is assumed that the following functions are already defined: Kernel_SE, Kernel_RQ, Kernel_Matern_1_2,
        Kernel_Matern_3_2, Kernel_Matern_5_2, Kernel_Polynomial2, Kernel_Periodic, SumKernels, KernelMatrix,
        LogMarginalLikelihood, MLEHyperparameters.
    A constant zero prior GP mean is assumed.
    If PreviousResult is not None and its label matches the current kernel, the previous hyperparameter values
        are used as the first initial point for MLEHyperparameters, while all random restarts use the standard initial
        point and perturbations unchanged.
    """
    #---------------------------------- Define Collection Containing (Single) Kernels ----------------------------------
    KernelFamily_1D = {"SE":   dict(kernel = Kernel_SE,          parameter_names = ["l_SE", "sigmaf2_SE"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                       "RQ":   dict(kernel = Kernel_RQ,          parameter_names = ["l_RQ", "alpha_RQ", "sigmaf2_RQ"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                          UniversalInitialParameter]),
                       "M12":  dict(kernel = Kernel_Matern_1_2,  parameter_names = ["l_M12", "sigmaf2_M12"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                       "M32":  dict(kernel = Kernel_Matern_3_2,  parameter_names = ["l_M32", "sigmaf2_M32"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                       "M52":  dict(kernel = Kernel_Matern_5_2,  parameter_names = ["l_M52", "sigmaf2_M52"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                       "Poly": dict(kernel = Kernel_Polynomial2, parameter_names = ["c_Poly", "sigmaf2_Poly"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter]),
                       "Per":  dict(kernel = Kernel_Periodic,    parameter_names = ["lmbd_P", "l_P", "sigmaf2_P"],
                                    initial_parameters = [UniversalInitialParameter, UniversalInitialParameter,
                                                          UniversalInitialParameter]),
                       }
    # {} creates a dictionary (= an ordered collection) that stores data values in "key: value" pairs
    # dict() also creates a dictionary. Hence, every kernel (name) is associated with a dictionary containing kernel
    #   information.


    #----------------------------- Collect Kernels & Sums of 2 Kernels into One Dictionary -----------------------------
    KernelSimulations_1D = []  # Store the MLE information for different kernels
    KernelNames_1D = list(KernelFamily_1D.keys())
    # .keys() returns (a view object containing) the keys of the dictionary, i.e.: dict_keys(['SE', 'RQ', 'M12', 'M32',
    #       'M52', 'Poly', 'Per'])
    # list() makes it a list which results in ['SE', 'RQ', 'M12', 'M32', 'M52', 'Poly', 'Per']

    ### Add every (single) kernel, with information defined in KernelFamily_1D, to KernelSimulations_1D
    for name in KernelNames_1D:
        if len(X_Input) < 10 and 'Per' in name:  # Skip Periodic kernels when n < 10 (too few data points)
            continue
        KernelSimulations_1D.append(
            dict(
                label = name,
                kernel = KernelFamily_1D[name]['kernel'],
                parameter_names = KernelFamily_1D[name]['parameter_names'],
                initial_parameters = np.array(KernelFamily_1D[name]['initial_parameters'], dtype = float),
            )
        )

    ### Add sums of two kernels, with corresponding information, to KernelSimulations_1D
    for i in range(len(KernelNames_1D)):
        for j in range(i + 1, len(KernelNames_1D)):
            Sum_label = f"{KernelNames_1D[i]}+{KernelNames_1D[j]}"

            # Skip Periodic sum-kernels when n < 10, because is ill-behaved for small sample sizes
            if len(X_Input) < 10 and 'Per' in Sum_label:
                continue

            Sum_2Kernels = SumKernels(KernelFamily_1D[ KernelNames_1D[i] ]['kernel'],
                                      KernelFamily_1D[ KernelNames_1D[j] ]['kernel'])
            Sum_parameter_names = (KernelFamily_1D[ KernelNames_1D[i] ]['parameter_names'] +
                                   KernelFamily_1D[ KernelNames_1D[j] ]['parameter_names'])
            Sum_initial_parameters = np.array(KernelFamily_1D[ KernelNames_1D[i] ]['initial_parameters'] +
                                              KernelFamily_1D[ KernelNames_1D[j] ]['initial_parameters'], dtype = float)

            KernelSimulations_1D.append(
                dict(
                    label = Sum_label,
                    kernel = Sum_2Kernels,
                    parameter_names = Sum_parameter_names,
                    initial_parameters = Sum_initial_parameters,
                )
            )


    #--------- Estimate (Hyperparameters by) Maximum Marginal Likelihood for Every Kernel in KernelFamily_1D ----------
    PriorMean_ = np.zeros_like(y, dtype = float)
    KernelChoiceMLE_1D = []
    for kernelinfo in KernelSimulations_1D:
        label = kernelinfo['label']
        kernelfunction = kernelinfo['kernel']
        parameter_names = kernelinfo['parameter_names']
        initial_parameters = kernelinfo['initial_parameters']

        ### Warm start: if PreviousResult matches this kernel label, use its hyperparameters as first initial point
        if (PreviousResult is not None and
                PreviousResult.get('Label') == label and
                'HyperparametersValues' in PreviousResult):
            initial_parameters = np.asarray(PreviousResult['HyperparametersValues'], dtype = float)

        ### Estimate Hyperparameter by ML
        hyperparameters_MLE = MLEHyperparameters(
            initial_parameters = initial_parameters,
            X_Input = X_Input,
            y = y,
            PriorMeanVector = PriorMean_,
            KernelFunction = kernelfunction,
            ParameterNames = parameter_names,
        )

        ### Compute the corresponding Log Marginal Likelihood Value
        LMLValue = LogMarginalLikelihood(
            loghyperparameters = np.log(hyperparameters_MLE),
            X_Input = X_Input,
            y = y,
            PriorMeanVector = PriorMean_,
            KernelFunction = kernelfunction,
            ParameterNames = parameter_names,
        )

        KernelChoiceMLE_1D.append(
            dict(
                label = label,
                kernel = kernelfunction,
                parameter_names = parameter_names,
                hyperparameters_MLE = hyperparameters_MLE,
                LML = float(LMLValue),
            )
        )

    #--------------------------------- Select Best Kernel with Corresponding Attributes --------------------------------
    KernelChoiceMLE_1D_Sorted = sorted(KernelChoiceMLE_1D, key = lambda d: d['LML'], reverse = True)
    BestKernelChoice_1D = KernelChoiceMLE_1D_Sorted[0]  # The kernel with the highest LML
    BestKernelChoice_1D_label = BestKernelChoice_1D['label']
    BestKernelChoice_1D_kernel = BestKernelChoice_1D['kernel']
    BestKernelChoice_1D_parameternames = BestKernelChoice_1D['parameter_names']
    BestKernelChoice_1D_hyperparametersValues = BestKernelChoice_1D['hyperparameters_MLE']
    BestKernelChoice_1D_hyperparameters = dict(zip(BestKernelChoice_1D_parameternames,
                                                   BestKernelChoice_1D_hyperparametersValues))
    BestKernelChoice_1D_LML = BestKernelChoice_1D['LML']

    return {"Label": BestKernelChoice_1D_label, "Kernel": BestKernelChoice_1D_kernel,
            "Hyperparameters": BestKernelChoice_1D_hyperparameters,
            "HyperparametersValues": BestKernelChoice_1D_hyperparametersValues,
            "HyperparametersNames": BestKernelChoice_1D_parameternames,
            "LML": BestKernelChoice_1D_LML,
            "List": KernelChoiceMLE_1D_Sorted, "BestKernelChoice": BestKernelChoice_1D}
