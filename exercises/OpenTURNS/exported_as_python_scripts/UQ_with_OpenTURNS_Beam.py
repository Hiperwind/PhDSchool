# %% [markdown]
# # Uncertainty quantification using OpenTURNS
# 
# This notebook presents the main steps of the uncertainty quantification framework. 
# 
# The analytical case of the cantilever beam in flexion illustrates the tools offered by OpenTURNS to perform a typical uncertainty quantification study.  

# %% [markdown]
# ## 1. Analytical case: cantilever beam in flexion
# 
# We are interested in the vertical deviation $Y$ of a diving board created by a young diver. 
# 
# We consider a child whose weight generates a force $F$ approximately equal to 300N, considered hereafter as a random variable. 
# 
# <img src="beam.png"  width=300>
# 
# 
# The Following inputs are considered as independent random variables: 
# 
# 
#  * $E$ : Young modulus (Pa), &nbsp;&nbsp;        Beta($\alpha = 0.9, \beta = 3.5, a=65.0 \times 10^9, b=75.0 \times 10^9$)
#  * $F$ : Loading (N), &nbsp;&nbsp;               Lognormal($\mu_F=300.0, \sigma_F=30.0$)
#  * $L$ : Length of beam ($m$), &nbsp;&nbsp;        Uniform(min=2.5, max= 2.6)
#  * $I$ : Moment of inertia ($m^4$), &nbsp;&nbsp; Beta($\alpha = 2.5, \beta = 4.0, a = 1.3 \times 10^{-7}, b = 1.7 \times 10^{-7}$).
# 
# The vertical deviation is given by the following expression: 
# 
# $$
# Y  = \frac{F\, L^3}{3 \, E \, I}
# $$

# %%
import numpy as np
import pandas as pd
import openturns as ot
import copulogram as cp 
import matplotlib.pyplot as plt
import openturns.viewer as viewer

# %% [markdown]
# ### [Step A]: model definition

# %%
deflection_function = ot.SymbolicFunction(['E', 'F', 'L', 'I'], ['F*L^3/(3*E*I)'])
# Testing the function
x_test = [7e+10, 300, 2.5, 1.5e-07]
print(f"deflection({x_test}) = {deflection_function(x_test)}")

# %% [markdown]
# ### [Step B]: uncertainty model definition

# %%
E = ot.Beta(0.9, 3.5, 6.5e+10, 7.5e+10)
F = ot.LogNormalMuSigma(300, 30, 0.0).getDistribution()
L = ot.Uniform(2.5, 2.6)
I = ot.Beta(2.5, 4., 1.3e-07, 1.7e-07)
marginals = [E, F, L, I]
joint_distribution = ot.ComposedDistribution(marginals)
joint_distribution.setDescription(['E', 'F', 'L', 'I'])
dim = joint_distribution.getDimension()
joint_distribution

# %% [markdown]
# ### [Step C]: uncertainty propagation

# %% [markdown]
# #### Monte Carlo propagation

# %%
size = 1000
x_MC_sample = joint_distribution.getSample(size)
y_MC_sample = deflection_function(x_MC_sample)

# %% [markdown]
# Other sampling methods available.
# 
# **For mean estimation:** 
# - 1D Deterministic quadratures (Gauss-Legendre, Gauss-Legendre, Féjer)
# - multi-D Deterministic quadratures (regular, sparse grids)
# - Quasi-Monte Carlo, Randomized Quasi-Monte Carlo 
# - Latin Hypercube Sampling
# - Optimal Latin Hypercube Sampling
# 
# **For rare event estimation:**
# - FORM/SORM
# - Importance Sampling
# - Line Sampling
# - Subset Sampling
# - Cross-entropy Importance Sampling
# - Non-Adaptive Importance Sampling

# %% [markdown]
# #### Data visualization
# 

# %%
data = pd.DataFrame(np.array(x_MC_sample), columns=list(joint_distribution.getDescription()))
data['Y'] = np.array(y_MC_sample)
copulogram = cp.Copulogram(data)
copulogram.draw(alpha=0.6, marker='.');

# %%
threshold = 0.22 # Material limit 1: cracks appear
#threshold = 0.3 # Material limit 2: rupture

# %%
fig, ax = plt.subplots(figsize=(5, 4))
N, bins, patches = ax.hist(data['Y'], bins=20, density=True, alpha=0.6, color='C2', label="Deflection")
for i in range(17, 20):
    patches[i].set_facecolor('C3')
plt.axvline(data['Y'].mean(), color='k', label="Mean Deflection")
plt.axvline(threshold, color='C3', label="Threshold")
plt.xlabel("Y")
plt.ylabel("PDF")
plt.legend();

# %% [markdown]
# #### Exact propagation (arithmetic of probability distributions)
# 
# *Disclaimer: this approach is only possible for simple analytical models and used here to obtain reference values.*

# %%
exact_Y = (F * L ** 3) / (3 * E * I)
exact_mean = exact_Y.getMean()[0]
exact_std = exact_Y.getStandardDeviation()[0]
exact_pf = exact_Y.computeProbability(ot.Interval([threshold], [1], [True], [False]))

# %% [markdown]
# #### Monte Carlo mean convergence 

# %%
data['Y_mean'] = np.cumsum(data['Y']) / (data.index + 1)
data['Y_std'] = data['Y'].expanding().std()
data['Y_mean_lower_IC_95'] = data['Y_mean'] - (1.96 * data['Y_std'] / np.sqrt(data.index + 1)) 
data['Y_mean_upper_IC_95'] = data['Y_mean'] + (1.96 * data['Y_std'] / np.sqrt(data.index + 1))
data

# %%
MC_mean = data.loc[size-1, 'Y_mean']
print(f"Monte Carlo mean: {MC_mean}")
print(f"Exact mean\t: {exact_mean}")

# %%
plt.figure(figsize=(5, 4))
start_plot = 10
plt.plot(data.index[start_plot:], data.loc[start_plot:, 'Y_mean'], label="MC mean")
plt.fill_between(data.index[start_plot:], 
                data.loc[start_plot:, 'Y_mean_lower_IC_95'], 
                data.loc[start_plot:, 'Y_mean_upper_IC_95'], 
                label="MC mean IC",
                alpha=0.4)
plt.xlabel("Size")
plt.ylabel("Y mean")
plt.legend();

# %% [markdown]
# #### Monte Carlo failure probability convergence

# %%
data['is_failed'] = 0
data.loc[data[data['Y'] > threshold].index, 'is_failed'] = 1
data

# %%
import seaborn as sns
from matplotlib.colors import to_rgba
# Define tailored binary colorbar
my_palette = sns.color_palette([to_rgba('C2', 0.5), to_rgba('C3', 1.)], as_cmap=True)

copulogram = cp.Copulogram(data[['E', 'F', 'L', 'I', 'Y', 'is_failed']])
copulogram.draw(hue='is_failed', hue_colorbar=my_palette, marker='.', kde_on_marginals=False);

# %%
MC_pf = data['is_failed'].mean()
print(f"Monte Carlo pf \t: {MC_pf}")
print(f"Exact pf \t: {exact_pf}")

# %% [markdown]
# ### [Step C'] Global sensitivity analysis 
# 
# #### Sobol' indices

# %%
sie = ot.SobolIndicesExperiment(joint_distribution, size)
input_design = sie.generate()
input_design.setDescription(joint_distribution.getDescription())
output_design = deflection_function(input_design)
sensitivity_analysis = ot.JansenSensitivityAlgorithm(input_design, output_design, size)
# Get results
sobol_first_order = sensitivity_analysis.getFirstOrderIndices()
sobol_tolal = sensitivity_analysis.getTotalOrderIndices()
# Draw results 
sensitivity_analysis.draw()

# %% [markdown]
# ### Kriging surrogate model
# 
# Let us consider small-data case (e.g., costly numerical model).

# %%
size = 10
x_train = joint_distribution.getSample(size)
y_train = deflection_function(x_train)

# %%
# Define the basis
basis = ot.ConstantBasisFactory(dim).build()
# Define the covariance model
covariance_model = ot.SquaredExponential(dim)
covariance_model.setScale(x_train.getMax())
# Define the kriging model
kriging = ot.KrigingAlgorithm(x_train, y_train, covariance_model, basis)
# Note that the amplitude parameter is computed analytically, we only need to set bounds on the scale parameter.
scale_optimization_bounds = ot.Interval([1.0, 1.0, 1.0, 1.0e-10], [1.0e11, 1.0e3, 1.0e1, 1.0e-5])
kriging.setOptimizationBounds(scale_optimization_bounds)
# Fit the kriging model and get results
kriging.run()
kriging_results = kriging.getResult()
kriging_predictor = kriging_results.getMetaModel()

# %%
# Validation on the large Monte Carlo previous input-output dataset
validation = ot.MetaModelValidation(x_MC_sample, y_MC_sample, kriging_predictor)
q2 = validation.computePredictivityFactor()[0]
print(f"Predictivity coefficient: Q2 = {q2:.2f}")

# %%
# Get the residual
residual = validation.getResidualSample()
# Get the histogram of residual
residual_histogram = validation.getResidualDistribution(False)
graph = residual_histogram.drawPDF()
view = viewer.View(graph)

# %%
# Draw the validation graph
qq_plot = validation.drawValidation()
view = viewer.View(qq_plot)

# %%



