# %% [markdown]
# # Uncertainty quantification using OpenTURNS
# 
# This notebook presents the main steps of the uncertainty quantification framework. 
# 
# The analytical case of the flooding model illustrates the tools offered by OpenTURNS to perform a typical uncertainty quantification study.  

# %% [markdown]
# ## 2. Analytical case: flooding model
# 
# The following figure presents a dyke protecting industrial facilities. When the river level exceeds the dyke height, flooding occurs. 
# The model is based on a crude simplification of the 1D hydrodynamical equations of Saint-Venant under the assumptions of uniform and constant flow rate and large rectangular sections.
# 
# <img src="flooding_section.png"  width=300>
# 
# 
# The Following inputs are considered as independent random variables: 
# 
# 
#  * $Q$ : Maximal annual flow [$m^3 s^{-1}$], &nbsp;&nbsp;        Gumbel(mode=1330, scale=500), Q > 0
#  * $K_s$ : Strickler coefficient [$m^{1/3} s^{-1}$], &nbsp;&nbsp;               Lognormal($\mu_F=30.0, \sigma_F=7.5$), $K_s$ > 0
#  * $B$ : River width [$m$], &nbsp;&nbsp;        Lognormal($\mu_F=200.0, \sigma_F=10.0$)
#  * $\alpha$ : River slope [deg.], &nbsp;&nbsp; Normal($\mu = 10^{-3}, \sigma = 10^{-4}$).
# 
# 
# The water level is determined by a simplified 1-dimensional hydraulic model:
# 
# $$
# Y  = \left(\frac{Q}{K_s \cdot B \cdot \sqrt{\alpha}} \right)^{\frac35}
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
water_height_function = ot.SymbolicFunction(['Q', 'Ks', 'B', 'Alpha'], ['(Q / (Ks * B * sqrt(Alpha)))^(3/5)'])
# Testing the function
x_test = [1330, 30, 200, 0.001]
print(f"Water height({x_test}) = {water_height_function(x_test)}")

# %% [markdown]
# ### [Step B]: uncertainty model definition

# %%
Q = ot.Gumbel(1330, 500)
Q = ot.TruncatedDistribution(Q, 0., ot.TruncatedDistribution.LOWER)
Ks = ot.LogNormalMuSigma(30, 7.5).getDistribution()
Ks = ot.TruncatedDistribution(Ks, 0., ot.TruncatedDistribution.LOWER)
B = ot.LogNormalMuSigma(200, 10).getDistribution()
Alpha = ot.Normal(1e-3, 1e-4)
marginals = [Q, Ks, B, Alpha]
joint_distribution = ot.ComposedDistribution(marginals)
joint_distribution.setDescription(['Q', 'Ks', 'B', 'Alpha'])
dim = joint_distribution.getDimension()
joint_distribution

# %% [markdown]
# ### [Step C]: uncertainty propagation

# %% [markdown]
# #### Monte Carlo propagation

# %%
size = 1000
x_MC_sample = joint_distribution.getSample(size)
y_MC_sample = water_height_function(x_MC_sample)

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
threshold = 10

# %%
fig, ax = plt.subplots(figsize=(5, 4))
N, bins, patches = ax.hist(data['Y'], bins=20, density=True, alpha=0.6, color='C2', label="Deflection")
for i in range(16, 20):
    patches[i].set_facecolor('C3')
plt.axvline(data['Y'].mean(), color='k', label="Mean Deflection")
plt.axvline(threshold, color='C3', label="Threshold")
plt.xlabel("Y")
plt.ylabel("PDF")
plt.legend();

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

copulogram = cp.Copulogram(data[['Q', 'Ks', 'B', 'Alpha', 'is_failed']])
copulogram.draw(hue='is_failed', hue_colorbar=my_palette, marker='.', kde_on_marginals=False);

# %%
MC_pf = data['is_failed'].mean()
print(f"Monte Carlo pf \t: {MC_pf}")

# %% [markdown]
# ### [Step C'] Global sensitivity analysis 
# 
# #### Sobol' indices

# %%
sie = ot.SobolIndicesExperiment(joint_distribution, size)
input_design = sie.generate()
input_design.setDescription(joint_distribution.getDescription())
output_design = water_height_function(input_design)
sensitivity_analysis = ot.JansenSensitivityAlgorithm(input_design, output_design, size)
# Get results
sobol_first_order = sensitivity_analysis.getFirstOrderIndices()
sobol_tolal = sensitivity_analysis.getTotalOrderIndices()
# Draw results 
graph = sensitivity_analysis.draw()
view = viewer.View(graph)

# %% [markdown]
# ### Kriging surrogate model
# 
# Let us consider small-data case (e.g., costly numerical model).

# %%
size = 10
x_train = joint_distribution.getSample(size)
y_train = water_height_function(x_train)

# %%
print(x_MC_sample.getMin(), x_MC_sample.getMax())

# %%
# Define the basis
basis = ot.QuadraticBasisFactory(dim).build()
# Define the covariance model
covariance_model = ot.SquaredExponential(dim)
covariance_model.setScale(x_train.getMax())
# Define the kriging model
kriging = ot.KrigingAlgorithm(x_train, y_train, covariance_model, basis)
# Note that the amplitude parameter is computed analytically, we only need to set bounds on the scale parameter.
scale_optimization_bounds = ot.Interval([1.0, 1.0, 1.0, 1.0e-6], [1.0e3, 1.0e2, 1.0e3, 1.0e-2])
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



