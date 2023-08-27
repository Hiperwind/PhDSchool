# %% [markdown]
# # Reliability analysis using OpenTURNS
# 
# This notebook presents the main steps of the uncertainty quantification framework. 
# 
# The analytical case of the flooding model illustrates the tools offered by OpenTURNS to perform a typical reliability analysis study.  

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
# 
# Function in OpenTURNS can be defined either using `ot.SymbolicFunction` when the function is explicit, otherwise it can be done using `ot.PythonFunction`

# %%
water_height_function = ot.SymbolicFunction(['Q', 'Ks', 'B', 'Alpha'], ['(Q / (Ks * B * sqrt(Alpha)))^(3/5)'])
# Testing the function
x_test = [1330, 30, 200, 0.001]
print(f"Water height({x_test}) = {water_height_function(x_test)}")

# %% [markdown]
# ### [Step B]: uncertainty model definition
# 
# See the large catalogue of marginals and copulas available in OpenTURNS: 
# 
# http://openturns.github.io/openturns/latest/user_manual/probabilistic_modelling.html

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
# - Line Sampling (soon available in OpenTURNS)
# - Subset Sampling
# - Directional Sampling
# - Cross-entropy Importance Sampling
# - Non-Adaptive Importance Sampling
#   
# This notebook illustrates some rare event estimation (reliability analysis) methods on this case. 

# %% [markdown]
# #### Data visualization
# 

# %%
data = pd.DataFrame(np.array(x_MC_sample), columns=list(joint_distribution.getDescription()))
data['Y'] = np.array(y_MC_sample)
copulogram = cp.Copulogram(data)
copulogram.draw(alpha=0.6, marker='.');

# %%
threshold = 16.

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
# Monte Carlo failure probability with 1000 samples (zero is found for low probabilities, corresponding to low thresholds)
MC_pf = data['is_failed'].mean()
print()
print(f"Monte Carlo pf with 1000 samples: {MC_pf}")


# %% [markdown]
# ## OpenTURNS reliability analysis methods
# 
# See the following API page regarding the methods used below: 
# 
# https://openturns.github.io/openturns/latest/user_manual/threshold_probability_simulation_algorithms.html

# %% [markdown]
# ### Define a generic `ThresholdEvent`
# 
# The `ThresholdEvent` object gathers all the inputs needed for a reliability problem

# %%
# threshold = 10.0
X = ot.RandomVector(joint_distribution)
Y = ot.CompositeRandomVector(water_height_function, X)
failure_event = ot.ThresholdEvent(Y, ot.GreaterOrEqual(), threshold)

# %%
standard_event = ot.StandardEvent(failure_event)

# cv for simulation methods

cv = 0.05

## Monte Carlo analysis using OpenTURNS (for any probability and threshold)

mc_experiment = ot.MonteCarloExperiment()
algoMC = ot.ProbabilitySimulationAlgorithm(standard_event, mc_experiment)
algoMC.setMaximumCoefficientOfVariation(cv)
algoMC.setMaximumOuterSampling(int(1e7))

in_num_LSF_calls = water_height_function.getEvaluationCallsNumber()

algoMC.run()
result = algoMC.getResult()
Pf = result.getProbabilityEstimate()
num_LSF_calls = water_height_function.getEvaluationCallsNumber() - in_num_LSF_calls

print("Monte Carlo Probability Pf = ", Pf)
print("CV = ", result.getCoefficientOfVariation())
print("Monte Carlo Confidence interval : ", "Lower bound =", 0.8*Pf, "    Upper bound = ", 1.2*Pf)
print("Number of LSF calls = ", num_LSF_calls)
print("Verified Coefficient of variation (= 1/sqrt(N*Pf)) =", 1/(num_LSF_calls * Pf)**0.5)
print()

# %%
palette = ot.Drawable.BuildDefaultPalette(10)
graph = algoMC.drawProbabilityConvergence(0.95)
graph.setColors([palette[3], palette[1], palette[1]])
view = viewer.View(graph)

# %% [markdown]
# ### FORM / SORM

# %%
# Many other optimization algorithms (i.e., solvers) available.
# See: https://openturns.github.io/openturns/latest/user_manual/optimization.html
# AbdoRackwitz is considered as the most efficient solver for FORM evaluations
 
solver = ot.AbdoRackwitz()
starting_point = joint_distribution.getMean()
# See FORM API: https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.FORM.html
FORM_algo = ot.FORM(solver, failure_event, starting_point)
FORM_algo.run()
FORM_result = FORM_algo.getResult()
beta = FORM_result.getHasoferReliabilityIndex()
importance_factors = FORM_result.getImportanceFactors()
u_star = FORM_result.getStandardSpaceDesignPoint()
FORM_pf = FORM_result.getEventProbability()
print(f"FORM pf: {FORM_pf:.2e}")

# %%
# See SORM API: https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.SORM.html
SORM_algo = ot.SORM(solver, failure_event, starting_point)
SORM_algo.run()# %%
standard_event = ot.StandardEvent(failure_event)
SORM_result = SORM_algo.getResult()
beta = SORM_result.getHasoferReliabilityIndex()
importance_factors = SORM_result.getImportanceFactors()
u_star = SORM_result.getStandardSpaceDesignPoint()
SORM_pf = SORM_result.getEventProbabilityBreitung()
print(f"SORM pf: {SORM_pf:.2e}")
print()

# %% [markdown]
# threshold = 10.0
# ### Importance Sampling / Quasi-Monte Carlo / LHS

# %%
# Set random seed to exactly repeat results 
#ot.RandomGenerator.SetSeed(0)

# %%
# Define an experiment (i.e., a sampling method among MC, IS, QMC, LHS)

## FORM - Importance sampling
importance_density = ot.Normal(u_star, [1.0] * dim) # Standard Gaussian centered around the FORM design point
is_experiment = ot.ImportanceSamplingExperiment(importance_density)

# Quasi-Monte Carlo
qmc_experiment = ot.LowDiscrepancyExperiment()
#qmc_experiment.setRandomize(True) # Randomized option

# LHS
lhs_experiment = ot.LHSExperiment()
# %%

#lhs_experiment.setAlwaysShuffle(True) # Randomized option

# FORM-Importance Sampling
algo = ot.ProbabilitySimulationAlgorithm(standard_event, is_experiment) # The user can change the experiment with one defined above 
# Algorithm stopping criterions
algo.setMaximumOuterSampling(int(1e4))
algo.setBlockSize(1)
algo.setMaximumCoefficientOfVariation(cv)
# Perform the simulation
algo.run()
simulation_results = algo.getResult()
simulation_pf = simulation_results.getProbabilityEstimate()
print(f'FORM-IS failure probability = {simulation_pf:.2e}')

# %%
palette = ot.Drawable.BuildDefaultPalette(10)
graph = algo.drawProbabilityConvergence(0.95)
graph.setColors([palette[3], palette[1], palette[1]])
view = viewer.View(graph)

# %% [markdown]
# ### Subset sampling

# %%
SS_algo = ot.SubsetSampling(failure_event)
SS_algo.setMaximumOuterSampling(int(1e4))
SS_algo.setBlockSize(1)
SS_algo.setConditionalProbability(0.1)
SS_algo.run()
SS_results = SS_algo.getResult()
levels = SS_algo.getThresholdPerStep()
SS_pf = SS_results.getProbabilityEstimate()
print(f'Subset sampling failure probability = {SS_pf:.2e}')
print("Number of steps for Subset Sampling: ",SS_algo.getStepsNumber())
print("Precision for Subset Sampling: ", SS_algo.getCoefficientOfVariationPerStep())

# %%
graph = SS_algo.drawProbabilityConvergence(0.95)
graph.setColors([palette[3], palette[1], palette[1]])
view = viewer.View(graph)

# %%



