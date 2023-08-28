#! /usr/bin/env python3

import openturns as ot
import openturns.viewer as viewer
import copy
from matplotlib import pylab as plt
from math import log, sqrt
from scipy.optimize import minimize

# Helpful utility for drawing
max_curve_cnt = 7
def createHSVColor(indexCurve, maximumNumberOfCurves):
    """Create a HSV color for the indexCurve-th curve
    from a sample with maximum size equal to maximumNumberOfCurves"""
    color = ot.Drawable.ConvertFromHSV(
        indexCurve * 360.0 / maximumNumberOfCurves, 1.0, 1.0
    )
    return color

def getBasicStatistics(sample):
    min_val = sample.getMin()[0]
    quartile_1 = sample.computeQuantile(0.25)[0]
    median_val = sample.computeMedian()[0]
    mean_val = sample.computeMean()[0]
    quartile_3 = sample.computeQuantile(0.75)[0]
    max_val = sample.getMax()[0]

    return (min_val, quartile_1, median_val, mean_val, quartile_3, max_val)

def getBivariateStatistics(sample):
    # Linear Correlation
    lc=sample.computeLinearCorrelation()[0,1]
    # Kendall's Tau
    kt=sample.computeKendallTau()[0,1]
    # Spearman's Rho
    sr=sample.computePearsonCorrelation()[0,1]
    # return
    return (lc,kt,sr)

# Load the csv data
###################

sample = ot.Sample.ImportFromCSVFile("WaveData_Example.csv",",")
# Select only the Height data
sample_H = sample[:,0]
sample_T = sample[:,3]
sample_HT = sample[:,(0,3)]
sample_TH = sample[:,(3,0)]

# Exploratory data analysis
###########################

# print summary of sample_H
stats_h=getBasicStatistics(sample_H)
print("Basic statistics for H")
print("min, Q1, median, mean, Q3, max")
print(stats_h)

# Show a histogram of the H data
graph_h = ot.HistogramFactory().build(sample_H).drawPDF()
graph_h.setColors(["black"])
graph_h.setLegends(["hist"])
view = viewer.View(graph_h)
plt.show()

# print summary of sample_T
stats_t=getBasicStatistics(sample_T)
print("Basic statistics for T")
print("min, Q1, median, mean, Q3, max")
print(stats_t)

# Show a histogram of the T data
graph_t = ot.HistogramFactory().build(sample_T).drawPDF()
graph_t.setColors(["black"])
graph_t.setLegends(["hist"])
view = viewer.View(graph_t)
plt.show()

print("Basic statistics for both H & T")
print("linear_correlation, Kendall's Tau, Spearman's Rho")
print(getBivariateStatistics(sample_HT))

# Scatter plot
scatter_graph = ot.Graph("H-T data", "H", "T", True, "")
cloud = ot.Cloud(sample_HT)
scatter_graph.add(cloud)
view = viewer.View(scatter_graph)
plt.show()

# Maximum likelihood fit of marginal distributions with 3 parameter model
#########################################################################

# Use a package that gives a liklihood maximumization
factory_mle_wbl3_h = ot.MaximumLikelihoodFactory(ot.WeibullMin())
# Give bounds for the optimization
# 3.5556343 1.7025580 0.3997116
lowerBound=[1.0, 1.0, -1.0]
upperBound=[5.0, 3.0, 1.0]
bounds = ot.Interval(lowerBound, upperBound, [True,True,True], [True,True,True])
factory_mle_wbl3_h.setOptimizationBounds(bounds)
# Set the starting point for the optimization
solver = factory_mle_wbl3_h.getOptimizationAlgorithm()
solver.setStartingPoint([4.0,2.0,0.0])
# solve the distribution
dist_mle_wbl3_h = factory_mle_wbl3_h.build(sample_H)
print(dist_mle_wbl3_h)
# Graph the results
graph_mle_wbl3_h = dist_mle_wbl3_h.drawPDF()
graph_h.add(graph_mle_wbl3_h)
view = viewer.View(graph_h)
plt.show()

# Use a package that gives a liklihood maximumization
factory_mle_ln_t = ot.MaximumLikelihoodFactory(ot.LogNormal())
# Set gamma = 0
##factory_mle_ln_t.setKnownParameter([0.0],[2])
# Give bounds for the optimization
lowerBound=[0.5, 0.01, 2.0]
upperBound=[4.0, 2.0, 5.0]
bounds = ot.Interval(lowerBound, upperBound, [True,True,True], [True,True,True])
factory_mle_ln_t.setOptimizationBounds(bounds)
# Set the starting point for the optimization
solver = factory_mle_ln_t.getOptimizationAlgorithm()
# Use method of moments to create guess
guess_gamma=stats_t[0]-0.01
guess_mu=log(stats_t[2]-guess_gamma)
guess_sigma=sqrt(2.0*(log(stats_t[3]-guess_gamma)-guess_mu))
print(guess_mu,guess_sigma,guess_gamma)
# Set the starting point
solver.setStartingPoint([guess_mu,guess_sigma,guess_gamma])
# solve the distribution
dist_mle_ln_t = factory_mle_ln_t.build(sample_T)
print(dist_mle_ln_t)
# Graph the results
graph_mle_ln_t = dist_mle_ln_t.drawPDF()
graph_t.add(graph_mle_ln_t)
view = viewer.View(graph_t)
plt.show()

# Combine marginals assuming they are independent
#########################################################################

dist_bivar_indep = ot.ComposedDistribution([dist_mle_wbl3_h,dist_mle_ln_t])
graph_bivar_indep = dist_bivar_indep.drawPDF()
scatter_graph.add(graph_bivar_indep)
view = viewer.View(scatter_graph)
plt.show()

# Lets create some fits with different copulas
#########################################################################

factory_bivar_norm = ot.MaximumLikelihoodFactory(ot.ComposedDistribution([ot.WeibullMin(),ot.LogNormal()], ot.NormalCopula()))
dist_bivar_norm = factory_bivar_norm.build(sample_HT)
print(dist_bivar_norm)
scatter_graph = ot.Graph("H-T data", "H", "T", True, "")
cloud = ot.Cloud(sample_HT)
scatter_graph.add(cloud)
scatter_graph.add(dist_bivar_norm.drawPDF())
view = viewer.View(scatter_graph)
plt.show()

factory_bivar_gumbel = ot.MaximumLikelihoodFactory(ot.ComposedDistribution([ot.WeibullMin(),ot.LogNormal()], ot.GumbelCopula()))
dist_bivar_gumbel = factory_bivar_gumbel.build(sample_HT)
print(dist_bivar_gumbel)
scatter_graph = ot.Graph("H-T data", "H", "T", True, "")
cloud = ot.Cloud(sample_HT)
scatter_graph.add(cloud)
scatter_graph.add(dist_bivar_gumbel.drawPDF())
view = viewer.View(scatter_graph)
plt.show()

factory_bivar_clayton = ot.MaximumLikelihoodFactory(ot.ComposedDistribution([ot.WeibullMin(),ot.LogNormal()], ot.ClaytonCopula()))
dist_bivar_clayton = factory_bivar_clayton.build(sample_HT)
print(dist_bivar_clayton)
scatter_graph = ot.Graph("H-T data", "H", "T", True, "")
cloud = ot.Cloud(sample_HT)
scatter_graph.add(cloud)
scatter_graph.add(dist_bivar_clayton.drawPDF())
view = viewer.View(scatter_graph)
plt.show()

# Define a conditional model
#########################################################################

# This is the log likelihood objective function for minimization
def get_log_likelihood_of_conditional(par):

    a0=par[0]
    a1=par[1]
    a2=par[2]
    b0=par[3]
    b1=par[4]
    b2=par[5]
    c0=par[6]
    c1=par[7]
    c2=par[8]
    ln = ot.LogNormal()
    retval=0.0
    for H, T in sample_HT:
        mu = a0+a1*H**a2
        sg = b0+b1*H**b2
        gm = c0+c1*H**c2
        #print(mu,sg,gm)
        ln.setParameter([mu,sg,gm])
        p = ln.computePDF(T)
        if p<=0.0:
            print("domain error")
            retval-=300.0
        else:
            retval+=log(p)
    print(par,-retval)
    return -retval

# This is the gradient of the log likelihood objective function for minimization
def get_log_likelihood_of_conditional_gradient(par):

    a0=par[0]
    a1=par[1]
    a2=par[2]
    b0=par[3]
    b1=par[4]
    b2=par[5]
    c0=par[6]
    c1=par[7]
    c2=par[8]
    ln = ot.LogNormal()
    retval=[0.0]*9
    for H, T in sample_HT:
        mu = a0+a1*H**a2
        sg = b0+b1*H**b2
        gm = c0+c1*H**c2

        ln.setParameter([mu,sg,gm])
        pnt = ot.Point(1,T)
        p = ln.computePDF(pnt)

        if p<=0.0:
            p=1.0e-300

        dmu_da0 = 1.0
        dmu_da1 = H**a2
        dmu_da2 = a1*a2*H**(a2-1.0)

        dsg_db0 = 1.0
        dsg_db1 = H**b2
        dsg_db2 = b1*b2*H**(b2-1.0)

        dgm_dc0 = 1.0
        dgm_dc1 = H**c2
        dgm_dc2 = c1*c2*H**(c2-1.0)

        dp_dTheta = ln.computePDFGradient(pnt)
        retval[0]-=dmu_da0*dp_dTheta[0]/p
        retval[1]-=dmu_da1*dp_dTheta[0]/p
        retval[2]-=dmu_da2*dp_dTheta[0]/p
        retval[3]-=dsg_db0*dp_dTheta[1]/p
        retval[4]-=dsg_db1*dp_dTheta[1]/p
        retval[5]-=dsg_db2*dp_dTheta[1]/p
        retval[6]-=dgm_dc0*dp_dTheta[2]/p
        retval[7]-=dgm_dc1*dp_dTheta[2]/p
        retval[8]-=dgm_dc2*dp_dTheta[2]/p

    return retval

# Set the boundaries of the problem
bnds = [(0.1,1.0),(0.1,2.0),(0.01,1.0),(0.000,1.0),(0.01,1.0),(-1.0,1.0),(0.000,5.0),(0.00,2.0),(-1.0,3.0)]
# Try a different set of starting points (inspired by the solution of the same problem solved with R)
#strt = [0.497464740,1.138225864,0.222807320,0.003084484,0.178687761,-0.091843939]
#strt = [0.49,1.13,0.22,0.0030,0.17,-0.091]
strt = [0.5,1.0,0.2,0.003,0.2,-0.1,2.0,0.0,1.0]

# Try different optimization algorithms
#sln = minimize(get_log_likelihood_of_conditional, strt, method='SLSQP', bounds=bnds, tol=1.0e-100 , options={'ftol':1.0e-100})
#sln = minimize(get_log_likelihood_of_conditional, strt, method='L-BFGS-B', bounds=bnds)
sln = minimize(get_log_likelihood_of_conditional, strt, method='L-BFGS-B', bounds=bnds, jac=get_log_likelihood_of_conditional_gradient)
#sln = minimize(get_log_likelihood_of_conditional, strt, method='SLSQP', bounds=bnds, jac=get_log_likelihood_of_conditional_gradient)
cond_par = sln.x

# Define a function that describes the conditional relationship using both stochastic and parametric variables
f_H_T_raw = ot.SymbolicFunction(['h','a0','a1','a2','b0','b1','b2','c0','c1','c2'], ['a0+a1*h^a2', 'b0+b1*h^b2', 'c0+c1*h^c2'])
# Define the parameters (variables 1-9)
f_H_T = ot.ParametricFunction(f_H_T_raw, [1,2,3,4,5,6,7,8,9], cond_par)
# Define the base distribution for T
T_given_H = ot.LogNormal()
# Copy our original weibull distribution for H
H_dist = copy.deepcopy(dist_mle_wbl3_h)
# Define a multivariate distribution with conditional dependence
TH_dist = ot.BayesDistribution(T_given_H, H_dist, f_H_T)
print(TH_dist)
scatter_graph = ot.Graph("T-H data", "T", "H", True, "")
cloud = ot.Cloud(sample_TH)
scatter_graph.add(cloud)
scatter_graph.add(TH_dist.drawPDF())
view = viewer.View(scatter_graph)
plt.show()

