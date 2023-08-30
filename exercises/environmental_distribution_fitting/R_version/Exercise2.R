### Exercise 2 - multivariate model

setwd("C:/Users/erva/OneDrive - DNV/DNV/Presentations/HIPERWIND_PhDSummerSchool_2023/")

### Read in the data
data = read.csv("WaveData_Example.csv", header = T)
HsTz = data[, c("Hm0", "Tm02")]

###Exploratory data analysis
summary(HsTz)
par(mfrow=c(2,1))
hist(HsTz[,1], freq=F, border = "grey", main=expression(H[S])); lines(density(HsTz[,1]))
legend("topright", legend = summary(HsTz)[,1], bty="n")
hist(HsTz[,2], freq=F, border = "grey", main = expression(T[Z])); lines(density(HsTz[,2]))
legend("topright", legend = summary(HsTz)[,2], bty="n")

dev.new()
plot(HsTz, xlab=expression(H[S]), ylab = expression(T[Z]), main = "Scatterplot")
cor_lin = cor(HsTz)[1,2]
cor_Kendall= cor(HsTz, method = "kendall")[1,2]
cor_Spearman = cor(HsTz, method = "spearman")[1,2]
legend("bottomright", c(paste("Linear correlation:", round(cor_lin, 3)), paste("Kendall's tau:", round(cor_Kendall, 3)), paste("Spearman's rho:", round(cor_Spearman, 3))), bty="n")

require(MASS)
KernDens2d = kde2d(HsTz[,1], HsTz[,2], n = 100)
contour(KernDens2d, xlab = expression(H[S]), ylab = expression(T[Z]), main = "Bivariate contour plot")
contour(KernDens2d, levels = c(0.001, 0.0001), add=T)
#require(Emcdf)
#plotcdf(as.matrix(HsTz[1:1000,]))
require(rgl)
persp3d(KernDens2d, theta = 20, phi = 30, main = "Empirical density", xlab = "Hs", ylab = "Tz", col=heat.colors(100, alpha = 0.95))

### Marginal models

require(FAdist)		### Density function for 3-parameter Weibull
logL_Wei3 = function(data, par){
	shape = par[1]; scale = par[2]; thres = par[3]
	sum(dweibull3(data, shape, scale, thres, log=T))
	
}
minuslogL_Wei3 = function(data, par){
	-logL_Wei3(data, par)
}

Wei3_hat = function(data){
	mu = mean(log(data))
	sigma2 = var(log(data))
	shape.guess = 1.2 / sqrt(sigma2)
	scale.guess = exp(mu + (0.572/shape.guess))
	thres.guess = min(data)-0.001

	nlm(function(par) minuslogL_Wei3(data, par), c(shape.guess, scale.guess, thres.guess), hessian=T)
}

theta_hat_w3 = Wei3_hat(HsTz[,1])

logL_lnorm = function(data, par){
	mu = par[1]; sigma = par[2]
	l = log(dlnorm(data, meanlog=mu, sdlog=sigma))
	sum(l)
}
minuslogL_lnorm = function(data, par){
	-logL_lnorm(data, par)
}

theta_hat_lnorm = nlm(function(par) minuslogL_lnorm(HsTz[,2], par), c(mean(log(HsTz[,2])), sd(log(HsTz[,2]))), hessian=T)

par(mfrow=c(2,1))
hist(HsTz[,1], freq=F, border = "grey", main=expression(H[S])); lines(density(HsTz[,1]))
curve(dweibull3(x, shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2],  thres = theta_hat_w3$estimate[3]), col=2, add=T, lwd=2)
legend("topright", legend = summary(HsTz)[,1], bty="n")
hist(HsTz[,2], freq=F, border = "grey", main = expression(T[Z]), ylim = c(0, 0.25)); lines(density(HsTz[,2]))
legend("topright", legend = summary(HsTz)[,2], bty="n")
curve(dlnorm(x, theta_hat_lnorm$estimate[1], theta_hat_lnorm$estimate[2]), add=T, col=2, lwd=2)

### Saving marginal parameters in a list
marginalPara = list(wei3 = list(shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2],  thres = theta_hat_w3$estimate[3]), lnorm = list(meanlog = theta_hat_lnorm$estimate[1], sdlog = theta_hat_lnorm$estimate[2]))
par(mfrow=c(1,1))


###Build joint model by assuming independence
### Plot empirical density contours
contourlevels = c(0.01, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.0005, 0.00001)
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Joint models", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)

### Copare with independent model
joint_ind = function(x, y) dweibull3(x, shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2],  thres = theta_hat_w3$estimate[3])*dlnorm(y, theta_hat_lnorm$estimate[1], theta_hat_lnorm$estimate[2])
z_ind = matrix(ncol=201, nrow = 201)
for (i in 0:200)
	for(j in 0:200)
		z_ind[i+1, j+1] = joint_ind(i/10, j/10)

contour(z_ind, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=2, add=T, drawlabels=F, lwd=2)

### Build conditional model using ML
### Use same marginal, but need to fit the conditional log-normal

logL_lnorm_cond = function(data, par){
	a0 = par[1]; a1=par[2]; a2 = par[3]; b0 = par[4]; b1 = par[5]; b2 = par[6]
	mu = function(x) a0 + a1*x^(a2)
	sigma = function(x) b0 + b1*exp(b2*x)
	l = dlnorm(data[,2], meanlog=mu(data[,1]), sdlog=sigma(data[,1]), log=T)
	sum(l)
}
minuslogL_lnorm_cond = function(data, par){
	-logL_lnorm_cond(data, par)
}

theta_hat_lnorm_cond = nlm(function(par) minuslogL_lnorm_cond(HsTz, par), c(1, 1, 1, 1, 1, 1), hessian=T)

### Copare with independent model
mu_cond = function(x) theta_hat_lnorm_cond$estimate[1] + theta_hat_lnorm_cond$estimate[2]*x^(theta_hat_lnorm_cond$estimate[3])
sigma_cond = function(x) theta_hat_lnorm_cond$estimate[4] + theta_hat_lnorm_cond$estimate[5]*exp(theta_hat_lnorm_cond$estimate[6]*x)

joint_cond = function(x, y) dweibull3(x, shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2],  thres = theta_hat_w3$estimate[3])*dlnorm(y, mu_cond(x), sigma_cond(x))
z_cond = matrix(ncol=201, nrow = 201)
for (i in 0:200)
	for(j in 0:200)
		z_cond[i+1, j+1] = joint_cond(i/10, j/10)

contour(z_cond, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), xlim = c(0, 20), ylim = c(0, 20), levels = contourlevels, col=3, add=T, drawlabels=F, lwd=2)

### Fit copula models to the bivariate data
### Try with normal, Gumbel, Frank and Clayton copula
### May use library copula
require(copula)

Normal_cop = fitCopula(normalCopula(dim=2), pobs(HsTz), method = "ml")
Gumbel_cop = fitCopula(evCopula("gumbel"), pobs(HsTz), method = "ml")
Frank_cop = fitCopula(archmCopula("frank"), pobs(HsTz), method = "ml", lower = -10)
Clayton_cop = fitCopula(archmCopula("clayton"), pobs(HsTz), method = "ml")
Joe_cop = fitCopula(archmCopula("joe"), pobs(HsTz), method = "ml")
HR_cop = fitCopula(evCopula("huslerReiss"), pobs(HsTz), method = "ml")


###Define multivariate probability distributions from the copula and the marginals
joint_Normal = mvdc(Normal_cop@copula, c("weibull3", "lnorm"), paramMargins= marginalPara)
joint_Gumbel = mvdc(Gumbel_cop@copula, c("weibull3", "lnorm"), paramMargins= marginalPara)
joint_Frank = mvdc(Frank_cop@copula, c("weibull3", "lnorm"), paramMargins= marginalPara)
joint_Clayton = mvdc(Clayton_cop@copula, c("weibull3", "lnorm"), paramMargins= marginalPara)
joint_Joe = mvdc(Joe_cop@copula, c("weibull3", "lnorm"), paramMargins= marginalPara)
joint_HR = mvdc(HR_cop@copula, c("weibull3", "lnorm"), paramMargins= marginalPara)

z_Normal = matrix(ncol=201, nrow = 201); z_Gumbel = matrix(ncol=201, nrow = 201); z_Frank = matrix(ncol=201, nrow = 201); z_Clayton = matrix(ncol=201, nrow = 201); 
z_Joe = matrix(ncol=201, nrow = 201); z_HR = matrix(ncol=201, nrow = 201)

for (i in 0:200)
	for(j in 0:200){
		z_Normal[i+1, j+1] = dMvdc(c(i/10, j/10), joint_Normal)
		z_Gumbel[i+1, j+1] = dMvdc(c(i/10, j/10), joint_Gumbel)
		z_Frank[i+1, j+1] = dMvdc(c(i/10, j/10), joint_Frank)
		z_Clayton[i+1, j+1] = dMvdc(c(i/10, j/10), joint_Clayton)
		z_Joe[i+1, j+1] = dMvdc(c(i/10, j/10), joint_Joe)
		z_HR[i+1, j+1] = dMvdc(c(i/10, j/10), joint_HR)
}

contour(z_Normal, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=4, add=T, drawlabels=F, lwd=2)
contour(z_Gumbel, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=5, add=T, drawlabels=F, lwd=2)
contour(z_Frank, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=6, add=T, drawlabels=F, lwd=2)
contour(z_Clayton, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=7, add=T, drawlabels=F, lwd=2)
contour(z_Joe, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col="skyblue", add=T, drawlabels=F, lwd=2)
contour(z_HR, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col="orange", add=T, drawlabels=F, lwd=2)
legend("bottomright", c("Independent", "Conditionl", "Normal copula", "Gumbel copula", "Frank copula", "Clayton copula", "Joe copula", "HR copula"), bty="n", text.col=c(2:7, "skyblue", "orange"), cex=0.8)

### Compare models by AIC

AIC = list()
AIC[["Ind"]] = 2*(3+2) + 2*(theta_hat_w3$minimum + theta_hat_lnorm$minimum)
AIC[["Cond"]] = 2*(3 + 6) + 2*(theta_hat_w3$minimum + theta_hat_lnorm_cond$minimum)
AIC[["Normal"]] = AIC[["Ind"]] - 2*Normal_cop@loglik + 2*1
AIC[["Gumbel"]] = AIC[["Ind"]] - 2*Gumbel_cop@loglik + 2*1
AIC[["Frank"]] = AIC[["Ind"]] - 2*Frank_cop@loglik + 2*1
AIC[["Clayton"]] = AIC[["Ind"]] - 2*Clayton_cop@loglik + 2*1 
AIC[["Joe"]] = AIC[["Ind"]] - 2*Joe_cop@loglik + 2*1
AIC[["HR"]] = AIC[["Ind"]] - 2*HR_cop@loglik + 2*1


### Illustrate results by some plots
par(mfrow=c(2, 4))
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Independent model", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_ind, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=2, add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Ind"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Conditional model", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_cond, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), xlim = c(0, 20), ylim = c(0, 20), levels = contourlevels, col=3, add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Cond"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Normal copula", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_Normal, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=4, add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Normal"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Gumbel copula", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_Gumbel, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=5, add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Gumbel"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Frank copula", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_Frank, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=6, add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Frank"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Clayton copula", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_Clayton, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col=7, add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Clayton"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Joe copula", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_Joe, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col="skyblue", add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["Joe"]], 2)), bty="n") 
contour(kde2d(HsTz[,1], HsTz[,2]), levels = contourlevels, drawlabels=F, col="grey", xlim = c(0, 20), ylim = c(0, 20), main = "Hussler-Reiss copula", xlab = expression(H[S]), ylab = expression(T[Z]), lwd=2)
contour(z_HR, x = seq(0, 20, by=0.1), y = seq(0, 20, by=0.1), levels = contourlevels, col="orange", add=T, drawlabels=F, lwd=2)
legend("bottomright", paste("AIC = ", round(AIC[["HR"]], 2)), bty="n") 






