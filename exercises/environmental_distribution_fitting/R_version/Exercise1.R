### Exercise 1: Fitting distributions to data (univariate)

setwd("C:/Users/erva/OneDrive - DNV/DNV/Presentations/HIPERWIND_PhDSummerSchool_2023/")

### Read in the data
data = read.csv("WaveData_Example.csv", header = T)
SWH = data[, "Hm0"]

### Visually inspect the data
hist(SWH, freq=F, border = "grey", ylim = c(0, 0.45))
lines(density(SWH))

### Some features: asymmetric data
### Support only for positive values
### May obviously exclude some models such as normal distribution etc... 

### 1. Fit a log-normal distribution to data by MoM
### Calculate sample moments and insert in formula for parameter estimates

m1 = mean(SWH); m2 = mean(SWH^2)

MoM_lnorm = c(log(m1^2/sqrt(m2)), sqrt(log(m2/m1^2)))

### Also, using package fitdistrplus to check
require(fitdistrplus)
MoM_lnorm2 = fitdist(SWH, method = "mme", distr="lnorm")

MoM_lnorm; MoM_lnorm2$estimate

logL_lnorm = function(data, par){
	mu = par[1]; sigma = par[2]
	l = log(dlnorm(data, meanlog=mu, sdlog=sigma))
	sum(l)
}
minuslogL_lnorm = function(data, par){
	-logL_lnorm(data, par)
}

MLE_lnorm = nlm(function(par) minuslogL_lnorm(SWH, par), c(MoM_lnorm), hessian = T)

curve(dlnorm(x, meanlog= MoM_lnorm[1], sdlog=MoM_lnorm[2]), col=2, add=T)
curve(dlnorm(x, meanlog= MLE_lnorm$estimate[1], sdlog=MLE_lnorm$estimate[2]), col=3, add=T)


### Fit Weibull using ML

logL_wei = function(data, par){
	a = par[1]; b = par[2]
	l = log(b) -b*log(a) -(data/a)^b + (b-1)*log(data)
	sum(l)
}
minuslogL_wei = function(data, par){
	(-logL_wei(data, par))
}

MLE_wei = nlm(function(par) minuslogL_wei(SWH, par), c(1, 1), hessian=T)

### MLE estimate and standard error for Weibull fit
MLE_wei_hat = MLE_wei$estimate
MLE_wei_se = sqrt(diag(solve(MLE_wei$hessian)))

### 95% confidence intervals
MLE_wei_ci95 = rbind(MLE_wei_hat[1] + 1.96*c(-1, 1)*MLE_wei_se[1], MLE_wei_hat[2] + 1.96*c(-1, 1)*MLE_wei_se[2])

### check with fitdistrplus
MLE_wei2 = fitdist(SWH, distr="weibull", method = "mle")

MLE_wei_hat; MLE_wei_se; MLE_wei2

###Fit a Weibull by maximum Goodness-of-fit

CvM_wei = fitdist(SWH, distr="weibull", method = "mge", gof="CvM")
KS_wei = fitdist(SWH, distr="weibull", method = "mge", gof="KS")
AD_wei = fitdist(SWH, distr="weibull", method = "mge", gof="AD")
AD2R_wei = fitdist(SWH, distr="weibull", method = "mge", gof="AD2R")

###Fit a Weibull by quantile matching
QM_wei = fitdist(SWH, distr="weibull", method = "qme", probs=c(0.75, 0.99))


###Fitting a gamma distribution by ML
logL_gamma = function(data, par){
	a = par[1]; b = par[2]
	l = dgamma(data, a, b, log=T)
	sum(l)
}
minuslogL_gamma = function(data, par){
	-logL_gamma(data, par)
}

MLE_gamma = nlm(function(par) minuslogL_gamma(SWH, par), c(1, 1), hessian=T)

MLE_exp = fitdist(SWH, distr="exp", method = "mle")


### Extra: Fit a 3-parameter Weibull by MLE

	llik.weibull3 = function(shape, scale, thres, x)           # The log-likelihood of the 3-parameter Weibull distribution
	{
		require(FAdist)						     # Include density function for 3-parameter Weibull
		sum(dweibull3(x, shape, scale, thres, log=T))
	}
	EPS = sqrt(.Machine$double.eps)  # epsilon for very small numbers

	thetahat.weibull3 = function(x)
	{
		if(any(x <= 0)) stop("x values must be positive")

		toptim = function(theta) - llik.weibull3(theta[1], theta[2], theta[3], x)
	
		mu = mean(log(x))
		sigma2 = var(log(x))
		shape.guess = 1.2 / sqrt(sigma2)
		scale.guess = exp(mu + (0.572/shape.guess))
		thres.guess = min(x)-0.001

		res = nlminb(c(shape.guess, scale.guess, thres.guess), toptim, lower=EPS, control=list(trace=10))

		c(shape=res$par[1], scale=res$par[2], thres=res$par[3])
	}
	
MLE_wei3 = thetahat.weibull3(SWH)


### Estimate median, 95% and 99% quantiles from the fitted distributions
### Only for the MLE models
quant_empirical = quantile(p=c(0.5, 0.95, 0.99), SWH)
quant_lnorm = qlnorm(p=c(0.5, 0.95, 0.99), meanlog = MLE_lnorm$estimate[1], sdlog = MLE_lnorm$estimate[2])
quant_wei = qweibull(p=c(0.5, 0.95, 0.99), scale = MLE_wei_hat[1], shape = MLE_wei_hat[2])
quant_gamma = qgamma(p=c(0.5, 0.95, 0.99), MLE_gamma$estimate[1], MLE_gamma$estimate[2])
quant_exp = qexp(p=c(0.5, 0.95, 0.99), MLE_exp$estimate)
quant_wei3 = qweibull3(p=c(0.5, 0.95, 0.99), shape = MLE_wei3["shape"], scale = MLE_wei3["scale"], thres = MLE_wei3["thres"])

quant_empirical; quant_lnorm; quant_wei; quant_gamma; quant_exp; quant_wei3

### Visual model checking by comparing fitted densities (only MLE)
hist(SWH, freq=F, border = "grey", ylim = c(0, 0.45))
lines(density(SWH))
curve(dlnorm(x, meanlog= MLE_lnorm$estimate[1], sdlog=MLE_lnorm$estimate[2]), col=2, add=T, lwd=2)
curve(dweibull(x, scale = MLE_wei_hat[1], shape = MLE_wei_hat[2]), col=3, add=T, lwd=2)
curve(dgamma(x, MLE_gamma$estimate[1], MLE_gamma$estimate[2]), col=4, add=T, lwd=2)
curve(dexp(x, MLE_exp$estimate), col=5, add=T, lwd=2)
curve(dweibull3(x, shape = MLE_wei3["shape"], scale = MLE_wei3["scale"], thres = MLE_wei3["thres"]), col=6, add=T, lwd=2)
abline(v=c(quant_lnorm[3], quant_wei[3], quant_gamma[3], quant_exp[3], quant_wei3[3]), col=2:6, lty=2)
legend("topright", c("Log-normal", "Weibull", "Gamma", "Exponential", "3p-Weibull"), bty="n", text.col=2:6)

### Visual model checking by CDF
plot(ecdf(SWH))
curve(plnorm(x, meanlog= MLE_lnorm$estimate[1], sdlog=MLE_lnorm$estimate[2]), col=2, add=T, lwd=2)
curve(pweibull(x, scale = MLE_wei_hat[1], shape = MLE_wei_hat[2]), col=3, add=T, lwd=2)
curve(pgamma(x, MLE_gamma$estimate[1], MLE_gamma$estimate[2]), col=4, add=T, lwd=2)
curve(pexp(x, MLE_exp$estimate), col=5, add=T, lwd=2)
curve(pweibull3(x, shape = MLE_wei3["shape"], scale = MLE_wei3["scale"], thres = MLE_wei3["thres"]), col=6, add=T, lwd=2)
legend("bottomright", c("Log-normal", "Weibull", "Gamma", "Exponential", "3p Weibull", ""), bty="n", text.col=2:6)

### Zoom in on the tail
plot(ecdf(SWH), ylim = c(0.975, 1))
curve(plnorm(x, meanlog= MLE_lnorm$estimate[1], sdlog=MLE_lnorm$estimate[2]), col=2, add=T, lwd=2)
curve(pweibull(x, scale = MLE_wei_hat[1], shape = MLE_wei_hat[2]), col=3, add=T, lwd=2)
curve(pgamma(x, MLE_gamma$estimate[1], MLE_gamma$estimate[2]), col=4, add=T, lwd=2)
curve(pexp(x, MLE_exp$estimate), col=5, add=T, lwd=2)
curve(pweibull3(x, shape = MLE_wei3["shape"], scale = MLE_wei3["scale"], thres = MLE_wei3["thres"]), col=6, add=T, lwd=2)
abline(v=c(quant_lnorm[3], quant_wei[3], quant_gamma[3], quant_exp[3], quant_wei3[3]), col=2:6, lty=2)
abline(h=0.99, col="grey")
legend("bottomright", c("Log-normal", "Weibull", "Gamma", "Exponential", "3p Weibull"), bty="n", text.col=2:5)

### Prepare QQ-plots
require(car)
par(mfrow=c(2, 3))
qqPlot(SWH, distribution="lnorm", meanlog= MLE_lnorm$estimate[1], sdlog=MLE_lnorm$estimate[2], main = "QQ plot log-normal") 
qqPlot(SWH, distribution="weibull", scale = MLE_wei_hat[1], shape = MLE_wei_hat[2], main = "QQ plot Weibull")
qqPlot(SWH, distribution="gamma", shape = MLE_gamma$estimate[1], rate = MLE_gamma$estimate[2], main = "QQ plot Gamma")
qqPlot(SWH, distribution="exp", rate = MLE_exp$estimate, main = "QQ plot Exponential")
qqPlot(SWH, distribution="weibull3", shape = MLE_wei3["shape"], scale = MLE_wei3["scale"], thres = MLE_wei3["thres"], main = "QQ plot 3p Weibull")
qqPlot(SWH, distribution="weibull", scale = QM_wei$estimate["scale"], shape = QM_wei$estimate["shape"], main = "QQ plot Weibull \n Quantile matching")

### Compare models by AIC
### Formula for AIC: 2*k - 2*lnL_max - Choose the one with lowest value if AIC

AIC_lnorm = 2*length(MLE_lnorm$estimate) + 2*minuslogL_lnorm(SWH, MLE_lnorm$estimate)
AIC_wei = 2*length(MLE_wei_hat) + 2*MLE_wei$minimum
AIC_gamma = 2*length(MLE_gamma$estimate) + 2*MLE_gamma$minimum
AIC_exp = 2 - 2*sum(dexp(SWH, MLE_exp$estimate, log=T))
AIC_wei3 = 2*length(MLE_wei3) - 2*llik.weibull3(shape = MLE_wei3["shape"], scale = MLE_wei3["scale"], thres = MLE_wei3["thres"], SWH)

AIC_lnorm; AIC_wei; AIC_gamma; AIC_exp; AIC_wei3

### Log-normal distribution has lowest AIC, followed by gamma distribuion.  



