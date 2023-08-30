### Exercise 3: Extreme value analysis

setwd("C:/Users/erva/OneDrive - DNV/DNV/Presentations/HIPERWIND_PhDSummerSchool_2023/")

### Read in the data
data = read.csv("WaveData_Example.csv", header = T)
SWH = data[, "Hm0"]

### Fit a 3-parameter Weibull distribution to the data

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

theta_hat_w3 = Wei3_hat(SWH)

### Alternative fitting method for same distribution - 2nd order Anderson-Darling statistic
require(fitdistrplus)
theta_hat_w3_AD = fitdist(SWH, distr="weibull3", method = "mge", gof="AD2R", start = list(shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2], thres = theta_hat_w3$estimate[3]))

### Estimate 10-, 20- and 100-year return values
q_3h = 1- 1/(365.25*8*c(10, 20, 100))      # Extreme quantiles for 3-hourly data

Wei3_rv = qweibull3(q_3h, shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2], thres = theta_hat_w3$estimate[3])
Wei3_AD_rv = qweibull3(q_3h, shape = theta_hat_w3_AD$estimate[1], scale = theta_hat_w3_AD$estimate[2], thres = theta_hat_w3_AD$estimate[3])


### Illustrate the results
hist(SWH, border = "grey", freq = F, xlim = c(0, 24), ylim = c(0, 0.3), main = "EVA", yaxt = "n")
lines(density(SWH))
curve(dweibull3(x, shape = theta_hat_w3$estimate[1], scale = theta_hat_w3$estimate[2], thres = theta_hat_w3$estimate[3]), col=2, add=T)
curve(dweibull3(x, shape = theta_hat_w3_AD$estimate[1], scale = theta_hat_w3_AD$estimate[2], thres = theta_hat_w3_AD$estimate[3]), col=3, add=T)
abline(v=Wei3_rv, col=2, lty=2:4)
abline(v=Wei3_AD_rv, col=3, lty=2:4)

###Annual maximum analysis
### Extract maxima within 1 year

extractBM = function(x, b){
	blocks = split(x, ceiling(seq_along(x)/b))
	bm = matrix(unlist(lapply(blocks, function(x) cbind(which.max(x), max(x)))), ncol=2, byrow=T)
	for(i in 1:nrow(bm)) bm[i,1] = bm[i,1] + (i-1)*b
	bm
}

AM = extractBM(SWH, 365.25*8)

### Fit a GEV distribution. Use package fitdistrplus to fit. 

theta_hat_GEV = fitdist(AM[,2], distr="gev", method = "mle", start = list(shape = 1, scale = 1, location = 0))
q_am = 1- 1/c(10, 20, 100)
GEV_rv = qgev(q_am, shape = theta_hat_GEV$estimate["shape"], scale = theta_hat_GEV$estimate["scale"], location = theta_hat_GEV$estimate["location"])

### Check if a Gumbel reduction seems appropriate by looking at the shape 
### parameter and assessing whether it is significantly different from 0

theta_hat_GEV$estimate["shape"]+1.96*theta_hat_GEV$sd["shape"]*c(-1, 1)
### 95% confidence interval contains 0. Cannot reject the Gumbel model based on the data

theta_hat_Gumbel = fitdist(AM[,2], distr="gumbel", method = "mle", start = list(scale = 1, location = 0))
Gumbel_rv = qgumbel(q_am, scale = theta_hat_Gumbel$estimate["scale"], location = theta_hat_Gumbel$estimate["location"])

par(new=T)
plot(density(AM[,2]), xlim = c(0, 22), ylim = c(0, 0.7), axes = F, xlab = "", main = "")
curve(dgev(x, shape = theta_hat_GEV$estimate["shape"], scale = theta_hat_GEV$estimate["scale"], location = theta_hat_GEV$estimate["location"]), col=4, add=T)
curve(dgumbel(x, scale = theta_hat_Gumbel$estimate["scale"], location = theta_hat_Gumbel$estimate["location"]), add=T, col=5)
abline(v=GEV_rv, col=4, lty=2:4)
abline(v=Gumbel_rv, col=5, lty=2:4)

### POT modelling
### We disregard de-clustering for now

POT_8 = SWH[SWH>8]
POT_10 = SWH[SWH>10]
POT_12 = SWH[SWH>12]

### Estimate quantiles for POT data
years = length(SWH)/(365.25*8)
ny = c(sum(SWH>8)/years, sum(SWH>10)/years, sum(SWH>12)/years)

q_POT = rbind(1-1/(ny[1]*c(10, 20, 100)), 1-1/(ny[2]*c(10, 20, 100)), 1-1/(ny[3]*c(10, 20, 100)))

###Fit a GPD model to the data
require(POT)

theta_hat_GPD8 = fitgpd(POT_8, threshold=10, est = "mle")
theta_hat_GPD10 = fitgpd(POT_10, threshold=10, est = "mle")
theta_hat_GPD12 = fitgpd(POT_12, threshold=12, est = "mle")

POT_8_rv = qgpd(q_POT[1,], loc=10, scale =theta_hat_GPD8$fitted.values["scale"], shape = theta_hat_GPD8$fitted.values["shape"])
POT_10_rv = qgpd(q_POT[2,], loc=10, scale =theta_hat_GPD10$fitted.values["scale"], shape = theta_hat_GPD10$fitted.values["shape"])
POT_12_rv = qgpd(q_POT[3,], loc=12, scale =theta_hat_GPD12$fitted.values["scale"], shape = theta_hat_GPD12$fitted.values["shape"])

lines(density(POT_10), col="grey")
lines(density(POT_12), col="grey")
curve(dgpd(x, loc=10, scale =theta_hat_GPD10$fitted.values["scale"], shape = theta_hat_GPD10$fitted.values["shape"]), col=6, add=T)
curve(dgpd(x, loc=12, scale =theta_hat_GPD12$fitted.values["scale"], shape = theta_hat_GPD12$fitted.values["shape"]), col=7, add=T)
abline(v=POT_10_rv, col=6, lty=2:4)
abline(v=POT_12_rv, col=7, lty=2:4)

lines(density(POT_8), col="grey")
curve(dgpd(x, loc=8, scale =theta_hat_GPD8$fitted.values["scale"], shape = theta_hat_GPD8$fitted.values["shape"]), col="orange", add=T)
abline(v=POT_8_rv, col="orange", lty=2:4)



### Summarizes results
sink("EVA results.txt")
cat("Summary of EVA results: \n" )
print(matrix(c(Wei3_rv, Wei3_AD_rv, GEV_rv, Gumbel_rv, POT_8_rv, POT_10_rv, POT_12_rv), ncol=3, byrow=T, dimnames = list(c("3p Weibull, MLE", "3p Weibull, GoF", "AM: GEV", "AM: Gumbel", "POT>8", "POT>10", "POT>12"), c("10-year", "20-year", "100-year"))))
sink()
