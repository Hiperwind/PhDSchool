### Exercise 4: Environmental contours

setwd("C:/Users/erva/OneDrive - DNV/DNV/Presentations/HIPERWIND_PhDSummerSchool_2023/")


### Read in the data
### 30 years of data
require(data.table)
##data = read.csv("WaveData_Example.csv", header = T)
data = fread("WaveData_Example.csv", header = T)
HsTz = data[, c("Hm0", "Tm02")]

colnames(HsTz) = c("hs", "tp")

### Install ECSADES software

#require("devtools")
#devtools::install_github(repo="ECSADES/ecsades-r", branch='master', subdir='ecsades', dependencies = TRUE)

require(ecsades)

help(ecsades)

### Fit the conditional model to the data
npy = 365.25*8
rp = c(10, 20, 100)
Wln = fit_wln(data = HsTz, npy=npy)

### Calculate direct sampling contours
DSC = estimate_dsc(Wln, output_rp = rp)

plot_ec(DSC, raw_data = HsTz)

### Calculate I-FORM contours
IFC = estimate_iform(Wln, output_rp = rp)
dev.new()
plot_ec(IFC, raw_data = HsTz)

### Plot in same plot to compare
plot(DSC[which(DSC[,1]==100), 2:3], type="l", lwd=2, main = "Contour comparison")
lines(DSC[which(DSC[,1]==20), 2:3], lwd=2, col=2)
lines(DSC[which(DSC[,1]==10), 2:3], type="l", lwd=2, col=3)

lines(IFC[which(IFC[,1]==100), 2:3], lwd=2, lty=2)
lines(IFC[which(IFC[,1]==20), 2:3], lwd=2, lty=2, col=2)
lines(IFC[which(IFC[,1]==10), 2:3], lwd=2, lty=2, col=3)
legend("bottomright", c("Direct sampling", "I-FORM"), bty="n", lty=1:2)
legend("topleft", c("10-year", "20-year", "100-year"), bty="n", col=3:1, text.col=3:1)

### HT-model
data(noaa_ww3)

noaa_ht = fit_ht(data = noaa_ww3, npy = nrow(noaa_ww3)/10, p_margin_thresh = 0.95, p_dep_thresh = 0.95)
sample_ht = sample_jdistr(jdistr = noaa_ht, sim_year = 1000)

plot(noaa_ww3[,2:3], main = "Sampling from a conditional extremes model", xlim = c(0, 12), ylim = c(5, 28))
points(sample_ht[which(sample_ht[,1]>6), ], col=2)
points(sample_ht[which(sample_ht[,2]>20), ], col=3)
points(noaa_ww3[,2:3])
abline(v=6, lty=2, col=2)
abline(h=20, lty=2, col=3)
legend("topright", c("10 years of data", "hs > 6m from 1000 year of simulated data", "tp > 20m from 1000 years of simulated data"), bty="n", col=1:3, text.col=1:3, pch=1)



