##################################################
# INTRODUCTION TO R IN RSTUDIO 
##################################################

# 1. You should use R for statistical programming: more flexible than Stata, with richer stat libraries than Matlab and Python. 
# 2. If you use R you should use R studio (friendly interface with different colours making it easier to read)
# 3. Good programming practice:
#     (1) ALWAYS name your variables something descriptive
#     (2) indent your code
#     (3) make everything into a function if it can possibly be a function (can be done on a second pass)
#     (4) almost never hardcode a number or copy (pass it to global environment) and paste pieces of code 
#     (5) check that those canned functions from packages do what you think they do 
#       -- especially when they encounter NAs
#     (6) write unit tests and debug regularly and systematically (check each piece works)
#     (7) remember R is case sensitive!

### Package Installation and Loading 

# Base R has a system library made up of basic math and stat commands,
# if you want to use a specific function, it may be in a certain package that you need to download separately

# chooseCRANmirror(ind=90) #set the CRAN repository for the session, not always necessary but often helpful

# Here's how you install and load 1 package (can also be done manually)
install.packages("lmtest") # only need to do this once so that the content is saved on your device
library(lmtest) # do this at the start of every session that you need to use it, otherwise R wont know where the function/data is 

# Here's how to install and many at once (also these are good for econometrics)
packages_many <- c("sandwich","ggplot2", "reshape", "boot", "zoo", 
                   "quantreg", "dummies", "stargazer", "plm", "rgl", "MASS", "parallel", "RColorBrewer", "openxlsx", "readstata13", "Hmisc")
install.packages(packages_many)

# Since you only need to run this once it's better to write an if statment to check if they are already installed
if (!all(packages_many %in% rownames(installed.packages()))) install.packages(packages_many)

# to use the library()/require() command on the vector we need to use the lapply (list apply) function
lapply(packages_many, require, character.only = TRUE)
# this means "apply this function to the elements of this list or more restricted data type"

### Preparing the workspace

# Clear the workspace! (Essential when running scripts on remote servers, who knows whats there from your previous sessions) 
rm(list=ls())
# This removes the data in the global environment
# Can use rm(something_specific) to just remove that object

# Set the working directory (file where data is taken from and stored to)
setwd("/Users/sofia.borodichsuarez/Documents/TA/Sess1")
# Typically do this in the console! NOT IN THE SCRIPT! 
# Each user of the code does this in their own console before running scripts.
# Your code will not run on another machine if you use setwd() in a script

# If you are working with code that prints out a lot of text in console, open a log file (where the output goes instead of the screen)
# sink("example_log_file.txt") # Redirect console output to file
# the log file will contain everything from here until you run the command dev.off() or sink()

# Logging can be useful, but not essential (good for debugging)
# Unlike STATA, we can save all objects/results at once and call them again when needed


### R BASICS

# If you need help with any function, or want to know how it works use ?functionname

# The <- operator is how R does assignment (kinda like =def that Prof. uses in class)
ex_scalar <-  5 
ex_vector <-  c(1,2,3,4)
ex_matrix <- matrix(rep(0,9), nrow=3, ncol=3) # fill a 3 by 3 matrix with 0s
ex_dataframe <- data.frame(ex_vector, 2*ex_vector, -2+5*ex_vector)
ex_list <- list(ex_dataframe, ex_vector, "Anything", 17)

# dataframes and lists are more flexible data types, and can hold multiple different column types
# lists are the most flexible but use the most memory to carry around
# as soon as possible you should use the "unlist()" function to get rid of it!

# Subsetting and Extraction
subsetted_ex_vector <- ex_vector[2:4] # elements 2,3,4
extracted_from_ex_vector <- ex_vector[2] # element 2 only
ex_vector_with_removed <- ex_vector[-4] # element 4 removed
extracted_from_ex_list <- ex_list[[2]] # element 2 of list which is a vector
extracted_and_subsetted_from_ex_list <- ex_list[[2]][1] # element 1 of vector that is the 2nd element of the list
subvector_of_ex_matrix <- ex_matrix[,1] # 1st col of matrix ([i,j]: element in i row, j col)

# Simple transformations
demeaned_ex_vector <- ex_vector - mean(ex_vector, na.rm=TRUE)
ex_vector_elementwise_squared <- ex_vector^2

## Loading Data ##

# Unlike STATA, R supports multiple data sets in memory
ex_data1 <-read.csv("example1.csv") # loading example csv file
ex_data2 <-read.xlsx("example2.xlsx", sheet = "Sheet1", colNames = FALSE) # loading example excel file (also for xls)
ex_data3 <-read.dta("example3.dta") # loading example stata file
#look at the arguments of these read functions to read the data as you want it
# e.g. argument header=T/F allows you to indicate if the first row of data is the title/ variable name

# When using arguments names then you can put them in any order
# otherwise without R gives the value to the redetermined order in the function definition

# Exploring the data
View(data) # look at data
head(data) # only display first few rows
summary(data) # summarize data (gives discriptive statistic of each variable)

# Saving data to memory
write.csv(ex_dataframe, "egresults.csv", row.names = FALSE) # Most common and simplest way to save your data and results in csv format
save(ex_dataframe, file = "egresults.RData", compress = "xz") # R format with very good compression allowing to save multiple data sets at once
load("egresults.RData") # Loads all objects from that file in the memory

### Data Generation / Random Number Generation
# Sometimes you want to simulate data instead of using a real data frame
N <- 3000
true_beta_0 <- 2
true_beta_1 <- 3
true_sigma <- 5

# SUPER IMPORTANT for recreatable results
# Set a seed (any number) which will tell R to generate the same random numbers
set.seed(1)

# To generate from most distributions you will use d,p,q or r and the shortened name of the dist.
# d: density (PDF), p: distribution function (CDF), q: quantile function and r: generates random deviates.
x <- rbinom(n=N, size=1, prob = 0.5) # N binomial random variables
y <- rnorm(n=N, true_beta_0 + true_beta_1*x, sd = true_sigma) # N Gaussian normal r.v.s
w <- rnorm(n=N, x+4, sd = 5)

# CDF, PDF and quantile functions of normal distribution
pnorm(0) # CDF(0)
dnorm(0) # PDF(0)
dnorm(0, 1, 2) # PDF_X(0) where X~N(1,4) (R uses SD instead of Var)

qnorm(0.95) # Below which number do we have 95% of the distribution mass
qnorm(0.025) # Left critical value for two-sided hypothesis testing at 5% significance level
qnorm(0.975) # Right critical value....
qnorm(seq(0.01, 0.99, 0.01)) # Displaying percentiles of standard normal distribution

# CDF, PDF, right tail probability, quantile of chi-squared distribution
pchisq(3.84, 1) # CDF of chi-squared with 1 degree of freedom at point 3.84
dchisq(2, 4)
pchisq(3.84, 1, lower.tail = FALSE) # Right tail probability
qchisq(0.95, 1)

# Plotting normal Gaussian with different param values
x <- seq(-4, 4, length.out = 201)
#par(mar = c(2, 4, 0.5, 0.5))
windows() # opens new plotting window in windows, for mac use command quartz()
plot(x, dnorm(x), type = "l", ylim = c(0, 0.6), bty = "n", ylab = "Normal density", xlab = "", col = "black", lwd = 5)
lines(x, dnorm(x, mean = 1), type = "l", col = "red", lwd = 5) # changing mean mu shifts centre of distribution
lines(x, dnorm(x, mean = 0, sd = 2), type = "l", col = "blue", lwd = 5) # changing sd affects the width or the dispersion from the mean
lines(x, dnorm(x, mean = -1.5, sd = 0.8), type = "l", col = "purple", lwd = 5)
legend("topright", c(expression(mu == 0 ~ ", " ~ sigma == 1),
                     expression(mu == 1 ~ ", " ~ sigma == 1),
                     expression(mu == 0 ~ ", " ~ sigma == 2),
                     expression(mu == -1.5 ~ ", " ~ sigma == 0.8)), lwd = 5, col = c("black", "red", "blue", "purple"), bty = "n")

# Plotting Chi squared with different param values
x <- seq(0, 15, length.out = 201)
#par(mar = c(2, 4.5, 0.5, 0.5))
windows()
plot(x, dchisq(x, 1), type = "l", ylim = c(0, 0.5), bty = "n", lwd = 5, ylab = expression(chi[df]^2 ~ "density"))
lines(x, dchisq(x, 2), lty = 2, lwd = 5)
mydf <- c(3, 5, 7, 10, 15) # here df is dataframe not degree of freedom
cols <- colorRampPalette(c("red", "green"))(length(mydf))
for (i in 1:length(mydf)) lines(x, dchisq(x, mydf[i]), col = cols[i], lwd = 5)
legend("topright", paste0("df = ", 1:2), lwd = 5, lty = 1:2, bty = "n")
legend("top", paste0("df = ", mydf), lwd = 5, col = cols, bty = "n")
# As the degrees of freedom increase this resembles Gaussian distribution
# In fact, as d.f.->Inf Chi-squared r.v.s converge in distribution to Gaussian
# Let's see for 100 d.f.

x <- seq(50, 150, length.out = 501)
#par(mar = c(2, 4.5, 0.5, 0.5))
windows()
plot(x, dchisq(x, 100), type = "l", bty = "n", lwd = 5, ylab = "Density")
lines(x, dnorm(x, mean = 100, sd = sqrt(2 * 100)), col = "red", lwd = 5)
legend("topright", c(expression(N(100,sqrt(200))), expression(chi[100]^2)), lwd = 5, col = c("red", "black"), bty = "n")

# F distribution is also common in practice
# Lets plot the Chi squared distribution curves divided by their respective d.f.

x <- seq(0, 6, length.out = 201)
windows()
#par(mar = c(2, 4.5, 0.5, 0.5))
plot(x, dchisq(x, 1), type = "l", ylim = c(0, 1.5), bty = "n", lwd = 5, ylab = expression(chi[df]^2 / df ~ "density"))
lines(x, dchisq(x * 2, 2) * 2, lty = 2, lwd = 5)
mydf <- c(3, 5, 7, 10, 15, 100)
cols <- colorRampPalette(c("red", "green"))(length(mydf))
for (i in 1:length(mydf)) lines(x, dchisq(x * mydf[i], mydf[i]) * mydf[i], col = cols[i], lwd = 5)
legend("topright", paste0("df = ", 1:2), lwd = 5, lty = 1:2, bty = "n")
legend("top", paste0("df = ", mydf), lwd = 5, col = cols, bty = "n")

# So F_{k,n}=(Chi-sq_k / k) / (Chi-sq_n / n)
# this ratio converges in distribution to Chi-sq_k / k as n goes to Inf
# Lets look at k*F_{k,n} for large n=100 and arbitrary k=5, it should resemble Chi-sq_5
x <- seq(0, 10, length.out = 501)
windows()
#par(mar = c(2, 4.5, 0.5, 0.5))
plot(x, df(x / 5, 5, 100) / 5, col = "red", type = "l", ylim = c(0, 0.25), xlim = range(x), bty = "n", lwd = 5, ylab = expression(chi[5]^2 ~ " and 5*" ~ F[5 * ", 100"] ~ " density"))
lines(x, dchisq(x, df = 5), lwd = 5)
legend("topright", paste0("n = ", 100), lwd = 5, col = "red", bty = "n")

setwd("Econometrics_TA")
# Lets look at some real (income) data
d <- read.dta13("NLS80.DTA")
d$lwage <- NULL # create an empty variable to fill with log(wage) values

# Good practice to check density plots of continuous variables
plot(density(d$wage), bty = "n")
d$lwage <- log(d$wage) # This might be a poor choice is someone's wage is equal to 0; log(0)=-Inf
d$lwage <- log(d$wage + 1) # This transformation maps 0 wage to 0 lwage
plot(density(d$lwage), bty = "n")


# Tabulation command useful for discrete variables
table(d$hours)
table(d$educ, d$black) # Tab two variables at once
# Also try tab educ feduc
table(d$feduc)
table(d$feduc, useNA = "ifany") # By default, table ignores missing data so indicate if you want it shown
plot(table(d$feduc), bty = "n") # Add a nice histogram

summary(d)
summary(d[, c("wage", "age")]) # Summarise only two variables
for (i in unique(d$black)) { # Reproduce the summary of all variables for each distinct value of black
  print(paste0("black = ", i, ", n = ", sum(d$black == i)))
  print(summary(d[d$black == i, ]))
}
# Replace unique(d$black) with sort(unique(d$black)) to get the effect of STATA's by black, sort

# Any functions other than mean and SD, even better than STATA's tabstat
apply(d[, c("wage", "hours")], 2, function(x) c(mean = mean(x), med = median(x), quantile(x, c(0.25, 0.75))))
# directly get any quantile or statistic of any of the variables you ask for

# Now generate hourly wage (our variable of interest)
d$wh <- d$wage / d$hours
# indicating d$___ adds it to the specific data frame, without this it would just be a variable by itself

# Check some correlations
cor(d[, c("wh", "age", "educ")])
# Test the significance of correlations using a function from library(Hmisc)
rcorr(as.matrix(d[, c("wh", "age", "educ")])) # Exact result form STATA
# first matrix gives the correlations and the second the p-values for the hyp test H0: true correlation=0, H1: corr=/=0

# Get all these statistics for hourly wage by age (or any discrete variable) 
aggregate(wh ~ age, data = d, FUN = function(x) c(mean = mean(x), sd = sd(x), median = median(x), n = length(x), quantile(x, c(0.025, 0.05, 0.25, 0.75, 0.95, 0.975))))


# MODEL 0: Simple linear regression

mod0 <- lm(wh ~ educ + exper + tenure + age + married + black + south + urban + sibs + brthord, data = d)
summary(mod0) # Inference with wrong standard errors and useless statistics!!!! Avoid!
# Use correct standard errors
lmtest::coeftest(mod0, vcov. = vcovHC) # By default, uses HC3 which corrects for small-sample and outliers
lmtest::coeftest(mod0, vcov. = function(x) vcovHC(x, type = "HC1")) # Replicating STATA results
lmtest::coeftest(mod0, vcov. = function(x) vcovHC(x, type = "HC0")) # White's asymptotic estimator
HAC_vcov <- vcovHAC(mod0) # heteroskedaticity and autocorrelation robust (HAC) s.e. also available

# here I had some issues with overlapping functions in different packages so I indicate the package it's in then :: then the function name

# you can also tell LM to regress the first column on everything else in a dataframe
# there are 2 ways to do this:
reg_all <- lm(d)
reg_all <- lm(d$wh ~ ., d)
# or include everything interacted with a certain variable e.g. treatment:
reg_interacted <- lm(wh ~ .*educ, d)

# R will coerce a character variable into a factor for regression
eg_character <- rep(c("good", "bad","ugly"), N/3)
eg_factor <- factor(rep(c("good", "bad","ok"), N/3)) # this is an example of acceptable hardcoding
# or you can do it yourself via factor() then regress as normal
# It is NOT automatic if the categories are stored as numerical! 
eg_numerical <- rep(c(1,2,3), N/3)
# you can fix that with the dummies package or by coercing the numerical to a factor yourself
# Use factor() to convert a numeric variable into a set of dummies

# Let's include a quadratic term
# MODEL 1: The dependence on age is not linear but quadratic now:
# Unlike STATA, one need not generate new variables and use up memory-just use I()
mod1 <- lm(wh ~ educ + exper + tenure + age + I(age^2) + married + black + south + urban + sibs + brthord, data = d)
lmtest::coeftest(mod1, vcov. = vcovHC)


# If you factor a variable with several levels you get many singleton dummies
mod2 <- lm(wh ~ educ + exper + tenure + age + I(age^2) + married + black + south + urban + factor(sibs) + brthord, data = d)
lmtest::coeftest(mod2, vcov. = vcovHC)
# So we have to use simpler functions to not get NAs
lmtest::coeftest(mod2, vcov. = function(x) vcovHC(x, type = "HC1")) # Like in STATA vce(robust) 


# MODEL 5: Let's look at the log wage
mod5 <- lm(lwage ~ hours + educ + exper + tenure + age + I(age^2) + married + black + south + urban + sibs + brthord, data = d)
lmtest::coeftest(mod5, vcov. = vcovHC)
# If log(1+b)~b does not hold because b is not so close to zero, one can try the following approximation (in per cent)
(exp(mod5$coefficients["black"]) - 1) * 100 # Value in per cent

# One need not re-estimate a model like in STATA; just use the saved object
d$yhat <- predict(mod5) # Throws an error because some values are missing
d$yhat <- predict(mod5, newdata = d) # Compute predicted/fitted values of wh and save them as yhat
d$res <- residuals(mod5) # For the same reason, does not work because the estimation sample is smaller
d$res <- d$wh - d$yhat # Compute residuals and save them as res

plot(density(mod5$residuals), bty = "n") # Look at the distribution of residuals
hist(mod5$residuals, breaks = "FD")
rug(mod5$residuals)
# Check if the residuals are homogeneous, because if there are two humps, it
# might indicate that there are two distinct sub-populations in the sample


# In order to produce beautiful tables, one can use `library(stargazer)`
# It requires models and a list of correct standard errors; otherwise it will produce incorrect SEs
stargazer(mod1, mod2, se = list(sqrt(diag(vcovHC(mod1))), sqrt(diag(vcovHC(mod2)))), type = "text")
# in type you can pick html or latex format settings so the tables can be copied and pasted into your files directly


# A simple hypothesis test
# Let's see if there is a difference between log(wage) for black and non-black individuals
t.test(d$lwage[d$black == 1], d$lwage[d$black == 0], alternative = "two.sided") #avoid
# The problem again is with the non-robust standard errors which must be corrected
race.wage.formula <- lm(lwage ~ black, d) # the formula/model that we are testing
s <- lmtest::coeftest(race.wage.formula, vcov = sandwich::vcovHC)
s["black", "t value"] # gives you the correct t stat
1 - pnorm(s["black", "t value"]) # gives the correct p value

