# R code for kernel estimator of pdf_X and E(Y|X).
# Gautam Tripathi
# March 03, 2017

rm(list = ls()) # Clear workspace.
ptm <- proc.time() # Start clock.
set.seed(100) # Set seed for replication. 12345678

n <- 10000 # Number of observations.

# Generate the data.
X <- rnorm(n, mean = 0, sd = 1)
U <- rnorm(n, mean = 0, sd = 1)
Y <- X^2 + U

# Generate the x-values at which pdf_X(x) and E(Y|X=x) are to be estimated
Xgrid <- seq(-5, 5, by = 0.01) # Array of length m := length(Xgrid).

gaussian.kernel <- function(u)
  # Gaussian kernel.
{
  return(dnorm(u))
}

uniform.kernel <- function(u)
  # Uniform kernel.
{
  return(0.5 * ifelse((u >= -1) & (u <= 1), 1, 0))
}

my.ksmooth <- function(Ydata, Xdata, xvalues, N, h, kernel)
  # Function for estimating pdf_X(x) and E(Y|X=x).
{
  Dmatrix <- outer(Xdata, xvalues, "-") # n x m
  Kmatrix <- kernel(Dmatrix / h)
  den <- colSums(Kmatrix) / (N * h)
  num <- colSums(sweep(Kmatrix, 1, Ydata, "*")) / (N * h) # Sweeps array Ydata across columns of Kmatrix.
  est <- list(fhat = den, muhat = num / den)
  return(est)
}

muhat.LOO <- function(Ydata, Xdata, N, h, kernel)
  # Function for estimating the LOO estimator of muhat. 
  # Returns muhat_{-1}(X_1), ..., muhat_{-n}(X_n).
{
  Dmatrix <- outer(Xdata, Xdata, "-") # n x n
  Kmatrix <- kernel(Dmatrix / h)
  diag(Kmatrix) <- 0 # Set diagonal elements of Kmatrix to 0.
  num <- colSums(sweep(Kmatrix, 1, Ydata, "*")) / ((N - 1) * h)
  den <- colSums(Kmatrix) / ((N - 1) * h)
  return(num / den)
}

cv.muhat <- function(Ydata, Xdata, N, h, kernel)
  # Cross-validation function for Nadaraya-Watson estimator of E(Y|X).
{
  muhat.LOO.values <- muhat.LOO(Ydata, Xdata, N, h, kernel)
  ase.LOO <- mean((Ydata - muhat.LOO.values)^2)
  return(ase.LOO)
}

bwGrid <- seq(0.1, 1, by = 0.01)
CV.values.with.gaussian.kernel <- matrix(NA, length(bwGrid), 1)
CV.values.with.uniform.kernel <- matrix(NA, length(bwGrid), 1)

for (i in seq_along(bwGrid))
{
  CV.values.with.gaussian.kernel[i, 1] <- cv.muhat(Y, X, n, bwGrid[i], gaussian.kernel)
  CV.values.with.uniform.kernel[i, 1] <- cv.muhat(Y, X, n, bwGrid[i], uniform.kernel)
}

# Crossvalidated bandwidths for muhat.
cv.bw.with.gaussian.kernel <- bwGrid[which.min(CV.values.with.gaussian.kernel)]
cv.bw.with.uniform.kernel <- bwGrid[which.min(CV.values.with.uniform.kernel)]

# Obtain fhat and muhat.
fhat.muhat.with.gaussian.kernel <- my.ksmooth(Y, X, Xgrid, n, cv.bw.with.gaussian.kernel, gaussian.kernel)
fhat.with.gaussian.kernel <- fhat.muhat.with.gaussian.kernel$fhat
muhat.with.gaussian.kernel <- fhat.muhat.with.gaussian.kernel$muhat

fhat.muhat.with.uniform.kernel <- my.ksmooth(Y, X, Xgrid, n, cv.bw.with.uniform.kernel, uniform.kernel)
fhat.with.uniform.kernel <- fhat.muhat.with.uniform.kernel$fhat
muhat.with.uniform.kernel <- fhat.muhat.with.uniform.kernel$muhat

runningtime <- proc.time() - ptm # Record running time.
cat("total run time (sec) =", round(runningtime[3], 1), "\n")

# Format data to be reported in the plots.
bw.gaussian <- bquote(bandwidth == .(cv.bw.with.gaussian.kernel))
bw.uniform <- bquote(bandwidth == .(cv.bw.with.uniform.kernel))
report.n <- bquote(n == .(n))
report.runtime <- bquote(Time(sec) == .(runningtime))

# Plot fhat and muhat to a pdf file.
pdf(file = "den+reg-5+LOO-graphs_R.pdf") # Open PDF file for writing.

par(oma = c(0, 0, 3, 0)) # Set margins: bottom=left=right=0, top=3 for main title.
par(mfrow = c(2, 2)) # 2 x 2 plots.

# Begin plots.
plot(Xgrid, fhat.with.gaussian.kernel, type = "l", lty = "dashed", xlab = "x", ylab = "fhat(x)")
title(main = "gaussian kernel", font.main = 1) # Individual title.
legend('top', legend = bw.gaussian, bty = 'n')

plot(Xgrid, fhat.with.uniform.kernel, type = "l", lty = "dashed", xlab = "x", ylab = "fhat(x)")
title(main = "uniform kernel", font.main = 1)
legend('top', legend = bw.uniform, bty = 'n')

plot(Xgrid, muhat.with.gaussian.kernel, type = "l", lty = "dashed", xlab = "x", ylab = "muhat(x)")
legend('top', legend = bw.gaussian, bty = 'n')

plot(Xgrid, muhat.with.uniform.kernel, type = "l", lty = "dashed", xlab = "x", ylab = "muhat(x)")
legend('top', legend = bw.uniform, bty = 'n')

top.plot.title <- list("Density and regression estimates", report.n, report.runtime)
mtext(do.call(expression, top.plot.title), outer = TRUE, line = 1:-1) # Main title on top.
dev.off() # Close file.

# Plot cross-validation functions to a pdf file.
pdf(file = "den+reg-5+LOO-cv_R.pdf")

par(oma = c(0, 0, 2, 0)) # Set margins: bottom=left=right=0, top=2 for main title.
par(mfrow = c(1, 2)) # 1 x 2 plots.

plot(bwGrid, CV.values.with.gaussian.kernel, type = "l", lty = "dashed")
title(main = "gaussian kernel", font.main = 1)
legend('topright', legend = bw.gaussian, bty = 'n')

plot(bwGrid, CV.values.with.uniform.kernel, type = "l", lty = "dashed")
title(main = "uniform kernel", font.main = 1)
legend('topright', legend = bw.uniform, bty = 'n')
mtext("LS crossvalidation function for the Nadaraya-Watson estimator", outer = TRUE)
dev.off() 