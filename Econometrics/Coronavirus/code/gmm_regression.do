* Load the dataset
use "C:\Users\leocr\Projects\Economics\Econometrics\Coronavirus\data\italy_data.dta", clear

describe

* Create the time variable
gen time = _n  
tsset time    

* Generate Y and lagged X variables
gen Y = Cumulative_cases if Cumulative_cases > 0
gen X = L.Y

* Ensure no zero or negative X values
drop if X <= 0

gen X2 = L2.X
gen X3 = L3.X
gen X4 = L4.X
gen X5 = L5.X
gen XlnX = ln(X)*X
gen XlnX2 = X*ln(X)^2
gen XlnX3 = X*ln(X)^3


* Ensure no missing values
drop if missing(Y) | missing(X) | missing(X2) | missing(X3) | missing(X4) | missing(X5)

* Ensure no zero or negative X values
drop if X <= 0

* Run GMM with initial values
gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2) iterate(2000)  nolog

ereturn list

estat overid

eststo X_X2

gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3) iterate(2000) nolog

ereturn list

estat overid

eststo X_X2_X3


gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3 X4) iterate(2000) nolog

ereturn list

estat overid

eststo X_X2_X3_X4

gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3 XlnX) iterate(2000) nolog

ereturn list

estat overid

eststo X_X2_X3_XlnX

esttab X_X2 X_X2_X3 X_X2_X3_X4 X_X2_X3_XlnX using gmm_models.tex, se stats(J J_df rank) label ///
    title("GMM Regression Table") ///
    mtitle("N_1 N_2" "N_1 N_2 N_3" "N_1 N_2 N_3 N_4" "N_1 N_2 N_3 NlnN") ///
    replace