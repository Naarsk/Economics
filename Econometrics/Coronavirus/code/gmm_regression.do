* Load the dataset
use "C:\Users\leocr\Projects\Economics\Econometrics\Coronavirus\data\italy_data.dta", clear

describe

* Create the time variable
gen time = _n  
tsset time    

* Generate Y and lagged X variables
gen Y = Cumulative_cases if Cumulative_cases > 0
gen X = L.Y
gen W1 = Average_temperature
gen W2 = Transit_stations
gen W3 = Retail_and_recreation 

* Ensure no zero or negative X values
drop if X <= 0

gen X2 = L2.Y
gen X3 = L3.Y
gen X4 = L4.Y
gen X5 = L5.Y
gen XlnX = ln(X)*X

* Ensure no missing values
drop if missing(Y) | missing(X) | missing(X2) | missing(X3) | missing(X4) | missing(X5)


**# GMM S2 #2
gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3 X4 W1 W2 W3) iterate(2000)  nolog

ereturn list

estat overid

eststo W

**# GMM S3 #3

gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3) iterate(2000) nolog

ereturn list

estat overid

eststo S3

**# GMM S4 #4

gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3 X4) iterate(2000) nolog

ereturn list

estat overid

eststo S4

**# GMM Sln #5

gmm (Y - {alpha1=4}*X - {alpha2=-3}*X^({epsilon=0.1}+1)),  instruments(X X2 X3 XlnX) iterate(2000) nolog

ereturn list

estat overid

eststo Sln

**# Print table #6

esttab S2 S3 S4 Sln using gmm_models.tex, se stats(J J_df rank) label ///
    title("GMM Regression Table") ///
    mtitle("S2" "S3" "S4" "Sln") ///
    replace
	

