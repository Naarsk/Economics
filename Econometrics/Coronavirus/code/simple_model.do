* Load the dataset
use "C:\Users\leocr\Projects\Economics\Econometrics\Coronavirus\data\italy_data.dta", clear

describe

* Create the time variable
gen time = _n  
tsset time    

* Generate Y and lagged X variables
gen Y = Cumulative_cases if Cumulative_cases > 0
gen Z = Cumulative_moving_average
gen X = L.Y

drop if X <= 0
gen XlnX = ln(X)*X

gen W1 = Average_temperature
gen W2 = Transit_stations
gen W3 = Retail_and_recreation 


* Ensure no missing values
drop if missing(Y) | missing(X) | missing(W1) | missing(W2) | missing(W3) | missing(Z)

**# GMM1 #1
gmm (Y - {gamma1=1.7}*X - {gamma2=-0.06}*XlnX),  instruments(X W1) iterate(2000)  nolog

ereturn list

estat overid

eststo GMM1

**# GMM2 #2
gmm (Y - {gamma1=1.7}*X - {gamma2=-0.06}*XlnX),  instruments(X W2) iterate(2000)  nolog

ereturn list

estat overid

eststo GMM2


**# GMM3 #3
gmm (Y - {gamma1=1.7}*X - {gamma2=-0.06}*XlnX),  instruments(X XlnX W1 W2 W3) iterate(2000)  nolog

ereturn list

estat overid

eststo GMM3

**# GMM4 #4
gmm (Y - {gamma1=1.7}*X - {gamma2=-0.06}*XlnX),  instruments(W1 W2 W3) iterate(2000)  nolog

ereturn list

estat overid

eststo GMM4

esttab GMM1 GMM2 GMM3 GMM4 using "gmm_result.tex", replace label se star(* 0.10 ** 0.05 *** 0.01) s(N J J_df, label("Observations" "J-stat" "DoF"))
