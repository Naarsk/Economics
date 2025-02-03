* Load the dataset
use "C:\Users\leocr\Projects\Economics\Econometrics\Coronavirus\data\italy_data.dta", clear

describe

* Create the time variable
gen time = _n  // Create a time index if it's not already there
tsset time      // Set the dataset as time-series

* Generate the lead variable Y = Cumulative_cases_t
gen Y = Cumulative_cases if Cumulative_cases > 0

* Generate the lagged variable X = Cumulative_cases_t-1 as lag of Cumulative_case
gen X = L.Y  

* Option 1: Drop observations with missing Y or X
drop if missing(Y) | missing(X)

* Run the nonlinear regression
nl (Y = {alpha1=1.8}*X + {alpha2=-0.8}*X^({epsilon=0.07}+1)), nolog

* Option 2 (alternative): restrict the sample in the nl command
* nl (Y = {alpha1=1.7}*X + {alpha2=-0.2}*X^({epsilon=0.1}+1)) if !missing(Y, X), nolog
