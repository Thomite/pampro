
* This is a standard post-processing script for Actigraph analyses
* It takes total time spent in count cutpoints and expresses them as a proportion of valid wear time

* Get a list of variables to adjust
quietly desc counts_0_* counts_*_99999_*, varlist
local varlist `r(varlist)'

* As a safety measure, ignore some variables commonly outputted
local ignore_vars "id timestamp wear_sum valid_sum counts_summary_n counts_summary_sum counts_summary_min counts_summary_max counts_summary_p50 counts_summary_mean counts_summary_std"
local safe : list varlist - ignore_vars 

foreach varname in `safe' {

	di "`varname'"
	quietly replace `varname' = `varname' / wear_sum
}
