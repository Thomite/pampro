
* This is a script to handle the full datetime stamp of PAMPRO in Stata

* Separate the datetime stamp into date and time
gen date = substr(timestamp, 1, 10)
gen time = substr(timestamp, 12, 15)

* Extract the separate day, month and year components
gen day = substr(timestamp, 1, 2)
gen month = substr(timestamp, 4, 2)
gen year = substr(timestamp, 7, 4)
tostring day month year, replace

* Cast the strings to Stata date and time formats
* Note that Stata times lack representation below the second level
gen double stata_date = date(date, "DMY")
format stata_date %td
gen double stata_time = clock(time, "hms#")

* Use Stata's date abilities to get day of week and year
gen day_of_week = dow(stata_date)
gen day_of_year = doy(stata_date)
