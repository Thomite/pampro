'''
This is a simple script to consolidate results from batch runs into 1 file.

Usage:
py consolidate_results.py "/path/to/results" "daily"
This will append together all results with the daily prefix into 1 file called daily_all.csv
'''

import glob
import sys

folder = sys.argv[1]
prefix = sys.argv[2]

# List all matching files
files = glob.glob(folder + "/" + prefix + "*.csv")

print("Found {} files matching {}/{}*.csv".format(len(files),folder,prefix))

# Create the output file
output = open(folder + "/" + prefix + "_all.csv", "w")

# Copy all of the data from the first file, including header
first_file = open(files[0], "r")
output.write(first_file.read())
first_file.close()

print files[0]

for filename in files:

	# Copy the data of every other file, excluding the header
	fh = open(filename, "r")
	fh.readline()
	data = fh.read()
	fh.close()

	output.write(data)

output.close()