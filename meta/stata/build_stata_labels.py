import numpy as np
import sys

# This quick script generates a Stata do-file to label a PAMPRO dataset using the data dictionary

dd_file = sys.argv[1]

do_file = dd_file.replace("data_dictionary", "label_variables")
do_file = do_file.replace(".csv", ".do")

print("Creating " + do_file + " from " + dd_file)

data = np.loadtxt(dd_file, delimiter=",", skiprows=1, dtype="S", unpack=True).astype("U")
output = open(do_file, "w")

for variable,description in zip(data[0,], data[1,]):
    #print("label variable {} \"{}\"".format(variable, description))
    output.write(("cap label variable {} \"{}\" \n".format(variable, description)))

output.close()
