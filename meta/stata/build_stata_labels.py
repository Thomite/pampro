import numpy as np

# This quick script generates a Stata do-file to label a PAMPRO dataset using the data dictionary

dd_file = "Z:/data/EPIC3/_results/data_dictionary_1.csv"
do_file = dd_file.replace("data_dictionary", "do_file")
do_file = do_file.replace(".csv", ".do")

data = np.loadtxt("Z:/data/EPIC3/_results/data_dictionary_1.csv", delimiter=",", skiprows=1, dtype=str, unpack=True)
output = file(do_file, "w")

for variable,description in zip(data[0,], data[1,]):
    #print("label variable {} \"{}\"".format(variable, description))
    output.write(("label variable {} \"{}\" \n".format(variable, description))

output.close()