
from datetime import datetime, date, time, timedelta
from pampro import data_loading, Time_Series, Channel, channel_inference, triaxial_calibration

# Change filenames as appropriate

# Read the data - yields 1 channel per axis
ts, header = data_loading.load("/pa/data/BIOBANK/example.cwa", "Axivity")

x, y, z = ts.get_channels(["X", "Y", "Z"])

# Autocalibrate the raw acceleration data
x, y, z, calibration_diagnostics = triaxial_calibration.calibrate(x, y, z)

# Infer some sample level information - Vector Magnitude (VM), Euclidean Norm Minus One (ENMO)
vm = channel_inference.infer_vector_magnitude(x, y, z)
enmo = channel_inference.infer_enmo(vm)

# Create a time series object and add channels to it
ts.add_channels([vm, enmo])

# Uncomment this line to write the raw data as CSV
#ts.write_channels_to_file("C:/Data/3.csv")

# Request some interesting statistics - mean of ENMO
stats = {"ENMO":[("generic", ["mean"])]}

# Get the above statistics on an hourly level - returned as channels
hourly_results = ts.piecewise_statistics(timedelta(hours=1), statistics=stats)

# Write the hourly analysis to a file
hourly_results.write_channels_to_file("/pa/data/BIOBANK/example_output.csv")

# Visualise the hourly ENMO signal
hourly_results.draw([["ENMO_mean"]], file_target="/pa/data/BIOBANK/example.png")
