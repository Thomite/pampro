
from datetime import datetime, date, time, timedelta
from pampro import Time_Series, Channel, channel_inference, triaxial_calibration


# Read the data - yields 1 channel per axis
x, y, z = Channel.load_channels("/pa/data/BIOBANK/example.cwa", "Axivity")

print(x.data)
print(x.timestamps)

# Autocalibrate the raw acceleration data
x, y, z, (cal_params), (results), (misc) = triaxial_calibration.calibrate(x, y, z)

# Infer some sample level information - Vector Magnitude (VM), Euclidean Norm Minus One (ENMO)
vm = channel_inference.infer_vector_magnitude(x, y, z)
enmo = channel_inference.infer_enmo(vm)

# Create a time series object and add channels to it
ts = Time_Series.Time_Series("Axivity")
ts.add_channels([x, y, z, vm, enmo])

# Uncomment this line to write the raw data as CSV
#ts.write_channels_to_file("C:/Data/3.csv")

# Request some interesting statistics - mean of ENMO
stats = {"ENMO":["mean"]}

# Get the above statistics on an hourly level - returned as channels
hourly_results = ts.piecewise_statistics(timedelta(hours=1), statistics=stats, time_period=x.timeframe)

# Add the result channels to a new time series object, and draw them
ts_visualisation = Time_Series.Time_Series("Axivity")
ts_visualisation.add_channels(hourly_results)

# Write the hourly analysis to a file
ts_visualisation.write_channels_to_file("/pa/data/BIOBANK/example_output.csv")

# Visualise the hourly ENMO signal
ts_visualisation.draw_experimental([["ENMO_mean"]], file_target="/pa/data/BIOBANK/example.png")
