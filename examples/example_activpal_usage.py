
from datetime import datetime, date, time, timedelta
from pampro import Time_Series, Channel, channel_inference, triaxial_calibration

# Change the filenames as appropriate

# Load sample activPAL data
x, y, z = Channel.load_channels("/pa/data/STVS/_data/activpal_data/714952C-AP1335893 18Nov13 10-00am for 7d 23h 14m.datx", "activPAL")

# Autocalibrate the raw acceleration data
x, y, z, (cal_params), (results), (misc) = triaxial_calibration.calibrate(x, y, z)

# Infer some sample level info from the three channels - VM, ENMO, Pitch & Roll
vm = channel_inference.infer_vector_magnitude(x, y, z)
enmo = channel_inference.infer_enmo(vm)
pitch, roll = channel_inference.infer_pitch_roll(x, y, z)

# Create a time series object and add all signals to it
ts = Time_Series.Time_Series("activPAL")
ts.add_channels([x,y,z,vm,enmo,pitch,roll])


# Request some stats about the time series
# In this case: mean ENMO, pitch and roll, and 10 degree cutpoints of pitch and roll
angle_levels = [[-90,-80],[-80,-70],[-70,-60],[-60,-50],[-50,-40],[-40,-30],[-30,-20],[-20,-10],[-10,0],[0,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80],[80,90]]
stat_dict = {"Pitch":angle_levels+["mean"], "Roll":angle_levels+["mean"], "ENMO":["mean"]}

# Get the output at 15 minute level
quarter_hourly_results = ts.piecewise_statistics(timedelta(minutes=15), stat_dict)

# Create a time series object to put the results in
ts_output = Time_Series.Time_Series("activPAL output")
ts_output.add_channels(quarter_hourly_results)

# Write the 15m results to a file
ts_output.write_channels_to_file("/pa/data/STVS/_results/example_output.csv")

# Visualise the mean ENMO, pitch and roll signals
ts_output.draw([["ENMO_mean"],["Pitch_mean","Roll_mean"]], file_target="/pa/data/STVS/example.png")
