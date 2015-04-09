
from datetime import datetime, date, time, timedelta
from pampro import Time_Series, Channel, channel_inference, Bout


# Request some interesting statistics - mean, min and max of the counts signal
# ...plus basic cutpoints for Sedentary, Light, and Moderate to Vigorous
cutpoints = [[0,99],[100,2999],[3000,99999]]
stats = {"AG_Counts": ["mean", "min", "max"] + cutpoints}

# Load Actigraph data
counts, header = Channel.load_channels("/pa/data/Tom/pampro/data/example_actigraph.DAT", "Actigraph", datetime_format="%m/%d/%Y")

ts = Time_Series.Time_Series("Actigraph")
ts.add_channel(counts)

print(counts.timestamps)

# Get a list of bouts where the monitor was & wasn't worn
nonwear_bouts, wear_bouts = channel_inference.infer_nonwear_actigraph(counts, zero_minutes=timedelta(minutes=90))

# Use that list to get a list of days of valid & invalid time
invalid_bouts, valid_bouts = channel_inference.infer_valid_days(counts, wear_bouts)

# Since the cutpoints defined above only count positive data, negative values will be ignored
# Where the monitor wasn't worn, set the count value to -1
# Where the monitor wasn't valid, set the count value to -2
counts.fill_windows(nonwear_bouts, fill_value=-1)
counts.fill_windows(nonwear_bouts, fill_value=-2)

# Get the summary level results
summary_results = ts.summary_statistics(statistics=stats)

# Create a time series object, put the summary level results in it, write them to a file
ts_output = Time_Series.Time_Series("Output")
ts_output.add_channels(summary_results)
ts_output.write_channels_to_file("/pa/data/ICAD/example_output.csv")

# Draw the counts signal
ts.draw_experimental([["AG_Counts"]], file_target="/pa/data/ICAD/example.png")
