import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
from datetime import datetime, date, time, timedelta
from scipy import stats
import random
import copy



from pampropy import Time_Series, Channel, channel_inference, Bout

execution_start = datetime.now()

ts = Time_Series.Time_Series("Actigraph")


# Load sample Actiheart data
filename = os.path.join(os.path.dirname(__file__), '..', 'data/random.DAT')

chans = Channel.load_channels(filename, "Actigraph")
counts = chans[0]

ts.add_channel(counts)
ts.add_channels(channel_inference.infer_nonwear_actigraph(counts))

lower = [0,0]#,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
upper = [50,100]#,150,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000]
lengths = [1]#,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100,110,120,150,180]
names = []
times = []

for low,high in zip(lower,upper):
	for length in lengths:
		bouts = counts.bouts(low,high,length)

		name = "_{}_{}_mt{}".format(low,high,length)
		names.append(name)

		total_time = Bout.total_time(bouts)
		times.append(total_time)

		subset = counts.subset_using_bouts(bouts, name, substitute_value=-1)
		ts.add_channel(subset)
		


timestats = {}
for cutpoint,time in zip(names,times):
	print cutpoint, time
	
	vals = cutpoint.split("_")
	#print vals
	timestats[cutpoint] = [[int(vals[1]),int(vals[2])]]

timestats = {}

timestats["Wear_only"] = [[0,50]]
timestats["AG_Counts"] = [[0,50]]
# Open the output file and print the header
file_output = open("V:/P5_PhysAct/People/Tom/pampropy/data/summary_ag_output.csv", "w")

# Formulate the file header based on the variables requested
file_header = ""
for k,v in timestats.items():
	for stat in v:
		if isinstance(stat, list):
			variable_name = str(k) + "_" + str(stat[0]) + "_" + str(stat[1])
		else:
			variable_name = str(k) + "_" + str(stat)
		
		file_header = file_header + "," + variable_name
file_header = file_header[1:]
file_output.write("id,timestamp," + file_header + "\n")

#result_chans = ts.summary_statistics(statistics=timestats )
result_chans = ts.piecewise_statistics( timedelta(minutes=10), timestats )
output = Time_Series.Time_Series("random.dat")
output.add_channels(result_chans)
output.write_channels_to_file(file_target=file_output)
output.write_channels_to_file(file_target=file_output)
#output.write_channels_to_file(file_target=os.path.join(os.path.dirname(__file__), '..', 'data/actigraph.csv'))
file_output.close()



execution_end = datetime.now()
print("{} to {} = {}".format( execution_start, execution_end, execution_end - execution_start))

ts.draw_separate()

