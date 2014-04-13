import numpy as np
from datetime import datetime, date, time, timedelta

class Channel(object):

	def __init__(self, name):
		
		self.name = name
		self.size = 0
		self.timeframe = 0
		self.data = []
		self.timestamps = []

	def set_contents(self, data, timestamps):
		
		self.data = data
		self.timestamps = timestamps
		self.size = len(self.data)

		self.timeframe = self.timestamps[0], self.timestamps[self.size-1], (self.timestamps[self.size-1]-self.timestamps[0]), self.size

	def normalise(self, floor=0, ceil=1):

		max_value = max(self.data)
		min_value = min(self.data)
		self.data = ((ceil - floor) * (self.data - min_value))/(max_value - min_value) + floor

def load_channels(source, source_type):

	if (source_type == "Actiheart"):

		activity, ecg  = np.loadtxt(source, delimiter=',', unpack=True, skiprows=15, usecols=[1,2])

		first_lines = []
		f = open(source, 'r')
		for i in range(0,13):
			s = f.readline().strip()
			first_lines.append(s)
		f.close()

		line8 = first_lines[8]
		test = line8.split(",")
		dt = datetime.strptime(test[1], "%d-%b-%Y  %H:%M")
		one_minute = timedelta(minutes=1)

		timestamp_list = []
		for i in range(0,len(activity)):
			timestamp_list.append(dt)
			dt = dt + one_minute

		timestamps = np.array(timestamp_list)

		#cutoff1 = datetime.strptime("16-Mar-2014 13:00", "%d-%b-%Y  %H:%M")
		#cutoff2 = datetime.strptime("17-Mar-2014 13:00", "%d-%b-%Y  %H:%M")
		#indices2 = np.where((timestamps > cutoff1) & (timestamps < cutoff2))
		#ecg2 = ecg[indices2]
		#timestamps2 = timestamps[indices2]

		indices1 = np.where(ecg > 1)
		activity = activity[indices1]
		ecg = ecg[indices1]
		timestamps = timestamps[indices1]

		actiheart_activity = Channel("Actiheart-Activity")
		actiheart_activity.set_contents(activity, timestamps)

		actiheart_ecg = Channel("Actiheart-ECG")
		actiheart_ecg.set_contents(ecg, timestamps)

		return [actiheart_activity, actiheart_ecg]