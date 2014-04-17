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

	def get_window(self, datetime_start, datetime_end):

		indices = np.where((self.timestamps >= datetime_start) & (self.timestamps < datetime_end))
		#print(indices)
		#print(len(indices[0]))
		return indices[0]

	def bouts(self, low, high, minimum_length=0, return_indices=False):

		state = 0
		start_index = 0
		end_index = 1
		bouts = []

		for i, value in enumerate(self.data):

			if state == 0:

				if value >= low and value <= high:

					state = 1
					start_index = i
					end_index = i

			else:

				if value >= low and value <= high:

					end_index = i

				else:
				
					state = 0
					if (end_index - start_index + 1 >= minimum_length):
						if return_indices:
							bouts.append([self.timestamps[start_index], self.timestamps[end_index], start_index, end_index])	
						else:
							bouts.append([self.timestamps[start_index], self.timestamps[end_index]])
	
		return bouts

	def subset_using_bouts(self, bout_list, name):
		# Given a list of bouts, create a new channel from this taking only the data from inside those bouts
		c = Channel(name)

		c.set_contents(np.zeros(self.size), self.timestamps)

		#print(self.data[2345])

		for bout in bout_list:
			#print(bout)

			indices = np.where((self.timestamps >= bout[0]) & (self.timestamps < bout[1]))

			#c.data[bout[2]:bout[3]] = self.data[bout[2]:bout[3]]
			c.data[indices] = self.data[indices]

		return c


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

		#indices1 = np.where(ecg > 1)
		#activity = activity[indices1]
		#ecg = ecg[indices1]
		#timestamps = timestamps[indices1]

		actiheart_activity = Channel("Actiheart-Activity")
		actiheart_activity.set_contents(activity, timestamps)

		actiheart_ecg = Channel("Actiheart-ECG")
		actiheart_ecg.set_contents(ecg, timestamps)

		return [actiheart_activity, actiheart_ecg]