import numpy as np
from datetime import datetime, date, time, timedelta
import Annotation
import copy

class Channel(object):

	def __init__(self, name):
		
		self.name = name
		self.size = 0
		self.timeframe = 0
		self.data = []
		self.timestamps = []
		self.annotations = []
		self.draw_properties = {}

	def clone(self):

		return copy.deepcopy(self)

	def set_contents(self, data, timestamps):
		
		self.data = data
		self.timestamps = timestamps

		self.calculate_timeframe()

	def calculate_timeframe(self):

		self.size = len(self.data)
		self.timeframe = self.timestamps[0], self.timestamps[self.size-1], (self.timestamps[self.size-1]-self.timestamps[0]), self.size

	def add_annotation(self, annotation):

		self.annotations.append(annotation)

	def add_annotations(self, annotations):

		for a in annotations:
			self.add_annotation(a)

	def normalise(self, floor=0, ceil=1):

		max_value = max(self.data)
		min_value = min(self.data)
		self.data = ((ceil - floor) * (self.data - min_value))/(max_value - min_value) + floor

	def get_window(self, datetime_start, datetime_end):

		#indices = np.where((self.timestamps >= datetime_start) & (self.timestamps < datetime_end))
		#print(indices)
		#print(len(indices[0]))
		#return indices[0]
		# 7 mins 10 seconds
		start = np.searchsorted(self.timestamps, datetime_start, 'left')
		end = np.searchsorted(self.timestamps, datetime_end, 'right')
		return np.arange(start, end-1)

	def window_statistics(self, start_dts, end_dts, statistics):

		indices = self.get_window(start_dts, end_dts)

		pretty_timestamp = start_dts.strftime("%d/%m/%Y %H:%M:%S:%f")

		output_row = [pretty_timestamp]
		if (len(indices) > 0):
			
			for stat in statistics:
				if stat == "mean":
					output_row.append(np.mean(self.data[indices]))
				elif stat == "sum":
					output_row.append(sum(self.data[indices]))
				elif stat == "std":
					output_row.append(np.std(self.data[indices]))
				elif stat == "min":
					output_row.append(np.min(self.data[indices]))
				elif stat == "max":
					output_row.append(np.max(self.data[indices]))
				elif stat == "n":
					output_row.append(len(indices))
				elif isinstance(stat, list):

					indices2 = np.where((self.data[indices] >= stat[0]) & (self.data[indices] < stat[1]))[0]
					output_row.append(len(indices2))

				else:
					output_row.append(-1)
		else:
			for i in range(len(statistics)):
				output_row.append(-1)	

		return output_row

	

	def append_data(self, timestamp, data_row):
		self.timestamps.append(timestamp)
		self.data.append(data_row)

	def piecewise_statistics(self, window_size, statistics=["mean"], time_period=False):

		if time_period == False:
			start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
			end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
		else:
			start = time_period[0]
			end = time_period[1]
		# ------------------------------

		#print start , "---", end

		channel_list = []
		for var in statistics:
			
			channel = Channel(self.name + "_" + var)
			channel_list.append(channel)

		window = window_size
		start_dts = start
		end_dts = start + window

		while start_dts < end:
			
			results = self.window_statistics(start_dts, end_dts, statistics)
			for i in range(1,len(results)):
				
				channel_list[i-1].append_data(start_dts, results[i])

			start_dts = start_dts + window
			end_dts = end_dts + window

		for channel in channel_list:
			channel.calculate_timeframe()
			channel.data = np.array(channel.data)
			channel.timestamps = np.array(channel.timestamps)

		return channel_list


	def channel_statistics(self, statistics=["mean"], file_target=False):

		start = self.timeframe[0]
		end = self.timeframe[1]

		# ------------------------------
		output = []
		if not file_target == False:
			file_output = open(file_target, 'w')

			# Print the header
			file_output.write("timestamp,")
			for index,var in enumerate(statistics):
				if not isinstance(var,list):
					file_output.write(var)
				else:
					file_output.write("mte_"+str(var[0])+"_lte_"+str(var[1]))

				if (index < len(statistics)-1):
						file_output.write(",")
				else:
					file_output.write("\n")

		if not file_target == False:
			
			results = self.window_statistics(start, end, statistics)
			for index,var in enumerate(results):
				file_output.write(str(var))
				if (index < len(results)-1):
					file_output.write(",")
				else:
					file_output.write("\n")
			
		else:
			
			output.append(self.window_statistics(start, end, statistics))


		if not file_target == False:
			file_output.close()
		else:
			return output

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

			indices = self.get_window(bout[0], bout[1])

			#c.data[bout[2]:bout[3]] = self.data[bout[2]:bout[3]]
			c.data[indices] = self.data[indices]

		return c

	def moving_average(self, size):

		averaged = []
		half = (size-1)/2

		for i in range(0,self.size):
			total = 0
			contributors = 0
			for j in range(i-half,i+half):
				if (j >= 0) & (j < self.size):
					contributors+=1
					total += self.data[j]
			averaged.append(total/contributors)

		result = Channel(self.name + "_ma")
		result.set_contents(np.array(averaged), self.timestamps)
		return result

	def moving_std(self, size):

		averaged = []
		half = (size-1)/2

		for i in range(0,self.size):

			low = max(0,i-half)
			high = min(self.size,i+half)
			
			averaged.append(np.std(self.data[low:high]))

		result = Channel(self.name + "_mstd")
		result.set_contents(np.array(averaged), self.timestamps)
		return result

	def time_derivative(self):
		
		result = Channel(self.name + "_td")
		result.set_contents(np.diff(self.data), self.timestamps[:-1])
		return result

	def absolute(self):

		result = Channel(self.name + "_abs")
		result.set_contents(np.abs(self.data), self.timestamps)
		return result

def load_channels(source, source_type, datetime_format="%d/%m/%Y %H:%M:%S:%f", datetime_column=0):

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
		one_minute = timedelta(seconds=15)

		timestamp_list = []
		for i in range(0,len(activity)):
			timestamp_list.append(dt)
			dt = dt + one_minute

		timestamps = np.array(timestamp_list)

		indices1 = np.where(ecg > 1)
		activity = activity[indices1]
		ecg = ecg[indices1]
		timestamps2 = timestamps[indices1]

		actiheart_activity = Channel("AH_Activity")
		actiheart_activity.set_contents(activity, timestamps2)

		actiheart_ecg = Channel("AH-ECG")
		actiheart_ecg.set_contents(ecg, timestamps2)

		return [actiheart_activity, actiheart_ecg]

	elif (source_type == "activPAL"):

		ap_timestamp, ap_x, ap_y, ap_z = np.loadtxt(source, delimiter=',', unpack=True, skiprows=5, dtype={'names':('ap_timestamp','ap_x','ap_y','ap_z'), 'formats':('S16','f8','f8','f8')})
		print("A")
		dt = datetime.strptime("30-Dec-1899", "%d-%b-%Y")

		ap_timestamps = []
		for val in ap_timestamp:

			test = val.split(".")

			while len(test[1]) < 10:
				test[1] = test[1] + "0"

			finaltest = dt + timedelta(days=int(test[0]), microseconds=int(test[1])*8.64)
			ap_timestamps.append(finaltest)

		ap_timestamps = np.array(ap_timestamps)
		print("B")
		x = Channel("AP_X")
		y = Channel("AP_Y")
		z = Channel("AP_Z")

		ap_x = (ap_x-128.0)/64.0
		ap_y = (ap_y-128.0)/64.0
		ap_z = (ap_z-128.0)/64.0

		x.set_contents(np.array(ap_x, dtype=np.float64), ap_timestamps)
		y.set_contents(np.array(ap_y, dtype=np.float64), ap_timestamps)
		z.set_contents(np.array(ap_z, dtype=np.float64), ap_timestamps)
		print("C")
		return [x,y,z]

	elif (source_type == "GeneActiv_CSV"):

		ga_timestamp, ga_x, ga_y, ga_z, ga_lux, ga_event, ga_temperature = np.genfromtxt(source, delimiter=',', unpack=True, skip_header=80, dtype=str)

		ga_x = np.array(ga_x, dtype=np.float64)
		ga_y = np.array(ga_y, dtype=np.float64)
		ga_z = np.array(ga_z, dtype=np.float64)
		ga_lux = np.array(ga_lux, dtype=np.int32)
		ga_event = np.array(ga_event, dtype=np.bool_)
		ga_temperature = np.array(ga_temperature, dtype=np.float32)

		ga_timestamps = []

		for i in range(0, len(ga_timestamp)):
			ts = datetime.strptime(ga_timestamp[i], "%Y-%m-%d %H:%M:%S:%f")
			ga_timestamps.append(ts)
		ga_timestamps = np.array(ga_timestamps)

		x = Channel("GA_X")
		y = Channel("GA_Y")
		z = Channel("GA_Z")
		lux = Channel("GA_Lux")
		event = Channel("GA_Event")
		temperature = Channel("GA_Temperature")

		x.set_contents(ga_x, ga_timestamps)
		y.set_contents(ga_y, ga_timestamps)
		z.set_contents(ga_z, ga_timestamps)
		lux.set_contents(ga_lux, ga_timestamps)
		event.set_contents(ga_event, ga_timestamps)
		temperature.set_contents(ga_temperature, ga_timestamps)

		return [x,y,z,lux,event,temperature]

	elif (source_type == "CSV"):

		f = open(source, 'r')
		s = f.readline().strip()
		f.close()

		test = s.split(",")

		source_split = source.split("/")
		
		data = np.loadtxt(source, delimiter=',', skiprows=1, dtype='str')
		
		#print(data.shape)
		#print(data[:,0])

		timestamps = []
		for date_row in data[:,datetime_column]:
			timestamps.append(datetime.strptime(date_row, datetime_format))
		timestamps = np.array(timestamps)

		data_columns = list(range(0,len(test)))
		del data_columns[datetime_column]
		print data_columns
		channels = []
		for col in data_columns:
			print col
			name = source_split[-1] + " - " + test[col]
			c = Channel(name)
			c.set_contents(np.array(data[:,col], dtype=np.float64), timestamps)
			channels.append(c)

		return channels