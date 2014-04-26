import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta

class Time_Series(object):

	def __init__(self, name):
		
		self.name = name
		self.channels = []
		self.number_of_channels = 0
		self.channel_lookup = {}
		self.earliest = 0
		self.latest = 0

	def add_channel(self, channel):

		self.number_of_channels = self.number_of_channels + 1
		self.channels.append(channel)

		tf = channel.timeframe

		if (self.number_of_channels == 1):
			self.earliest = tf[0]
			self.latest = tf[1]

		else:
			if self.earliest > tf[0]:
				self.earliest = tf[0]
			if self.latest < tf[1]:
				self.latest = tf[1]

		self.channel_lookup[channel.name] = channel

		print("Added channel {} to time series {}.".format(channel.name, self.name))
		#print("Earliest: {}, latest: {}".format(self.earliest, self.latest))

	def add_channels(self, new_channels):

		for c in new_channels:
			self.add_channel(c)

	def get_channel(self, channel_name):

		return self.channel_lookup[channel_name]


	def piecewise_statistics(self, window_size, file_target, statistics=["mean"], time_period=False):

		if time_period == False:
			start = self.earliest - timedelta(hours=self.earliest.hour, minutes=self.earliest.minute, seconds=self.earliest.second, microseconds=self.earliest.microsecond)
			end = self.latest + timedelta(hours=23-self.latest.hour, minutes=59-self.latest.minute, seconds=59-self.latest.second, microseconds=999999-self.latest.microsecond)
		else:
			start = time_period[0]
			end = time_period[1]
		# ------------------------------
		#print start, "--|--", end
		for channel in self.channels:
			target = file_target + channel.name + ".csv"
			channel.piecewise_statistics(window_size, statistics=statistics, time_period=[start,end], file_target=target)


	def write_channels_to_file(self, file_target, channel_list=False):

		channel_sources = []

		if channel_list == False:
			channel_sources = self.channels
		else:
			for channel_name in channel_list:
				channel_sources.append(self.get_channel(channel_name))

		file_output = open(file_target, 'w')

		# Print the header
		file_output.write("timestamp,")
		for index,chan in enumerate(channel_sources):
			file_output.write(chan.name)
			if (index < len(channel_sources)-1):
				file_output.write(",")
			else:
				file_output.write("\n")


		for i in range(0,len(channel_sources[0].data)):

			pretty_timestamp = channel_sources[0].timestamps[i].strftime("%d/%m/%Y %H:%M:%S:%f")
			file_output.write(pretty_timestamp + ",")
			
			for n,chan in enumerate(channel_sources):

				file_output.write(str(chan.data[i]))
				if n < len(channel_sources)-1:
					file_output.write(",")
				else:
					file_output.write("\n")
		
		file_output.close()

	def draw(self, channels=False, time_period=False):

		fig = plt.figure(figsize=(15,10))

		ax = fig.add_subplot(1,1,1)

		channel_list = []
		if channels==False:
			channel_list = self.channels
		else:
			for c in channels:
				channel_list.append(self.get_channel(c))
		

		for channel in channel_list:

			if time_period==False:
				ax.plot(channel.timestamps, channel.data, alpha=0.9, label=channel.name)
			else:
				indices = channel.get_window(time_period[0], time_period[1])
				ax.plot(channel.timestamps[indices], channel.data[indices], label=channel.name, **channel.draw_properties)

		legend = ax.legend(loc='upper right')

		fig.tight_layout()

		plt.show()

	def draw_normalised(self, channels=False, time_period=False):

		fig = plt.figure(figsize=(15,10))

		ax = fig.add_subplot(1, 1, 1)

		channel_list = []
		if channels==False:
			channel_list = self.channels
		else:
			for c in channels:
				channel_list.append(self.get_channel(c))
		

		for channel in channel_list:
			max_value = max(channel.data)
			min_value = min(channel.data)
			 
			if time_period==False:
				indices = channel.get_window(time_period[0], time_period[1])
				if (len(indices) > 0):
					ax.plot(channel.timestamps[indices], ((1 - 0) * (channel.data[indices] - min_value))/(max_value - min_value) + 0, label=channel.name, **channel.draw_properties)
			else:
				ax.plot(channel.timestamps, ((1 - 0) * (channel.data - min_value))/(max_value - min_value) + 0, label=channel.name, **channel.draw_properties)

		legend = ax.legend(loc='upper right')

		fig.tight_layout()

		plt.show()

	def draw_separate(self, channels=False, time_period=False, file_target=False):

		fig = plt.figure(figsize=(15,10))

		channel_list = []
		if channels==False:
			channel_list = self.channels
		else:
			for c in channels:
				channel_list.append(self.get_channel(c))

		for index, channel in enumerate(channel_list):
			ax = fig.add_subplot(len(channel_list), 1, 1+index)
			

			if time_period==False:
				ax.plot(channel.timestamps, channel.data, label=channel.name, **channel.draw_properties)
				ax.set_xlim(self.earliest, self.latest)
			else:
				indices = channel.get_window(time_period[0], time_period[1])
				if (len(indices) > 0):
					ax.plot(channel.timestamps[indices[0]:indices[-1]], channel.data[indices[0]:indices[-1]], label=channel.name, **channel.draw_properties)
				ax.set_xlim(time_period[0], time_period[1])

			for a in channel.annotations:
				ax.axvspan(xmin=a.start_timestamp, xmax=a.end_timestamp, **a.draw_properties)
				#print(a.start_timestamp)

			legend = ax.legend(loc='upper right')
		fig.tight_layout()

		if file_target==False:
			plt.show()
		else:
			plt.savefig(file_target, dpi=300, frameon=False)