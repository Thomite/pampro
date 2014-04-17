import matplotlib.pyplot as plt

class Time_Series(object):

	def __init__(self):
		
		self.channels = []
		self.number_of_channels = 0
	
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

		print("Added channel {} to time series.".format(channel.name))
		print("Earliest: {}, latest: {}".format(self.earliest, self.latest))

	def add_channels(self, new_channels):

		for c in new_channels:
			self.add_channel(c)

	def draw(self):

		fig = plt.figure(figsize=(15,10))

		ax = fig.add_subplot(1,1,1)

		for channel in self.channels:

			ax.plot(channel.timestamps, channel.data, alpha=0.9, label=channel.name)

		legend = ax.legend(loc='upper right')

		fig.tight_layout()

		plt.show()

	def draw_normalised(self):

		fig = plt.figure(figsize=(15,10))

		ax = fig.add_subplot(1,1,1)

		for channel in self.channels:
			max_value = max(channel.data)
			min_value = min(channel.data)
			 
			ax.plot(channel.timestamps, ((1 - 0) * (channel.data - min_value))/(max_value - min_value) + 0, label=channel.name)

		legend = ax.legend(loc='upper right')

		fig.tight_layout()

		plt.show()

	def draw_separate(self):

		fig = plt.figure(figsize=(15,10))

		for index, channel in enumerate(self.channels):
			ax = fig.add_subplot(len(self.channels),1,1+index)
			ax.set_xlim(self.earliest, self.latest)
			ax.plot(channel.timestamps, channel.data, label=channel.name)
			legend = ax.legend(loc='upper right')
		fig.tight_layout()

		plt.show()

