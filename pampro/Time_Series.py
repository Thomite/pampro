import matplotlib.pyplot as plt
from datetime import datetime, date, time, timedelta
import numpy as np


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

        self.calculate_timeframe()

        self.channel_lookup[channel.name] = channel

        #print("Added channel {} to time series {}.".format(channel.name, self.name))
        #print("Earliest: {}, latest: {}".format(self.earliest, self.latest))

    def add_channels(self, new_channels):

        for c in new_channels:
            self.add_channel(c)

    def get_channel(self, channel_name):

        return self.channel_lookup[channel_name]

    def calculate_timeframe(self):

        chan = self.channels[0]
        self.earliest, self.latest = chan.timeframe

        for chan in self.channels[1:]:

            tf = chan.timeframe
            if tf[0] < self.earliest:
                self.earliest = tf[0]
            if tf[1] > self.latest:
                self.latest = tf[1]

    def rename_channel(self, current_name, desired_name):

        chan = self.get_channel(current_name)
        chan.name = desired_name
        self.channel_lookup.pop(current_name, None)
        self.channel_lookup[desired_name] = chan


    def build_statistics_channels(self, bouts, statistics):

        result_channels = []

        for channel_name,stats in statistics.items():
            if channel_name in self.channel_lookup.keys():
                channels = self.get_channel(channel_name).build_statistics_channels(bouts, statistics=stats)
                result_channels = result_channels + channels
            else:
                print("Warning: {} not in {}".format(channel_name, self.name))
        return result_channels

    def piecewise_statistics(self, window_size, statistics, time_period=False):

        result_channels = []

        for channel_name,stats in statistics.items():
            if channel_name in self.channel_lookup.keys():
                channels = self.get_channel(channel_name).piecewise_statistics(window_size, statistics=stats, time_period=time_period)
                result_channels = result_channels + channels
            else:
                print("Warning: {} not in {}".format(channel_name, self.name))
        return result_channels

    def summary_statistics(self, statistics):

        results = []

        for channel_name,stats in statistics.items():
            if channel_name in self.channel_lookup.keys():
                channel_results = self.get_channel(channel_name).summary_statistics(statistics=stats)
                results = results + channel_results
            else:
                print("Warning: {} not in {}".format(channel_name, self.name))
        return results


    def restrict_timeframe(self, start, end):

        for channel in self.channels:
            channel.restrict_timeframe(start,end)

        self.calculate_timeframe()


    def append(self, time_series):

        for channel in time_series.channels:

            match = self.get_channel(channel.name)
            match.append(channel)

            tf = channel.timeframe

            if self.earliest > tf[0]:
                self.earliest = tf[0]
            if self.latest < tf[1]:
                self.latest = tf[1]


    def write_channels_to_file(self, file_target, channel_list=False):

        channel_sources = []

        if channel_list == False:
            channel_sources = self.channels
        else:
            for channel_name in channel_list:
                channel_sources.append(self.get_channel(channel_name))

        file_output = False
        if isinstance(file_target, str):

            file_output = open(file_target, 'w')

            # Print the header
            file_output.write("id,timestamp,")
            for index,chan in enumerate(channel_sources):
                file_output.write(chan.name)
                if (index < len(channel_sources)-1):
                    file_output.write(",")
                else:
                    file_output.write("\n")
        else:

            file_output = file_target

        #for chan in channel_sources:
        #    print(chan.name)
        #    print(len(chan.data))
        #    print(len(chan.timestamps))

        for i in range(0,len(channel_sources[0].data)):

            pretty_timestamp = channel_sources[0].timestamps[i].strftime("%d/%m/%Y %H:%M:%S:%f")
            file_output.write(self.name + "," + pretty_timestamp + ",")

            for n,chan in enumerate(channel_sources):

                file_output.write(str(chan.data[i]))
                if n < len(channel_sources)-1:
                    file_output.write(",")
                else:
                    file_output.write("\n")

        if isinstance(file_target, str):
            file_output.close()


    def draw(self, channel_combinations, time_period=False, file_target=False):

        fig = plt.figure(figsize=(15,10), frameon=False)
        fig.patch.set_facecolor('#FFFFFF')

        axes = [fig.add_subplot(len(channel_combinations), 1, 1+index) for index in range(len(channel_combinations))]

        for channels, axis in zip(channel_combinations, axes):

            if time_period == False:
                axis.set_xlim(self.earliest, self.latest)
            else:
                axis.set_xlim(time_period[0], time_period[1])

            for c in channels:
                self.get_channel(c).draw(axis)

            legend = axis.legend(loc='upper right')
            legend.get_frame().set_alpha(0.5)
            legend.draw_frame(False)
            axis.grid()

        fig.tight_layout()

        if file_target==False:
            return fig
        else:
            plt.savefig(file_target, dpi=300, frameon=False, facecolor='#FFFFFF')

            plt.clf()
