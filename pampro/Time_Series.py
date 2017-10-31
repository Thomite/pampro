import matplotlib
matplotlib.use("Agg") # So that plotting doesn't require X and can be done remotely

import matplotlib.pyplot as plt
from matplotlib import rcParams

from datetime import datetime, date, time, timedelta
import numpy as np
from collections import OrderedDict



class Time_Series(object):

    def __init__(self, name="Blank"):

        self.name = name
        self.channels = []
        self.number_of_channels = 0
        self.channel_lookup = {}
        self.earliest = 0
        self.latest = 0
        self.time_period = (self.earliest, self.latest)

    def __iter__(self):
        self.iter_index = 0
        return self

    def __next__(self):
        if self.iter_index == self.number_of_channels:
            raise StopIteration
        else:
            self.iter_index += 1
            return self.channels[self.iter_index-1]

    def __getitem__(self, key):

        index_type = str(type(key))
        if "int" in index_type:
            return self.channels[key]
        else:
            return self.channel_lookup[key]

    def add_channel(self, channel):
        """ Insert a Channel into the Time_Series and index it appropriately. """

        self.number_of_channels = self.number_of_channels + 1
        self.channels.append(channel)

        self.calculate_timeframe()

        self.channel_lookup[channel.name] = channel

    def add_channels(self, new_channels):
        """ Iteratively call add_channel on each Channel given. """

        for c in new_channels:
            self.add_channel(c)

    def get_channel(self, channel_name):

        return self.channel_lookup[channel_name]

    def get_channels(self, channel_names):

        return [self.get_channel(c) for c in channel_names]

    def calculate_timeframe(self):
        """ Find the earliest and latest observations contained by Channels inside this Time_Series. """

        chan = self.channels[0]
        self.earliest, self.latest = chan.timeframe

        for chan in self.channels[1:]:

            tf = chan.timeframe
            if tf[0] < self.earliest:
                self.earliest = tf[0]
            if tf[1] > self.latest:
                self.latest = tf[1]

        self.time_period = (self.earliest, self.latest)

    def rename_channel(self, current_name, desired_name):
        """ Change the name of a contained Channel. """

        chan = self.get_channel(current_name)
        chan.name = desired_name
        self.channel_lookup.pop(current_name, None)
        self.channel_lookup[desired_name] = chan


    def build_statistics_channels(self, bouts, statistics, common_time=False, name=""):

        if name == "":
            name = self.name
        result_ts = Time_Series(name)

        for channel_name,stats in statistics.items():
            if channel_name in self.channel_lookup.keys():

                if common_time:
                    # Once the first channel has been queried, inherit its indexes so the rest are faster
                    self.get_channel(channel_name).inherit_time_properties(self.get_channel(list(statistics.keys())[0]))

                ts = self.get_channel(channel_name).build_statistics_channels(bouts, statistics=stats, name=name)
                result_ts.add_channels(ts.channels)

            else:
                print("Warning: {} not in {}".format(channel_name, self.name))

        return result_ts

    def piecewise_statistics(self, window_size, statistics, time_period=False, common_time=False, name=""):

        if name == "":
            name = self.name
        result_ts = Time_Series(name)

        for channel_name,stats in statistics.items():
            if channel_name in self.channel_lookup.keys():

                if common_time:
                    # Once the first channel has been queried, inherit its indexes so the rest are faster
                    self.get_channel(channel_name).inherit_time_properties(self.get_channel(list(statistics.keys())[0]))

                ts = self.get_channel(channel_name).piecewise_statistics(window_size, statistics=stats, time_period=time_period)
                result_ts.add_channels(ts.channels)

            else:
                print("Warning: {} not in {}".format(channel_name, self.name))

        return result_ts

    def summary_statistics(self, statistics, time_period=False, name=""):

        if name == "":
            name = self.name
        result_ts = Time_Series(name)

        for channel_name,stats in statistics.items():
            if channel_name in self.channel_lookup.keys():
                channel_results = self.get_channel(channel_name).summary_statistics(statistics=stats, time_period=time_period)
                result_ts.add_channels(channel_results.channels)
            else:
                print("Warning: {} not in {}".format(channel_name, self.name))
        return result_ts


    def restrict_timeframe(self, start, end):
        """ Calls restrict_timeframe on each Channel inside this Time_Series. """

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


    def write_channels_to_file(self, file_target, channel_list=False, timestamp_format="%d/%m/%Y %H:%M:%S:%f"):

        channel_sources = []

        if channel_list == False:
            channel_sources = self.channels
        else:
            for channel_name in channel_list:
                channel_sources.append(self.get_channel(channel_name))

        # Error checking - only allows data to be written if all channels have equal number of data/timestamps
        for c1,c2 in zip(channel_sources, channel_sources[1:]):
            if (len(c1.data) != len(c2.data)) or (len(c1.timestamps) != len(c2.timestamps)) or (len(c1.data) != len(c1.timestamps)):
                raise Exception("Channels must have equal amount of data to be written to a file.")


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

            pretty_timestamp = channel_sources[0].timestamps[i].strftime(timestamp_format)
            file_output.write(self.name + "," + pretty_timestamp + ",")

            for n,chan in enumerate(channel_sources):

                file_output.write(str(chan.data[i]))
                if n < len(channel_sources)-1:
                    file_output.write(",")
                else:
                    file_output.write("\n")

        if isinstance(file_target, str):
            file_output.close()


    def draw(self, channel_combinations, time_period=False, file_target=False, width=6.3, height=4.35):

        try:
            rcParams['font.size'] = '8'

            rcParams['legend.frameon'] = True
            rcParams['legend.fontsize'] = "x-small"
            rcParams['legend.labelspacing'] = 0.1

            rcParams['xtick.labelsize'] = "x-small"
            rcParams['ytick.labelsize'] = "x-small"

            rcParams['grid.linewidth'] = 0.2

            rcParams['figure.dpi'] = 300
            rcParams['figure.facecolor'] = "white"

            rcParams['axes.linewidth'] = 0.25
            rcParams['legend.framealpha'] = 0.7

        except:
            print("Encountered an error whilst customising the appearance of charts. This may be because you are running an outdated version of matplotlib.")

        fig = plt.figure(figsize=(width,height), frameon=False)
        fig.patch.set_facecolor('#FFFFFF')

        if time_period == False:
            axis_xlim = (self.earliest, self.latest)
        else:
            axis_xlim = (time_period[0], time_period[1])

        axes = [fig.add_subplot(len(channel_combinations), 1, 1+index) for index in range(len(channel_combinations))]

        for channels, axis in zip(channel_combinations, axes):

            axis.set_xlim(axis_xlim)

            for c in channels:
                self.get_channel(c).draw(axis, time_period=axis_xlim)

            handles, labels = axis.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            legend = axis.legend(by_label.values(), by_label.keys(), loc='upper right')
            legend.get_frame().set_linewidth(0.0)
            axis.grid()

        fig.tight_layout()

        if file_target==False:
            return fig
        else:
            plt.savefig(file_target, dpi=300, frameon=False, facecolor='#FFFFFF')
            plt.close("all")
