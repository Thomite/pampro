from pampro import Time_Series, Channel, channel_inference, Bout, pampro_utilities, time_utilities, batch_processing
import numpy as np

from datetime import datetime, date, time, timedelta
from pampro import Bout, Channel
import copy

import time
from datetime import datetime

class Bout_Collection(object):

    def __init__(self, name, bouts=[]):

        self.name = name
        self.size = 0
        self.timeframe = 0
        self.bouts = bouts
        self.draw_properties = {}

    def clone(self):

        return copy.deepcopy(self)

    def bouts_involved(self, window):

        return [b for b in self.bouts if b.overlaps(window)]


    def window_statistics(self, start_dts, end_dts, statistics):

        window = Bout.Bout(start_dts, end_dts)
        bouts = self.bouts_involved(window)

        output_row = []
        if (len(bouts) > 0):

            for stat in statistics:

                if stat[0] == "generic":

                    for val1 in stat[1]:
                        if val1 == "sum":

                            intersection = Bout.bout_list_intersection([window],bouts)
                            Bout.cache_lengths(intersection)
                            sum_seconds = Bout.total_time(intersection).total_seconds()
                            output_row.append(sum_seconds)

                        elif val1 == "mean":

                            intersection = Bout.bout_list_intersection([window],bouts)
                            Bout.cache_lengths(intersection)
                            sum_seconds = Bout.total_time(intersection).total_seconds()
                            output_row.append( sum_seconds / len(bouts) )

                        elif val1 == "n":

                            output_row.append( len(bouts) )

                        else:
                            print("nooooooooo")
                            print(stat)
                            print(statistics)
                            output_row.append(-1)

                elif stat[0] == "sdx":

                    # ("sdx", [10,20,30,40,50,60,70,80,90])

                    sdx_results = sdx(bouts, stat[1])
                    for r in sdx_results:
                        output_row.append(r)

        else:
            # No bouts in this Bout_Collection overlapping this window
            # There was no data for the time period
            # Output -1 for each missing variable
            for i in range(self.expected_results(statistics)):
                output_row.append(-1)


        return output_row

    def build_statistics_channels(self, windows, statistics):

        channel_list = []

        for stat in statistics:
            #print(stat)
            channel_names = pampro_utilities.design_variable_names(self.name, stat)
            #print(channel_names)
            for cn in channel_names:
                channel_list.append(Channel.Channel(cn))

        num_expected_results = len(channel_list)

        for window in windows:

            results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)
            if len(results) != num_expected_results:
                raise Exception("Incorrect number of statistics yielded. {} expected, {} given. Channel: {}. Statistics: {}.".format(num_expected_results, len(results), self.name, statistics))

            for i in range(len(results)):
                #print len(results)
                channel_list[i].append_data(window.start_timestamp, results[i])

        for channel in channel_list:
            channel.calculate_timeframe()
            channel.data = np.array(channel.data)
            channel.timestamps = np.array(channel.timestamps)

        return channel_list

    def piecewise_statistics(self, window_size, statistics=[("generic", "mean")], time_period=False):

        if time_period == False:
            start = self.timeframe[0] - timedelta(hours=self.timeframe[0].hour, minutes=self.timeframe[0].minute, seconds=self.timeframe[0].second, microseconds=self.timeframe[0].microsecond)
            end = self.timeframe[1] + timedelta(hours=23-self.timeframe[1].hour, minutes=59-self.timeframe[1].minute, seconds=59-self.timeframe[1].second, microseconds=999999-self.timeframe[1].microsecond)
        else:
            start = time_period[0]
            end = time_period[1]

        #print("Piecewise statistics: {}".format(self.name))

        windows = []

        start_dts = start
        end_dts = start + window_size

        while start_dts < end:

            window = Bout.Bout(start_dts, end_dts)
            windows.append(window)

            start_dts = start_dts + window_size
            end_dts = end_dts + window_size

        return self.build_statistics_channels(windows, statistics)

    def summary_statistics(self, statistics=[("generic", "mean")]):

        windows = [Bout.Bout(self.timeframe[0], self.timeframe[1])]
        #results = self.window_statistics(self.timeframe[0], self.timeframe[1], statistics)

        return self.build_statistics_channels(windows, statistics)

    def expected_results(self, statistics):
        """ Calculate the number of expected results for this statistics request """

        expected = 0
        for stat in statistics:
            if stat[0] == "generic":
                expected += len(stat[1])
            elif stat[0] == "sdx":
                expected += len(stat[1])
        return expected

    def cache_lengths(self):

        for bout in self.bouts:
            bout.length = bout.end_timestamp - bout.start_timestamp

    def total_time(self):
        total = timedelta(minutes=0)

        self.cache_lengths()

        for bout in self.bouts:
            total += bout.length

        return total


    def limit_to_lengths(self, min_length=False, max_length=False, cached=False, sorted=False):

        if not cached:
            self.cache_lengths()

        within_length = []
        for bout in self.bouts:
            #bout_length = bout.end_timestamp - bout.start_timestamp
            if (min_length==False or bout.length >= min_length) and (max_length==False or bout.length <= max_length):
                within_length.append(bout)

            else:
                if sorted:
                    break

        self.bouts = within_length


def sdx(bouts, percentages):

    total_time_minutes = Bout.total_time(bouts).total_seconds()/60

    Bout.cache_lengths(bouts)
    bouts.sort(key=lambda x : x.length)

    highest_length_minutes = int(bouts[-1].length.total_seconds()/60)

    targets_minutes = [int((total_time_minutes)/100.0 * percentage) for percentage in percentages]
    results = []

    #print("Number of bouts: ", len(bouts))
    #print("Total time mins: ", total_time_minutes)
    #print("Highest length mins", highest_length_minutes)
    #print(targets_minutes)

    current_target_index = 0
    target_minutes = targets_minutes[current_target_index]
    for length in range(1, highest_length_minutes+1):

        included_bouts = [b for b in bouts if b.length.total_seconds()/60 <= length]
        #print(included_bouts)
        total_included_time_minutes = Bout.total_time(included_bouts).total_seconds()/60

        #print(length, total_included_time_minutes)
        while total_included_time_minutes >= target_minutes:

            #print(">target_minutes", target_minutes)
            #length is the result
            results.append(length)
            current_target_index += 1
            if current_target_index == len(targets_minutes):
                target_minutes = 999999999
            else:
                target_minutes = targets_minutes[current_target_index]

        if current_target_index == len(targets_minutes):
            break

    #print(results)
    return results
