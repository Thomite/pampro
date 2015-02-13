from pampro import Time_Series, Channel, channel_inference, Bout, pampro_utilities, time_utilities, batch_processing
import numpy as np    
import scipy as sp    
from datetime import datetime, date, time, timedelta    
from pampro import Bout, Channel
import copy    
    
from struct import *    
from math import *    
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

                if stat == "sum":    

                    intersection = Bout.bout_list_intersection([window],bouts)
                    Bout.cache_lengths(intersection)
                    sum_seconds = Bout.total_time(intersection).total_seconds()
                    output_row.append(sum_seconds)    

                elif stat == "mean":
                    
                    intersection = Bout.bout_list_intersection([window],bouts)
                    Bout.cache_lengths(intersection)
                    sum_seconds = Bout.total_time(intersection).total_seconds()
                    output_row.append( sum_seconds / len(bouts) ) 
                    
                elif stat == "n":    

                    output_row.append( len(bouts) )

                else:    

                    output_row.append(-1)    
        else:    
            # No bouts in this Bout_Collection overlapping this window
            num_missings = len(statistics)    
    
            for i in range(num_missings):    
                output_row.append(-1)    
    
    
        return output_row   

    def build_statistics_channels(self, windows, statistics):    
    
        channel_list = []    
        for var in statistics:    
 
            channel = Channel.Channel(self.name+"_"+str(var))    
            channel_list.append(channel)    
    
 
        for window in windows:    
    
            results = self.window_statistics(window.start_timestamp, window.end_timestamp, statistics)    
            for i in range(len(results)):    
                #print len(results)    
                channel_list[i].append_data(window.start_timestamp, results[i])    
    
        for channel in channel_list:    
            channel.calculate_timeframe()    
            channel.data = np.array(channel.data)    
            channel.timestamps = np.array(channel.timestamps)    
    
        return channel_list    

    def piecewise_statistics(self, window_size, statistics=["mean"], time_period=False):    
    
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
        
    def summary_statistics(self, statistics=["mean"]):    
    
        windows = [Bout.Bout(self.timeframe[0], self.timeframe[1])]    
        #results = self.window_statistics(self.timeframe[0], self.timeframe[1], statistics)    
    
        return self.build_statistics_channels(windows, statistics) 
    
    
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