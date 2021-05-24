# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.signal import butter, lfilter

from .Channel import *

def low_pass_filter(input_channel, low, frequency=1, order=1):

    nyq = 0.5 * frequency
    low_nyq = low / nyq
    b, a = butter(order, low_nyq)
    y = lfilter(b, a, input_channel.data)

    output_channel = Channel(input_channel.name + "_LPF_" + str(low))
    output_channel.set_contents(y, input_channel.timestamps)

    output_channel.inherit_time_properties(input_channel)

    return output_channel

def high_pass_filter(input_channel, high, frequency=1, order=1):

    nyq = 0.5 * frequency
    high_nyq = high / nyq
    b, a = butter(order, high_nyq, btype="high")
    y = lfilter(b, a, input_channel.data)

    output_channel = Channel(input_channel.name + "_HPF_" + str(high))
    output_channel.set_contents(y, input_channel.timestamps)

    output_channel.inherit_time_properties(input_channel)

    return output_channel
