from pampro import Channel
import numpy as np
from scipy.signal import butter, lfilter

def low_pass_filter(input_channel, low, frequency=1, order=1):

    nyq = 0.5 * frequency
    low_nyq = low / nyq
    b, a = butter(order, low_nyq)
    y = lfilter(b, a, input_channel.data)

    output_channel = Channel.Channel(input_channel.name + "_LPF_" + str(low))
    output_channel.set_contents(y, input_channel.timestamps)

    return output_channel

def high_pass_filter(input_channel, high, frequency=1, order=1):

    nyq = 0.5 * frequency
    high_nyq = high / nyq
    b, a = butter(order, high_nyq, btype="high")
    y = lfilter(b, a, input_channel.data)

    output_channel = Channel.Channel(input_channel.name + "_HPF_" + str(high))
    output_channel.set_contents(y, input_channel.timestamps)

    if input_channel.sparsely_timestamped:
        output_channel.indices = input_channel.indices

    return output_channel
