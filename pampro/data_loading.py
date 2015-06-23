import numpy as np
import scipy as sp
from datetime import datetime, date, time, timedelta
from pampro import Time_Series, Channel, Bout, pampro_utilities
import copy
from struct import *
from math import *
import time
from datetime import datetime
import sys
import io
import re
import string
from scipy.io.wavfile import write
import zipfile
from collections import OrderedDict

# Axivity import code adapted from source provided by Open Movement: https://code.google.com/p/openmovement/. Their license terms are reproduced here in full, and apply only to the Axivity related code:
# Copyright (c) 2009-2014, Newcastle University, UK. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def twos_comp(val, bits):

    if( (val&(1<<(bits-1))) != 0 ):
        val = val - (1<<bits)
    return val

def byte(value):
    return (value + 2 ** 7) % 2 ** 8 - 2 ** 7

def ushort(value):
    return value % 2 ** 16

def short(value):
    return (value + 2 ** 15) % 2 ** 16 - 2 ** 15

def axivity_read_timestamp(stamp):
    stamp = unpack('I', stamp)[0]
    year = ((stamp >> 26) & 0x3f) + 2000
    month = (stamp >> 22) & 0x0f
    day   = (stamp >> 17) & 0x1f
    hours = (stamp >> 12) & 0x1f
    mins  = (stamp >>  6) & 0x3f
    secs  = (stamp >>  0) & 0x3f
    try:
        t = datetime(year, month, day, hours, mins, secs)
    except ValueError:
        t = None
    return t

def axivity_read_timestamp_raw(stamp):
    year = ((stamp >> 26) & 0x3f) + 2000
    month = (stamp >> 22) & 0x0f
    day   = (stamp >> 17) & 0x1f
    hours = (stamp >> 12) & 0x1f
    mins  = (stamp >>  6) & 0x3f
    secs  = (stamp >>  0) & 0x3f
    try:
        t = datetime(year, month, day, hours, mins, secs)
    except ValueError:
        t = None
    return t

def axivity_read(fh, bytes):
    data = fh.read(bytes)
    if len(data) == bytes:
        return data
    else:
        raise IOError

def axivity_parse_header(fh):
    blockSize = unpack('H', axivity_read(fh,2))[0]
    performClear = unpack('B', axivity_read(fh,1))[0]
    deviceId = unpack('H', axivity_read(fh,2))[0]
    sessionId = unpack('I', axivity_read(fh,4))[0]
    shippingMinLightLevel = unpack('H', axivity_read(fh,2))[0]
    loggingStartTime = axivity_read(fh,4)
    loggingEndTime = axivity_read(fh,4)
    loggingCapacity = unpack('I', axivity_read(fh,4))[0]
    allowStandby = unpack('B', axivity_read(fh,1))[0]
    debuggingInfo = unpack('B', axivity_read(fh,1))[0]
    batteryMinimumToLog = unpack('H', axivity_read(fh,2))[0]
    batteryWarning = unpack('H', axivity_read(fh,2))[0]
    enableSerial = unpack('B', axivity_read(fh,1))[0]
    lastClearTime = axivity_read(fh,4)
    samplingRate = unpack('B', axivity_read(fh,1))[0]
    lastChangeTime = axivity_read(fh,4)
    firmwareVersion = unpack('B', axivity_read(fh,1))[0]

    reserved = axivity_read(fh,22)

    annotationBlock = axivity_read(fh,448 + 512)

    if len(annotationBlock) < 448 + 512:
        annotationBlock = ""

    annotation = ""
    for x in annotationBlock:
        if x != 255 and x != ' ':
            if x == '?':
                x = '&'
            annotation += str(x)
    annotation = annotation.strip()

    annotationElements = annotation.split('&')
    annotationNames = {'_c': 'studyCentre', '_s': 'studyCode', '_i': 'investigator', '_x': 'exerciseCode', '_v': 'volunteerNum', '_p': 'bodyLocation', '_so': 'setupOperator', '_n': 'notes', '_b': 'startTime', '_e': 'endTime', '_ro': 'recoveryOperator', '_r': 'retrievalTime', '_co': 'comments'}
    annotations = dict()
    for element in annotationElements:
        kv = element.split('=', 2)
        if kv[0] in annotationNames:
            annotations[annotationNames[kv[0]]] = kv[1]

    for x in ('startTime', 'endTime', 'retrievalTime'):
        if x in annotations:
            if '/' in annotations[x]:
                annotations[x] = time.strptime(annotations[x], '%d/%m/%Y')
            else:
                annotations[x] = time.strptime(annotations[x], '%Y-%m-%d %H:%M:%S')

    annotations = annotations
    deviceId = deviceId
    sessionId = sessionId
    lastClearTime = axivity_read_timestamp(lastClearTime)
    lastChangeTime = axivity_read_timestamp(lastChangeTime)
    firmwareVersion = firmwareVersion if firmwareVersion != 255 else 0

    return {"frequency":samplingRate}

def parse_header(header, type, datetime_format):

    header_info = OrderedDict()

    if type == "Actiheart":

        for i,row in enumerate(header):
            try:
                values = row.split(",")
                header_info[values[0]] = values[1]
            except:
                pass

        time1 = datetime.strptime(header[-2].split(",")[0], "%H:%M:%S")
        time2 = datetime.strptime(header[-1].split(",")[0], "%H:%M:%S")
        header_info["epoch_length"] = time2 - time1

        header_info["start_date"] = datetime.strptime(header_info["Started"], "%d-%b-%Y  %H:%M")

        if "Start trimmed to" in header_info:
            header_info["Start trimmed to"] = datetime.strptime(header_info["Start trimmed to"], "%Y-%m-%d %H:%M")


        for i,row in enumerate(header):

            if row.split(",")[0] == "Time":
                header_info["data_start"] = i+1
                break

    elif type == "Actigraph":

        test = header[2].split(" ")
        timeval = datetime.strptime(test[-1], "%H:%M:%S")
        start_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
        header_info["start_time"] = start_time

        test = header[3].split(" ")
        start_date = test[-1].replace("-", "/")

        try:
            start_date = datetime.strptime(start_date, datetime_format)
        except:
            start_date = datetime.strptime(start_date, "%d/%m/%Y")
        header_info["start_date"] = start_date

        test = header[4].split(" ")
        delta = datetime.strptime(test[-1], "%H:%M:%S")
        epoch_length = timedelta(hours=delta.hour, minutes=delta.minute, seconds=delta.second)
        header_info["epoch_length"] = epoch_length

        start_datetime = start_date + start_time
        header_info["start_datetime"] = start_datetime

        mode = 0
        splitup = header[8].split(" ")
        if "Mode" in splitup:
            index = splitup.index("Mode")
            mode = splitup[index + 2]
        header_info["mode"] = int(mode)

    elif type == "GT3X+_CSV":

        test = header[2].split(" ")
        timeval = datetime.strptime(test[-1], "%H:%M:%S")
        start_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
        header_info["start_time"] = start_time

        test = header[0].split(" ")
        if "Hz" in test:
            index = test.index("Hz")
            hz = int(test[index-1])
            epoch_length = timedelta(seconds=1) / hz
            header_info["epoch_length"] = epoch_length

        if "format" in test:
            index = test.index("format")
            format = test[index+1]
            format = format.replace("dd", "%d")
            format = format.replace("MM", "%m")
            format = format.replace("yyyy", "%Y")

            start_date = datetime.strptime(header[3].split(" ")[2], format)
            header_info["start_date"] = start_date

        start_datetime = start_date + start_time
        header_info["start_datetime"] = start_datetime

    elif type == "GeneActiv":

        header_info["start_datetime"] = header[21][11:]

        #print(header_info["start_datetime"])

        if header_info["start_datetime"] == "0000-00-00 00:00:00:000":
            header_info["start_datetime_python"] = datetime.strptime("0001-01-01", "%Y-%m-%d")
        else:
            header_info["start_datetime_python"] = datetime.strptime(header_info["start_datetime"], "%Y-%m-%d %H:%M:%S:%f")

        header_info["device_id"] = header[1].split(":")[1]
        header_info["firmware"] = header[4][24:]
        header_info["calibration_date"] = header[5][17:]
        header_info["frequency"] = float(header[19].split(":")[1].replace(" Hz", ""))
        header_info["epoch"] = timedelta(seconds=1) / int(header_info["frequency"])

        header_info["x_gain"] = float(header[47].split(":")[1])
        header_info["x_offset"] = float(header[48].split(":")[1])
        header_info["y_gain"] = float(header[49].split(":")[1])
        header_info["y_offset"] = float(header[50].split(":")[1])
        header_info["z_gain"] = float(header[51].split(":")[1])
        header_info["z_offset"] = float(header[52].split(":")[1])

        header_info["number_pages"] = int(header[57].split(":")[1])

        return header_info

    return header_info

def convert_actigraph_timestamp(t):
    return datetime(*map(int, [t[6:10],t[3:5],t[0:2],t[11:13],t[14:16],t[17:19],int(t[20:])*1000]))

def load(source, source_type, datetime_format="%d/%m/%Y %H:%M:%S:%f", datetime_column=0, ignore_columns=False, unique_names=False):

    header = OrderedDict()
    channels = []
    ts = Time_Series.Time_Series("")

    if (source_type == "Actiheart"):

        first_lines = []
        f = open(source, 'r')
        for i in range(0,30):
            s = f.readline().strip()
            first_lines.append(s)
        f.close()

        header_info = parse_header(first_lines, "Actiheart", "%d-%b-%Y  %H:%M")

        start_date = header_info["start_date"]
        epoch_length = header_info["epoch_length"]
        data_start = header_info["data_start"]
        #timestamp_list = []
        #for i in range(0,len(activity)):
        #    timestamp_list.append(start_date)
        #    start_date = start_date + epoch_length

        activity, ecg  = np.loadtxt(source, delimiter=',', unpack=True, skiprows=data_start, usecols=[1,2])

        timestamp_list = [start_date+i*epoch_length for i in range(len(activity))]
        timestamps = np.array(timestamp_list)


        indices1 = []

        if "Start trimmed to" in header_info:
            indices1 = np.where((ecg > 0) & (timestamps > header_info["Start trimmed to"]))
        else:
            indices1 = np.where(ecg > 0)

        activity = activity[indices1]
        ecg = ecg[indices1]
        timestamps2 = timestamps[indices1]

        actiheart_activity = Channel.Channel("AH_Activity")
        actiheart_activity.set_contents(activity, timestamps2)

        actiheart_ecg = Channel.Channel("AH_ECG")
        actiheart_ecg.set_contents(ecg, timestamps2)

        header = header_info
        channels = [actiheart_activity, actiheart_ecg]

    elif (source_type == "activPAL_CSV"):

        ap_timestamp, ap_x, ap_y, ap_z = np.loadtxt(source, delimiter=',', unpack=True, skiprows=5, dtype={'names':('ap_timestamp','ap_x','ap_y','ap_z'), 'formats':('S16','f8','f8','f8')})
        #print("A")
        dt = datetime.strptime("30-Dec-1899", "%d-%b-%Y")

        ap_timestamps = []
        for val in ap_timestamp:

            test = val.split(".")

            while len(test[1]) < 10:
                test[1] = test[1] + "0"

            finaltest = dt + timedelta(days=int(test[0]), microseconds=int(test[1])*8.64)
            ap_timestamps.append(finaltest)

        ap_timestamps = np.array(ap_timestamps)
        #print("B")
        x = Channel.Channel("AP_X")
        y = Channel.Channel("AP_Y")
        z = Channel.Channel("AP_Z")

        ap_x = (ap_x-128.0)/64.0
        ap_y = (ap_y-128.0)/64.0
        ap_z = (ap_z-128.0)/64.0

        x.set_contents(np.array(ap_x, dtype=np.float64), ap_timestamps)
        y.set_contents(np.array(ap_y, dtype=np.float64), ap_timestamps)
        z.set_contents(np.array(ap_z, dtype=np.float64), ap_timestamps)
        #print("C")
        channels = [x,y,z]


    elif (source_type == "activPAL"):

        f = open(source, "rb")
        data = f.read()
        filesize = len(data)
        data = io.BytesIO(data)


        #print("File read in")

        #print("filesize", filesize)

        A = unpack('1024s', data.read(1024))[0]

        #print((A[276]) << 8 | (A[277]))
        #print((A[278]) << 8 | (A[279]))


        start_time = str((A[256])).rjust(2, "0") + ":" + str((A[257])).rjust(2, "0") + ":" + str((A[258])).rjust(2, "0")
        start_date = str((A[259])).rjust(2, "0") + "/" + str((A[260])).rjust(2, "0") + "/" + str(2000 + (A[261]))

        end_time = str((A[262])).rjust(2, "0") + ":" + str((A[263])).rjust(2, "0") + ":" + str((A[264])).rjust(2, "0")
        end_date = str((A[265])).rjust(2, "0") + "/" + str((A[266])).rjust(2, "0") + "/" + str(2000 + (A[267]))

        start = start_date + " " + start_time
        end = end_date + " " + end_time

        #print(start_date, start_time)
        #print(end_date, end_time)

        # Given in Hz
        sampling_frequency = A[35]

        # 0 = 2g, 1 = 4g (therefore double the raw values), 2 = 8g (therefore quadruple the raw values)
        dynamic_range = A[38]
        dynamic_multiplier = 2**dynamic_range

        start_python = datetime.strptime(start, "%d/%m/%Y %H:%M:%S")
        end_python = datetime.strptime(end, "%d/%m/%Y %H:%M:%S")
        duration = end_python - start_python

        num_records = duration.total_seconds() / (timedelta(seconds=1)/sampling_frequency).total_seconds()

        #print("expected records:", num_records)
        #print("sampling frequency", (A[35]))

        #print("header end", A[1012:1023])


        n = 0
        data_cache = False
        # extract timestamp

        x = np.zeros(num_records)
        y = np.zeros(num_records)
        z = np.zeros(num_records)

        x.fill(-1.1212121212121)
        y.fill(-1.1212121212121)
        z.fill(-1.1212121212121)

        last_a,last_b,last_c = 0,0,0

        while n < num_records and data.tell() < filesize:

            try:
                data_cache = data.read(3)
                a,b,c = unpack('ccc', data_cache)
                a,b,c = ord(a), ord(b), ord(c)

                #print(a,b,c)

                # activPAL writes TAIL but these values could legitimately turn up
                if a == 116 and b == 97 and c == 105:
                    # if a,b,c spell TAI

                    d = ord(unpack('c', data.read(1))[0])
                    # and if d == T, so TAIL just came up
                    if d == 108:
                        #print ("Found footer!")
                        #print (data.tell())
                        remainder = data.read()
                    else:
                    # Otherwise TAI came up coincidently
                    # Meaning a,b,c was a legitimate record, and we just read in a from another record
                    # So read in the next 2 bytes and record 2 records: a,b,c and d,e,f
                        e,f = unpack('cc', data.read(2))
                        e,f = ord(e), ord(f)
                        x[n] = a
                        y[n] = b
                        z[n] = c
                        n += 1
                        x[n] = d
                        y[n] = e
                        z[n] = f
                        n += 1

                else:
                    if a == 0 and b == 0:
                        # repeat last abc C-1 times
                        x[n:(n+c+1)] = [last_a for val in range(c+1)]
                        y[n:(n+c+1)] = [last_b for val in range(c+1)]
                        z[n:(n+c+1)] = [last_c for val in range(c+1)]
                        n += c+1
                    else:
                        x[n] = a
                        y[n] = b
                        z[n] = c
                        n += 1

                last_a, last_b, last_c = a,b,c


            except:
                """
                print("Exception tell", data.tell())
                print(str(sys.exc_info()))
                print("data_cache:", data_cache)
                print("len(data_cache)", len(data_cache))

                for a in data_cache:
                    print(a)
                """
                break


        x.resize(n)
        y.resize(n)
        z.resize(n)

        x = (x-128.0)/64.0
        y = (y-128.0)/64.0
        z = (z-128.0)/64.0

        if dynamic_multiplier > 1:
            x *= dynamic_multiplier
            y *= dynamic_multiplier
            z *= dynamic_multiplier

        delta = timedelta(seconds=1)/sampling_frequency
        timestamps = np.array([start_python + delta*i for i in range(n)])

        x_channel = Channel.Channel("X")
        y_channel = Channel.Channel("Y")
        z_channel = Channel.Channel("Z")

        x_channel.set_contents(x, timestamps)
        y_channel.set_contents(y, timestamps)
        z_channel.set_contents(z, timestamps)

        channels = [x_channel, y_channel, z_channel]

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

        x = Channel.Channel("GA_X")
        y = Channel.Channel("GA_Y")
        z = Channel.Channel("GA_Z")
        lux = Channel.Channel("GA_Lux")
        event = Channel.Channel("GA_Event")
        temperature = Channel.Channel("GA_Temperature")

        x.set_contents(ga_x, ga_timestamps)
        y.set_contents(ga_y, ga_timestamps)
        z.set_contents(ga_z, ga_timestamps)
        lux.set_contents(ga_lux, ga_timestamps)
        event.set_contents(ga_event, ga_timestamps)
        temperature.set_contents(ga_temperature, ga_timestamps)

        channels = [x,y,z,lux,event,temperature]

    elif (source_type == "Actigraph"):

        first_lines = []
        f = open(source, 'r')
        for i in range(0,10):
            s = f.readline().strip()
            first_lines.append(s)


        header_info = parse_header(first_lines, "Actigraph", datetime_format)

        time = header_info["start_datetime"]
        epoch_length = header_info["epoch_length"]
        mode = header_info["mode"]

        count_list = []
        timestamp_list = []

        line = f.readline().strip()
        while (len(line) > 0):

            counts = line.split()
            for index, c in enumerate(counts):
            #    print index, index % 2
                if mode == 0 or mode == 4 or (mode == 1 and index % 2 == 0) or (mode == 3 and index % 2 == 0):
                    #print("eh?")
                    count_list.append(int(c))
                    timestamp_list.append(time)
                    time = time + epoch_length
                #else:
                    #print("eh?")


            line = f.readline().strip()
        f.close()

        timestamps = np.array(timestamp_list)
        counts = np.array(count_list)

        #print timestamps[0], timestamps[-1]
        #print sum(counts)



        chan = Channel.Channel("AG_Counts")
        chan.set_contents(counts, timestamps)

        channels = [chan]
        header = header_info

    elif (source_type == "GT3X+_CSV"):

        first_lines = []
        f = open(source, 'r')
        for i in range(0,10):
            s = f.readline().strip()
            first_lines.append(s)
        f.close()

        header_info = parse_header(first_lines, "GT3X+_CSV", "")

        time = header_info["start_datetime"]
        epoch_length = header_info["epoch_length"]

        timestamps = np.genfromtxt(source, delimiter=',', converters={0:convert_actigraph_timestamp}, skip_header=11, usecols=(0))

        x,y,z = np.genfromtxt(source, delimiter=',', skip_header=11, usecols=(1,2,3), unpack=True)


        x_chan = Channel.Channel("X")
        y_chan = Channel.Channel("Y")
        z_chan = Channel.Channel("Z")

        x_chan.set_contents(x, timestamps)
        y_chan.set_contents(y, timestamps)
        z_chan.set_contents(z, timestamps)

        channels = [x_chan,y_chan,z_chan]
        header = header_info

    elif (source_type == "GT3X+_CSV_ZIP"):

        filename = source.split("/")[-1].replace(".zip", ".csv")
        archive = zipfile.ZipFile(source)
        file_handle = archive.open(filename)

        first_lines = []
        for i in range(0,10):
            s = file_handle.readline().strip().decode("utf-8")
            #print(s)
            first_lines.append(s)

        #print(first_lines)

        header_info = parse_header(first_lines, "GT3X+_CSV", "")

        time = header_info["start_datetime"]
        epoch_length = header_info["epoch_length"]

        file_handle = archive.open(filename)
        timestamps = np.genfromtxt(file_handle, delimiter=',', converters={0:convert_actigraph_timestamp}, skip_header=11, usecols=(0))


        file_handle = archive.open(filename)
        x,y,z = np.genfromtxt(file_handle, delimiter=',', skip_header=11, usecols=(1,2,3), unpack=True)



        x_chan = Channel.Channel("X")
        y_chan = Channel.Channel("Y")
        z_chan = Channel.Channel("Z")

        x_chan.set_contents(x, timestamps)
        y_chan.set_contents(y, timestamps)
        z_chan.set_contents(z, timestamps)

        channels = [x_chan,y_chan,z_chan]
        header = header_info

    elif (source_type == "CSV"):

        f = open(source, 'r')
        s = f.readline().strip()
        f.close()

        test = s.split(",")

        source_split = source.split("/")


        data = np.loadtxt(source, delimiter=',', skiprows=1, dtype='S').astype("U")


        timestamps = []
        for date_row in data[:,datetime_column]:
            #print(date_row)
            #print(str(date_row))
            #print(type(date_row))
            timestamps.append(datetime.strptime(date_row, datetime_format))
        timestamps = np.array(timestamps)

        data_columns = list(range(0,len(test)))
        del data_columns[datetime_column]

        if ignore_columns != False:
            for ic in ignore_columns:
                del data_columns[ic]

        #print data_columns

        channels = []
        for col in data_columns:
            #print col
            if unique_names:
                name = source_split[-1] + " - " + test[col]
            else:
                name = test[col]
            c = Channel.Channel(name)
            c.set_contents(np.array(data[:,col], dtype=np.float64), timestamps)
            channels.append(c)


    elif (source_type == "Axivity"):

        channel_x = Channel.Channel("X")
        channel_y = Channel.Channel("Y")
        channel_z = Channel.Channel("Z")

        raw_bytes = open(source, "rb").read()
        #print("Number of bytes:", len(raw_bytes))
        #print("/512 = ", len(raw_bytes)/512)

        fh = io.BytesIO(raw_bytes)

        n = 0
        num_samples = 0

        start = datetime(2014, 1, 1)

        # Rough number of pages expected = length of file / size of block (512 bytes)
        # Rough number of samples expected = pages * 120
        # Add 1% buffer just to be cautious - it's trimmed later
        estimated_size = int(((len(raw_bytes)/512)*120)*1.01)

        #print("Estimated number of observations:", estimated_size)

        axivity_x = np.empty(estimated_size)
        axivity_y = np.empty(estimated_size)
        axivity_z = np.empty(estimated_size)
        axivity_timestamps = np.empty(estimated_size, dtype=type(start))

        sample_rates = []

        try:
            header = axivity_read(fh,2)

            while len(header) == 2:

                if header == b'MD':
                    #print('MD')
                    axivity_parse_header(fh)
                elif header == b'UB':
                    #print('UB')
                    blockSize = unpack('H', axivity_read(fh,2))[0]
                elif header == b'SI':
                    #print('SI')
                    pass
                elif header == b'AX':

                    packetLength, deviceId, sessionId, sequenceId, sampleTimeData, light, temperature, events,battery,sampleRate, numAxesBPS, timestampOffset, sampleCount = unpack('HHIIIHHcBBBhH', axivity_read(fh,28))

                    sample_rates.append(sampleRate)

                    sampleTime = axivity_read_timestamp_raw(sampleTimeData)

                    if packetLength != 508:
                        continue

                    if sampleTime == None:
                        continue

                    if sampleRate == 0:
                        continue

                    if ((numAxesBPS >> 4) & 15) != 3:
                        print('[ERROR: num-axes not expected]')

                    if (numAxesBPS & 15) == 2:
                        bps = 6
                    elif (numAxesBPS & 15) == 0:
                        bps = 4

                    timestamp = sampleTime
                    freq = 3200 / (1 << (15 - sampleRate & 15))
                    if freq <= 0:
                        freq = 1
                    offsetStart = float(-timestampOffset) / float(freq)


                    time0 = timestamp + timedelta(milliseconds=offsetStart)
                    sampleOffset = (timedelta(seconds=1) / sampleCount)


                    for sample in range(sampleCount):

                        x,y,z,t = 0,0,0,0

                        if bps == 6:

                            x,y,z = unpack('hhh', fh.read(6))
                            x,y,z = x/256.0,y/256.0,z/256.0

                        elif bps == 4:
                            temp = unpack('I', fh.read(4))[0]
                            temp2 = (6 - byte(temp >> 30))
                            x = short(short((ushort(65472) & ushort(temp << 6))) >> temp2) / 256.0
                            y = short(short((ushort(65472) & ushort(temp >> 4))) >> temp2) / 256.0
                            z = short(short((ushort(65472) & ushort(temp >> 14))) >> temp2) / 256.0

                            # Optimisation:
                            # Cache value of ushort(65472) ?


                        t = sample*sampleOffset + time0

                        axivity_x[num_samples] = x
                        axivity_y[num_samples] = y
                        axivity_z[num_samples] = z
                        axivity_timestamps[num_samples] = t
                        num_samples += 1

                    checksum = unpack('H', axivity_read(fh,2))[0]

                else:
                    #pass
                    print("Unrecognised header", header)

                header = axivity_read(fh,2)

                n=n+1
        except IOError:
            pass

        print(sum(sample_rates)/len(sample_rates))
        axivity_x.resize(num_samples)
        axivity_y.resize(num_samples)
        axivity_z.resize(num_samples)
        axivity_timestamps.resize(num_samples)

        channel_x.set_contents(axivity_x, axivity_timestamps)
        channel_y.set_contents(axivity_y, axivity_timestamps)
        channel_z.set_contents(axivity_z, axivity_timestamps)

        channels = [channel_x,channel_y,channel_z]

    elif (source_type == "Axivity_ZIP"):

        channel_x = Channel.Channel("X")
        channel_y = Channel.Channel("Y")
        channel_z = Channel.Channel("Z")
        channel_light = Channel.Channel("Light")
        channel_temperature = Channel.Channel("Temperature")

        #print("Opening file")
        archive = zipfile.ZipFile(source, "r")

        without_filepath = source.split("/")[-1]

        cwa_not_zip = without_filepath.replace(".zip", ".cwa")

        handle = archive.open(cwa_not_zip)

        raw_bytes = handle.read()
        #print("Number of bytes:", len(raw_bytes))
        #print("/512 = ", len(raw_bytes)/512)

        fh = io.BytesIO(raw_bytes)

        n = 0
        num_samples = 0
        num_pages = 0

        start = datetime(2014, 1, 1)

        # Rough number of pages expected = length of file / size of block (512 bytes)
        # Rough number of samples expected = pages * 120
        # Add 1% buffer just to be cautious - it's trimmed later
        estimated_size = int(((len(raw_bytes)/512)*120)*1.01)

        #print("Estimated number of observations:", estimated_size)

        axivity_x = np.empty(estimated_size)
        axivity_y = np.empty(estimated_size)
        axivity_z = np.empty(estimated_size)
        axivity_light = np.empty(len(raw_bytes)/512*1.01)
        axivity_temperature = np.empty(len(raw_bytes)/512*1.01)
        axivity_timestamps = np.empty((len(raw_bytes)/512)*1.01, dtype=type(start))
        axivity_indices = np.empty(len(raw_bytes)/512*1.01)
        sample_rates = []
        file_header = OrderedDict()

        try:
            header = axivity_read(fh,2)

            while len(header) == 2:

                if header == b'MD':
                    #print('MD')
                    file_header = axivity_parse_header(fh)
                elif header == b'UB':
                    #print('UB')
                    blockSize = unpack('H', axivity_read(fh,2))[0]
                elif header == b'SI':
                    #print('SI')
                    pass
                elif header == b'AX':

                    packetLength, deviceId, sessionId, sequenceId, sampleTimeData, light, temperature, events, battery, sampleRate, numAxesBPS, timestampOffset, sampleCount = unpack('HHIIIHHcBBBhH', axivity_read(fh,28))
                    sample_rates.append(sampleRate)
                    timestamp = axivity_read_timestamp_raw(sampleTimeData)

                    if packetLength != 508 or timestamp == None or sampleRate == 0:
                        continue

                    if ((numAxesBPS >> 4) & 15) != 3:
                        print('[ERROR: num-axes not expected]')

                    if (numAxesBPS & 15) == 2:
                        bps = 6
                    elif (numAxesBPS & 15) == 0:
                        bps = 4

                    freq = 3200 / (1 << (15 - sampleRate & 15))
                    if freq <= 0:
                        freq = 1
                    offsetStart = float(-timestampOffset) / float(freq)

                    time0 = timestamp + timedelta(milliseconds=offsetStart)
                    axivity_indices[num_pages] = num_samples
                    axivity_timestamps[num_pages] = time0
                    axivity_light[num_pages] = light
                    axivity_temperature[num_pages] = temperature
                    num_pages += 1

                    for sample in range(sampleCount):

                        x,y,z = 0,0,0

                        if bps == 6:

                            x,y,z = unpack('hhh', fh.read(6))
                            x,y,z = x/256.0,y/256.0,z/256.0

                        elif bps == 4:
                            temp = unpack('I', fh.read(4))[0]
                            temp2 = (6 - byte(temp >> 30))
                            x = short(short((ushort(65472) & ushort(temp << 6))) >> temp2) / 256.0
                            y = short(short((ushort(65472) & ushort(temp >> 4))) >> temp2) / 256.0
                            z = short(short((ushort(65472) & ushort(temp >> 14))) >> temp2) / 256.0

                            # Optimisation:
                            # Cache value of ushort(65472) ?


                        #t = sample*sampleOffset + time0

                        axivity_x[num_samples] = x
                        axivity_y[num_samples] = y
                        axivity_z[num_samples] = z

                        num_samples += 1



                    checksum = unpack('H', axivity_read(fh,2))[0]

                else:
                    #pass
                    print("Unrecognised header", header)

                header = axivity_read(fh,2)

                n=n+1
        except IOError:
            pass

        print(sum(sample_rates)/len(sample_rates))
        axivity_x.resize(num_samples)
        axivity_y.resize(num_samples)
        axivity_z.resize(num_samples)
        axivity_timestamps.resize(num_pages)
        axivity_indices.resize(num_pages)
        axivity_temperature.resize(num_pages)
        axivity_light.resize(num_pages)

        channel_x.set_contents(axivity_x, axivity_timestamps)
        channel_y.set_contents(axivity_y, axivity_timestamps)
        channel_z.set_contents(axivity_z, axivity_timestamps)

        channel_light.set_contents(axivity_light, axivity_timestamps)
        channel_temperature.set_contents(axivity_temperature, axivity_timestamps)

        for c in [channel_x, channel_y, channel_z]:
            c.indices = axivity_indices
            c.sparsely_timestamped = True


        channels = [channel_x, channel_y, channel_z, channel_light, channel_temperature]
        header = file_header

    elif (source_type == "GeneActiv"):

        f = open(source, "rb")
        data = io.BytesIO(f.read())
        #print("File read in")

        first_lines = [data.readline().strip().decode() for i in range(59)]
        #print(first_lines)
        header_info = parse_header(first_lines, "GeneActiv", "")
        print(header_info)

        n = header_info["number_pages"]
        obs_num = 0
        num = 300
        x_values = np.empty(num*n)
        y_values = np.empty(num*n)
        z_values = np.empty(num*n)
        #time_values = np.array([header_info["start_datetime_python"]])
        #time_values = np.resize(time_values, num*n)

        ga_timestamps = np.empty(header_info["number_pages"]+1, dtype=type(header_info["start_datetime_python"]))
        ga_indices = np.empty(header_info["number_pages"]+1)

        # For each page
        for i in range(n):

            #xs,ys,zs,times = read_block(data, header_info)
            lines = [data.readline().strip().decode() for i in range(9)]
            page_time = datetime.strptime(lines[3][10:29], "%Y-%m-%d %H:%M:%S") + timedelta(microseconds=int(lines[3][30:])*1000)

            ga_timestamps[i] = page_time
            ga_indices[i] = obs_num

            # For each 12 byte measurement in page (300 of them)
            for j in range(num):

                #time = page_time + (j * header_info["epoch"])

                block = data.read(12)

                x = int(block[0:3], 16)
                y = int(block[3:6], 16)
                z = int(block[6:9], 16)

                x, y, z = twos_comp(x, 12), twos_comp(y, 12), twos_comp(z, 12)
                #print(x,y,z)
                x_values[obs_num] = x
                y_values[obs_num] = y
                z_values[obs_num] = z
                #time_values[obs_num] = time
                obs_num += 1


            excess = data.read(2)

        # Timestamp the final observation
        ga_timestamps[-1] = page_time + (num*(timedelta(seconds=1)/header_info["frequency"]))
        ga_indices[-1] = obs_num

        #print("A")
        x_values = np.array([(x * 100.0 - header_info["x_offset"]) / header_info["x_gain"] for x in x_values])
        y_values = np.array([(y * 100.0 - header_info["y_offset"]) / header_info["y_gain"] for y in y_values])
        z_values = np.array([(z * 100.0 - header_info["z_offset"]) / header_info["z_gain"] for z in z_values])

        x_channel = Channel.Channel("X")
        y_channel = Channel.Channel("Y")
        z_channel = Channel.Channel("Z")

        x_channel.set_contents(x_values, ga_timestamps)
        y_channel.set_contents(y_values, ga_timestamps)
        z_channel.set_contents(z_values, ga_timestamps)

        #print("B")
        for c in [x_channel, y_channel, z_channel]:
            c.indices = ga_indices
            c.sparsely_timestamped = True
            c.frequency = header_info["frequency"]
        #print("C")
        channels = [x_channel, y_channel, z_channel]
        header = header_info

    ts.add_channels(channels)
    return ts, header
