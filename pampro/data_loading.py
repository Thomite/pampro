# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

# python-uos-activpal open source python module distributed under GNU General Public License v2.0
# Source: https://pypi.org/project/uos-activpal/

import uos_activpal.io.raw as ap_read

import io
from itertools import chain
from .diagnostics import *

# CWA Metadata Reader by Dan Jackson, 2017.
# Source: https://github.com/digitalinteraction/openmovement/tree/master/Software/AX3/cwa-convert/python
# Adapted for use with pampro by Ella Hutchinson - 22/03/2018

# Axivity import code adapted from source provided by Open Movement: https://code.google.com/p/openmovement/. Their license terms are reproduced here in full, and apply only to the Axivity related code:
# Copyright (c) 2009-2014, Newcastle University, UK. All rights reserved.
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def twos_comp(val, bits):
    if ((val & (1 << (bits - 1))) != 0):
        val = val - (1 << bits)
    return val


def byte(value):
    return (value + 2 ** 7) % 2 ** 8 - 2 ** 7


def ushort(value):
    return value % 2 ** 16


def short(value):
    return (value + 2 ** 15) % 2 ** 16 - 2 ** 15


# conversion of 'light' value from raw axivity file to lux
def convert_axivity_light(value):
    power = int(value)/341
    return 10**power


# conversion of 'temperature' value from raw Axivity file to degrees C NOTE: this conversion may be incorrect - awaiting communication from OpenMovement Oct19
def convert_axivity_temp(value):
    factor = int(value)-171
    return factor/3.413


def axivity_read_timestamp(stamp):
    stamp = unpack('I', stamp)[0]
    year = ((stamp >> 26) & 0x3f) + 2000
    month = (stamp >> 22) & 0x0f
    day = (stamp >> 17) & 0x1f
    hours = (stamp >> 12) & 0x1f
    mins = (stamp >> 6) & 0x3f
    secs = (stamp >> 0) & 0x3f
    try:
        t = datetime(year, month, day, hours, mins, secs)
    except ValueError:
        t = None
    return t


def axivity_read_timestamp_raw(stamp):
    year = ((stamp >> 26) & 0x3f) + 2000
    month = (stamp >> 22) & 0x0f
    day = (stamp >> 17) & 0x1f
    hours = (stamp >> 12) & 0x1f
    mins = (stamp >> 6) & 0x3f
    secs = (stamp >> 0) & 0x3f
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

# Function to convert a string from camelCase (even with spaces between words) to snake_case
def convert_case(string, first_cap_re, all_cap_re):
    """
    # regular expressions compiled for use in the convert_case function
    first_cap_re = re.compile('(.)([A-Z][a-z]+)')
    all_cap_re = re.compile('([a-z0-9])([A-Z])')
    """
    stripped = ''.join(string.split())
    s1 = first_cap_re.sub(r'\1_\2', stripped)
    return all_cap_re.sub(r'\1_\2', s1).lower()


# Function that makes use of convert_string() to convert the keys of a dictionary to snake_case, returning a new dictionary.
# special cases and the addition of prefixes are allowed for
def convert_dict_keys(old_dict, special_cases={}, prefix=None):

    """ Takes a dictionary, creates and returns a new dictionary with the keys converted

    old_dict: the input dictionary to be converted
    special_cases: a dictionary of special cases of key names in the form {old_key_name: required_key_name}
    prefix: a prefix to be applied to all key names, e.g "file"

    """

    # regular expressions compiled to pass to the the convert_case function
    first_cap_re = re.compile('(.)([A-Z][a-z]+)')
    all_cap_re = re.compile('([a-z0-9])([A-Z])')

    # create a new dictionary
    new_dict = {}

    for key, value in old_dict.items():
        # check if key is a 'special case', if so return the value of special_cases[key] as new key
        if key in special_cases.keys():
            new_key = special_cases[key]
        else:
        # create the new key by using convert_case()
            new_key = convert_case(key, first_cap_re, all_cap_re)

        # if a prefeix is given, add to the front of the new key
        if prefix is not None:
            new_key = prefix + "_" + new_key

        # create a dictionary item in new_dict of the new key with the original value
        new_dict[new_key] = value

    return new_dict


def parse_header(header, type, datetime_format):
    header_info = OrderedDict()

    if type == "Actiheart":

        delimiter = "\t"
        if "," in header[0]:
            delimiter = ","

        safe = {"\t": "tab", ",": "comma"}
        header_info["delimiter"] = safe[delimiter]

        for i, row in enumerate(header):
            try:
                values = row.split(delimiter)

                if ":" not in values[0]:
                    header_info[values[0]] = values[1]
                    # print("["+str(values[0])+"]")
            except:
                pass

        time1 = datetime.strptime(header[-2].split(delimiter)[0], "%H:%M:%S")
        time2 = datetime.strptime(header[-1].split(delimiter)[0], "%H:%M:%S")
        header_info["epoch_length"] = time2 - time1

        header_info["start_date"] = datetime.strptime(header_info["Started"], "%d-%b-%Y  %H:%M")

        if "Start trimmed to" in header_info:
            header_info["Start trimmed to"] = datetime.strptime(header_info["Start trimmed to"], "%Y-%m-%d %H:%M")

        for i, row in enumerate(header):

            if row.split(delimiter)[0] == "Time":
                header_info["data_start"] = i + 1
                break

    elif type == "Actigraph":

        # Use lines 2 and 3 to get start date and time
        test = header[2].split(" ")
        timeval = datetime.strptime(test[-1], "%H:%M:%S")
        start_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
        header_info["start_time"] = str(start_time)
        test = header[3].split(" ")
        start_date = test[-1].replace("-", "/")

        # Use lines 5 and 6 to get download date and time
        test = header[5].split(" ")
        timeval = datetime.strptime(test[-1], "%H:%M:%S")
        download_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
        header_info["download_time"] = str(download_time)
        test = header[6].split(" ")
        download_date = test[-1].replace("-", "/")

        test = header[1].split(":")
        header_info["serial_number"] = test[1].strip()

        header_info["version_string"] = header[0].replace("-", "")

        # Try to interpret the two dates using the user-provided format
        try:
            start_date = datetime.strptime(start_date, datetime_format)
            download_date = datetime.strptime(download_date, datetime_format)
        except:
            raise Exception("The given datetime format ({}) is incompatible with the start or download date.".format(
                datetime_format))

        header_info["start_date"] = str(start_date)
        header_info["download_date"] = str(download_date)

        test = header[4].split(" ")
        delta = datetime.strptime(test[-1], "%H:%M:%S")
        epoch_length = timedelta(hours=delta.hour, minutes=delta.minute, seconds=delta.second)
        header_info["epoch_length_seconds"] = int(epoch_length.total_seconds())

        start_datetime = start_date + start_time
        header_info["start_datetime"] = start_datetime

        header_info["mode"] = 0

        try:
            splitup = header[8].split(" ")
            if "Mode" in splitup:
                index = splitup.index("Mode")
                mode = splitup[index + 2]

                header_info["mode"] = int(mode)
        except:
            pass


    elif type == "GT3X+_CSV":

        # find the device ID
        header_info["device"] = header[1].split(": ")[1]

        # Find start time
        test = header[2].split(" ")
        timeval = datetime.strptime(test[-1], "%H:%M:%S")
        start_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
        header_info["start_time"] = start_time

        # Find download time
        test = header[5].split(" ")
        timeval = datetime.strptime(test[-1], "%H:%M:%S")
        download_time = timedelta(hours=timeval.hour, minutes=timeval.minute, seconds=timeval.second)
        header_info["download_time"] = str(download_time)

        test = header[0].split(" ")
        if "Hz" in test:
            index = test.index("Hz")
            hz = int(test[index - 1])
            epoch_length = timedelta(seconds=1) / hz
            header_info["epoch_length"] = epoch_length
            header_info["frequency"] = hz

        # Find the date format
        if "format" in test:
            index = test.index("format")
            format = test[index + 1]
            format = format.replace("dd", "%d")
            format = format.replace("MM", "%m")
            format = format.replace("yyyy", "%Y")

            start_date = datetime.strptime(header[3].split(" ")[2], format)
            header_info["start_date"] = start_date

            download_date = datetime.strptime(header[6].split(" ")[2], format)
            header_info["download_date"] = download_date.strftime("%Y-%m-%d %H:%M:%S")

        start_datetime = start_date + start_time
        header_info["start_datetime"] = start_datetime
        download_datetime = download_date + download_time
        header_info["download_datetime"] = download_datetime.strftime("%Y-%m-%d %H:%M:%S")

        # find the monitor type, software and firmware
        line_split = header[0].split("By ")[1]
        header_info["monitor_type"] = (line_split.split(" ")[0]) + "_" + (line_split.split(" ")[1])
        #print(header_info["monitor_type"])

        if "ActiLife" in test:
            line_split1 = line_split.split("ActiLife ")[1]
            header_info["actilife"] = line_split1.split(" ")[0]
            #print(header_info["actilife"])

        if "Firmware" in test:
            line_split2 = line_split.split("Firmware ")[1]
            header_info["firmware"] = line_split2.split(" ")[0]
            #print(header_info["Firmware"])

        # find the mode
        header_info["mode"] = (header[8].split("Mode")[1]).strip(" = ")
        #print(header_info["mode"])


    elif type == "GeneActiv":

        header_info["start_datetime"] = header[21][11:]
        if header_info["start_datetime"] == "0000-00-00 00:00:00:000":
            header_info["start_datetime_python"] = datetime.strptime("0001-01-01", "%Y-%m-%d")
        else:
            header_info["start_datetime_python"] = datetime.strptime(header_info["start_datetime"],
                                                                     "%Y-%m-%d %H:%M:%S:%f")

        # Turns out the frequency might be written European style (, instead of .)
        splitted = header[19].split(":")
        sans_hz = splitted[1].replace(" Hz", "")
        comma_safe = sans_hz.replace(",", ".")
        header_info["frequency"] = float(comma_safe)
        header_info["epoch"] = str(timedelta(seconds=1) / int(header_info["frequency"]))

        # currently the GeneActive header has 59 lines, create a dictionary of all header items, with value -1 if no data
        for i in range(0, 58):
            if ":" in header[i]:
                a, b = header[i].split(":", maxsplit=1)
                var = a.lower()
                var2 = var.replace(" ", "_")
                if b == "":
                    header_info[var2] = "-1"
                else:
                    if var2.startswith(("x", "y", "z", "volts", "lux")):
                        header_info[var2] = float(b)
                    elif var2 == "number_of_pages":
                        header_info["number_pages"] = int(b)
                    elif var2 == "device_unique_serial_code":
                        header_info["device"] = str(b)
                    else:
                        header_info[var2] = b

        return header_info


    elif type == "XLO":

        # Start timestamp
        blah = header[7].split()
        sans_meridian = blah[3].replace("AM", "")
        sans_meridian = sans_meridian.replace("PM", "")
        dt = datetime.strptime(blah[1], "%d/%m/%Y")
        header_info["start_datetime_python"] = dt

        # Height and weight
        l3 = header[3].split(":")
        height = float(l3[1].strip().replace(" cm", "").split()[0])
        weight = float(l3[2].strip().replace(" kg", ""))
        header_info["height"] = height
        header_info["weight"] = weight

        return header_info

    return header_info


def convert_actigraph_timestamp(t):
    return datetime(*map(int, [t[6:10], t[3:5], t[0:2], t[11:13], t[14:16], t[17:19], int(t[20:]) * 1000]))


def load(source, source_type="infer", datetime_format="%d/%m/%Y %H:%M:%S:%f", datetime_column=0, ignore_columns=False,
         unique_names=False, hdf5_mode="r", hdf5_group="Raw", hdf5_groups=None, anomalies_file=None, compress=True):
    load_start = datetime.now()
    
    header = OrderedDict()
    channels = []
    ts = Time_Series("")

    # when the source_type is left blank, we can assume using the filename extension
    # throw an error if unsure
    extension_map = {
        "dat": "Actigraph", "DAT": "Actigraph", "Dat": "Actigraph",
        "csv": "CSV",
        "bin": "GeneActiv",
        "hdf5": "HDF5", "h5": "HDF5",
        "datx": "activPAL",
        "cwa": "Axivity", "CWA": "Axivity"
    }

    if source_type == "infer":

        extension = source.split(".")[-1]
        if extension in extension_map:

            source_type = extension_map[extension]
        else:

            raise Exception(
                "Cannot assume file type from extension ({}), specify source_type when trying to load this file.".format(
                    extension))

    if source_type == "Actiheart":

        first_lines = []
        f = open(source, 'r')
        for i in range(0, 30):
            s = f.readline().strip()
            first_lines.append(s)
        f.close()

        header_info = parse_header(first_lines, "Actiheart", "%d-%b-%Y  %H:%M")

        start_date = header_info["start_date"]
        epoch_length = header_info["epoch_length"]
        data_start = header_info["data_start"]

        mapping = {"comma": ",", "tab": "\t"}

        activity, ecg = np.loadtxt(source, delimiter=mapping[header_info["delimiter"]], unpack=True,
                                   skiprows=data_start, usecols=[1, 2])

        timestamp_list = [start_date + i * epoch_length for i in range(len(activity))]
        timestamps = np.array(timestamp_list)

        if "Start trimmed to" in header_info:
            indices1 = (timestamps > header_info["Start trimmed to"])
            activity = activity[indices1]
            ecg = ecg[indices1]
            timestamps = timestamps[indices1]

        ecg[(ecg <= 0)] = -1

        actiheart_activity = Channel("Chest")
        actiheart_activity.set_contents(activity, timestamps)

        actiheart_ecg = Channel("HR")
        actiheart_ecg.set_contents(ecg, timestamps)

        actiheart_ecg.missing_value = -1

        actiheart_ecg.draw_properties = {"c": [0.8, 0.05, 0.05]}
        actiheart_activity.draw_properites = {"c": [0.05, 0.8, 0.8]}

        header = header_info
        channels = [actiheart_activity, actiheart_ecg]

    elif source_type == "activPAL_CSV":

        file_handle = open(source, "r")
        first_lines = []
        for i in range(0,5):
            s = file_handle.readline().strip()
            first_lines.append(s)
                
        string = first_lines[2].replace('"', '')
        header["device"] = string.split(": ")[1]       
        
        ap_timestamp, ap_x, ap_y, ap_z = np.loadtxt(source, delimiter=',', unpack=True, skiprows=5, 
                                                    dtype={'names': ('ap_timestamp', 'ap_x', 'ap_y', 'ap_z'),
                                                           'formats': ('S16', 'f8', 'f8', 'f8')})
        
        dt = datetime.strptime("30-Dec-1899", "%d-%b-%Y")

        ap_timestamps = []
        for val in ap_timestamp:
        
            val = val.decode()

            test = val.split(".")

            while len(test[1]) < 10:
                test[1] = test[1] + "0"

            finaltest = dt + timedelta(days=int(test[0]), microseconds=int(test[1]) * 8.64)
            ap_timestamps.append(finaltest)

        ap_timestamps = np.array(ap_timestamps)
        # print("B")
        x = Channel("X")
        y = Channel("Y")
        z = Channel("Z")
        integrity_channel = Channel("Integrity")

        ap_x = (ap_x - 128.0) / 64.0
        ap_y = (ap_y - 128.0) / 64.0
        ap_z = (ap_z - 128.0) / 64.0

        integrity = np.zeros(len(ap_timestamps))
        x.set_contents(np.array(ap_x, dtype=np.float64), ap_timestamps)
        y.set_contents(np.array(ap_y, dtype=np.float64), ap_timestamps)
        z.set_contents(np.array(ap_z, dtype=np.float64), ap_timestamps)
        # print("C")
        integrity_channel.set_contents(integrity, ap_timestamps)
        integrity_channel.binary_data = True
        
        channels = [x, y, z, integrity_channel]

    elif source_type == "activPAL":

        metadata = ap_read.extract_metadata_from_file(source)

        header.update(metadata._asdict())
        
        f = open(source, "rb")
        data = f.read()
        filesize = len(data)
        data = io.BytesIO(data)

        A = unpack('1024s', data.read(1024))[0]

        start_time = str((A[256])).rjust(2, "0") + ":" + str((A[257])).rjust(2, "0") + ":" + str((A[258])).rjust(2, "0")
        start_date = str((A[259])).rjust(2, "0") + "/" + str((A[260])).rjust(2, "0") + "/" + str(2000 + (A[261]))

        end_time = str((A[262])).rjust(2, "0") + ":" + str((A[263])).rjust(2, "0") + ":" + str((A[264])).rjust(2, "0")
        end_date = str((A[265])).rjust(2, "0") + "/" + str((A[266])).rjust(2, "0") + "/" + str(2000 + (A[267]))

        start = start_date + " " + start_time
        end = end_date + " " + end_time

        # print(start_date, start_time)
        # print(end_date, end_time)

        # Given in Hz
        sampling_frequency = A[35]

        # 0 = 2g, 1 = 4g (therefore double the raw values), 2 = 8g (therefore quadruple the raw values)
        dynamic_range_code = A[38]
        dynamic_multiplier = 2 ** dynamic_range_code

        start_python = datetime.strptime(start, "%d/%m/%Y %H:%M:%S")
        end_python = datetime.strptime(end, "%d/%m/%Y %H:%M:%S")
        duration = end_python - start_python

        num_records = int(duration.total_seconds() / (timedelta(seconds=1) / sampling_frequency).total_seconds())

        # print("expected records:", num_records)
        # print("sampling frequency", (A[35]))
        # print("header end", A[1012:1023])

        n = 0
        data_cache = False
        # extract timestamp

        x = np.zeros(num_records)
        y = np.zeros(num_records)
        z = np.zeros(num_records)
        integrity = np.zeros(num_records)

        x.fill(-1.1212121212121)
        y.fill(-1.1212121212121)
        z.fill(-1.1212121212121)

        last_a, last_b, last_c = 0, 0, 0

        while n < num_records and data.tell() < filesize:
        
            try:
                data_cache = data.read(3)
                a, b, c = unpack('ccc', data_cache)
                a, b, c = ord(a), ord(b), ord(c)
    
                # print(a,b,c)
    
                # activPAL writes TAIL but these values could legitimately turn up
                if a == 116 and b == 97 and c == 105:
                    # if a,b,c spell TAI
    
                    d = ord(unpack('c', data.read(1))[0])
                    # and if d == T, so TAIL just came up
                    if d == 108:
                        # print ("Found footer!")
                        # print (data.tell())
                        remainder = data.read()
                    else:
                        # Otherwise TAI came up coincidently
                        # Meaning a,b,c was a legitimate record, and we just read in a from another record
                        # So read in the next 2 bytes and record 2 records: a,b,c and d,e,f
                        e, f = unpack('cc', data.read(2))
                        e, f = ord(e), ord(f)
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
                        x[n:(n + c + 1)] = [last_a for val in range(c + 1)]
                        y[n:(n + c + 1)] = [last_b for val in range(c + 1)]
                        z[n:(n + c + 1)] = [last_c for val in range(c + 1)]
                        n += c + 1
                    else:
                        x[n] = a
                        y[n] = b
                        z[n] = c
                        n += 1
    
                last_a, last_b, last_c = a, b, c
    
    
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

        integrity.resize(n)
        x.resize(n)
        y.resize(n)
        z.resize(n)

        x = (x - 128.0) / 64.0
        y = (y - 128.0) / 64.0
        z = (z - 128.0) / 64.0

        if dynamic_multiplier > 1:
            x *= dynamic_multiplier
            y *= dynamic_multiplier
            z *= dynamic_multiplier

        delta = timedelta(seconds=1) / sampling_frequency
        timestamps = np.array([start_python + delta * i for i in range(n)])

        x_channel = Channel("X")
        y_channel = Channel("Y")
        z_channel = Channel("Z")
        integrity_channel = Channel("Integrity")

        x_channel.set_contents(x, timestamps)
        y_channel.set_contents(y, timestamps)
        z_channel.set_contents(z, timestamps)
        integrity_channel.set_contents(integrity, timestamps)
        integrity_channel.binary_data = True
        
        for c in [x_channel, y_channel, z_channel, integrity_channel]:
            c.sparsely_timestamped = False
            c.frequency = sampling_frequency

        header["frequency"] = sampling_frequency
        header["dynamic_range_code"] = dynamic_range_code
        
        channels = [x_channel, y_channel, z_channel, integrity_channel]

    elif source_type == "GeneActiv_CSV":

        ga_timestamp, ga_x, ga_y, ga_z, ga_lux, ga_event, ga_temperature = np.genfromtxt(source, delimiter=',',
                                                                                         unpack=True, skip_header=80,
                                                                                         dtype=str)

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

        channels = [x, y, z, lux, event, temperature]

    elif source_type == "Actigraph":

        first_lines = []
        f = open(source, 'r')
        for i in range(0, 10):
            s = f.readline().strip()
            first_lines.append(s)

        header_info = parse_header(first_lines, "Actigraph", datetime_format)

        time = header_info["start_datetime"]
        epoch_length = timedelta(seconds=header_info["epoch_length_seconds"])
        mode = header_info["mode"]

        # If the mode is not one of those currently supported, raise an error
        if mode not in [0, 1, 3, 4, 5]:
            raise Exception("Mode {} is not currently supported.".format(mode))

        count_list = []
        timestamp_list = []

        line = f.readline().strip()
        while (len(line) > 0):
            counts = line.split()
            count_list = count_list + counts
            line = f.readline().strip()
        f.close()

        # Cast the strings to integers
        count_list = [int(c) for c in count_list]

        # If the mode implies the data is count, steps
        if mode == 1 or mode == 3 or mode == 4:

            count_list = [a for a, b in zip(*[iter(count_list)] * 2)]

        # If the mode implies the data is count X, count Y, count Z
        elif mode == 5:

            count_list = [a for a, b, c in zip(*[iter(count_list)] * 3)]

        timestamp_list = [time + t * epoch_length for t in range(len(count_list))]

        timestamps = np.array(timestamp_list)
        counts = np.abs(np.array(count_list))

        chan = Channel("AG_Counts")
        chan.set_contents(counts, timestamps)

        channels = [chan]
        header = header_info

    elif source_type.startswith("GT3X+_CSV"):

        first_lines = []
        
        # if source file is zipped, unzip to read:
        if source_type.endswith("_ZIP"):
            archive = zipfile.ZipFile(source)
            csv_not_zip = source.split("/")[-1].replace(".zip", ".csv")
            file_handle = archive.open(csv_not_zip)
            for i in range(0,10):
                s = file_handle.readline().strip().decode("utf-8")
                first_lines.append(s)

        else:
            file_handle = open(source, "r")
            for i in range(0,10):
                s = file_handle.readline().strip()
                first_lines.append(s)

        header_info = parse_header(first_lines, "GT3X+_CSV", "")

        time = header_info["start_datetime"]
        epoch_length = header_info["epoch_length"]

        if source_type.endswith("_ZIP"):
            file_handle = archive.open(csv_not_zip)
        else:
            file_handle = source
        
        timestamps = np.genfromtxt(file_handle, delimiter=',', converters={0:convert_actigraph_timestamp}, skip_header=11, usecols=(0))

        if source_type.endswith("_ZIP"):
            file_handle = archive.open(csv_not_zip)
        else:
            file_handle = source
        
        x,y,z = np.genfromtxt(file_handle, delimiter=',', skip_header=11, usecols=(1,2,3), unpack=True)
        
        integrity = np.zeros(len(x))

        file_handle.close()

        x_chan = Channel("X")
        y_chan = Channel("Y")
        z_chan = Channel("Z")
        integrity_chan = Channel("Integrity")

        x_chan.set_contents(x, timestamps)
        y_chan.set_contents(y, timestamps)
        z_chan.set_contents(z, timestamps)
        integrity_chan.set_contents(integrity, timestamps)
        integrity_chan.binary_data = True

        for c in [x_chan, y_chan, z_chan, integrity_chan]:
            c.frequency = header_info["frequency"]


        channels = [x_chan, y_chan, z_chan, integrity_chan]
        header = header_info

    elif source_type == "CSV":

        f = open(source, 'r')
        s = f.readline().strip()
        f.close()

        test = s.split(",")

        source_split = source.split("/")

        data = np.loadtxt(source, delimiter=',', skiprows=1, dtype='S').astype("U")

        timestamps = []
        for date_row in data[:, datetime_column]:
            # print(date_row)
            # print(str(date_row))
            # print(type(date_row))
            timestamps.append(datetime.strptime(date_row, datetime_format))
        timestamps = np.array(timestamps)

        data_columns = list(range(0, len(test)))
        del data_columns[datetime_column]

        if ignore_columns != False:
            for ic in ignore_columns:
                del data_columns[ic]

        # print data_columns

        channels = []
        for col in data_columns:
            # print col
            if unique_names:
                name = source_split[-1] + " - " + test[col]
            else:
                name = test[col]
            c = Channel(name)
            c.set_contents(np.array(data[:, col], dtype=np.float64), timestamps)
            channels.append(c)

    elif source_type.startswith("Axivity"):
        # if source file is zipped, unzip to read
        if source_type.endswith("_ZIP"):
            archive = zipfile.ZipFile(source, "r")
            cwa_not_zip = source.split("/")[-1].replace(".zip", ".cwa")
            handle = archive.open(cwa_not_zip)

        else:
            handle = open(source, "rb")

        channel_x = Channel("X")
        channel_y = Channel("Y")
        channel_z = Channel("Z")
        channel_light = Channel("Light")
        channel_temperature = Channel("Temperature")
        channel_battery = Channel("Battery")
        channel_integrity = Channel("Integrity")

        raw_bytes = handle.read()
        fh = io.BytesIO(raw_bytes)

        n = 0
        num_samples = 0
        num_pages = 0

        start = datetime(2014, 1, 1)

        # Rough number of pages expected = length of file / size of block (512 bytes)
        # Rough number of samples expected = pages * 120
        # Add 1% buffer just to be cautious - it's trimmed later
        estimated_num_pages = int(len(raw_bytes) / 512 * 1.01)
        estimated_num_samples = int(estimated_num_pages * 120)
        # print("Estimated number of samples:", estimated_num_samples)

        axivity_x = np.empty(estimated_num_samples)
        axivity_y = np.empty(estimated_num_samples)
        axivity_z = np.empty(estimated_num_samples)
        axivity_timestamps = np.empty(estimated_num_pages, dtype=type(start))
        axivity_indices = np.empty(estimated_num_pages)
        axivity_integrity = np.zeros(estimated_num_samples)

        # Check if data is to be separated into sample-level and page-level data:
        if compress is True:
            axivity_light = np.empty(estimated_num_pages)
            axivity_temperature = np.empty(estimated_num_pages)
            axivity_battery = np.empty(estimated_num_pages)

        # if not, then all data is expressed at sample level (i.e. uncompressed)
        else:
            axivity_light = np.empty(estimated_num_samples)
            axivity_temperature = np.empty(estimated_num_samples)
            axivity_battery = np.empty(estimated_num_samples)

        file_header = OrderedDict()
        file_header = parse_axivity_header(source)

        # extract the following from the header to use as fail-safe measures later on:
        # preserve the 'file_session_id'
        file_session_id = file_header['session_id']
        # preserve the 'first_sample_count'
        first_sample_count = file_header['first_sample_count']

        lastSequenceId = None
        lastTimestampOffset = None
        lastTimestamp = None

        try:
            header = axivity_read(fh, 2)

            while len(header) == 2:

                if header == b'MD':
                    #print('MD')
                    pass
                elif header == b'UB':
                    #print('UB')
                    blockSize = unpack('H', axivity_read(fh, 2))[0]
                elif header == b'SI':
                    #print('SI')
                    pass
                elif header == b'AX':

                    packet = axivity_read(fh, 510)

                    packetLength, deviceId, sessionId, sequenceId, sampleTimeData, light, temperature, events, battery, sampleRate, numAxesBPS, timestampOffset, sampleCount = unpack('HHIIIHHcBBBhH', packet[0:28])


                    # sector is equal to packet plus sector header
                    sector = b'AX' + packet

                    #calculate the checksum of the data sector
                    sector_checksum = checksum(sector)

                    if packetLength != 508 or sampleRate == 0:
                        continue

                    if ((numAxesBPS >> 4) & 15) != 3:
                        print('[ERROR: Axes!=3 not supported yet -- this will not work properly]')

                    if (numAxesBPS & 15) == 2:
                        bps = 6
                    elif (numAxesBPS & 15) == 0:
                        bps = 4

                    freq = 3200 / (1 << (15 - sampleRate & 15))
                    if freq <= 0:
                        freq = 1

                    timestamp_original = axivity_read_timestamp_raw(sampleTimeData)

                    if timestamp_original is None:
                        continue

                    # if top-bit set, we have a fractional date
                    if deviceId & 0x8000:
                        # Need to undo backwards-compatible shim by calculating how many whole samples the fractional part of timestamp accounts for.
                        timeFractional = (deviceId & 0x7fff) * 2  # use original deviceId field bottom 15-bits as 16-bit fractional time
                        timestampOffset += (timeFractional * int(
                            freq)) // 65536  # undo the backwards-compatible shift (as we have a true fractional)
                        timeFractional = float(timeFractional) / 65536

                        # Add fractional time to timestamp
                        timestamp = timestamp_original + timedelta(seconds=timeFractional)

                    else:

                        timestamp = timestamp_original

                    # --- Time interpolation ---
                    # Reset interpolator if there's a sequence break or there was no previous timestamp
                    if lastSequenceId == None or (lastSequenceId + 1) & 0xffff != sequenceId or lastTimestampOffset == None or lastTimestamp == None:
                        # Bootstrapping condition is a sample one second ago (assuming the ideal frequency)
                        lastTimestampOffset = timestampOffset - freq
                        lastTimestamp = timestamp - timedelta(seconds=1)
                        lastSequenceId = sequenceId - 1

                    localFreq = timedelta(seconds=(timestampOffset - lastTimestampOffset)) / (timestamp - lastTimestamp)
                    final_timestamp = timestamp + -timedelta(seconds=timestampOffset) / localFreq

                    # Update for next loop
                    lastSequenceId = sequenceId
                    lastTimestampOffset = timestampOffset - sampleCount
                    lastTimestamp = timestamp

                    axivity_indices[num_pages] = num_samples
                    axivity_timestamps[num_pages] = final_timestamp

                    '''# check for data integrity:
                    if sector_checksum == 0 and sessionId == file_session_id:
                        # if check passed then set 'validity' to '0' and extract the data
                        axivity_validity[num_pages] = 0
                    # if integrity check fails on checksum, set 'validity' to '1'
                    elif sector_checksum != 0:
                        axivity_validity[num_pages] = 1
                    # if integrity check fails on session id matching, set 'validity value' to '2'
                    elif sessionId != file_session_id:
                        axivity_validity[num_pages] = 2'''

                    # convert the light and temperature values to lux and degrees C values
                    light_converted = convert_axivity_light(light)
                    temp_converted = convert_axivity_temp(temperature)

                    if compress is True:
                        # save one value of light, battery and temperature per page
                        axivity_light[num_pages] = light_converted
                        axivity_temperature[num_pages] = temp_converted
                        axivity_battery[num_pages] = battery

                    for sample in range(sampleCount):
                        # index for the bytes per sample, depending on bytesPerSample(bps)
                        i = 28 + (sample * bps)

                        if bps == 4:
                            temp = unpack('I', packet[i:(i+4)])[0]
                            temp2 = (6 - (temp >> 30))
                            x = short(short((ushort(65472) & ushort(temp << 6))) >> temp2) / 256.0
                            y = short(short((ushort(65472) & ushort(temp >> 4))) >> temp2) / 256.0
                            z = short(short((ushort(65472) & ushort(temp >> 14))) >> temp2) / 256.0

                        elif bps == 6:
                            x, y, z = unpack('hhh', packet[i:(i+6)])
                            x, y, z = x / 256.0, y / 256.0, z / 256.0

                        axivity_x[num_samples] = x
                        axivity_y[num_samples] = y
                        axivity_z[num_samples] = z

                        if compress is not True:
                            # save one value of light, battery and temperature
                            axivity_light[num_samples] = light_converted
                            axivity_temperature[num_samples] = temp_converted
                            axivity_battery[num_samples] = battery

                        num_samples += 1

                    num_pages += 1

                else:
                    pass
                    # print("Unrecognised header", header)

                header = axivity_read(fh, 2)

                n = n + 1

        except IOError:
            # End of file
            pass

        # We created oversized arrays at the start, to make sure we could fit all the data in
        # Now we know how much data was there, we can shrink the arrays to size
        axivity_x.resize(num_samples)
        axivity_y.resize(num_samples)
        axivity_z.resize(num_samples)
        axivity_timestamps.resize(num_pages)
        axivity_indices.resize(num_pages)
        axivity_integrity.resize(num_samples)

        axivity_indices = axivity_indices.astype(int)

        # Map the page-level timestamps to the acceleration data "sparsely"
        channel_x.set_contents(axivity_x, axivity_timestamps, timestamp_policy="sparse")
        channel_y.set_contents(axivity_y, axivity_timestamps, timestamp_policy="sparse")
        channel_z.set_contents(axivity_z, axivity_timestamps, timestamp_policy="sparse")
        channel_integrity.set_contents(axivity_integrity, axivity_timestamps, timestamp_policy="sparse")
        channel_integrity.binary_data = True

        if compress is True:
            axivity_temperature.resize(num_pages)
            axivity_light.resize(num_pages)
            axivity_battery.resize(num_pages)

            # Map the page-level timestamps to the temperature, battery and light data
            channel_battery.set_contents(axivity_battery, axivity_timestamps, timestamp_policy="normal")
            channel_temperature.set_contents(axivity_temperature, axivity_timestamps, timestamp_policy="normal")
            channel_light.set_contents(axivity_light, axivity_timestamps, timestamp_policy="normal")

        else:
            axivity_temperature.resize(num_samples)
            axivity_light.resize(num_samples)
            axivity_battery.resize(num_samples)

            # Map the page-level timestamps to the temperature, battery and light data
            channel_battery.set_contents(axivity_battery, axivity_timestamps, timestamp_policy="sparse")
            channel_temperature.set_contents(axivity_temperature, axivity_timestamps, timestamp_policy="sparse")
            channel_light.set_contents(axivity_light, axivity_timestamps, timestamp_policy="sparse")

        # Approximate the frequency in hertz, based on the difference between the first and last timestamp
        approximate_frequency = timedelta(seconds=1) / (
                    (axivity_timestamps[-1] - axivity_timestamps[0]) / num_samples)
        file_header["approximate_frequency"] = approximate_frequency
        file_header["num_samples"] = num_samples
        file_header["num_pages"] = num_pages

        for c in [channel_x, channel_y, channel_z, channel_integrity]:
            c.indices = axivity_indices
            c.frequency = round(approximate_frequency, 2)

        channels = [channel_x, channel_y, channel_z, channel_temperature, channel_battery, channel_light,
                    channel_integrity]
        header = file_header
        try:
            handle.close()
        except:
            pass

    elif source_type == "GeneActiv":

        channel_x = Channel("X")
        channel_y = Channel("Y")
        channel_z = Channel("Z")
        channel_integrity = Channel("Integrity")
        channel_light = Channel("Light")
        channel_temperature = Channel("Temperature")
        channel_battery = Channel("Battery")

        # Open the file in read binary mode, read it into a data block
        f = open(source, "rb")
        data = io.BytesIO(f.read())
        # print("File read in")

        # First 59 lines contain header information
        first_lines = [data.readline().strip().decode() for i in range(59)]
        # print(first_lines)
        header_info = parse_header(first_lines, "GeneActiv", "")
        # print(header_info)

        num_pages = int(header_info["number_pages"])
        obs_num = 0
        page = 0
        # Data format contains 300 XYZ values per page
        num = 300
        x_values = np.empty(int(num * num_pages))
        y_values = np.empty(int(num * num_pages ))
        z_values = np.empty(int(num * num_pages))
        integrity = np.zeros(int(num * num_pages))
        ga_indices = np.empty(int(num_pages))

        # Check if data is to be separated into sample-level and page-level data:
        if compress is True:
            light_values = np.empty(int(num_pages))
            temperature_values = np.empty(int(num_pages))
            battery_values = np.empty(int(num_pages))

        else:
            light_values = np.empty(int(num * num_pages))
            temperature_values = np.empty(int(num * num_pages))
            battery_values = np.empty(int(num * num_pages))

        # we want a timestamp for each page
        page_timestamps = np.empty(int(num_pages), dtype=type(header_info["start_datetime_python"]))

        # For each page
        for i in range(num_pages):
            # xs,ys,zs,times = read_block(data, header_info)
            lines = [data.readline().strip().decode() for l in range(9)]
            if lines[0] == 'Recorded Data':

                # If the block data is not corrupt...
                try:
                    page_time = datetime.strptime(lines[3][10:29], "%Y-%m-%d %H:%M:%S") + timedelta(microseconds=int(lines[3][30:]) * 1000)
                    page_timestamps[page] = page_time

                    # read temperature, for page, in degrees C
                    temperature = lines[5].split(":")[1]
                    # read battery voltage for page
                    battery = lines[6].split(":")[1]

                    if compress is True:
                        # record temperature and battery at page level
                        temperature_values[page] = temperature
                        battery_values[page] = battery

                    # For each 12 byte measurement in page (300 of them)
                    for j in range(num):

                        block = data.read(12)
                        # Each of x,y,z are given by a 3-digit hexadecimal number
                        x = int(block[0:3], 16)
                        y = int(block[3:6], 16)
                        z = int(block[6:9], 16)
        
                        # 'Light' is included in the last 3-digit hexadecimal number
                        final_block = int(block[9:12], 16)
                        # Convert to 12 bit binary number
                        final_binary = "{0:b}".format(final_block).zfill(12)
                        # light is only the first 10 bits, so drop last 2 bits.
                        light_binary = final_binary[:-2]
                        # Then convert to an integer:
                        light = int(light_binary, 2)

                        if compress is True and j == 0:
                            # record the first observation of light at page level
                            light_values[page] = light

                        elif compress is True and j != 0:
                            pass

                        else:
                            # record light at sample level
                            light_values[obs_num] = light
      
                        x, y, z = twos_comp(x, 12), twos_comp(y, 12), twos_comp(z, 12)
                        x_values[obs_num] = x
                        y_values[obs_num] = y
                        z_values[obs_num] = z
                        obs_num += 1

                    ga_indices[page] = obs_num

                    excess = data.read(2)

                    page += 1
                    
                except ValueError:
                    pass
                
            else:
                pass    

        # in cases where the number of pages in incorrect, trim page-timestamps and ga_indices arrays, 
        # using 'page' which is a counter of the number of pages, or 'obs_num' which is a counter of samples
        safe_timestamps = np.resize(page_timestamps, page)  
        safe_indices = np.resize(ga_indices, page)
        safe_x = np.resize(x_values, obs_num)
        safe_y = np.resize(y_values, obs_num)
        safe_z = np.resize(z_values, obs_num)
        safe_integrity = np.resize(integrity, obs_num)
        safe_indices = safe_indices.astype(int)

        if compress is True:
            safe_light = np.resize(light_values, page)
            safe_temperature = np.resize(temperature_values, page)
            safe_battery = np.resize(battery_values, page)

        else:
            safe_light = np.resize(light_values, obs_num)
            safe_temperature = np.resize(temperature_values, obs_num)
            safe_battery = np.resize(battery_values, obs_num)

        # calibrate the x, y, z and light data using the monitor's given calibration parameters.
        safe_x = np.array([(x * 100.0 - header_info["x_offset"]) / header_info["x_gain"] for x in safe_x])
        safe_y = np.array([(y * 100.0 - header_info["y_offset"]) / header_info["y_gain"] for y in safe_y])
        safe_z = np.array([(z * 100.0 - header_info["z_offset"]) / header_info["z_gain"] for z in safe_z])
        safe_light = np.array([(light * header_info["lux"]) / header_info["volts"] for light in safe_light])

        # Map the page-level timestamps to the x, y,and z data "sparsely"
        channel_x.set_contents(safe_x, safe_timestamps, timestamp_policy="sparse")
        channel_y.set_contents(safe_y, safe_timestamps, timestamp_policy="sparse")
        channel_z.set_contents(safe_z, safe_timestamps, timestamp_policy="sparse")
        channel_integrity.set_contents(safe_integrity, safe_timestamps, timestamp_policy="sparse")
        channel_integrity.binary_data = True

        if compress is True:
            # Map the page-level timestamps to the temperature, battery and light data "normally"
            channel_temperature.set_contents(safe_temperature, safe_timestamps, timestamp_policy="normal")
            channel_battery.set_contents(safe_battery, safe_timestamps, timestamp_policy="normal")
            channel_light.set_contents(safe_light, safe_timestamps, timestamp_policy="normal")

        else:
            # Map the page-level timestamps to the temperature, battery and light data "sparsely"
            channel_temperature.set_contents(safe_temperature, safe_timestamps, timestamp_policy="sparse")
            channel_battery.set_contents(safe_battery, safe_timestamps, timestamp_policy="sparse")
            channel_light.set_contents(safe_light, safe_timestamps, timestamp_policy="sparse")

        for c in [channel_x, channel_y, channel_z, channel_integrity]:
            c.indices = safe_indices
            c.frequency = header_info["frequency"]

        channels = [channel_x, channel_y, channel_z, channel_temperature, channel_battery, channel_light, channel_integrity]
        header = header_info

    elif source_type == "XLO":

        # First 15 lines contain generic header info
        first_lines = []
        f = open(source, 'r')
        for i in range(15):
            s = f.readline().strip()
            first_lines.append(s)

        header_info = parse_header(first_lines, "XLO", "%d/%m/%Y %H:%M:%S")
        data = np.loadtxt(f, delimiter="\t", dtype="S").astype("U")
        f.close()

        # Skip the "empty" artefacts
        good_rows = data[:, 0] == '   -    '
        data = data[good_rows]

        # Timestamps
        start = header_info["start_datetime_python"]
        mins = [int(t.strip().split(":")[0]) for t in data[:, 2]]
        secs = [int(t.strip().split(":")[1]) for t in data[:, 2]]
        time = [m * 60 + s for m, s in zip(mins, secs)]
        timestamps = np.array([start + timedelta(seconds=m * 60 + s) for m, s in zip(mins, secs)])

        varlist = ['T-body', 'Pmean', 'Time', 't-ph', 'RPM', 'Load', 'Speed', 'Elev.', 'VTex', 'VTin', 't-ex', 't-in',
                   'BF', "V'E", "V'O2", "V'CO2", 'RER', 'FIO2', 'FICO2', 'FEO2', 'FECO2', 'FETO2', 'FETCO2', 'PEO2',
                   'PECO2', 'PETO2', 'PETCO2', 'EqO2', 'EqCO2', 'VDe', 'VDc/VT', 'VDe/VT']
        ignore = ['T-body', 'Pmean', 'Time', 't-ph', 'RPM', 'Load', 'Speed', 'Elev.']
        for i, var in enumerate(varlist):

            if not var in ignore:
                chan = Channel(var)

                missings = data[:, i] == '   -    '
                data[missings, i] = 0

                chan.set_contents(data[:, i].astype("float"), timestamps)
                channels.append(chan)

                if var in ["V'O2", "V'CO2"]:
                    chan.data /= header_info["weight"]

        header = header_info

    elif source_type == "HDF5":

        f = h5py.File(source, hdf5_mode)

        if hdf5_groups is None:
            raw_group = f[hdf5_group]
            ts = load_time_series(raw_group)

            header = dictionary_from_attributes(raw_group)

            header["hdf5_file"] = f

        else:
            for group in hdf5_groups:
                raw_group = f[group]
                ts_temp = load_time_series(raw_group)

                for channel in ts_temp:
                    channels.append(channel)

                # For the first loaded group create a header dictionary from the group attributes
                if len(header.keys()) == 0:
                    header = dictionary_from_attributes(raw_group)
                    header["hdf5_file"] = f

    # Check for an anomalies csv file corresponding to this data file, read in the file to a list of dictionaries if it exists
    anomalies = []
    if anomalies_file is not None:
        anomalies_df = pd.read_csv(anomalies_file)
        anomalies = anomalies_df.to_dict('records')
        channels_fixed = fix_anomalies(anomalies, channels)

        ts.add_channels(channels_fixed)

    # channels is a list of Channel objects, set above according to the file format
    else:
        ts.add_channels(channels)

    # Calculate how long it took to load this file
    load_end = datetime.now()
    load_duration = (load_end - load_start).total_seconds()

    header["generic_num_channels"] = ts.number_of_channels
    header["generic_first_timestamp"] = ts.earliest.strftime("%d/%m/%Y %H:%M:%S:%f")
    header["generic_last_timestamp"] = ts.latest.strftime("%d/%m/%Y %H:%M:%S:%f")
    header["generic_num_samples"] = len(ts.channels[0].data)
    header["generic_loading_time_seconds"] = load_duration
    header["generic_processing_timestamp"] = load_start.strftime("%d/%m/%Y %H:%M:%S:%f")

    return ts, header


def parse_axivity_header(file):

    info = cwa_info(file)

    # create dictionaries from the header information sections
    dict1 = (info["file"])
    file_dict = convert_dict_keys(dict1, prefix="file")

    dict2 = (info["first"])
    first_dict = convert_dict_keys(dict2, prefix="first")

    dict3 = (info["last"])
    last_dict = convert_dict_keys(dict3, prefix="last")

    dict4 = (info["header"])["metadata"]
    meta_dict = convert_dict_keys(dict4)

    dict5 = (info["header"])
    del dict5["metadata"]
    header_dict = convert_dict_keys(dict5, special_cases={"sampleRate": "programmed_sample_rate"})

    header_info = dict(
        chain(file_dict.items(), header_dict.items(), meta_dict.items(), first_dict.items(), last_dict.items()))

    return header_info


def read_timestamp(data):
    value = unpack('<I', data)[0]
    if value == 0x00000000:  # Infinitely in past = 'always before now'
        return 0
    if value == 0xffffffff:  # Infinitely in future = 'always after now'
        return -1
    # bit pattern:  YYYYYYMM MMDDDDDh hhhhmmmm mmssssss
    year = ((value >> 26) & 0x3f) + 2000
    month = (value >> 22) & 0x0f
    day = (value >> 17) & 0x1f
    hours = (value >> 12) & 0x1f
    mins = (value >> 6) & 0x3f
    secs = (value >> 0) & 0x3f
    try:
        dt = datetime(year, month, day, hours, mins, secs)
        timestamp = (dt - datetime(1970, 1, 1)).total_seconds()
        return timestamp
    # return str(datetime.fromtimestamp(timestamp))
    # return time.strptime(t, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None


def timestamp_string(timestamp):
    if timestamp == 0:
        return "0"
    if timestamp < 0:
        return "-1"
    # return str(datetime.fromtimestamp(timestamp))
    return datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S:%f")[:23]


# Local "URL-decode as UTF-8 string" function
def urldecode(input):
    output = bytearray()
    nibbles = 0
    value = 0
    # Each input character
    for char in input:
        if char == '%':
            # Begin a percent-encoded hex pair
            nibbles = 2
            value = 0
        elif nibbles > 0:
            # Parse the percent-encoded hex digits
            value *= 16
            if char >= 'a' and char <= 'f':
                value += ord(char) + 10 - ord('a')
            elif char >= 'A' and char <= 'F':
                value += ord(char) + 10 - ord('A')
            elif char >= '0' and char <= '9':
                value += ord(char) - ord('0')
            nibbles -= 1
            if nibbles == 0:
                output.append(value)
        elif char == '+':
            # Treat plus as space (application/x-www-form-urlencoded)
            output.append(ord(' '))
        else:
            # Preserve character
            output.append(ord(char))
    return output.decode('utf-8')


def cwa_parse_metadata(data):
    # Metadata represented as a dictionary
    metadata = {}

    # Shorthand name expansions
    shorthand = {
        "_c": "study_centre",
        "_s": "study_code",
        "_i": "investigator",
        "_x": "exercise_code",
#        "_v": "volunteer_number",
        "_p": "body_location",
        "_so": "setup_operator",
        "_n": "study_notes",
        "_b": "start_time",
        "_e": "end_time",
#        "_ro": "recovery_operator",
#        "_r": "retrieval_time",
        "_sn": "subject_notes",
#        "_co": "comments",
        "_sc": "subject_code",
        "_se": "sex",
        "_h": "height",
        "_w": "weight",
        "_ha": "handedness",
    }
    # create a list of metadata variables
    meta_vars = shorthand.values()

    # CWA File has 448 bytes of metadata at offset 64
    if sys.version_info[0] < 3:
        encString = str(data)
    else:
        encString = str(data, 'ascii')

    # Remove any trailing spaces, null, or 0xFF bytes
    encString = encString.rstrip('\x20\xff\x00')

    # Name-value pairs separated with ampersand
    nameValues = encString.split('&')

    # Each name-value pair separated with an equals
    for nameValue in nameValues:
        parts = nameValue.split('=')
        # Name is URL-encoded UTF-8
        name = urldecode(parts[0])
        if len(name) > 0:
            value = None

            if len(parts) > 1:
                # Value is URL-encoded UTF-8
                value = urldecode(parts[1])

            # Expand shorthand names
            name = shorthand.get(name, name)

            # Store metadata name-value pair
            metadata[name] = value

    # If metadata variable is not already a key in the metadata dictionary, create it and set value to "-1" (null code)
    for var in meta_vars:
        if var in metadata.keys():
            pass
        else:
            metadata[var] = "-1"

    # Metadata dictionary
    return metadata
    

# 16-bit checksum (should sum to zero)
def checksum(data):
    sum = 0
    for i in range(0, len(data), 2):
        value = data[i] | (data[i + 1] << 8)
        sum = (sum + value) & 0xffff
    return sum
 

def cwa_header(block):
    header = {}
    if len(block) >= 512:
        packetHeader = block[0:2]  # @ 0  +2   ASCII "MD", little-endian (0x444D)
        packetLength = unpack('<H', block[2:4])[
            0]  # @ 2  +2   Packet length (1020 bytes, with header (4) = 1024 bytes total)
        if packetHeader[0] == ord('M') and packetHeader[1] == ord('D') and packetLength >= 508:
            header['packet_length'] = packetLength
            # unpack() <=little-endian, bB=s/u 8-bit, hH=s/u 16-bit, iI=s/u 32-bit
            # header['reserved1'] = unpack('B', block[4:5])[0]            # @ 4  +1   (1 byte reserved)
            header['device'] = unpack('<H', block[5:7])[0]  # @ 5  +2   Device identifier
            deviceIdUpper = unpack('<H', block[11:13])[0]    			# @11  +2   (2 bytes reserved)
            if deviceIdUpper != 0xffff:
                header['device'] |= deviceIdUpper << 16
            header['session_id'] = unpack('<I', block[7:11])[0]  # @ 7  +4   Unique session identifier
            # header['reserved2'] = unpack('<H', block[11:13])[0]        # @11  +2   (2 bytes reserved)
            header['logging_start'] = read_timestamp(block[13:17])  # @13  +4   Start time for delayed logging
            header['logging_end'] = read_timestamp(block[17:21])  # @17  +4   Stop time for delayed logging
            # header['loggingCapacity'] = unpack('<I', block[21:25])[0]    # @21  +4   (Deprecated: preset maximum number of samples to collect, 0 = unlimited)
            # header['reserved3'] = block[25:36]                        # @25  +11  (11 bytes reserved)
            rateCode = unpack('B', block[36:37])[
                0]  # @36  +1   Sampling rate code, frequency (3200/(1<<(15-(rate & 0x0f)))) Hz, range (+/-g) (16 >> (rate >> 6)).
            header['last_change'] = read_timestamp(block[37:41])  # @37  +4   Last change metadata time
            header['firmware_revision'] = unpack('B', block[41:42])[0]  # @41  +1   Firmware revision number
            # header['timeZone'] = unpack('<H', block[42:44])[0]        # @42  +2   (Unused: originally reserved for a "Time Zone offset from UTC in minutes", 0xffff = -1 = unknown)
            # header['reserved4'] = block[44:64]                        # @44  +20  (20 bytes reserved)
            header['metadata'] = cwa_parse_metadata(block[
                                                    64:512])  # @64  +448 "Annotation" meta-data (448 ASCII characters, ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs)
            # header['reserved'] = block[512:1024]                        # @512 +512 Reserved for device-specific meta-data (512 bytes, ASCII characters, ignore trailing 0x20/0x00/0xff bytes, url-encoded UTF-8 name-value pairs, leading '&' if present?)

            # Timestamps
            header['logging_start_time'] = timestamp_string(header['logging_start'])
            header['logging_end_time'] = timestamp_string(header['logging_end'])
            header['last_change_time'] = timestamp_string(header['last_change'])

            # Parse rateCode
            header['frequency'] = (3200 / (1 << (15 - (rateCode & 0x0f))))
            header['sample_range'] = (16 >> (rateCode >> 6))

    return header


def cwa_data(block):
    data = {}
    if len(block) >= 512:
        packetHeader = block[0:2]  # @ 0  +2   ASCII "AX", little-endian (0x5841)
        packetLength = unpack('<H', block[2:4])[
            0]  # @ 2  +2   Packet length (508 bytes, with header (4) = 512 bytes total)
        if packetHeader[0] == ord('A') and packetHeader[1] == ord('X') and packetLength == 508:
            deviceFractional = unpack('<H', block[4:6])[
                0]  # @ 4  +2   Top bit set: 15-bit fraction of a second for the time stamp, the timestampOffset was already adjusted to minimize this assuming ideal sample rate; Top bit clear: 15-bit device identifier, 0 = unknown;
            data['session_id'] = unpack('<I', block[6:10])[0]  # @ 6  +4   Unique session identifier, 0 = unknown
            data['sequence_id'] = unpack('<I', block[10:14])[
                0]  # @10  +4   Sequence counter (0-indexed), each packet has a new number (reset if restarted)
            timestamp = read_timestamp(block[14:18])  # @14  +4   Last reported RTC value, 0 = unknown
            data['light'] = unpack('<H', block[18:20])[0]                # @18  +2   Last recorded light sensor value in raw units, 0 = none
            data['temperature'] = unpack('<H', block[20:22])[0]        # @20  +2   Last recorded temperature sensor value in raw units, 0 = none
            # data['events'] = unpack('B', block[22:23])[0]                # @22  +1   Event flags since last packet, b0 = resume logging, b1 = reserved for single-tap event, b2 = reserved for double-tap event, b3 = reserved, b4 = reserved for diagnostic hardware buffer, b5 = reserved for diagnostic software buffer, b6 = reserved for diagnostic internal flag, b7 = reserved)
            data['battery'] = unpack('B', block[23:24])[0]            # @23  +1   Last recorded battery level in raw units, 0 = unknown
            rateCode = unpack('B', block[24:25])[
                0]  # @24  +1   Sample rate code, frequency (3200/(1<<(15-(rate & 0x0f)))) Hz, range (+/-g) (16 >> (rate >> 6)).
            numAxesBPS = unpack('B', block[25:26])[
                0]  # @25  +1   0x32 (top nibble: number of axes = 3; bottom nibble: packing format - 2 = 3x 16-bit signed, 0 = 3x 10-bit signed + 2-bit exponent)
            timestampOffset = unpack('<h', block[26:28])[
                0]  # @26  +2   Relative sample index from the start of the buffer where the whole-second timestamp is valid
            data['sample_count'] = unpack('<H', block[28:30])[
                0]  # @28  +2   Number of accelerometer samples (80 or 120 if this sector is full)
            # rawSampleData[480] = block[30:510]                        # @30  +480 Raw sample data.  Each sample is either 3x 16-bit signed values (x, y, z) or one 32-bit packed value (The bits in bytes [3][2][1][0]: eezzzzzz zzzzyyyy yyyyyyxx xxxxxxxx, e = binary exponent, lsb on right)
            # checksum = unpack('<H', block[510:512])[0]                # @510 +2   Checksum of packet (16-bit word-wise sum of the whole packet should be zero)

            # range = 16 >> (rateCode >> 6)
            frequency = 3200 / (1 << (15 - (rateCode & 0x0f)))

            timeFractional = 0;
            # if top-bit set, we have a fractional date
            if deviceFractional & 0x8000:
                # Need to undo backwards-compatible shim by calculating how many whole samples the fractional part of timestamp accounts for.
                timeFractional = (
                                 deviceFractional & 0x7fff) << 1  # use original deviceId field bottom 15-bits as 16-bit fractional time
                timestampOffset += (timeFractional * int(
                    frequency)) >> 16  # undo the backwards-compatible shift (as we have a true fractional)

            # Add fractional time to timestamp
            timestamp += timeFractional / 65536

            data['timestamp'] = timestamp
            data['timestamp_offset'] = timestampOffset

            data['timestampTime'] = timestamp_string(data['timestamp'])

            if numAxesBPS & 0x0f == 0x00:
                data['samples_per_sector'] = 120
            elif numAxesBPS & 0x0f == 0x02:
                data['samples_per_sector'] = 80
    return data


def cwa_info(filename):
    file = {}
    header = {}
    first = {}
    last = {}

    file['filename'] = os.path.basename(filename)

    # Header
    with open(filename, "rb") as f:
        sectorSize = 512

        # File length
        f.seek(0, 2)
        fileSize = f.tell()

        # Read header
        headerSize = 1024
        f.seek(0)
        headerBytes = f.read(headerSize)
        header = cwa_header(headerBytes)
        headerSize = header['packet_length'] + 4

        # Read first data sector
        f.seek(headerSize)
        firstBytes = f.read(sectorSize)
        first = cwa_data(firstBytes)

        # Read last data sector
        f.seek(fileSize - sectorSize)
        lastBytes = f.read(sectorSize)
        last = cwa_data(lastBytes)

        # Update file metadata
        file['size'] = fileSize
        file['sector_size'] = sectorSize
        file['header_size'] = headerSize
        if fileSize >= headerSize:
            file['num_sectors'] = (fileSize - headerSize) // 512
        else:
            file['num_sectors'] = 0

        # Samples per sector
        samplesPerSector = 0
        if 'samples_per_sector' in first:
            samplesPerSector = first['samples_per_sector']
        if 'samples_per_sector' in last:
            samplesPerSector = last['samples_per_sector']
        file['samples_per_sector'] = samplesPerSector

        # Estimate total number of samples
        file['num_samples'] = file['num_sectors'] * samplesPerSector

        duration = 0
        if 'timestamp' in first and 'timestamp' in last:
            duration = last['timestamp'] - first['timestamp']
        file['duration'] = duration

        # Mean rate (assuming no breaks)
        meanRate = 0
        if duration != 0:
            meanRate = file['num_samples'] / duration
        file['mean_rate'] = meanRate

    # Parse metadata
    info = {}
    info['file'] = file
    info['header'] = header
    info['first'] = first
    info['last'] = last

    # Metadata dictionary
    return info


# Test function
if __name__ == "__main__":
    import json
    import os

    mode = 'json'  # '-mode:json', '-mode:ldjson', '-mode:size_rate'
    for filename in sys.argv[1:]:
        try:
            if filename[0:6] == '-mode:':
                mode = filename[6:]
                continue
            info = cwa_info(filename)
            if mode == 'json':
                print(json.dumps(info, indent=4, sort_keys=True))
            elif mode == 'ldjson':
                print(json.dumps(info))
            elif mode == 'size_rate':
                print('%s,%s,%s,%s,%s' % (
                info['file']['name'], info['file']['size'] / 1024 / 1024, info['file']['duration'] / 60 / 60 / 24,
                info['header']['sampleRate'], info['file']['meanRate']))
            else:
                print('ERROR: Unknown output mode: %s' % mode)
        except Exception as e:
            print('Exception ' + e.__doc__ + ' -- ' + e.message)


def fast_load(source, source_type):
    """ This fast data loading option is currently valid for Axivity and GeneActiv files only, and will load in just the page-level data.
    This results in a shorter data set and faster loading time, use for Quality Control purposes ONLY.
    """

    load_start = datetime.now()

    header = OrderedDict()
    channels = []
    ts = Time_Series("")

    if source_type.startswith("Axivity"):
        # if source file is zipped, unzip to read
        if source_type.endswith("_ZIP"):
            archive = zipfile.ZipFile(source, "r")
            cwa_not_zip = source.split("/")[-1].replace(".zip", ".cwa")
            handle = archive.open(cwa_not_zip)

        else:
            handle = open(source, "rb")

        channel_x = Channel("X")
        channel_y = Channel("Y")
        channel_z = Channel("Z")
        channel_light = Channel("Light")
        channel_temperature = Channel("Temperature")
        channel_battery = Channel("Battery")
        channel_integrity = Channel("Integrity")

        raw_bytes = handle.read()
        fh = io.BytesIO(raw_bytes)

        n = 0
        num_pages = 0

        start = datetime(2014, 1, 1)

        # Rough number of pages expected = length of file / size of block (512 bytes)
        # Add 1% buffer just to be cautious - it's trimmed later
        estimated_num_pages = int(len(raw_bytes) / 512 * 1.01)


        axivity_x = np.empty(estimated_num_pages)
        axivity_y = np.empty(estimated_num_pages)
        axivity_z = np.empty(estimated_num_pages)
        axivity_light = np.empty(estimated_num_pages)
        axivity_temperature = np.empty(estimated_num_pages)
        axivity_battery = np.empty(estimated_num_pages)
        axivity_timestamps = np.empty(estimated_num_pages, dtype=type(start))
        axivity_indices = np.empty(estimated_num_pages)
        axivity_integrity = np.empty(estimated_num_pages)

        file_header = OrderedDict()
        file_header = parse_axivity_header(source)

        file_session_id = file_header['session_id']

        lastSequenceId = None
        lastTimestampOffset = None
        lastTimestamp = None

        try:
            header = axivity_read(fh, 2)

            while len(header) == 2:

                if header == b'MD':
                    #print('MD')
                    pass
                elif header == b'UB':
                    #print('UB')
                    blockSize = unpack('H', axivity_read(fh, 2))[0]
                elif header == b'SI':
                    #print('SI')
                    pass
                elif header == b'AX':
                    #print('AX')

                    packet = axivity_read(fh, 510)
                    
                    packetLength, deviceId, sessionId, sequenceId, sampleTimeData, light, temperature, events, battery, sampleRate, numAxesBPS, timestampOffset, sampleCount = unpack('HHIIIHHcBBBhH', packet[0:28])
                    
                    sector = b'AX' + packet
                    
                    sector_checksum = checksum(sector)

                    if packetLength != 508 or sampleRate == 0:
                        continue

                    if ((numAxesBPS >> 4) & 15) != 3:
                        print('[ERROR: Axes!=3 not supported yet -- this will not work properly]')

                    if (light & 0xfc00) != 0:
                        print('[ERROR: Scale not supported yet -- this will not work properly]')

                    if (numAxesBPS & 15) == 2:
                        bps = 6
                    elif (numAxesBPS & 15) == 0:
                        bps = 4

                    freq = 3200 / (1 << (15 - sampleRate & 15))
                    if freq <= 0:
                        freq = 1

                    timestamp_original = axivity_read_timestamp_raw(sampleTimeData)

                    if timestamp_original is None:
                        continue


                    # if top-bit set, we have a fractional date
                    if deviceId & 0x8000:
                        # Need to undo backwards-compatible shim by calculating how many whole samples the fractional part of timestamp accounts for.
                        timeFractional = (
                                             deviceId & 0x7fff) * 2  # use original deviceId field bottom 15-bits as 16-bit fractional time
                        timestampOffset += (timeFractional * int(
                            freq)) // 65536  # undo the backwards-compatible shift (as we have a true fractional)
                        timeFractional = float(timeFractional) / 65536

                        # Add fractional time to timestamp
                        timestamp = timestamp_original + timedelta(seconds=timeFractional)

                    else:

                        timestamp = timestamp_original

                    # --- Time interpolation ---
                    # Reset interpolator if there's a sequence break or there was no previous timestamp
                    if lastSequenceId == None or (
                                lastSequenceId + 1) & 0xffff != sequenceId or lastTimestampOffset == None or lastTimestamp == None:
                        # Bootstrapping condition is a sample one second ago (assuming the ideal frequency)
                        lastTimestampOffset = timestampOffset - freq
                        lastTimestamp = timestamp - timedelta(seconds=1)
                        lastSequenceId = sequenceId - 1

                    localFreq = timedelta(seconds=(timestampOffset - lastTimestampOffset)) / (timestamp - lastTimestamp)
                    final_timestamp = timestamp + -timedelta(seconds=timestampOffset) / localFreq
                    
                    # Update for next loop
                    lastSequenceId = sequenceId
                    lastTimestampOffset = timestampOffset - sampleCount
                    lastTimestamp = timestamp

                    axivity_indices[num_pages] = num_pages
                    axivity_timestamps[num_pages] = final_timestamp

                    '''# check for data integrity:
                    if sector_checksum == 0 and sessionId == file_session_id:
                        # if check passed then set 'validity' to '0' and extract the data
                        axivity_validity[num_pages] = 0
                    # if integrity check fails on checksum, set 'validity' to '1'
                    elif sector_checksum != 0:
                        axivity_validity[num_pages] = 1
                    # if integrity check fails on session id matching, set 'validity value' to '2'
                    elif sessionId != file_session_id:
                        axivity_validity[num_pages] = 2'''
                        
                        
                    # convert the light and temperature values to lux and degrees C values
                    light_converted = convert_axivity_light(light)
                    temp_converted = convert_axivity_temp(temperature)
                    axivity_light[num_pages] = light_converted
                    axivity_temperature[num_pages] = temp_converted
                    axivity_battery[num_pages] = battery

                    if bps == 6:

                        x, y, z = unpack('hhh', packet[28:34])
                        x, y, z = x / 256.0, y / 256.0, z / 256.0

                    elif bps == 4:

                        temp = unpack('I', packet[28:32])[0]
                        temp2 = (6 - (temp >> 30))
                        x = short(short((ushort(65472) & ushort(temp << 6))) >> temp2) / 256.0
                        y = short(short((ushort(65472) & ushort(temp >> 4))) >> temp2) / 256.0
                        z = short(short((ushort(65472) & ushort(temp >> 14))) >> temp2) / 256.0

                    # save only the first values of x,y,z
                    axivity_x[num_pages] = x
                    axivity_y[num_pages] = y
                    axivity_z[num_pages] = z
                    
                    num_pages += 1
                    
                else:
                    pass
                    #print("Unrecognised header", header)

                header = axivity_read(fh, 2)

                n = n + 1
                
        except IOError:
            # End of file
            pass

        # We created oversized arrays at the start, to make sure we could fit all the data in
        # Now we know how much data was there, we can shrink the arrays to size

        axivity_temperature.resize(num_pages)
        axivity_light.resize(num_pages)
        axivity_battery.resize(num_pages)
        axivity_timestamps.resize(num_pages)
        axivity_indices.resize(num_pages)
        axivity_indices = axivity_indices.astype(int)
        axivity_integrity.resize(num_pages)

        axivity_x.resize(num_pages)
        axivity_y.resize(num_pages)
        axivity_z.resize(num_pages)
        # Map the page-level timestamps to the acceleration data "normally"
        channel_x.set_contents(axivity_x, axivity_timestamps, timestamp_policy="normal")
        channel_y.set_contents(axivity_y, axivity_timestamps, timestamp_policy="normal")
        channel_z.set_contents(axivity_z, axivity_timestamps, timestamp_policy="normal")
        # Map the page-level timestamps to the temperature, battery and light data
        channel_temperature.set_contents(axivity_temperature, axivity_timestamps, timestamp_policy="normal")
        channel_battery.set_contents(axivity_battery, axivity_timestamps, timestamp_policy="normal")
        channel_light.set_contents(axivity_light, axivity_timestamps, timestamp_policy="normal")
        channel_integrity.set_contents(axivity_integrity, axivity_timestamps, timestamp_policy="normal")
        channel_integrity.binary_data = True

        file_header["num_pages"] = num_pages

        for c in [channel_x, channel_y, channel_z, channel_integrity]:
            c.indices = axivity_indices
            c.frequency = file_header["frequency"]

        channels = [channel_x, channel_y, channel_z, channel_temperature, channel_battery, channel_light, channel_integrity]
        header = file_header
        try:
            handle.close()
        except:
            pass


    elif source_type == "GeneActiv":

        channel_x = Channel("X")
        channel_y = Channel("Y")
        channel_z = Channel("Z")
        channel_integrity = Channel("Integrity")
        channel_light = Channel("Light")
        channel_temperature = Channel("Temperature")
        channel_battery = Channel("Battery")

        # Open the file in read binary mode, read it into a data block
        f = open(source, "rb")
        data = io.BytesIO(f.read())
        # print("File read in")

        # First 59 lines contain header information
        first_lines = [data.readline().strip().decode() for i in range(59)]
        # print(first_lines)
        header_info = parse_header(first_lines, "GeneActiv", "")
        # print(header_info)

        num_pages = int(header_info["number_pages"])
        page = 0
        # Data format contains 300 XYZ values per page
        num = 300
        x_values = np.empty(int(num_pages))
        y_values = np.empty(int(num_pages))
        z_values = np.empty(int(num_pages))
        integrity = np.zeros(int(num_pages))
        light_values = np.empty(int(num_pages))
        temperature_values = np.empty(int(num_pages))
        battery_values = np.empty(int(num_pages))

        ga_indices = np.empty(int(num_pages))

        # we want a timestamp for each page to assign to the data
        page_timestamps = np.empty(int(num_pages), dtype=type(header_info["start_datetime_python"]))

        # For each page
        for i in range(num_pages):
            lines = [data.readline().strip().decode() for l in range(9)]
            if lines[0] == 'Recorded Data':

              page_time = datetime.strptime(lines[3][10:29], "%Y-%m-%d %H:%M:%S") + timedelta(microseconds=int(lines[3][30:])*1000)
              page_timestamps[page] = page_time
  
              # read temperature, for page, in degrees C
              temperature = lines[5].split(":")[1]
              # read battery voltage for page
              battery = lines[6].split(":")[1]
  
              # record temperature and battery at page level
              temperature_values[page] = temperature
              battery_values[page] = battery
  
              block = data.read(12)
  
              # Each of x,y,z are given by a 3-digit hexadecimal number
              x = int(block[0:3], 16)
              y = int(block[3:6], 16)
              z = int(block[6:9], 16)
  
              # 'Light' is included in the last 3-digit hexadecimal number
              final_block = int(block[9:12], 16)
              # Convert to 12 bit binary number
              final_binary = "{0:b}".format(final_block).zfill(12)
              # light is only the first 10 bits, so drop last 2 bits.
              light_binary = final_binary[:-2]
              # Then convert to an integer:
              light = int(light_binary, 2)
  
              x, y, z = twos_comp(x, 12), twos_comp(y, 12), twos_comp(z, 12)
  
              x_values[page] = x
              y_values[page] = y
              z_values[page] = z
              light_values[page] = light
  
              # skip over the remaining samples for the page
              bytes_to_skip = 12 * (num - 1)
              skip = data.read(bytes_to_skip)
  
              excess = data.read(2)
  
              ga_indices[page] = page
              page += 1
            
            else:
              pass
              
        # in cases where the number of pages in incorrect, trim page-timestamps and ga_indices arrays, 
        # using 'page' which is a counter of the number of pages
        safe_timestamps = np.resize(page_timestamps, page)  
        safe_integrity = np.resize(integrity, page)
        safe_indices = np.resize(ga_indices, page)

        # calibrate the x, y, z and light data using the monitor's given calibration parameters.
        x_values = np.array([(x * 100.0 - header_info["x_offset"]) / header_info["x_gain"] for x in x_values])
        y_values = np.array([(y * 100.0 - header_info["y_offset"]) / header_info["y_gain"] for y in y_values])
        z_values = np.array([(z * 100.0 - header_info["z_offset"]) / header_info["z_gain"] for z in z_values])
        light_values = np.array([(light * header_info["lux"]) / header_info["volts"] for light in light_values])

        # Map the second-level timestamps to the acceleration data "sparsely"
        channel_x.set_contents(x_values, safe_timestamps, timestamp_policy="normal")
        channel_y.set_contents(y_values, safe_timestamps, timestamp_policy="normal")
        channel_z.set_contents(z_values, safe_timestamps, timestamp_policy="normal")

        # Map the page-level timestamps to the temperature, battery and light data "normally"
        channel_temperature.set_contents(temperature_values, safe_timestamps, timestamp_policy="normal")
        channel_battery.set_contents(battery_values, safe_timestamps, timestamp_policy="normal")
        channel_light.set_contents(light_values, safe_timestamps, timestamp_policy="normal")
        channel_integrity.set_contents(safe_integrity, safe_timestamps, timestamp_policy="normal")
        channel_integrity.binary_data = True

        for c in [channel_x, channel_y, channel_z, channel_integrity]:
            c.indices = safe_indices
            c.frequency = header_info["frequency"]

        channels = [channel_x, channel_y, channel_z, channel_temperature, channel_battery, channel_light, channel_integrity]
        header = header_info
    
    # channels is a list of Channel objects, set above according to the file format
    ts.add_channels(channels)

    # Calculate how long it took to load this file
    load_end = datetime.now()
    load_duration = (load_end - load_start).total_seconds()

    first_timestamp = ts.earliest.strftime("%d/%m/%Y %H:%M:%S:%f")
    last_timestamp = ts.latest.strftime("%d/%m/%Y %H:%M:%S:%f")
    header["QC_first_timestamp"] = first_timestamp
    header["QC_last_timestamp"] = last_timestamp
    header["QC_num_channels"] = ts.number_of_channels
    
    file_duration = ts.latest - ts.earliest
    header["QC_file_duration"] = file_duration.total_seconds()
    header["QC_num_samples"] = len(ts.channels[0].data)
    header["QC_loading_time_seconds"] = load_duration
    header["QC_processing_timestamp"] = load_start.strftime("%d/%m/%Y %H:%M:%S:%f")

    return ts, header
