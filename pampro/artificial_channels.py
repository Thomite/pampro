# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from .Channel import *


def sine_wave(channel, wavelength_minutes=1440.0):

	sine_data = np.sin([2.0*np.pi*(float(t.time().hour*60+t.time().minute)/ wavelength_minutes) for t in channel.timestamps])

	sine_wave = Channel.Channel(channel.name + "_sine")
	sine_wave.set_contents(sine_data, channel.timestamps)
	return sine_wave


def cosine_wave(channel, wavelength_minutes=1440.0):

	cosine_data = np.cos([2.0*np.pi*(float(t.time().hour*60+t.time().minute)/wavelength_minutes) for t in channel.timestamps])

	cosine_wave = Channel.Channel(channel.name + "_cosine")
	cosine_wave.set_contents(cosine_data, channel.timestamps)
	return cosine_wave
