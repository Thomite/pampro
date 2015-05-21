import numpy as np
from pampro import Channel


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
