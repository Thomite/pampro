# pampro - physical activity monitor processing
# Copyright (C) 2019  MRC Epidemiology Unit, University of Cambridge
#   
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
#   
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#   
# You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

from datetime import datetime, date, time, timedelta

def start_of_day(datetimestamp):
	''' Trim the excess hours, minutes, seconds and microseconds off a datetime stamp '''
	start = datetimestamp - timedelta(hours=datetimestamp.hour, minutes=datetimestamp.minute, seconds=datetimestamp.second, microseconds=datetimestamp.microsecond)
	return start

def end_of_day(datetimestamp):
	''' Add the excess hours, minutes, seconds and microseconds onto a datetime stamp '''
	end = datetimestamp + timedelta(hours=23-datetimestamp.hour, minutes=59-datetimestamp.minute, seconds=59-datetimestamp.second, microseconds=999999-datetimestamp.microsecond)
	return end

def start_of_hour(datetimestamp):
	''' Trim the excess minutes, seconds and microseconds off a datetime stamp '''
	start = datetimestamp - timedelta(minutes=datetimestamp.minute, seconds=datetimestamp.second, microseconds=datetimestamp.microsecond)
	return start

def end_of_hour(datetimestamp):
	''' Add the excess minutes, seconds and microseconds onto a datetime stamp '''
	end = datetimestamp + timedelta(minutes=59-datetimestamp.minute, seconds=59-datetimestamp.second, microseconds=999999-datetimestamp.microsecond)
	return end

def start_of_minute(datetimestamp):
	''' Trim the excess seconds and microseconds off a datetime stamp '''
	start = datetimestamp - timedelta(seconds=datetimestamp.second, microseconds=datetimestamp.microsecond)
	return start

def end_of_minute(datetimestamp):
	''' Add the excess seconds and microseconds onto a datetime stamp '''
	end = datetimestamp + timedelta(seconds=59-datetimestamp.second, microseconds=999999-datetimestamp.microsecond)
	return end

def is_midnight(datetimestamp):
	return datetimestamp.hour == 0 & datetimestamp.minute == 0 & datetimestamp.second == 0 & datetimestamp.microsecond==0
