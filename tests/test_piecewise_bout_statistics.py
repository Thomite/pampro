
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile21.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

def teardown_func():
    pass


def test_hourly():

    one_bouts = counts.bouts(1,1)
    bc = Bout_Collection.Bout_Collection("1_bouts", bouts=one_bouts)

    hourly_ts = bc.piecewise_statistics(timedelta(hours=1), statistics=[("generic", ["sum", "n"])], time_period=counts.timeframe)

    #
    hourly_sum = hourly_ts["1_bouts_sum"]
    assert(hourly_sum.data[0] == 30*60)
    assert(hourly_sum.data[1] == 30*60)
    assert(hourly_sum.data[2] == 30*60)
    assert(hourly_sum.data[3] == 60*60)
    assert(hourly_sum.data[4] == 15*60)
    assert(hourly_sum.data[5] == 29*60)
    assert(hourly_sum.data[6] == 30*60)
    assert(hourly_sum.data[7] == 7*60)

    # Number of bouts per hour goes 1 1 2 1 1 3 30 2
    hourly_n = hourly_ts["1_bouts_n"]
    assert(hourly_n.data[0] == 1)
    assert(hourly_n.data[1] == 1)
    assert(hourly_n.data[2] == 2)
    assert(hourly_n.data[3] == 1)
    assert(hourly_n.data[4] == 1)
    assert(hourly_n.data[5] == 3)
    assert(hourly_n.data[6] == 30)
    assert(hourly_n.data[7] == 2)

test_hourly.setup = setup_func
test_hourly.teardown = teardown_func


def test_daily():

    one_bouts = counts.bouts(1,1)
    bc = Bout_Collection.Bout_Collection("1_bouts", bouts=one_bouts)

    daily_ts = bc.piecewise_statistics(timedelta(days=1), statistics=[("generic", ["sum", "n"])], time_period=counts.timeframe)

    # 231 total minutes of 1s
    daily_sum = daily_ts["1_bouts_sum"]
    assert(daily_sum.data[0] == 231*60)

    # There are 38 bouts of 1s
    daily_n = daily_ts["1_bouts_n"]
    assert(daily_n.data[0] == 38)


test_daily.setup = setup_func
test_daily.teardown = teardown_func
