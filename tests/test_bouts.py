
from pampro import data_loading, Time_Series, Channel, Bout
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile16.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass



def test_bouts():

    # There are 8 bouts of 0s
    zero_bouts = counts.bouts(0,0)
    assert(len(zero_bouts) == 8)

    # There are 8 bouts of 1s
    one_bouts = counts.bouts(1,1)
    assert(len(one_bouts) == 8)

    # Since there are only 1s and 0s in the file, there should be 1 bout of 0 to 1
    both_bouts = counts.bouts(0,1)
    assert(len(both_bouts) == 1)

    # The timestamps of that 1 bout should match the start and end of the channel timestamps
    # But "end" of bout occurs 1 minute after end of channel
    assert(both_bouts[0].start_timestamp == counts.timestamps[0])
    assert(both_bouts[0].end_timestamp == counts.timestamps[-1]+timedelta(minutes=1))

    # Changing the max value shouldn't change anything
    bouts = counts.bouts(0,23)
    assert(len(bouts) == 1)

    # Same for the minimum value
    bouts = counts.bouts(-340,23)
    assert(len(bouts) == 1)

    # Should be no bouts 2 or above
    bouts = counts.bouts(2,23)
    assert(len(bouts) == 0)

    # Same for below 0
    bouts = counts.bouts(-32323,-2)
    assert(len(bouts) == 0)

    # The data is in 1 minute epochs
    total_zero_time = Bout.total_time(zero_bouts)
    total_one_time = Bout.total_time(one_bouts)
    total_both_time = Bout.total_time(both_bouts)

    assert(total_zero_time == timedelta(minutes=10*30))
    assert(total_one_time == timedelta(minutes=16*30))
    assert(total_both_time == total_zero_time + total_one_time)

    # Integer seconds spent at 0 should be 300 minutes * 60 = 18000 seconds
    total_zero_time_seconds = total_zero_time.total_seconds()
    assert(total_zero_time_seconds == 10*30*60)

    # Inverting bouts within a period
    # Since the file is 0s and 1s, the total time - the time spent @ 0 should = time spent @ 1
    not_zero_bouts = Bout.time_period_minus_bouts((counts.timestamps[0],counts.timestamps[-1]+timedelta(minutes=1)), zero_bouts)
    total_not_zero_time = Bout.total_time(not_zero_bouts)
    assert(total_not_zero_time == total_one_time)


test_bouts.setup = setup_func
test_bouts.teardown = teardown_func
