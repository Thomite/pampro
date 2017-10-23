
from pampro import data_loading, Time_Series, Channel, Bout, channel_inference
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile19.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass


def test_bouts():

    one_bouts = counts.bouts(1,1)

    # There are 7 bouts
    assert(len(one_bouts) == 7)

    # Their length is 1, 2, 3, .. 7
    assert(Bout.total_time(one_bouts) == timedelta(minutes=7+6+5+4+3+2+1))

    # Keeping bouts >= i minutes means there should be 7-(i-1) left
    for i in range(1,7):
        i_or_longer = Bout.limit_to_lengths(one_bouts, min_length=timedelta(minutes=i))
        assert(len(i_or_longer) == 7-(i-1))

    # One manual check
    three_or_longer = Bout.limit_to_lengths(one_bouts, min_length=timedelta(minutes=3))
    assert(len(three_or_longer) == 5)

    # This should exclude the 1 bout at exactly 3 minutes
    three_plus_bit_or_longer = Bout.limit_to_lengths(one_bouts, min_length=timedelta(minutes=3, seconds=1))
    assert(len(three_plus_bit_or_longer) == 4)

    # No bouts should be this long
    eight_or_longer = Bout.limit_to_lengths(one_bouts, min_length=timedelta(minutes=8))
    assert(len(eight_or_longer) == 0)

    # There is nothing above 1 in the file, should be 0 bouts
    two_bouts = counts.bouts(2,989)
    assert(len(two_bouts) == 0)

test_bouts.setup = setup_func
test_bouts.teardown = teardown_func
