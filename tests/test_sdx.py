
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile22.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

def teardown_func():
    pass

def test_sdx():

    #File consists of bouts of 1s in the following lengths:
    # 1 1 2 2 3 4 5 5 6 6 7 7 8 8 8 8 8 8 9 10 11 11 12

    one_bouts = counts.bouts(1,1)

    # 23 bouts
    assert(len(one_bouts) == 23)

    bc = Bout_Collection.Bout_Collection("1_bouts", bouts=one_bouts)

    summary_ts = Time_Series.Time_Series("")
    summary_ts.add_channels(bc.summary_statistics(statistics=[("generic", ["sum", "n"]), ("sdx", list(range(10,100,10)))]))

    #for c in summary_ts.channels:
    #    print(c.name, c.data[0])
    # Correct SD(x) values independently determined from above bouts
    assert(summary_ts.get_channel("1_bouts_sd10").data[0] == 5)
    assert(summary_ts.get_channel("1_bouts_sd20").data[0] == 6)
    assert(summary_ts.get_channel("1_bouts_sd30").data[0] == 7)
    assert(summary_ts.get_channel("1_bouts_sd40").data[0] == 8)
    assert(summary_ts.get_channel("1_bouts_sd50").data[0] == 8)
    assert(summary_ts.get_channel("1_bouts_sd60").data[0] == 8)
    assert(summary_ts.get_channel("1_bouts_sd70").data[0] == 9)
    assert(summary_ts.get_channel("1_bouts_sd80").data[0] == 11)
    assert(summary_ts.get_channel("1_bouts_sd90").data[0] == 11)

    # bout collection still contains 23 bouts
    assert(summary_ts.get_channel("1_bouts_n").data[0] == 23)

    # total time in bouts should be 150 minutes (150*60 = 9000 in seconds)
    assert(summary_ts.get_channel("1_bouts_sum").data[0] == 150*60)

test_sdx.setup = setup_func
test_sdx.teardown = teardown_func
