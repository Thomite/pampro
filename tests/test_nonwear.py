
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection, channel_inference
from datetime import datetime, timedelta
import os

ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile18.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass


def test_nonwear_amount():

    # File contains 24 hours of 1s, then 15 hours of 0s, then 9 hours of 1s, then 24 hours of 1s

    nonwear_bouts, wear_bouts = channel_inference.infer_nonwear_actigraph(counts)

    # There is 1 nonwear bout and 2 wear bouts surrounding it
    assert(len(nonwear_bouts) == 1)
    assert(len(wear_bouts) == 2)

    Bout.cache_lengths(nonwear_bouts)
    Bout.cache_lengths(wear_bouts)

    nw_bout = nonwear_bouts[0]

    # The nonwear bout is 15 hours long
    assert(nw_bout.length == timedelta(hours=15))

    # Summarise the data before deleting the nonwear
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # Number of 1s = 24 hours then 9 hours then 24 hours
    assert(summary_before.get_channel("AG_Counts_sum").data[0] == (24+9+24)*60)

    # 15 hours of 0s
    assert(summary_before.get_channel("AG_Counts_0_0").data[0] == 15*60)

    # Sum should = number of 1s
    assert(summary_before.get_channel("AG_Counts_1_1").data[0] == (24+9+24)*60)

    # n should be 3 days = 1440*3 = 24*3*60
    assert(summary_before.get_channel("AG_Counts_n").data[0] == 24*3*60)

    # Missing should be 0
    assert(summary_before.get_channel("AG_Counts_missing").data[0] == 0)

    counts.delete_windows(nonwear_bouts)

    # Summarise the data after deleting the nonwear
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # Sum shouldn't have changed
    assert(summary_after.get_channel("AG_Counts_sum").data[0] == (24+9+24)*60)

    # All the 0s were nonwear, so there should now be no 0s
    assert(summary_after.get_channel("AG_Counts_0_0").data[0] == 0)

    # And the number of 1s shouldn't have changed
    assert(summary_after.get_channel("AG_Counts_1_1").data[0] == (24+9+24)*60)

    # n should have reduced by 15 hours = 15*60
    assert(summary_after.get_channel("AG_Counts_n").data[0] == (24+9+24)*60)

    # missing should have gone up by 15 hours = 15*60
    assert(summary_after.get_channel("AG_Counts_missing").data[0] == 15*60)

test_nonwear_amount.setup = setup_func
test_nonwear_amount.teardown = teardown_func


def test_invalid_days():

    # File contains 24 hours of 1s, then 15 hours of 0s, then 24 hours of 1s

    nonwear_bouts, wear_bouts = channel_inference.infer_nonwear_actigraph(counts)

    # Get invalid/valid days where valid criterion is default (10 hours)
    invalid_windows, valid_windows = channel_inference.infer_valid_days(counts, wear_bouts)

    # One of the days should be invalid
    assert(len(invalid_windows) == 1)

    # The invalid bout should be exactly 1 day long
    assert(invalid_windows[0].length == timedelta(days=1))

    # Two valid days
    assert(len(valid_windows) == 2)

    # Now get invalid/valid windows where need only 9 hours
    invalid_windows, valid_windows = channel_inference.infer_valid_days(counts, wear_bouts, valid_criterion=timedelta(hours=9))

    # None of the days should be invalid
    assert(len(invalid_windows) == 0)

    # Three valid days
    assert(len(valid_windows) == 3)

test_invalid_days.setup = setup_func
test_invalid_days.teardown = teardown_func


def test_nonwear_positions():

    # Case 1: Nonwear at very beginning of file
    ts1, header1 = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile23.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts1 = ts1.get_channel("AG_Counts")
    nonwear_bouts1, wear_bouts1 = channel_inference.infer_nonwear_actigraph(counts1)

    # Case 2: Nonwear in middle of file
    ts2, header2 = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile24.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts2 = ts2.get_channel("AG_Counts")
    nonwear_bouts2, wear_bouts2 = channel_inference.infer_nonwear_actigraph(counts2)

    # Case 3: Nonwear at very end of file
    ts3, header3 = data_loading.load(os.path.abspath(__file__).replace(os.path.basename(__file__), "") + "_data/testfile25.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts3 = ts3.get_channel("AG_Counts")
    nonwear_bouts3, wear_bouts3 = channel_inference.infer_nonwear_actigraph(counts3)

    # They should all have the same duration of wear & nonwear
    assert(Bout.total_time(nonwear_bouts1) == timedelta(hours=2))
    assert(Bout.total_time(nonwear_bouts1) == Bout.total_time(nonwear_bouts2))
    assert(Bout.total_time(nonwear_bouts1) == Bout.total_time(nonwear_bouts3))
    assert(Bout.total_time(wear_bouts1) == Bout.total_time(wear_bouts2))
    assert(Bout.total_time(wear_bouts1) == Bout.total_time(wear_bouts3))

    # Delete the relevant nonwear bouts from each channel
    counts1.delete_windows(nonwear_bouts1)
    counts2.delete_windows(nonwear_bouts2)
    counts3.delete_windows(nonwear_bouts3)

    # Total data should be equal
    assert(sum(counts1.data) == sum(counts2.data))
    assert(sum(counts1.data) == sum(counts3.data))

    # Summary level mean should also be the same
    s1 = counts1.summary_statistics()[0]
    s2 = counts2.summary_statistics()[0]
    s3 = counts3.summary_statistics()[0]
    assert(s1.data[0] == s2.data[0])
    assert(s1.data[0] == s3.data[0])
