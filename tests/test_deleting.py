
from pampro import data_loading, Time_Series, Channel, Bout, Bout_Collection, channel_inference
from datetime import datetime, timedelta


ts, header, counts = False, False, False

def setup_func():
    global ts
    global header
    global counts
    ts, header = data_loading.load("_data/testfile20.dat", "Actigraph", datetime_format="%d/%m/%Y")
    counts = ts.get_channel("AG_Counts")

    #print(counts.data)

def teardown_func():
    pass


def test_a():
    # Case A
    # Both timestamps preceed data

    origin = counts.timestamps[0]

    start = origin - timedelta(days=2)
    end = origin - timedelta(days=1)

    # Summarise the data before deletion
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # All values should be identical, loop through them and assert equality
    suffixes = "sum n missing 0_0 0_1 1_1".split(" ")

    for suffix in suffixes:
        assert(summary_before.get_channel("AG_Counts_" + suffix).data[0] == summary_after.get_channel("AG_Counts_" + suffix).data[0])



def test_b():
    # Case B
    # First timestamp preceeds data, second doesn't

    origin = counts.timestamps[0]

    start = origin - timedelta(hours=12)
    end = origin + timedelta(hours=12)

    # Summarise the data before deletion
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # n should go down and missing should go up
    assert(summary_before.get_channel("AG_Counts_n").data[0] > summary_after.get_channel("AG_Counts_n").data[0])
    assert(summary_before.get_channel("AG_Counts_missing").data[0] < summary_after.get_channel("AG_Counts_missing").data[0])

    # Should only be 12 hours left
    assert(summary_after.get_channel("AG_Counts_n").data[0] == 12*60)

    # And 12 hours missing
    assert(summary_after.get_channel("AG_Counts_missing").data[0] == 12*60)


def test_c():
    # Case C
    # Both timestamps inside data
    origin = counts.timestamps[0]

    start = origin + timedelta(hours=6)
    end = origin + timedelta(hours=7)

    # Summarise the data before deletion
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # n should go down and missing should go up
    assert(summary_before.get_channel("AG_Counts_n").data[0] > summary_after.get_channel("AG_Counts_n").data[0])
    assert(summary_before.get_channel("AG_Counts_missing").data[0] < summary_after.get_channel("AG_Counts_missing").data[0])

    # Should only be 23 hours left
    assert(summary_after.get_channel("AG_Counts_n").data[0] == 23*60)

    # And 1 hours missing
    assert(summary_after.get_channel("AG_Counts_missing").data[0] == 1*60)

def test_d():
    # Case D
    # First timestamp inside, second timestamp succeeds data

    origin = counts.timestamps[-1]+timedelta(minutes=1)

    start = origin - timedelta(hours=12)
    end = origin + timedelta(hours=12)

    # Summarise the data before deletion
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # n should go down and missing should go up
    assert(summary_before.get_channel("AG_Counts_n").data[0] > summary_after.get_channel("AG_Counts_n").data[0])
    assert(summary_before.get_channel("AG_Counts_missing").data[0] < summary_after.get_channel("AG_Counts_missing").data[0])

    # Should only be 12 hours left
    assert(summary_after.get_channel("AG_Counts_n").data[0] == 12*60)

    # And 12 hours missing
    assert(summary_after.get_channel("AG_Counts_missing").data[0] == 12*60)


def test_e():
    # Case E
    # Both timestamps succeed data

    origin = counts.timestamps[-1]

    start = origin + timedelta(days=1)
    end = origin + timedelta(days=2)

    # Summarise the data before deletion
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after = Time_Series.Time_Series("")
    summary_after.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # All values should be identical, loop through them and assert equality
    suffixes = "sum n missing 0_0 0_1 1_1".split(" ")

    for suffix in suffixes:
        assert(summary_before.get_channel("AG_Counts_" + suffix).data[0] == summary_after.get_channel("AG_Counts_" + suffix).data[0])

def test_f():
    # Case F
    # Multiple deletions producing consistent results

    origin = counts.timestamps[0]

    # Delete first 2 hours
    start = origin
    end = origin + timedelta(hours=2)

    # Summarise the data before deletion
    summary_before = Time_Series.Time_Series("")
    summary_before.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after_a = Time_Series.Time_Series("")
    summary_after_a.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # Delete midday to 2pm
    start = origin + timedelta(hours=12)
    end = origin + timedelta(hours=14)

    counts.delete_windows([Bout.Bout(start,end)])

    # Summarise the data after deletion
    summary_after_b = Time_Series.Time_Series("")
    summary_after_b.add_channels(counts.summary_statistics(statistics=[("generic", ["sum", "n", "missing"]),("cutpoints", [[0,0],[0,1],[1,1]])]))

    # 20 hours left
    assert(summary_after_b.get_channel("AG_Counts_n").data[0] == 20*60)

    # 4 hours missing
    assert(summary_after_b.get_channel("AG_Counts_missing").data[0] == 4*60)

    # Sum data should be 20 1s
    assert(summary_after_b.get_channel("AG_Counts_sum").data[0] == 20*60)


for t in [test_a, test_b, test_c, test_d, test_e, test_f]:
    t.setup = setup_func
    t.teardown = teardown_func
