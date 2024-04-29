####################
# description:
# 
####################

# external imports
import pandas as pd

# internal imports
from config import *
from AMHa import AMHa


def amha():
    model = AMHa(pd.read_csv('data/transitions_AMHa.csv'), 
                 pd.read_csv('data/s_AMHa.csv'),
                 track_ids = [25075, 10546, 14675, 8588])
    model.run()
    model.plot_results()
    model.report_tracked()