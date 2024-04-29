####################
# description:
# 
####################

# external imports
import pandas as pd

# internal imports
from config import *
from AMHa2 import AMHa2


def amha2():
    model = AMHa2(pd.read_csv('data/transitions_AMHa2.csv'), 
                  pd.read_csv('data/s_AMHa2.csv'),
                  track_ids = [25075, 10546, 14675, 8588])
    model.run()
    model.plot_results()
    model.report_tracked()

amha2()