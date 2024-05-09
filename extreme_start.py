####################
# description:
# 
####################

# external imports
import sys
import pandas as pd

# internal imports
from config import *
from AMHa import AMHa
from AMHa2 import AMHa2

if len(sys.argv) < 2:
    print('This program needs 1 command line argument: drinker_type [A, M, H] (can be a sequence of multiple)')

start_types = list(sys.argv[1])

# import model_independent data
print('importing data from file...', end = '\r')
data = pd.read_csv('data/data.csv', converters = {'links': pd.eval})

for Model, name, in [(AMHa, 'AMHa'), (AMHa2, 'AMHa2')]:
    
    # import model-dependent data
    transitions = pd.read_csv(f'data/transitions_{name}.csv')
    s = pd.read_csv(f'data/s_{name}.csv')
    s = normalize_s(s)
    
    for start_type in start_types:       
            print(f'running {name} starting with only {start_type}...')
            
            # initialize model
            model = Model(data, transitions, s,
                          start_type = start_type)
            
            # run model
            model.run()

            # show results
            model.plot_results(name_addition = f'_only_{start_type}')
            model.report_tracked()