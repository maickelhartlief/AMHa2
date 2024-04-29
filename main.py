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

if __name__ == "__main__":
    models = []
    if sys.argv[1] in ['both', 'AMHa']:
        models.append((AMHa, 'AMHa'))
    if sys.argv[1] in ['both', 'AMHa2']:
        models.append((AMHa2, 'AMHa2'))

    use_s = []
    if sys.argv[2] in ['both', 'without']:
        use_s.append(False)
    if sys.argv[2] in ['both', 'with']:
        use_s.append(True)

    # import model_independent data
    print('importing data from file...', end = '\r')
    data = pd.read_csv('data/data.csv', converters = {'links': pd.eval})
    
    for Model, name, in models:
        
        # import model-dependent data
        transitions = pd.read_csv(f'data/transitions_{name}.csv')
        s = pd.read_csv(f'data/s_{name}.csv')
        for use_network in use_s:
            print(f'running {name} with{"" if use_network else "out"} social factors...')
            
            # initialize model
            model = Model(data,
                          transitions, 
                          s,
                          use_network = use_network)
            # run model
            model.run()

            # show results
            model.plot_results()
            model.report_tracked()

