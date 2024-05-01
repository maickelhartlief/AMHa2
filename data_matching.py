####################
# description:
# 
####################

# external imports
import pandas as pd

# internal imports
from config import *
from AMHa import AMHa
from AMHa2 import AMHa2

if __name__ == "__main__":
    # import model_independent data
    print('importing data from file...', end = '\r')
    data = pd.read_csv('data/data.csv', converters = {'links': pd.eval})
    
    errors = []
    for Model, name, in [(AMHa, 'AMHa'), (AMHa2, 'AMHa2')]:
        
        # import model-dependent data
        transitions = pd.read_csv(f'data/transitions_{name}.csv')
        s = pd.read_csv(f'data/s_{name}.csv')
        print(f'running data matching test on {name}...               ')
        
        # initialize model
        model = Model(data,
                      transitions, 
                      s,
                      n_steps = 5,
                      data_matching = True)
        # run model
        model.run()

        # show results
        model.plot_results(name_addition = '_data_matching')
        errors.append(model.errors)

    # plot errors for both models in 1 plot
    # TODO: implement this.

