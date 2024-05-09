####################
# description:
# 
####################

# external imports
import pandas as pd
from matplotlib import pyplot as plt

# internal imports
from config import *
from AMHa import AMHa
from AMHa2 import AMHa2

# import model_independent data
print('importing data from file...', end = '\r')
data = pd.read_csv('data/data.csv', converters = {'links': pd.eval})

errors = []
errors_ci = []
for Model, name in [(AMHa, 'AMHa'), (AMHa2, 'AMHa2')]:
    
    # import model-dependent data
    transitions = pd.read_csv(f'data/transitions_{name}.csv')
    s = pd.read_csv(f'data/s_{name}.csv')
    s = normalize_s(s)

    print(f'running data matching test on {name}...               ')
    
    # initialize model
    model = Model(data, transitions, s,
                  n_steps = 5, data_matching = True)
    # run model
    model.run()

    # show results
    model.plot_results(name_addition = '_data_matching')
    errors.append(np.array(model.errors))
    errors_ci.append(np.array(model.errors_ci))

# plot errors for both models in 1 plot
for error, ci, color, name in zip(errors, errors_ci, ['lightblue', 'brown'], ['AMHa', 'AMHa2']):
    x = range(2, len(error) + 2)
    plt.plot(x, error, color = color, label = name)
    plt.fill_between(x, (error - ci), (error + ci), color = color, alpha = 0.2)

plt.legend()
plt.title('total error of predictions per wave')
plt.ylabel('ratio of people')
plt.xlabel('wave')
plt.savefig(f'results/data_matching_errors')  
plt.clf() 

