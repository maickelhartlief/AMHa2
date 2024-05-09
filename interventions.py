####################
# description:
# 
####################

# external imports
import pandas as pd
import sys
from math import sqrt
from matplotlib import pyplot as plt

# internal imports
from config import *
from AMHa import AMHa
from AMHa2 import AMHa2


# check whether to run personal, social, or both modifiers
interventions = []
if len(sys.argv) < 3:
    print('This program needs 2 command line arguments: intervention [personal, social, both] and modifier type [A, H, both]')

if sys.argv[1] in ['both', 'personal']:
    interventions.append('personal')
if sys.argv[1] in ['both', 'social']:
    interventions.append('social')

# check whether to run intervention on abstainer or heavy drinker influence
modifier_types = []
if sys.argv[2] in ['both', 'A']:
    modifier_types.append('A')
if sys.argv[2] in ['both', 'H']:
    modifier_types.append('H')

# import model_independent data
print('importing data from file...', end = '\r')
data = pd.read_csv('data/data.csv', converters = {'links': pd.eval})

for intervention in interventions:
    for modifier_type in modifier_types:

        for Model, name, color in [(AMHa, 'AMHa', 'lightblue'), (AMHa2, 'AMHa2', 'brown')]:
                        
            # import model-dependent data
            transitions = pd.read_csv(f'data/transitions_{name}.csv')

            # import and normalize s
            s = normalize_s(pd.read_csv(f'data/s_{name}.csv'))        
            
            modifiers = np.concatenate((np.arange(.1, .99, .1), np.arange(1, 2.099, .2)))


            print(f'running {intervention} {modifier_type} intervention test on {name}...               ')

            endemic_H = pd.DataFrame(columns = ['modifier', 'H', 'ci'])
            
            for modifier in modifiers:

                # currently on this level to investigate resetting of rates afterapplying intervention
                if intervention == 'personal':
                    transitions = pd.read_csv(f'data/transitions_{name}.csv')
                elif intervention == 'social':
                    s = normalize_s(pd.read_csv(f'data/s_{name}.csv'))

                # initialize model
                model = Model(data, transitions, s)

                # prepare model for interventions
                model.apply_intervention(modifier = modifier, level = intervention, drinker_type = modifier_type)

                # run model
                model.run()

                # show results
                lapses, last_wave = model.report_tracked()
                last_wave_H = last_wave[last_wave['drinker_type'] == 'H']['count']

                ci = 1.96 * np.std(last_wave_H) / sqrt(model.n_runs)
                endemic_H.loc[len(endemic_H.index)] = [modifier, np.mean(last_wave_H), ci]
            
            x = endemic_H['modifier'].to_numpy()
            y = endemic_H['H'].to_numpy()
            ci = endemic_H['ci'].to_numpy()
            
            print(f'plotting {name}...')
            plt.plot(x, y, color = color, label = name)
            plt.fill_between(x, (y - ci), (y + ci), color = color, alpha = 0.2)
            
        plt.xscale("log")
        ticks = [.1, .5, 1, 2]
        plt.xticks(ticks, labels = ticks)
        
        plt.legend()
        plt.ylabel('ratio H')
        plt.xlabel('modifier')
        plt.title(f'ratio of heavy drinkers in an endemic state (step 30) \nfor different {intervention} transition modifiers towards {modifier_type}')
        plt.savefig(f'results/endemic_H_intervention_{intervention}_{modifier_type}')   
        plt.clf()  