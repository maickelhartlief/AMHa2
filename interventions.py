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

if __name__ == "__main__":
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
            for Model, name, in [(AMHa, 'AMHa'), (AMHa2, 'AMHa2')]:
                            
                # import model-dependent data
                transitions = pd.read_csv(f'data/transitions_{name}.csv')
                s = pd.read_csv(f'data/s_{name}.csv')
                print(f'running intervention test on {name}...               ')

                endemic_H = pd.DataFrame(columns = ['modifier', 'H', 'ci'])
                modifiers = np.concatenate((np.arange(.1, .99, .1), np.arange(1, 2.099, .2)))
                for modifier in modifiers:
                    # initialize model
                    model = Model(data,
                                  transitions, 
                                  s)

                    model.apply_intervention(modifier = modifier, level = intervention, drinker_type = modifier_type)

                    # run model
                    model.run()

                    # show results
                    #model.plot_results(name_addition = f'_intervention_{intervention}_{modifier_type}_{modifier}')
                    lapses, last_wave = model.report_tracked()
                    last_wave_H = last_wave[last_wave['drinker_type'] == 'H']['count']
                    
                    # make count into ratio...
                    last_wave_H /= model.total_per_wave[1]


                    ci = 1.96 * np.std(last_wave_H) / sqrt(model.n_runs)
                    endemic_H.loc[len(endemic_H.index)] = [modifier, sum(last_wave_H), ci]
                x = endemic_H['modifier'].to_numpy()
                y = endemic_H['H'].to_numpy()
                ci = endemic_H['ci'].to_numpy()
                plt.plot(x, y, color = 'brown')
                plt.fill_between(x, (y - ci), (y + ci), color = 'brown', alpha = 0.3)
                plt.xscale("log")
                ticks = [.1, .5, 1, 2]
                plt.xticks(ticks, labels = ticks)
                plt.ylabel('ratio H')
                plt.xlabel('modifier')
                plt.title(f'ratio of heavy drinkers in an endemic state (step 30) \nfor different {intervention} transition modifiers towards {modifier_type}')
                plt.savefig(f'results/{name}_endemic_H_intervention_{intervention}_{modifier_type}')   
                plt.clf()  


