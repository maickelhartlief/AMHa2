####################
# description:
# TODO: make it run several times to acount for stochasticity...
####################

# external imports
from matplotlib import pyplot as plt
import os.path
import pandas as pd

# ignore warnings (not good practise, but there are a lot of spammy deprecated 
# warnings and other things that are not applicable and clutter the output)
import warnings
warnings.filterwarnings("ignore")

# internal imports
from config import *
from AMHa import AMHa

class AMHa2(AMHa):

    # initialize same as parent
    def __init__(self, 
                 data,
                 transitions, 
                 s,
                 pops = [drinker_type_2 + drinker_type_1 for drinker_type_1 in drinker_types for drinker_type_2 in drinker_types],
                 update_attribute = 'drinker_type_min1',
                 n_runs = 5,
                 step_start = 2,
                 n_steps = 28,
                 use_network = True,
                 data_matching = False,
                 track_ids = []):
        super().__init__(data,
                         transitions, 
                         s, 
                         pops = pops, 
                         update_attribute = update_attribute, 
                         n_runs = n_runs,
                         step_start = step_start,
                         n_steps = n_steps,
                         use_network = use_network,
                         data_matching = data_matching,
                         track_ids = track_ids)

 
    def apply_intervention(self, modifier = 1, level = 'personal', drinker_type = 'A', history = ''):
        for history in drinker_types:
            super().apply_intervention(modifier = modifier, level = level, drinker_type = drinker_type, history = history)

    def update_node(self, node):
        # handle shifting the 'previous observation' part of the sequence and use AMHa method for the rest
        return self.g.nodes[node][self.update_attribute][-1] + super().update_node(node)

    def plot_results(self, name = 'AMHa2', name_addition = ''):
        ## sns has version issues and my laptop refuses to update so i am forced to do this is the most roundabout way ever....
        
        # convert results into 3-category format
        results = self.results
        results['drinker_type'] = results['drinker_type'].apply(lambda seq : seq[-1])
        stupid_list = []
        for run in results['run'].unique().tolist():
            for step in results['step'].unique().tolist():
                dumb_partition = results[(results['run'] == run) & (results['step'] == step)]
                stupid_list.append(dumb_partition.groupby('drinker_type').agg({'run': 'mean', 'step': 'mean', 'count' : 'sum'}).reset_index())
        results = pd.concat(stupid_list)
        
        # use plotting like in AMHa
        super().plot_results(name = name, results = results, name_addition = name_addition)