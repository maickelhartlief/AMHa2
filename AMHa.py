####################
# description:
####################

# external imports
import networkx as nx
from matplotlib import pyplot as plt
import pickle
import os.path
import pandas as pd
import seaborn as sns
from math import sqrt
# ignore warnings (not good practise, but there are a lot of spammy deprecated 
# warnings and other things that are not applicable and clutter the output)
import warnings
warnings.filterwarnings("ignore")

# internal imports
from config import *

class AMHa():

    def __init__(self, 
                 data,
                 transitions,
                 s,
                 pops = drinker_types,
                 update_attribute = 'drinker_type',
                 n_runs = 5,
                 step_start = 2,
                 n_steps = 28,
                 use_network = True,
                 data_matching = False,
                 start_type = 'data'):
        self.name = 'AMHa'
        self.data_matching = data_matching
        self.data = data
        self.step_start = step_start
        self.start_type = start_type
        self.update_attribute = update_attribute
        self.g = self.initialize_network()
        self.wave_pops = data[['wave', 'drinker_type', 'id']].groupby(['wave', 'drinker_type']).count().reset_index().rename(columns = {'id' : 'count'})
        self.total_per_wave = self.wave_pops[['wave', 'count']].groupby('wave').sum()['count'].tolist()

        # intialize results by generating entire dataframe
        self.results = pd.DataFrame({'run' : [list(range(n_runs))], 
                        'step' : [list(range(step_start, step_start + n_steps + 1))],
                        'drinker_type' : [pops],
                        'count' : -1})
        self.results = self.results.explode('run')
        self.results = self.results.explode('step')
        self.results = self.results.explode('drinker_type')

        self.errors = [0] * max(n_steps, 6)
        self.errors_ci = [0] * max(n_steps, 6)
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.transitions = transitions
        self.s = s
        self.pops = pops
        self.use_network = use_network

        # declare empty modifier settings. initializing is done in apply_intervention()
        self.modifier_level = ''
        self.modifier = ''
        self.modifier_type = ''

        # does not show which waves are missing from true data if there are any missing
        self.lapses = {'relapses' : 0, 'first_time_lapses' : 0, 'ids' : []}


    def initialize_network(self, verbose = True):
        if os.path.isfile(f'data/network_start_wave_{self.step_start}') and self.start_type == 'data':
            # load graph object from file
            if verbose:
                print('loading graph from file...                   ', end = '\r')
            return pickle.load(open(f'data/network_start_wave_{self.step_start}', 'rb'))
        if verbose:
            print('loading graph from data...                       ', end = '\r')

        # remake wave edge_data based on remaining data
        wave_start = self.data[(self.data['wave'] == self.step_start)][['id', 'links', 'drinker_type', 'drinker_type_prev']].dropna(subset = ['drinker_type_prev'])
        wave_start['from'] = wave_start['drinker_type_prev'] + wave_start['drinker_type']
        nodes = wave_start['id'].tolist()
        edges_raw = wave_start[['id', 'links']].explode('links').rename(columns = {'id' : 'from', 'links' : 'to'})
        edges = edges_raw.dropna().drop_duplicates()
        edges = edges.drop(edges[~edges['to'].isin(nodes)].index)

        # get nodes without edges
        non_lonely_nodes = set(edges['from'].tolist())
        all_nodes = set(nodes)
        lonely_nodes = list(all_nodes - non_lonely_nodes)

        # create initial model from edge data of wave 1
        g = nx.from_pandas_edgelist(edges, 'from', 'to', create_using = nx.DiGraph())
        g.add_nodes_from(lonely_nodes)

        # set node attributes:
        nx.set_node_attributes(g, {
            node : {
                'drinker_type' : wave_start[wave_start['id'] == node]['drinker_type'].iloc[0] if self.start_type == 'data' else self.start_type, 
                'drinker_type_min1' : wave_start[wave_start['id'] == node]['from'].iloc[0] if self.start_type == 'data' else self.start_type + self.start_type,
                'has_been_H' : 'H' in wave_start[wave_start['id'] == node]['from'].iloc[0]
            } for node in g.nodes
        })
        
        # save graph object to file
        if self.start_type == 'data':
            pickle.dump(g, open(f'data/network_start_wave_{self.step_start}', 'wb'))

        return g


    def apply_intervention(self, modifier = 1, level = 'personal', drinker_type = 'A', history = ''):
        self.modifier = modifier
        self.modifier_level = level
        self.modifier_type = drinker_type
        if level == 'social':
            # modify
            self.s.loc[self.s['link'] == history + self.modifier_type, 'ratio'] *= self.modifier
            mod_type = f'influence of {history}'
            # normalize
            self.s = normalize_s(self.s)
        elif level == 'personal':
            # modify
            from_type = history + ('H' if self.modifier_type == 'A' else 'A')
            self.transitions.loc[(self.transitions['to'] == self.modifier_type) & (self.transitions['from'] == from_type), 'ratio'] *= self.modifier
            # normalize
            condition = self.transitions['from'] == from_type
            self.transitions.loc[condition, 'ratio'] /= sum(self.transitions[condition]['ratio'])
            mod_type = f'{from_type}->'
        print(f'modified {self.modifier_level} {mod_type}{self.modifier_type} by factor of {self.modifier}')


    def run(self):
        for run in range(self.n_runs):
            # intialize network if not done in object initialization yet
            if run > 0:
                self.g = self.initialize_network()

            # simulate
            self.report_step(self.step_start, run)
            for step in range(self.step_start, self.step_start + self.n_steps):
                # take data as starting point for each step
                if self.data_matching:
                    step_start = self.step_start
                    self.step_start = step
                    self.g = self.initialize_network(verbose = False)
                    self.step_start = step_start
                    
                for node in self.g.nodes:
                    self.g.nodes[node][self.update_attribute] = self.update_node(node)
                self.report_step(step + 1, run)
    

    # update state of node stochastically
    def update_node(self, node):
        # get current state
        cur_type = self.g.nodes[node][self.update_attribute]
        # get transition ratios based on personal factors
        ratios = self.transitions[self.transitions['from'] == cur_type][['ratio', 'to']]
        if self.use_network:
            # modify transition ratios based on social factors    
            ratios = self.adapt_for_network(node, cur_type[-1], ratios)
        
        # pick next state
        next_type = np.random.choice(ratios['to'], 1, p = ratios['ratio'])[0]
        
        # track lapse
        if next_type == 'H':
            
            # mark lapse or relapse
            if cur_type[-1] == 'A':
                self.lapses['relapses' if self.g.nodes[node]['has_been_H'] else 'first_time_lapses'] += 1
                
            # remember H to distinguish later lapse and relapse
            self.g.nodes[node]['has_been_H'] = True

        return next_type


    def adapt_for_network(self, node, cur_type, ratios):
        # modify ratio with social influence
        ratios = ratios.set_index('to')
        for link_type in [self.g.nodes[edge[1]][self.update_attribute] for edge in self.g.edges(node)]:
            ratios = ratios.add(self.s[(self.s['link'] == link_type) & (self.s['from'] == cur_type)][['to', 'ratio']].set_index('to'))

        # counteracht possible negative probabilities
        while (ratios['ratio'] < 0).any():
            negative = ratios['ratio'].min()
            ratios['ratio'] -= negative
            ratios['ratio'] /= ratios['ratio'].sum()
            
        return ratios.reset_index()


    def report_step(self, step, run):
        print(f'simulating...    (run: {run + 1} / {self.n_runs},    step: {step - self.step_start + 1} / {self.n_steps})', end = '\r')
        # report and save stats
        stats = list(nx.get_node_attributes(self.g, self.update_attribute).values())
        for drinker_type in self.pops:
            stat = stats.count(drinker_type)
            self.results.loc[(self.results['run'] == run)
                           & (self.results['step'] == step)
                           & (self.results['drinker_type'] == drinker_type), 'count'] = stat
           

    def plot_true(self, drinker_type):
        true_pops = self.wave_pops[self.wave_pops['drinker_type'] == drinker_type]['count'].tolist()
        true_pops = [true_pop / total for true_pop, total in zip(true_pops, self.total_per_wave)]
        plt.scatter(list(range(1, 8)), 
                    true_pops, 
                    label = drinker_type, 
                    color = colors[drinker_type])
        return true_pops

    
    def report_tracked(self, results = None):
        # report observed lapses and relapses
        # NOTE: return last_wave does not work for data matching
        # account for possible child class overwrite of results
        if results is None:
            results = self.results

        total_steps = self.n_steps * self.n_runs
        print(f"{self.lapses['relapses'] / total_steps } relapses and {self.lapses['first_time_lapses'] / total_steps} first time lapses found per step on average!")
        last_wave = results[results['step'] == results['step'].max()]
        last_wave['count'] /= self.total_per_wave[self.step_start]
        return self.lapses, last_wave


    def plot_results(self, results = None, name_addition = ''):
        # account for possible child class overwrite of results
        if results is None:
            results = self.results
        
        # plot line per drinker type
        for drinker_type in drinker_types:
            # format results
            cur_results = results[results['drinker_type'] == drinker_type]
            if self.data_matching:
                for step in cur_results['step'].unique():
                    cur_results.loc[cur_results['step'] == step, 'count'] /= self.total_per_wave[max(int(step) - 2, 2)]
            else:
                cur_results['count'] /= self.total_per_wave[self.step_start]
            x = list(range(self.step_start, self.n_steps + self.step_start + 1))
            y = np.array([np.mean(cur_results[cur_results['step'] == step]['count']) for step in x])
            y_stds = [np.std(cur_results[cur_results['step'] == step]['count']) for step in x]
            ci = np.array([1.96 * y_std / sqrt(self.n_runs) for y_std in y_stds])
            
            # plot true
            true_pops = self.plot_true(drinker_type)
            
            # plot predicted
            if self.data_matching: # plot from each true point to the step after.
                for step in range(self.n_steps):
                    plt.plot(x[step : step + 2], 
                             [true_pops[step + 1], y[step + 1]], 
                             color = colors[drinker_type])
                    plt.fill_between(x[step : step + 2], 
                                     [true_pops[step + 1], (y - ci)[step + 1]], 
                                     [true_pops[step + 1], (y + ci)[step + 1]], 
                                     color = colors[drinker_type], alpha = 0.3)
            else: # regular plotting
                plt.plot(x, y, color = colors[drinker_type])
                plt.fill_between(x, (y - ci), (y + ci), color = colors[drinker_type], alpha = 0.3)
         
            # plot error per wave per category
            x = [wave for wave in x if 2 <= wave <= 7]
            plt.vlines(x = x, 
                       ymin = y[min(x) - self.step_start:max(x) - self.step_start + 1], 
                       ymax = true_pops[min(x) - 1:max(x)], 
                       color = colors[drinker_type],
                       linestyles = 'dashed')

        # calculate error
        true_pops = {}
        for drinker_type in drinker_types:
            true_pops[drinker_type] = self.wave_pops[self.wave_pops['drinker_type'] == drinker_type]['count'].tolist()
            true_pops[drinker_type] = [true_pop / total for true_pop, total in zip(true_pops[drinker_type], self.total_per_wave)]
        errors = {}
        for _, result in results[results['step'] < 8].iterrows():
            # get error of this prediction
            condition = (self.wave_pops['drinker_type'] == result['drinker_type']) & (self.wave_pops['wave'] == result['step'])
            cur_error = (result['count'] / self.total_per_wave[self.step_start] - true_pops[result['drinker_type']][int(result['step']) - 1]) ** 2
            self.errors[int(result['step']) - 2] += cur_error / self.n_runs
            # save individual error for ci later
            if result['step'] not in errors.keys():
                errors[result['step']] = []
            errors[result['step']].append(cur_error)
        # calculate error ci
        for error_key, error_value in errors.items():
            self.errors_ci[int(error_key) - 2] = 1.96 * np.std(error_value) / sqrt(self.n_runs)
        self.errors[0] = 0
        self.errors_ci[0] = 0
       
        plt.ylabel('number of people')
        plt.xlabel('wave')
        plt.legend()
        plt.savefig(f'results/{self.name}{name_addition}')   
        plt.clf()  