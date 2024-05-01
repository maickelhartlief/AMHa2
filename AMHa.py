####################
# description:
# TODO: make it run several times to acount for stochasticity...
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
                 track_ids = []):
        self.data_matching = data_matching
        self.data = data
        self.step_start = step_start
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
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.transitions = transitions
        self.s = s
        self.pops = pops
        self.update_attribute = update_attribute
        self.use_network = use_network

        # declare empty modifier settings. initializing is done in apply_intervention()
        self.modifier_level = ''
        self.modifier = ''
        self.modifier_type = ''

        # does not show which waves are missing from true data if there are any missing
        self.trackers = {track_id : {'true' : data[data['id'] == track_id]['drinker_type'].tolist()[1:], 'predicted' : []} for track_id in track_ids}
        self.lapses = {'relapses' : 0, 'first_time_lapses' : 0, 'ids' : []}


    def initialize_network(self, verbose = True):
        if os.path.isfile(f'data/network_start_wave_{self.step_start}'):
            # load graph object from file
            if verbose:
                print('loading graph from file...                   ', end = '\r')
            return pickle.load(open('data/network_start', 'rb'))
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
                'drinker_type' : wave_start[wave_start['id'] == node]['drinker_type'].iloc[0], 
                'drinker_type_min1' : wave_start[wave_start['id'] == node]['from'].iloc[0],
                'has_been_H' : 'H' in wave_start[wave_start['id'] == node]['from'].iloc[0]
            } for node in g.nodes
        })

        # save graph object to file
        pickle.dump(g, open(f'data/network_start_wave_{self.step_start}', 'wb'))

        return g


    def apply_intervention(self, modifier = 1, level = 'personal', drinker_type = 'A', history = ''):
        self.modifier = modifier
        self.modifier_level = level
        self.modifier_type = drinker_type
        # TODO: implement this
        if level == 'social':
            # modify
            self.s.loc[self.s['link'] == history + self.modifier_type, 's'] *= self.modifier
            print(f'modified {self.modifier_level} influence of {history + self.modifier_type} by factor of {self.modifier}')
        elif level == 'personal':
            # modify
            from_type = history + ('H' if self.modifier_type == 'A' else 'A')
            self.transitions.loc[(self.transitions['to'] == self.modifier_type) & (self.transitions['from'] == from_type), 'ratio'] *= self.modifier
            # normalize
            condition = self.transitions['from'] == from_type
            self.transitions.loc[condition, 'ratio'] /= sum(self.transitions[condition]['ratio'])
            print(f'modified {self.modifier_level} {from_type}->{self.modifier_type} by factor of {self.modifier}')


    def report_step(self, step, run):
        print(f'simulating...    (run: {run + 1} / {self.n_runs},    step: {step - self.step_start + 1} / {self.n_steps})', end = '\r')
        # report and save stats
        #print('  timestep:', step)
        stats = list(nx.get_node_attributes(self.g, self.update_attribute).values())
        for drinker_type in self.pops:
            stat = stats.count(drinker_type)
            #print(f'    {drinker_type}: {stat}')
            self.results.loc[(self.results['run'] == run)
                           & (self.results['step'] == step)
                           & (self.results['drinker_type'] == drinker_type), 'count'] = stat
        # TODO: turned this of for implementeing runs. reimplement?
        #for node in self.trackers.keys():
        #    self.trackers[node]['predicted'].append(self.g.nodes[node][self.update_attribute][-1])
        
        #print()


    def adapt_for_network(self, node, cur_type, ratios):
        # get links states
        link_counts = {}
        for link_type in [self.g.nodes[edge[1]][self.update_attribute] for edge in self.g.edges(node)]:
            link_counts[link_type] = link_counts.get(link_type, 0) + 1
        
        # add social factors
        for link_type, count in link_counts.items():
            s_to = self.s.loc[(self.s['link'] == link_type) & (self.s['from'] == cur_type), ['to', 's']]
            for drinker_type in drinker_types:
                ratios.loc[ratios['to'] == drinker_type, 'ratio'] += s_to.loc[s_to['to'] == drinker_type, 's'].iloc[0] * count
                if ratios[ratios['to'] == drinker_type]['ratio'].iloc[0] < 0:
                    ratios.loc[ratios['to'] == drinker_type, 'ratio'] = 0

        # normalize
        ratios['ratio'] /= sum(ratios['ratio'])

        return ratios


    # update state of node stochastically
    def update_node(self, node):
        # TODO: update with personal and social modifiers 
        #       (maybe only once by updating transitions and s?)

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
                # track agents. not sure if necessary
                if node not in self.lapses['ids']:
                    self.lapses['ids'].append(node)
                
            # remember H to distinguish later lapse and relapse
            self.g.nodes[node]['has_been_H'] = True
        return next_type


    def run(self):
        for run in range(self.n_runs):
            #print(f'run {run + 1}:')
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
            

    def plot_true(self, drinker_type):
        true_pops = self.wave_pops[self.wave_pops['drinker_type'] == drinker_type]['count'].tolist()
        true_pops = [true_pop / total for true_pop, total in zip(true_pops, self.total_per_wave)]
        plt.scatter(list(range(1, 8)), 
                    true_pops, 
                    label = drinker_type, 
                    color = colors[drinker_type])
        return true_pops

    
    def report_tracked(self):
        # report predicted and true sequence of tracked agents
        for node_id, track in self.trackers.items():
            print(f'person {node_id} had the following predicted record: {track["predicted"]}')
            print(f'versus the following true record: {track["true"]}')
        
        # report observed lapses and relapses
        total_steps = self.n_steps * self.n_runs
        print(f"{self.lapses['relapses'] / total_steps } relapses and {self.lapses['first_time_lapses'] / total_steps} first time lapses found per step on average!")
        # TODO: return last wave results
        last_wave = self.results[self.results['step'] == self.results['step'].max()]
        return self.lapses, last_wave


    def plot_results(self, name = 'AMHa', results = None, name_addition = ''):
        # account for possible child class overwrite of results
        if results is None:
            results = self.results
        
        # plot line per drinker type
        for drinker_type in drinker_types:
            # format results
            cur_results = results[results['drinker_type'] == drinker_type]
            cur_results['count'] /= self.total_per_wave[self.step_start - 1]
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
            plt.vlines(x = x, 
                       ymin = y[:min(6, len(y))], 
                       ymax = true_pops[1:min(6, len(y)) + 1], 
                       color = colors[drinker_type],
                       linestyles = 'dashed')

            # calculate error
            self.errors = [error_old + abs(pred - true) for error_old, pred, true in zip(self.errors, y, true_pops[1:])]

        plt.ylabel('number of people')
        plt.xlabel('wave')
        plt.legend()
        plt.savefig(f'results/{name}_populations{"_s" if self.use_network else ""}{name_addition}')   
        plt.clf()  

        # TODO: make this clearer by combining the different runs into 1 plot. 
        #       this requires an outside function and a return statement.
        plt.plot(range(self.step_start, len(self.errors) + self.step_start), self.errors)
        plt.title('total error of predictions per wave')
        plt.ylabel('number of people')
        plt.xlabel('wave')
        plt.savefig(f'results/{name}_errors{"_s" if self.use_network else ""}{name_addition}')  
        plt.clf() 