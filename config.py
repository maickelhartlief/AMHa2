####################
# description:
# 
####################

# external imports
import numpy as np

# internal imports


# waves of the study
waves = list(range(1, 8))

# pre-defined drinker types
# A = abstainer, M = moderate drinker, H = heavy drinker
drinker_types = ['A', 'M', 'H']

# limits of these types
drinker_type_bins = (-np.inf, 0, 14, np.inf) 

# modifyer to limits if participant is female
female_modifier = .5 # NOTE: assuming sex-dependent tolerance is linear

# order and color in which I want drinker types to appear in plots and tables
order = {'A' : 0, 'M' : 1, 'H' : 2}
colors = {'A' : 'green', 'M' : 'yellow', 'H' : 'red'}
color_mods = {'A' : .2, 'M' : .6, 'H' : 1}

# helper function to drop columns only when they are present
def drop_columns(data, columns):
    columns = [column for column in columns if column in data.columns]
    return data.drop(columns = columns)

def normalize_s(s):
    s = s.rename(columns = {'s' : 'ratio'})
    for link_type in s['link'].unique():
        for from_type in drinker_types:
            condition = (s['from'] == from_type) & (s['link'] == link_type)
            shift = sum(s[condition]['ratio'])
            s.loc[condition, 'ratio'] -= shift / 3
    # rename to match transition column name
    return s
