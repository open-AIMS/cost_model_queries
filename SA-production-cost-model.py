import numpy as np
import pandas as pd

# import SALib
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib import ProblemSpec

import matplotlib.pyplot as plt

samples_fn ="prod_model_samples_run2.csv"
samples_df = pd.read_csv(samples_fn)

factor_dict = {'num_devices': [1.0, 100000000.0],
                'colony_hold_time': [19.0, 23.0],
                'species_no': [8.0, 16.0],
                'col_spawn_gam_bun': [57600, 70400],
                'gam_bun_egg': [10.0, 14.0],
                'egg_embryo': [0.7, 0.9],
                'embryo_freeswim': [0.7, 0.9],
                'freeswim_settle': [0.7,0.9],
                'settle_just': [0.7, 0.9],
                'just_unit': [6.0, 9.0],
                'just_mature': [0.6, 0.9],
                '1YOEC_yield': [0.6, 0.8],
                'optimal_rear_dens': [1.0 , 3.0]}
is_cat = [True, True, True, True, True, False, False, False, False, True, False, False, True]

factor_names = [*factor_dict.keys()]
problem_dict = {'num_vars': len(is_cat), 'names': factor_names, 'bounds':[*factor_dict.values()]}
sp = ProblemSpec(problem_dict)

sp.samples=np.array(samples_df[factor_names])
breakpoint()
# First get sensitivity to setup cost
sp.set_results(np.array(samples_df['setupCost']))
sp.analyze_sobol()

axes = sp.plot()
axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

sp.analyze_pawn()
axes = sp.plot()
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

# SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)
sp.heatmap()
plt.show()

# Then get sensitivity to operational cost
sp = ProblemSpec(problem_dict)
sp.samples=np.array(samples_df[factor_names])

# Get sensitivity to operational cost
sp.set_results(np.array(samples_df['Cost']))
sp.analyze_sobol()

axes = sp.plot()
axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

sp.analyze_pawn()
axes = sp.plot()
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

# SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)
sp.heatmap()
plt.show()

# Get sensitivity to setup cost per coral
sp.set_results(np.array(samples_df['setupCost_percoral']))
sp.analyze_sobol()

axes = sp.plot()
axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

sp.analyze_pawn()
axes = sp.plot()
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

# SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)
sp.heatmap()
plt.show()

# Get sensitivity to operational cost per coral
sp.set_results(np.array(samples_df['Cost_percoral']))
sp.analyze_sobol()

axes = sp.plot()
axes[0].set_yscale('log')
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

sp.analyze_pawn()
axes = sp.plot()
fig = plt.gcf()  # get current figure
fig.set_size_inches(10, 4)
plt.tight_layout()
plt.show()

# SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)
sp.heatmap()
plt.show()
