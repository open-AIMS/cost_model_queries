import numpy as np
import pandas as pd

# import SALib
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib import ProblemSpec

import matplotlib.pyplot as plt

samples_fn ="CAD_model_samples_run2.csv"
samples_df = pd.read_csv(samples_fn)

factor_dict = {'num_devices': [1.0, 100000000.0],
            'port': [0.0, 4.0],
            'deployment_dur': [20.0, 28.0],
            'DAJ_a_r': [1.0, 4.0],
            'DAJ_c_s': [1.0, 4.0],
            'deck_space': [8.0, 15.0],
            'distance_from_port': [13.0, 200.0],
            '1YOEC_yield': [0.6, 0.8],
            'secs_per_dev': [0.0, 4.0],
            'bins_per_tender': [3.0, 7.0],
            'proportion': [0.4,0.6],
            'cape_ferg_price':[12000, 15000],
            'ship_endurance': [12.0, 16.0]}

is_cat = [True, True, True, True, True, True, False, False, False, True, False, False, True]
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

# First get sensitivity to setup cost
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

# Then get sensitivity to operational cost
sp = ProblemSpec(problem_dict)
sp.samples=np.array(samples_df[factor_names])

# First get sensitivity to setup cost
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

# Then get sensitivity to operational cost
sp = ProblemSpec(problem_dict)
sp.samples=np.array(samples_df[factor_names])

# First get sensitivity to setup cost
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
