import numpy as np
import pandas as pd

from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample

import matplotlib.pyplot as plt

from sampling.sampling_functions import production_problem_spec

samples_fn = "production_cost_samples.csv"
samples_df = pd.read_csv(samples_fn)

sp, factor_names, is_cat = production_problem_spec()
sp.samples = np.array(samples_df[factor_names])

# First get sensitivity to setup cost
sp.set_results(np.array(samples_df["setupCost"]))
sp.analyze_sobol()

axes = sp.plot()
axes[0].set_yscale("log")
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
sp.samples = np.array(samples_df[factor_names])

# Get sensitivity to operational cost
sp.set_results(np.array(samples_df["Cost"]))
sp.analyze_sobol()

axes = sp.plot()
axes[0].set_yscale("log")
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
