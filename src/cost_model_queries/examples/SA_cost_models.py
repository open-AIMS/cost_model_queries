import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sampling.sampling_functions import cost_sensitivity_analysis

samples_fn = "production_cost_samples.csv"
# Run SA for production model sample and save figures to figures folder
cost_sensitivity_analysis(samples_fn, "production")

samples_fn = "deployment_cost_samples.csv"
# Run SA for deployment model sample and save figures to figures folder
cost_sensitivity_analysis(samples_fn, "deployment")
