from src.sampling.sampling_functions import (
    production_problem_spec,
    deployment_problem_spec,
    cost_sensitivity_analysis,
)

samples_fn = "production_cost_samples.csv"
# Run SA for production model sample and save figures to figures folder
cost_sensitivity_analysis(production_problem_spec, samples_fn)

samples_fn = "deployment_cost_samples.csv"
# Run SA for deployment model sample and save figures to figures folder
cost_sensitivity_analysis(deployment_problem_spec, samples_fn)
