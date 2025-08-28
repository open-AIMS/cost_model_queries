import pandas as pd
import numpy as np
import math, numbers
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cost_model_queries.sampling.sampling_functions import (
    problem_spec,
    convert_factor_types,
    sample_deployment_cost,
    sample_production_cost
)

def test_sampling(cost_function, model_type, file_name):

    wb_file_path = os.path.abspath(os.getcwd()) + file_name

    # Generate sample
    N = 2**3

    # Generate problem spec, factor names and list of categorical factors to create factor sample
    sp, factor_specs = problem_spec(model_type)
    # Sample factors using sobal sampling
    sp.sample_sobol(N, calc_second_order=True)

    factors_df = pd.DataFrame(data=sp.samples, columns=factor_specs.factor_names)

    # Convert categorical factors to categories
    factors_df = convert_factor_types(factors_df, factor_specs.is_cat)
    check_types = [isinstance(factors_df[factor_specs.factor_names].iloc[[0]][k][0], (np.int64, np.int32)) for k in factor_specs.factor_names[factor_specs.is_cat]]

    # Check any conversions to Int have worked
    assert all(check_types)

    factors_df = cost_function(wb_file_path, factors_df, factor_specs)
    check_sample = [math.isnan(factors_df[factor_specs.factor_names].iloc[[0]][k][0]) for k in factor_specs.factor_names]

    # Check sampling has worked (no Nans due to sampling empty cells)
    assert not any(check_sample)

    check_types = [isinstance(factors_df[factor_specs.factor_names].iloc[[0]][k][0], numbers.Number) for k in factor_specs.factor_names]

    # Check sampling has only sampled numbers
    assert all(check_types)

# Test deployment model sampling
test_sampling(sample_deployment_cost, "deployment", "\\tests\\test_deployment_cost_model.xlsx")

# Test production model sampling
test_sampling(sample_production_cost, "production", "\\tests\\test_production_cost_model.xlsx")
