import win32com.client
import os
import pandas as pd

from src.sampling.sampling_functions import (
    problem_spec,
    convert_factor_types,
    sample_production_cost,
)

# Filename for saved samples
samples_save_fn = "production_cost_samples.csv"

# Path to cost model
file_name = "\\Cost Models\\3.5.2 CA Production Model.xlsx"
wb_file_path = os.path.abspath(os.getcwd()) + file_name

# Generate sample
N = 2**3

# Generate problem spec, factor names and list of categorical factors to create factor sample
sp, factor_specs = problem_spec("production")
# Sample factors using sobal sampling
sp.sample_sobol(N, calc_second_order=True)

factors_df = pd.DataFrame(data=sp.samples, columns=factor_specs.names)

# Convert categorical factors to categories
factors_df = convert_factor_types(factors_df, factor_specs.is_cat)

# Sample cost using factors sampled
xlApp = win32com.client.Dispatch("Excel.Application")  # Open workbook
wb = xlApp.Workbooks.Open(wb_file_path)
factors_df = sample_production_cost(wb, factors_df, factor_specs, N)
factors_df.to_csv(samples_save_fn, index=False)  # Save to CSV
wb.Close(True)  # Close workbook
