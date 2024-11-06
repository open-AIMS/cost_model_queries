import win32com.client
import pandas as pd
import os

from sampling_functions import deployment_problem_spec, convert_factor_types, sample_deployment_cost

# Filename for saved samples
samples_save_fn = 'deployment_cost_samples.csv'

# Path to cost model
file_name =  "\\Cost Models\\3.5.3 CA Deployment Model.xlsx"
wb_file_path =  os.path.abspath(os.getcwd())+file_name

# Generate sample
N = 2**10

# Generate problem spec, factor names and list of categorical factors to create factor sample
sp, factor_names, is_cat = deployment_problem_spec()
# Sample factors using sobal sampling
sp.sample_sobol(N, calc_second_order=True)

factors_df = pd.DataFrame(data=sp.samples, columns=factor_names)

# Convert categorical factors to categories
factors_df = convert_factor_types(factors_df, is_cat)

# Sample cost using factors sampled
xlApp = win32com.client.Dispatch("Excel.Application") # Open workbook
wb = xlApp.Workbooks.Open(wb_file_path)
factors_df = sample_deployment_cost(wb, factors_df, N)
factors_df.to_csv(samples_save_fn, index=False) # Save to CSV
wb.Close(True) # Close workbook