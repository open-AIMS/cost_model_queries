import win32com.client
import os
import numpy as np
import pandas as pd
import decimal

# import SALib
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib import ProblemSpec

import matplotlib.pyplot as plt

# Path to cost model
file_name =  "\\Cost Models\\3.5.2 CA Production Model.xlsx"
wb_file_path =  os.path.abspath(os.getcwd())+file_name

# Generate sample
N = 2**10

factor_dict = {'num_devices': [1.0, 100000000.0],
                'colony_hold_time': [19.0, 23.0],
                'species_no': [3.0, 24.0],
                'col_spawn_gam_bun': [57600, 70400],
                'gam_bun_egg': [10.0, 14.0],
                'egg_embryo': [0.7, 0.9],
                'embryo_freeswim': [0.7, 0.9],
                'freeswim_settle': [0.7,0.9],
                'settle_just': [0.7, 0.9],
                'just_unit': [6.0, 9.0],
                'just_mature': [0.7, 0.9],
                '1YOEC_yield': [0.6, 0.8],
                'optimal_rear_dens': [1.0 , 3.0]}
is_cat = [True, True, True, True, True, False, False, False, False, True, False, False, True]

factor_names = [*factor_dict.keys()]
problem_dict = {'num_vars': len(is_cat), 'names': factor_names, 'bounds':[*factor_dict.values()]}
sp = ProblemSpec(problem_dict)

sp.sample_sobol(N, calc_second_order=True)

xlApp = win32com.client.Dispatch("Excel.Application")
wb = xlApp.Workbooks.Open(wb_file_path)

def new_deploy_cost_calc(wb, val_df, idx):
    ws = wb.Sheets("Dashboard")
    ws.Cells(5,5).Value = val_df['num_devices'][idx]
    ws.Cells(10,5).Value = val_df['species_no'][idx]

    ws_3 = wb.Sheets("Conversions")
    ws_3.Cells(7,6).Value = val_df['col_spawn_gam_bun'][idx]
    ws_3.Cells(8,7).Value = val_df['gam_bun_egg'][idx]
    ws_3.Cells(9,8).Value = val_df['egg_embryo'][idx]
    ws_3.Cells(10,9).Value = val_df['embryo_freeswim'][idx]
    ws_3.Cells(11,10).Value = val_df['freeswim_settle'][idx]
    ws_3.Cells(12,11).Value = val_df['settle_just'][idx]
    ws_3.Cells(14,11).Value = val_df['just_unit'][idx]
    ws_3.Cells(14,15).Value = val_df['just_mature'][idx]
    ws_3.Cells(19,18).Value = val_df['1YOEC_yield'][idx]

    ws_4 = wb.Sheets("Logistics")
    ws_4.Cells(11,5).Value = val_df['optimal_rear_dens'][idx]

    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(13,12).Value
    setupCost = ws.Cells(11,12).Value
    YOEC_yield = ws.Cells(5,12).Value
    percoral_Cost = Cost/decimal.Decimal(YOEC_yield)
    percoral_setupCost = setupCost/decimal.Decimal(YOEC_yield)

    return [Cost, setupCost, percoral_Cost, percoral_setupCost, YOEC_yield]

vals_df = pd.DataFrame(data=sp.samples, columns=factor_names)

total_cost = np.zeros((N*(2*len(is_cat)+2), 5))

for (ic_ind, ic) in enumerate(is_cat):
    if ic:
        vals_df[vals_df.columns[ic_ind]] = np.ceil(vals_df[vals_df.columns[ic_ind]]).astype(int)

sp.samples=np.array(vals_df)

for idx_n in range(len(total_cost)):
    total_cost[idx_n, :] = new_deploy_cost_calc(wb, vals_df, idx_n)

sp.set_results(total_cost.sum(axis=1))

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

sp.heatmap()
plt.show()
# SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)

vals_df.insert(1,"Cost", total_cost[:,0])
vals_df.insert(1,"setupCost", total_cost[:,1])
vals_df.insert(1,"Cost_percoral", total_cost[:,2])
vals_df.insert(1,"setupCost_percoral", total_cost[:,3])
vals_df.insert(1,"1YOEC_per_dev", total_cost[:,4])
vals_df.to_csv('prod_model_samples_run2.csv', index=False)
wb.Close(True)
