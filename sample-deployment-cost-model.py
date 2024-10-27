import win32com.client
import numpy as np
import pandas as pd

import decimal
# import SALib
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib import ProblemSpec

import matplotlib.pyplot as plt

# Generate sample
N = 2**10

orig_distances_from_port = [27.0, 54.0, 129.6, 15.1]
factor_dict = {'num_devices': [1.0, 100000000.0],
            'port': [0.0, 4.0],
            'deployment_dur': [20.0, 28.0],
            'DAJ_a_r': [1.0, 4.0],
            'DAJ_c_s': [1.0, 4.0],
            'deck_space': [8.0, 15.0],
            'distance_from_port': [13.0, 200.0],
            '1YOEC_yield': [0.6, 0.8],
            'secs_per_dev': [0.0,4.0],
            'bins_per_tender': [3.0, 7.0],
            'proportion': [0.4,0.6],
            'cape_ferg_price':[12000, 15000],
            'ship_endurance': [12.0, 16.0]}

is_cat = [True, True, True, True, True, True, False, False, False, True, False, False, True]
factor_names = [*factor_dict.keys()]
problem_dict = {'num_vars': len(is_cat), 'names': factor_names, 'bounds':[*factor_dict.values()]}
sp = ProblemSpec(problem_dict)

sp.sample_sobol(N, calc_second_order=True)

xlApp = win32com.client.Dispatch("Excel.Application")
wb = xlApp.Workbooks.Open("C:\\Users\\rcrocker\\Documents\\Github\\cost-model-queries\\Cost Models\\3.5.3 CA Deployment Model.xlsx")

def new_deploy_cost_calc(wb, val_df, idx):
    ws = wb.Sheets("Dashboard")
    reef_key = ["Moore", "Davies", "Swains", "Keppel"]
    ws.Cells(5,4).Value = val_df['num_devices'][idx]
    ws.Cells(6,4).Value = reef_key[val_df['port'][idx]-1]
    ws.Cells(9,4).Value = val_df['DAJ_a_r'][idx]
    ws.Cells(10,4).Value = val_df['DAJ_c_s'][idx]
    ws.Cells(13,4).Value = val_df['deck_space'][idx]

    ws_2 = wb.Sheets("Lookup Tables")
    ws_2.Cells(19,7).Value = val_df['cape_ferg_price'][idx]
    ws_2.Cells(19,15).Value = val_df['ship_endurance'][idx]
    ws_2.Cells(54+val_df['port'][idx],8).Value = val_df['distance_from_port'][idx]

    ws_3 = wb.Sheets("Conversions")
    ws_3.Cells(25,5).Value = val_df['1YOEC_yield'][idx]

    ws_4 = wb.Sheets("Logistics")
    ws_4.Cells(84,6).Value = val_df['secs_per_dev'][idx]
    ws_4.Cells(84,9).Value = val_df['proportion'][idx]
    ws_4.Cells(83,9).Value = val_df['bins_per_tender'][idx]
    ws_4.Cells(38,4).Value = val_df['deployment_dur'][idx]

    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(6,11).Value
    setupCost = ws.Cells(11,11).Value
    YOEC_yield = ws.Cells(5,11).Value

    percoral_Cost = Cost/decimal.Decimal(YOEC_yield)
    percoral_setupCost = setupCost/decimal.Decimal(YOEC_yield)

    return [Cost, setupCost, percoral_Cost, percoral_setupCost, YOEC_yield]


vals_df = pd.DataFrame(data=sp.samples, columns=factor_names)

total_cost = np.zeros((N*(2*len(is_cat)+2),5))
for (ic_ind, ic) in enumerate(is_cat):
    if ic:
        vals_df[vals_df.columns[ic_ind]] = np.ceil(vals_df[vals_df.columns[ic_ind]]).astype(int)

sp.samples=np.array(vals_df)

for idx_n in range(len(total_cost)):
    total_cost[idx_n, :] = new_deploy_cost_calc(wb, vals_df, idx_n)

vals_df.insert(1,"Cost", total_cost[:,0])
vals_df.insert(1,"setupCost", total_cost[:,1])
vals_df.insert(1,"Cost_percoral", total_cost[:,2])
vals_df.insert(1,"setupCost_percoral", total_cost[:,3])
vals_df.insert(1,"1YOEC_per_dev", total_cost[:,4])
vals_df.to_csv('CAD_model_samples_run2.csv', index=False)
wb.Close(True)
