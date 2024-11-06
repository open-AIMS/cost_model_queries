import decimal
from SALib import ProblemSpec
import numpy as np


def calculate_deployment_cost(wb, val_df, idx):
    ws = wb.Sheets("Dashboard")
    reef_key = ["Moore", "Davies", "Swains", "Keppel"]
    ws.Cells(5, 4).Value = val_df["num_devices"][idx]
    ws.Cells(6, 4).Value = reef_key[val_df["port"][idx] - 1]
    ws.Cells(9, 4).Value = val_df["DAJ_a_r"][idx]
    ws.Cells(10, 4).Value = val_df["DAJ_c_s"][idx]
    ws.Cells(13, 4).Value = val_df["deck_space"][idx]

    ws_2 = wb.Sheets("Lookup Tables")
    ws_2.Cells(19, 7).Value = val_df["cape_ferg_price"][idx]
    ws_2.Cells(19, 15).Value = val_df["ship_endurance"][idx]
    ws_2.Cells(54 + val_df["port"][idx], 8).Value = val_df["distance_from_port"][idx]

    ws_3 = wb.Sheets("Conversions")
    ws_3.Cells(25, 5).Value = val_df["1YOEC_yield"][idx]

    ws_4 = wb.Sheets("Logistics")
    ws_4.Cells(84, 6).Value = val_df["secs_per_dev"][idx]
    ws_4.Cells(84, 9).Value = val_df["proportion"][idx]
    ws_4.Cells(83, 9).Value = val_df["bins_per_tender"][idx]
    ws_4.Cells(38, 4).Value = val_df["deployment_dur"][idx]

    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(6, 11).Value
    setupCost = ws.Cells(11, 11).Value
    YOEC_yield = ws.Cells(5, 11).Value

    percoral_Cost = Cost / decimal.Decimal(YOEC_yield)
    percoral_setupCost = setupCost / decimal.Decimal(YOEC_yield)

    return [Cost, setupCost, percoral_Cost, percoral_setupCost, YOEC_yield]


def calculate_production_cost(wb, val_df, idx):
    ws = wb.Sheets("Dashboard")
    ws.Cells(5, 5).Value = val_df["num_devices"][idx]
    ws.Cells(10, 5).Value = val_df["species_no"][idx]

    ws_3 = wb.Sheets("Conversions")
    ws_3.Cells(7, 6).Value = val_df["col_spawn_gam_bun"][idx]
    ws_3.Cells(8, 7).Value = val_df["gam_bun_egg"][idx]
    ws_3.Cells(9, 8).Value = val_df["egg_embryo"][idx]
    ws_3.Cells(10, 9).Value = val_df["embryo_freeswim"][idx]
    ws_3.Cells(11, 10).Value = val_df["freeswim_settle"][idx]
    ws_3.Cells(12, 11).Value = val_df["settle_just"][idx]
    ws_3.Cells(14, 11).Value = val_df["just_unit"][idx]
    ws_3.Cells(14, 15).Value = val_df["just_mature"][idx]
    ws_3.Cells(19, 18).Value = val_df["1YOEC_yield"][idx]

    ws_4 = wb.Sheets("Logistics")
    ws_4.Cells(11, 5).Value = val_df["optimal_rear_dens"][idx]

    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(13, 12).Value
    setupCost = ws.Cells(11, 12).Value
    YOEC_yield = ws.Cells(5, 12).Value
    percoral_Cost = Cost / decimal.Decimal(YOEC_yield)
    percoral_setupCost = setupCost / decimal.Decimal(YOEC_yield)

    return [Cost, setupCost, percoral_Cost, percoral_setupCost, YOEC_yield]


def _problem_spec(factor_dict, is_cat):
    factor_names = [*factor_dict.keys()]
    problem_dict = {
        "num_vars": len(is_cat),
        "names": factor_names,
        "bounds": [*factor_dict.values()],
    }
    sp = ProblemSpec(problem_dict)
    return ProblemSpec(problem_dict), factor_names, is_cat


def deployment_problem_spec():
    factor_dict = {
        "num_devices": [1.0, 100000000.0],
        "port": [0.0, 4.0],
        "deployment_dur": [20.0, 28.0],
        "DAJ_a_r": [1.0, 4.0],
        "DAJ_c_s": [1.0, 4.0],
        "deck_space": [8.0, 15.0],
        "distance_from_port": [13.0, 200.0],
        "1YOEC_yield": [0.6, 0.8],
        "secs_per_dev": [0.0, 4.0],
        "bins_per_tender": [3.0, 7.0],
        "proportion": [0.4, 0.6],
        "cape_ferg_price": [12000, 15000],
        "ship_endurance": [12.0, 16.0],
    }

    is_cat = [
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    return _problem_spec(factor_dict, is_cat)


def production_problem_spec():
    factor_dict = {
        "num_devices": [1.0, 100000000.0],
        "colony_hold_time": [19.0, 23.0],
        "species_no": [3.0, 24.0],
        "col_spawn_gam_bun": [57600, 70400],
        "gam_bun_egg": [10.0, 14.0],
        "egg_embryo": [0.7, 0.9],
        "embryo_freeswim": [0.7, 0.9],
        "freeswim_settle": [0.7, 0.9],
        "settle_just": [0.7, 0.9],
        "just_unit": [6.0, 9.0],
        "just_mature": [0.7, 0.9],
        "1YOEC_yield": [0.6, 0.8],
        "optimal_rear_dens": [1.0, 3.0],
    }
    is_cat = [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    return _problem_spec(factor_dict, is_cat)


def convert_factor_types(factor_df, is_cat):
    for ic_ind, ic in enumerate(is_cat):
        if ic:
            factor_df[factor_df.columns[ic_ind]] = np.ceil(
                factor_df[factor_df.columns[ic_ind]]
            ).astype(int)
    return factor_df


def _sample_cost(wb, factors_df, N, calculate_cost):
    total_cost = np.zeros((N * (2 * (factors_df.shape[1]) + 2), 5))
    for idx_n in range(len(total_cost)):
        total_cost[idx_n, :] = calculate_cost(wb, factors_df, idx_n)

    factors_df.insert(1, "Cost", total_cost[:, 0])
    factors_df.insert(1, "setupCost", total_cost[:, 1])
    factors_df.insert(1, "Cost_percoral", total_cost[:, 2])
    factors_df.insert(1, "setupCost_percoral", total_cost[:, 3])
    factors_df.insert(1, "1YOEC_per_dev", total_cost[:, 4])
    return factors_df


def sample_deployment_cost(wb, factors_df, N):
    return _sample_cost(wb, factors_df, N, calculate_deployment_cost)


def sample_production_cost(wb, factors_df, N):
    return _sample_cost(wb, factors_df, N, calculate_production_cost)
