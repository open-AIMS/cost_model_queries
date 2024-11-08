import decimal
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_deployment_cost(wb, factors):
    """
    Calculates set up and operational costs in the deployment cost model (wb), given a set of parameters to sample.

    Args:
        wb : The cost model as an excel workbook
        factors : Row of a pandas dataframe with factors to sample

    Returns:
        Cost: Operational cost
        setupCost: Setup cost
        percoral_Cost: Operational cost per 1YO coral
        percoral_setupCost: Setup cost per 1YO coral
        YOEC_yield: Number of 1YO corals produced
    """
    ws = wb.Sheets("Dashboard")
    reef_key = ["Moore", "Davies", "Swains", "Keppel"]
    port = factors["port"].iloc[0]
    ws.Cells(5, 4).Value = factors["num_devices"].iloc[0]
    ws.Cells(6, 4).Value = reef_key[port - 1]
    ws.Cells(9, 4).Value = factors["DAJ_a_r"].iloc[0]
    ws.Cells(10, 4).Value = factors["DAJ_c_s"].iloc[0]
    ws.Cells(13, 4).Value = factors["deck_space"].iloc[0]

    ws_2 = wb.Sheets("Lookup Tables")
    ws_2.Cells(19, 7).Value = factors["cape_ferg_price"].iloc[0]
    ws_2.Cells(19, 15).Value = factors["ship_endurance"].iloc[0]
    ws_2.Cells(54 + port, 8).Value = factors["distance_from_port"].iloc[0]

    ws_3 = wb.Sheets("Conversions")
    ws_3.Cells(25, 5).Value = factors["1YOEC_yield"].iloc[0]

    ws_4 = wb.Sheets("Logistics")
    ws_4.Cells(84, 6).Value = factors["secs_per_dev"].iloc[0]
    ws_4.Cells(84, 9).Value = factors["proportion"].iloc[0]
    ws_4.Cells(83, 9).Value = factors["bins_per_tender"].iloc[0]
    ws_4.Cells(38, 4).Value = factors["deployment_dur"].iloc[0]

    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(6, 11).Value
    setupCost = ws.Cells(11, 11).Value
    YOEC_yield = ws.Cells(5, 11).Value

    percoral_Cost = Cost / decimal.Decimal(YOEC_yield)
    percoral_setupCost = setupCost / decimal.Decimal(YOEC_yield)

    return [Cost, setupCost, percoral_Cost, percoral_setupCost, YOEC_yield]


def calculate_production_cost(wb, factors):
    """
    Calculates set up and operational costs in the production cost model (wb), given a set of parameters to sample.

    Args:
        wb : The cost model as an excel workbook
        factors : Row of a pandas dataframe with factors to sample

    Returns:
        Cost: Operational cost
        setupCost: Setup cost
        percoral_Cost: Operational cost per 1YO coral
        percoral_setupCost: Setup cost per 1YO coral
        YOEC_yield: Number of 1YO corals produced
    """
    ws = wb.Sheets("Dashboard")
    ws.Cells(5, 5).Value = factors["num_devices"].iloc[0]
    ws.Cells(10, 5).Value = factors["species_no"].iloc[0]

    ws_3 = wb.Sheets("Conversions")
    ws_3.Cells(7, 6).Value = factors["col_spawn_gam_bun"].iloc[0]
    ws_3.Cells(8, 7).Value = factors["gam_bun_egg"].iloc[0]
    ws_3.Cells(9, 8).Value = factors["egg_embryo"].iloc[0]
    ws_3.Cells(10, 9).Value = factors["embryo_freeswim"].iloc[0]
    ws_3.Cells(11, 10).Value = factors["freeswim_settle"].iloc[0]
    ws_3.Cells(12, 11).Value = factors["settle_just"].iloc[0]
    ws_3.Cells(14, 11).Value = factors["just_unit"].iloc[0]
    ws_3.Cells(14, 15).Value = factors["just_mature"].iloc[0]
    ws_3.Cells(19, 18).Value = factors["1YOEC_yield"].iloc[0]

    ws_4 = wb.Sheets("Logistics")
    ws_4.Cells(11, 5).Value = factors["optimal_rear_dens"].iloc[0]

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
    """
    Create a problem specification for sampling using SALib.

    Args:
        factor_dict : A dictionary of factor names and ranges
        is_cat : Boolian vector specifian whether each factor is categorical

    Returns:
        sp: ProblemSpec for sampling with SALib
        factor_names: List of factor names
        is_cat: Boolian vector specifian whether each factor is categorical.
    """
    factor_names = [*factor_dict.keys()]
    problem_dict = {
        "num_vars": len(is_cat),
        "names": factor_names,
        "bounds": [*factor_dict.values()],
    }
    return ProblemSpec(problem_dict), factor_names, is_cat


def deployment_problem_spec():
    """
    Create a problem specification for sampling deployment cost parameters using SALib.

    Args:
        factor_dict : A dictionary of factor names and ranges
        is_cat : Boolian vector specifian whether each factor is categorical
    """
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
    """
    Create a problem specification for sampling production cost parameters using SALib.

    Args:
        factor_dict : A dictionary of factor names and ranges
        is_cat : Boolian vector specifian whether each factor is categorical
    """
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


def convert_factor_types(factors_df, is_cat):
    """
    SALib samples floats, so convert categorical variables to integers by taking the ceiling.

    Args:
        factors_df : A dataframe of sampled factors
        is_cat : Boolian vector specifian whether each factor is categorical

    Returns:
        factors_df: Updated sampled factor dataframe with categorical factors as integers
    """
    for ic_ind, ic in enumerate(is_cat):
        if ic:
            factors_df[factors_df.columns[ic_ind]] = np.ceil(
                factors_df[factors_df.columns[ic_ind]]
            ).astype(int)

    return factors_df


def _sample_cost(wb, factors_df, N, calculate_cost):
    """
    Sample a cost model.

    Args:
        wb : A cost model as an escel workbook
        factors_df : Dataframe of ffactors to input in the cost model
        N: Number of samples input to SALib sampling function
        calculate_cost: Function to use to sample cost

    Returns:
        factors_df: Updated sampled factor dataframe with costs added
    """
    total_cost = np.zeros((N * (2 * (factors_df.shape[1]) + 2), 5))
    for idx_n in range(len(total_cost)):
        total_cost[idx_n, :] = calculate_cost(wb, factors_df.iloc[[idx_n]])

    factors_df.insert(1, "Cost", total_cost[:, 0])
    factors_df.insert(1, "setupCost", total_cost[:, 1])
    factors_df.insert(1, "Cost_percoral", total_cost[:, 2])
    factors_df.insert(1, "setupCost_percoral", total_cost[:, 3])
    factors_df.insert(1, "1YOEC_per_dev", total_cost[:, 4])
    return factors_df


def sample_deployment_cost(wb, factors_df, N):
    """
    Sample the deployment cost model.

    Args:
        wb : A cost model as an escel workbook
        factors_df : Dataframe of ffactors to input in the cost model
        N: Number of samples input to SALib sampling function

    Returns:
        factors_df: Updated sampled factor dataframe with costs added
    """
    return _sample_cost(wb, factors_df, N, calculate_deployment_cost)


def sample_production_cost(wb, factors_df, N):
    """
    Sample the production cost model.

    Args:
        wb : A cost model as an escel workbook
        factors_df : Dataframe of ffactors to input in the cost model
        N: Number of samples input to SALib sampling function

    Returns:
        factors_df: Updated sampled factor dataframe with costs added
    """
    return _sample_cost(wb, factors_df, N, calculate_production_cost)


def cost_sensitivity_analysis(
    problem_spec_func, samples_fn, figures_path=".\\figures\\"
):
    samples_df = pd.read_csv(samples_fn)
    sp, factor_names, is_cat = problem_spec_func()
    sp.samples = np.array(samples_df[factor_names])

    # First get sensitivity to setup cost
    sp.set_results(np.array(samples_df["setupCost"]))
    sp.analyze_sobol()

    axes = sp.plot()
    axes[0].set_yscale("log")
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(10, 4)
    plt.tight_layout()
    plt.savefig(figures_path + "setup_cost_sobol_SA.png")

    sp.analyze_pawn()
    axes = sp.plot()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(10, 4)
    plt.tight_layout()
    plt.savefig(figures_path + "setup_cost_pawn_barplot_SA.png")

    # SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)
    sp.heatmap()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(10, 4)
    plt.savefig(figures_path + "setup_cost_pawn_heatmap_SA.png")

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
    plt.savefig(figures_path + "operational_cost_sobol_SA.png")

    sp.analyze_pawn()
    axes = sp.plot()
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(10, 4)
    plt.tight_layout()
    plt.savefig(figures_path + "operational_cost_pawn_barplot_SA.png")

    # SALib.analyze.rsa.analyze(problem_dict, sp.samples, total_cost)
    sp.heatmap()
    fig.set_size_inches(10, 4)
    plt.savefig(figures_path + "operational_cost_pawn_heatmap_SA.png")
