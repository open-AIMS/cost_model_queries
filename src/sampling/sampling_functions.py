from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_deployment_cost(wb, factor_spec, factors):
    """
    Calculates set up and operational costs in the deployment cost model (wb), given a set of parameters to sample.

    Args:
        wb : The cost model as an excel workbook
        factor_spec : factor specification, as loaded from the config.csv
        factors : Row of a pandas dataframe with factors to sample

    Returns:
        Cost: Operational cost
        setupCost: Setup cost
    """
    reef_key = ["Moore", "Davies", "Swains", "Keppel"]
    port = factors["port"].iloc[0]

    for _, factor_row in factor_spec.iterrows():
        ws = wb.Sheets(factor_row.sheet)
        if factor_row.factor_names == "distance_from_port":
            ws.Cells(factor_row.cell_row + port, factor_row.cell_col).Value = factors[
                factor_row.factor_names
            ].iloc[0]
        elif factor_row.factor_names == "port":
            ws.Cells(factor_row.cell_row, factor_row.cell_col).Value = reef_key[
                port - 1
            ]
        else:
            ws.Cells(factor_row.cell_row, factor_row.cell_col).Value = factors[
                factor_row.factor_names
            ].iloc[0]

    ws = wb.Sheets("Dashboard")
    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(6, 11).Value
    setupCost = ws.Cells(11, 11).Value

    return [Cost, setupCost]


def calculate_production_cost(wb, factor_spec, factors):
    """
    Calculates set up and operational costs in the production cost model (wb), given a set of parameters to sample.

    Args:
        wb : The cost model as an excel workbook
        factor_spec : factor specification, as loaded from the config.csv
        factors : Row of a pandas dataframe with factors to sample

    Returns:
        Cost: Operational cost
        setupCost: Setup cost
    """
    for _, factor_row in factor_spec.iterrows():
        ws = wb.Sheets(factor_row.sheet)

        ws.Cells(factor_row.cell_row, factor_row.cell_col).Value = factors[
            factor_row.factor_names
        ].iloc[0]

    ws = wb.Sheets("Dashboard")
    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    Cost = ws.Cells(13, 12).Value
    setupCost = ws.Cells(11, 12).Value

    return [Cost, setupCost]


def load_config():
    return pd.read_csv("config.csv")


def problem_spec(cost_type):
    """
    Create a problem specification for sampling using SALib.

    Args:
        cost_type : String specifying cost model type, "production_params" or "deployment_params"

    Returns:
        sp: ProblemSpec for sampling with SALib
        factor_spec : factor specification, as loaded from the config.csv
    """
    if (cost_type != "production") & (cost_type != "deployment"):
        raise ValueError("Non-existent parameter type")

    factor_specs = load_config()
    factor_specs = factor_specs[factor_specs.cost_type == cost_type]
    factor_ranges = [
        factor_specs[["range_lower", "range_upper"]].iloc[k].values
        for k in range(factor_specs.shape[0])
    ]

    problem_dict = {
        "num_vars": factor_specs.shape[0],
        "names": [name for name in factor_specs.factor_names],
        "bounds": factor_ranges,
    }
    return ProblemSpec(problem_dict), factor_specs


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


def _sample_cost(wb, factors_df, factor_spec, N, calculate_cost):
    """
    Sample a cost model.

    Args:
        wb : A cost model as an escel workbook
        factors_df : Dataframe of ffactors to input in the cost model
        factor_spec : factor specification, as loaded from the config.csv
        N: Number of samples input to SALib sampling function
        calculate_cost: Function to use to sample cost

    Returns:
        factors_df: Updated sampled factor dataframe with costs added
    """
    total_cost = np.zeros((N * (2 * (factors_df.shape[1]) + 2), 2))
    for idx_n in range(len(total_cost)):
        total_cost[idx_n, :] = calculate_cost(wb, factor_spec, factors_df.iloc[[idx_n]])

    factors_df.insert(1, "Cost", total_cost[:, 0])
    factors_df.insert(1, "setupCost", total_cost[:, 1])
    return factors_df


def sample_deployment_cost(wb, factors_df, factor_spec, N):
    """
    Sample the deployment cost model.

    Args:
        wb : A cost model as an escel workbook
        factors_df : Dataframe of ffactors to input in the cost model
        factor_spec : factor specification, as loaded from the config.csv
        N: Number of samples input to SALib sampling function

    Returns:
        factors_df: Updated sampled factor dataframe with costs added
    """
    return _sample_cost(wb, factors_df, factor_spec, N, calculate_deployment_cost)


def sample_production_cost(wb, factors_df, factor_spec, N):
    """
    Sample the production cost model.

    Args:
        wb : A cost model as an escel workbook
        factors_df : Dataframe of factors to input in the cost model
        factor_spec : factor specification, as loaded from the config.csv
        N: Number of samples input to SALib sampling function

    Returns:
        factors_df: Updated sampled factor dataframe with costs added
    """
    return _sample_cost(wb, factors_df, factor_spec, N, calculate_production_cost)


def cost_sensitivity_analysis(samples_fn, cost_type, figures_path=".\\src\\figures\\"):
    """
    Perform a sensitvity analysis with costs as output from a set of samples.

    Args:
        samples_fn : Filename/path of the samples.
        cost_type : "production" or "deployment".
        figures_path: where to save figures from the sensitvity analysis.
    """
    samples_df = pd.read_csv(samples_fn)
    sp, factor_spec = problem_spec(cost_type)

    factor_names = factor_spec.factor_names.values
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
