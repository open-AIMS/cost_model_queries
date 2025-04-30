import win32com.client
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_deployment_cost(wb, factor_spec, factors):
    """
    Calculates set up and operational costs in the deployment cost model (wb), given a set of parameters to sample.

    Parameters
    ----------
        wb : Workbook
            The cost model as an excel workbook
        factor_spec : dataframe
            The factor specification, as loaded from the config.csv
        factors : dataframerow
            Row of a pandas dataframe with factors to sample

    Returns
    -------
        Cost: float
            Operational cost
        setupCost: float
            Setup cost
    """
    reef_key = ["Moore", "Davies", "Swains", "Keppel"]
    port = factors["port"].iloc[0]

    for _, factor_row in factor_spec[(factor_spec.factor_names!="Cost")&(factor_spec.factor_names!="setupCost")].iterrows():
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
    cost_cells = factor_spec.loc[factor_spec.factor_names=="Cost"]
    setupcost_cells = factor_spec.loc[factor_spec.factor_names=="setupCost"]
    Cost = ws.Cells(cost_cells.cell_row.iloc[0], cost_cells.cell_col.iloc[0]).Value
    setupCost = ws.Cells(setupcost_cells.cell_row.iloc[0], setupcost_cells.cell_col.iloc[0]).Value

    return [Cost, setupCost]


def calculate_production_cost(wb, factor_spec, factors):
    """
    Calculates set up and operational costs in the production cost model (wb), given a set of parameters to sample.

    Parameters
    ----------
        wb : Workbook
            The cost model as an excel workbook
        factor_spec : dataframe
            factor specification, as loaded from the config.csv
        factors : dataframerow
            Row of a pandas dataframe with factors to sample

    Returns
    -------
        Cost: float
            Operational cost
        setupCost: float
            Setup cost
    """
    for _, factor_row in factor_spec[(factor_spec.factor_names!="Cost")&(factor_spec.factor_names!="setupCost")].iterrows():
        ws = wb.Sheets(factor_row.sheet)

        ws.Cells(factor_row.cell_row, factor_row.cell_col).Value = factors[
            factor_row.factor_names
        ].iloc[0]

    ws = wb.Sheets("Dashboard")
    ws.EnableCalculation = True
    ws.Calculate()

    # get the new output
    cost_cells = factor_spec.loc[factor_spec.factor_names=="Cost"]
    setupcost_cells = factor_spec.loc[factor_spec.factor_names=="setupCost"]
    Cost = ws.Cells(cost_cells.cell_row.iloc[0], cost_cells.cell_col.iloc[0]).Value
    setupCost = ws.Cells(setupcost_cells.cell_row.iloc[0], setupcost_cells.cell_col.iloc[0]).Value

    return [Cost, setupCost]


def load_config(config_filepath="config.csv"):
    """
    Load configuration file for model sampling

    Parameters
    ----------
        config_filepath : str
            String specifying filepath of config file, default is the default package config file
    """
    return pd.read_csv(config_filepath)


def problem_spec(cost_type, config_filepath="config.csv"):
    """
    Create a problem specification for sampling using SALib.

    Parameters
    ----------
        cost_type : str
            String specifying cost model type, "production_params" or "deployment_params"
        config_filepath : str
            String specifying filepath of config file, default is the default package config file

    Returns
    -------
        sp: dict
            ProblemSpec for sampling with SALib
        factor_spec : dataframe
            factor specification, as loaded from the config.csv
    """
    if (cost_type != "production") & (cost_type != "deployment"):
        raise ValueError("Non-existent parameter type")

    factor_specs = load_config(config_filepath=config_filepath)
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

    Parameters
    ----------
        factors_df : dataframe
            A dataframe of sampled factors
        is_cat : list{bool}
            Boolian vector specifian whether each factor is categorical

    Returns:
        factors_df: Updated sampled factor dataframe with categorical factors as integers
    """
    for ic_ind, ic in enumerate(is_cat):
        if ic:
            factors_df[factors_df.columns[ic_ind]] = np.ceil(
                factors_df[factors_df.columns[ic_ind]]
            ).astype(int)

    return factors_df


def _sample_cost(wb_file_path, factors_df, factor_spec, N, calculate_cost, n_factors):
    """
    Sample a cost model.

    Parameters
    ----------
        wb_file_path : str
            Filepath to a cost model as an excel workbook
        factors_df : dataframe
            Dataframe of factors to input in the cost model
        factor_spec : dataframe
            factor specification, as loaded from the config.csv
        N : int
            Number of samples input to SALib sampling function
        calculate_cost: function
            Function to use to sample cost

    Returns
    -------
        factors_df : dataframe
            Updated sampled factor dataframe with costs added
    """
    xlApp = win32com.client.Dispatch("Excel.Application")  # Open workbook
    wb = xlApp.Workbooks.Open(wb_file_path)

    total_cost = np.zeros((N * (2 * n_factors + 2), 2))
    for idx_n in range(len(total_cost)):
        total_cost[idx_n, :] = calculate_cost(wb, factor_spec, factors_df.iloc[[idx_n]])

    factors_df.loc[:, "Cost"] = total_cost[:, 0]
    factors_df.loc[:, "setupCost"] = total_cost[:, 1]

    wb.Close(True)  # Close workbook
    return factors_df


def sample_deployment_cost(wb_file_path, factors_df, factor_spec, N, **kwargs):
    """
    Sample the deployment cost model.

    Parameters
    ----------
        wb_file_path : str
            Filepath to a cost model as an excel workbook
        factors_df : dataframe
            Dataframe of factors to input in the cost model
        factor_spec : dataframe
            factor specification, as loaded from the config.csv
        N : int
            Number of samples input to SALib sampling function

    Returns
    -------
        factors_df : dataframe
            Updated sampled factor dataframe with costs added
    """
    n_factors = kwargs.get('n_factors', factors_df.shape[1])
    return _sample_cost(wb_file_path, factors_df, factor_spec, N, calculate_deployment_cost, n_factors=n_factors)


def sample_production_cost(wb_file_path, factors_df, factor_spec, N, **kwargs):
    """
    Sample the production cost model.

    Parameters
    ----------
        wb_file_path : Workbook file path
            A cost model as an excel workbook
        factors_df : dataframe
            Dataframe of factors to input in the cost model
        factor_spec : dataframe
            Factor specification, as loaded from the config.csv
        N : int
            Number of samples input to SALib sampling function

    Returns
    -------
        factors_df : dataframe
            Updated sampled factor dataframe with costs added
    """
    n_factors = kwargs.get('n_factors', factors_df.shape[1])
    return _sample_cost(wb_file_path, factors_df, factor_spec, N, calculate_production_cost, n_factors=n_factors)


def cost_sensitivity_analysis(samples_fn, cost_type, figures_path=".\\src\\figures\\"):
    """
    Perform a sensitvity analysis with costs as output from a set of samples.

    Parameters
    ----------
        samples_fn : str
            Filename/path of the samples.
        cost_type : str
            "production" or "deployment".
        figures_path : str
            Path to save figures from the sensitvity analysis.
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
