import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import src.plotting.LM_diagnostics as lmd
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plotting.data_plotting import plot_predictors, plot_predicted_vs_actual

samples_fn = "production_cost_samples.csv"
samples_df = pd.read_csv(samples_fn)

### Model for setupCost ###
init_x = samples_df[
    samples_df.columns[
        (samples_df.columns != "Cost")
        & (samples_df.columns != "setupCost")
    ]
]
init_x["YOEC_yield"] = init_x["1YOEC_yield"]

# General review of potential relationships/correlations
ax, fig = plot_predictors(init_x, samples_df.setupCost)
fig.show()

formula = "setupCost ~ 0 + num_devices + species_no"
x = pd.concat([samples_df.setupCost, init_x], axis=1)
ols_model = smf.ols(formula=formula, data=x)
res = ols_model.fit()
print(res.summary())

# Calculate diagnostics
cls = lmd.LinearRegDiagnostic(res)
# Remove outliers
remove_inds = cls.get_influence_ids(n_i=30)
fill_vec = np.repeat(False, x.shape[0])
fill_vec[remove_inds] = True
x = x.drop(x[fill_vec].index)
ols_model = smf.ols(formula=formula, data=x)
res = ols_model.fit()
print(res.summary())

# Plot diagnostics
cls = lmd.LinearRegDiagnostic(res)
cls.residual_plot()
cls.qq_plot()
cls.scale_location_plot()

# Megaphone shape against num_devices suggests weighted least squares to amend heteroscedacity
ax, fig = plot_predictors(x, res.resid)
fig.show()

# First regress abs value of residuals against
res_ols = sm.OLS(np.abs(res.resid), x.setupCost)
res_fit_res = res_ols.fit()
print(res_fit_res.summary())
pred_var = res_fit_res.get_prediction()
mean_pred = pred_var.summary_frame()["mean"]
weights = 1 / (mean_pred) ** 2

# Fit WLS using estimated weights
mod_wls = sm.WLS(x.setupCost, x[["num_devices", "species_no"]], weights=weights)
res_wls = mod_wls.fit()
print(res_wls.summary())
fig, ax = plt.subplots(1, 1)
ax.scatter(x.setupCost, np.abs(res_wls.resid_pearson))
fig.show()

# Plot pred against actual
pred_var = res_wls.get_prediction().summary_frame()["mean"]
ax, fig = plot_predicted_vs_actual(x.setupCost, pred_var)
fig.show()

### Model for Cost ###
init_x = samples_df[
    samples_df.columns[
        (samples_df.columns != "Cost")
        & (samples_df.columns != "setupCost")
    ]
]
init_x["YOEC_yield"] = init_x["1YOEC_yield"]

# General review of potential relationships/correlations
ax, fig = plot_predictors(init_x, samples_df.Cost)
fig.show()

formula = "Cost ~ 0 + num_devices + just_mature"
x = pd.concat([samples_df.Cost, init_x], axis=1)
ols_model = smf.ols(formula=formula, data=x)
res = ols_model.fit()
print(res.summary())

# Plot diagnostics
cls = lmd.LinearRegDiagnostic(res)
cls.residual_plot()
cls.qq_plot()
cls.scale_location_plot()

# Megaphone shape against num_devices suggests weighted least squares to amend heteroscedacity
ax, fig = plot_predictors(x, np.abs(res.resid))
fig.show()
res_ols = sm.OLS(np.abs(res.resid), x.Cost)
res_fit_res = res_ols.fit()
print(res_fit_res.summary())
pred_var = res_fit_res.get_prediction()
mean_pred = pred_var.summary_frame()["mean"]
weights = 1 / (mean_pred) ** 2

# Fit WLS using estimated weights
mod_wls = sm.WLS(x.Cost, x[["num_devices", "species_no"]], weights=weights)
res_wls = mod_wls.fit()
print(res_wls.summary())
fig, ax = plt.subplots(1, 1)
ax.scatter(x.Cost, np.abs(res_wls.resid_pearson))
fig.show()

# Plot pred against actual
pred_var = res_wls.get_prediction().summary_frame()["mean"]
ax, fig = plot_predicted_vs_actual(x.Cost, pred_var)
fig.show()
