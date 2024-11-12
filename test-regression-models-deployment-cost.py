import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import src.plotting.LM_diagnostics as lmd

from src.plotting.data_plotting import plot_predictors, plot_predicted_vs_actual

samples_fn = "deployment_cost_samples.csv"
samples_df = pd.read_csv(samples_fn)

init_x = samples_df[
    samples_df.columns[
        (samples_df.columns != "Cost")
        & (samples_df.columns != "setupCost")
    ]
]

init_x["port"] = init_x["port"].astype("category")

# General review of potential relationships/correlations
ax, fig = plot_predictors(init_x, samples_df.Cost)
fig.show()

### Model for Cost ###
formula = "Cost ~ 0 + np.log(num_devices) + port + DAJ_a_r + deck_space + np.log(distance_from_port) + secs_per_dev + bins_per_tender + proportion + np.log(cape_ferg_price)"
x = pd.concat([np.log(samples_df.Cost), init_x], axis=1)
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

# Plot pred against actual
pred_var = res.get_prediction().summary_frame()["mean"]
ax, fig = plot_predicted_vs_actual(np.exp(x.Cost), np.exp(pred_var))
fig.show()

### Model for setupCost ###
# General review of potential relationships/correlations
ax, fig = plot_predictors(init_x, samples_df.setupCost)
fig.show()

formula = "setupCost ~ 0 + np.log(num_devices) + DAJ_a_r + DAJ_c_s + deck_space + np.log(distance_from_port) + secs_per_dev + bins_per_tender + proportion"
x = pd.concat([np.log(samples_df.setupCost), init_x], axis=1)
ols_model = smf.ols(formula=formula, data=x)
res = ols_model.fit()
print(res.summary())

# Caculate diagnostics
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

# Plot pred against actual
pred_var = res.get_prediction().summary_frame()["mean"]
ax, fig = plot_predicted_vs_actual(np.exp(x.setupCost), np.exp(pred_var))
fig.show()
