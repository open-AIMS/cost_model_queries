import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import plotting.LM_diagnostics as lmd

from plotting.data_plotting import plot_scatter_xy, plot_y_v_y

samples_fn = 'CAD_model_samples_run2.csv'
samples_df = pd.read_csv(samples_fn)

init_x = samples_df[samples_df.columns[(samples_df.columns!='Cost')&(samples_df.columns!='setupCost')&(samples_df.columns!='setupCost_percoral')&(samples_df.columns!='Cost_percoral')]]

init_x['port'] = init_x['port'].astype('category')
init_x['YOEC_yield'] = init_x['1YOEC_yield']

# General review of potential relationships/correlations
ax, fig = plot_scatter_xy(init_x, samples_df.Cost)
fig.show()

### Model for Cost ###
formula = 'Cost ~ 0 + np.log(num_devices) + port + DAJ_a_r + deck_space + np.log(distance_from_port) + secs_per_dev + bins_per_tender + proportion + np.log(cape_ferg_price)'
x = pd.concat([np.log(samples_df.Cost),init_x],axis=1)
ols_model = smf.ols(formula=formula, data=x)
res = ols_model.fit()
print(res.summary())

# Calculate diagnostics
cls = lmd.LinearRegDiagnostic(res)
# Remove outliers
remove_inds = cls.get_influence_ids(n_i=30)
fill_vec =  np.repeat(False, x.shape[0])
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
ax,fig = plot_y_v_y(np.exp(x.Cost), np.exp(pred_var))
fig.show()

### Model for setupCost ###
# General review of potential relationships/correlations
ax, fig = plot_scatter_xy(init_x, samples_df.setupCost)
fig.show()

formula = 'setupCost ~ 0 + np.log(num_devices) + DAJ_a_r + DAJ_c_s + deck_space + np.log(distance_from_port) + secs_per_dev + bins_per_tender + proportion'
x = pd.concat([np.log(samples_df.setupCost),init_x],axis=1)
ols_model = smf.ols(formula=formula, data=x)
res = ols_model.fit()
print(res.summary())

# Caculate diagnostics
cls = lmd.LinearRegDiagnostic(res)
# Remove outliers
remove_inds = cls.get_influence_ids(n_i=30)
fill_vec =  np.repeat(False, x.shape[0])
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
ax,fig = plot_y_v_y(np.exp(x.setupCost), np.exp(pred_var))
fig.show()
