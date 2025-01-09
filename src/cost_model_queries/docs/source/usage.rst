Usage
=====

Installation
------------

To use Cost Model Queries, the evironment must first be installed from the `environment.yml` file:

.. code-block:: console

    (base) $ pip install cost_model_queries

Code structure
--------------

The excel files containing the cost models need to be placed in the **Cost Models** folder in the project root.

Configuration
-------------

The file `config.csv` sets the parameters names, sampling ranges, and cells in the excel-based cost models. To adjust sampling ranges
and parameter names a new config file can be created and placed in the project root. The config file is a csv with the following columns:

    * `cost_type` : `production` or `deployment` model.
    * `sheet` : the sheet the parameter is found on in the excel-based cost model.
    * `factor_names` : a shortened name for the parameter.
    * `cell_row` : the row the parameter sits on in the model.
    * `cell_col` : the column the parameter sits in in the model.
    * `range_lower` : the lower limit of the parameter's sampling range.
    * `range_upper` : the upper limit of the parameter's sampling range.
    * `is_cat` : TRUE if the parameter is categorical, FALSE if not.

The config file is loaded by :py:func:`sampling.sampling_functions.problem_spec` and used to sample predictors and calculate costs for samples.

Sampling
--------

Parameter sampling is done using the `SALib <https://salib.readthedocs.io/en/latest/index.html>`_ package, with parameter sampling ranges defined in
`config.csv`. The input samples are then adusted to the correct types, using :py:func:`sampling.sampling_functions.convert_factor_types`.
In the example below, cost model sampling is carried out by the function, :py:func:`sampling.sampling_functions.sample_production_cost`,
which saves the samples as a csv, here specified as `production_cost_samples.csv`. An example for the deployment costs is included in
`sample_deployment_cost_model.py`.

.. literalinclude:: ../../examples/sample_production_cost_model.py
   :language: python
   :linenos:

Sensitvity analysis
-------------------

Sensitivity analysis can be carried out on the collected samples, again using the `SALib <https://salib.readthedocs.io/en/latest/index.html>`_
package. In the example below, the files `production_cost_samples.csv` and `deployment_cost_samples.csv` were generated using the sampling scripts
described above. The function :py:func:`sampling.sampling_functions.cost_sensitvity_analysis` generates a series of figures which are saved in
the **figures** folder, including bar plots and heatmaps of the Pawn and Sobol sensitvity analysis results.

.. literalinclude:: ../../examples/SA_cost_models.py
   :language: python
   :linenos:

Develop Regression Models
-------------------------

Several packages are included for developing and testing regression models for the sampled cost data. Models are available from the included
packages `statsmodels <https://www.statsmodels.org/stable/index.html>`_ and `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
For exploring potential models, predictors can be plotted against cost using :py:func:`plotting.data_plotting.plot_predictors`.
A series of functions for testing the assumptions of linear regression are also included in :py:mod:`plotting.LM_diagnostics`,
including QQplots, location vs. scale and residuals plots. The example below shows the process of fitting linear regression models
to samples from the deployment cost model and checking assumptions. An example for the production cost is included in `test_regression_models_production_cost.py`.

.. literalinclude:: ../../examples/test_regression_models_deployment_cost.py
   :language: python
   :linenos:
