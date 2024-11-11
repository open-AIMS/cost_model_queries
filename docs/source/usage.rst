Usage
=====

Installation
------------

To use Cost Model Queries, the evironment must first be installed from the `environment.yml` file:

.. code-block:: console

    (base) $ conda env create --name environment_name -f environment.yml
    (base) $ conda activate environment_name

Code structure
--------------

The excel files containing the cost models need to be placed in the **Cost Models** folder in the project root.

Configuration
-------------

The file `config.json` sets the parameters names, sampling ranges and other information. To adjust sampling ranges
and parameter names a new config file can be created and placed in the **src** folder. Parameters under the :py:obj:`production_params`
key represent the parameters in the production costs model, while those under :py:obj:`deployment_params` represent those in the
deployment costs model. :py:obj:`factors_dict` describes parameter ranges for sampling. :py:obj:`is_cat` is of the same length as the number
of parameters for a given model and contains `True` for categorical variables and `False` for continuous variables.

Sampling
--------

Parameter sampling is done using the `SALib <https://salib.readthedocs.io/en/latest/index.html>`_ package, with parameter sampling ranges defined in
`config.json` in :py:obj:`factors_dict`. The input samples are then adusted to the correct types, using :py:func:`sampling.sampling_functions.convert_factor_types`.
In the example below, cost model sampling is carried out by the function, :py:func:`sampling.sampling_functions.sample_production_cost`,
which saves the samples as a csv, here specified as `production_cost_samples.csv`. An example for the deployment costs is included in
`sample-deployment-cost-model.py`.

.. literalinclude:: ../../sample-production-cost-model.py
   :language: python
   :linenos:

Sensitvity analysis
-------------------

Sensitivity analysis can be carried out on the collected samples, again using the `SALib <https://salib.readthedocs.io/en/latest/index.html>`_
package. In the example below, the files `production_cost_samples.csv` and `deployment_cost_samples.csv` were generated using the sampling scripts
described above. The function :py:func:`sampling.sampling_functions.cost_sensitvity_analysis` generates a series of figures which are saved in
the **figures** folder, including bar plots and heatmaps of the Pawn and Sobol sensitvity analysis results.

.. literalinclude:: ../../SA-cost-models.py
   :language: python
   :linenos:

Develop Regression Models
-------------------------

Several packages are included for developing and testing regression models for the sampled cost data. Models are available from the included
packages `statsmodels <https://www.statsmodels.org/stable/index.html>`_ and `scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.
For exploring potential models, predictors can be plotted against cost using :py:func:`plotting.data_plotting.plot_predictors`.
A series of functions for testing the assumptions of linear regression are also included in :py:mod:`plotting.LM_diagnostics`,
including QQplots, location vs. scale and residuals plots. The example below shows the process of fitting linear regression models
to samples from the deployment cost model and checking assumptions. An example for the production cost is included in `test-regression-models-production-cost.py`.

.. literalinclude:: ../../test-regression-models-deployment-cost.py
   :language: python
   :linenos:
