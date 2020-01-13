ClimWIP
=======

A collection of functions to perform Climate model Weighting by Independence and Performance (ClimWIP).

Content
-------

* [Idea and Publications](#idea)
* [Requirements and Installation](#requirements)
* [Setup and Data Paths](#setup)
* [Usage and Testing](#usage)
* [Updates](#updates)
* [Contributors](#contributors)
* [Attribution](#attribution)
* [License](#license)


Idea and Publications
---------------------
Brunner, L. et al. (2019): Quantifying uncertainty in European climate projections using combined performance-independence weighting. _Eniron. Res. Lett._ DOI: <a href="https://doi.org/10.1088/1748-9326/ab492f">10.1088/1748-9326/ab492f</a>

Lorenz, R. et al. (2018): Prospects and caveats of weighting climate models for summer maximum temperature projections over North America. _Journal of Geophysical Research: Atmospheres_, 123, 4509–4526. DOI: <a href="http://doi.wiley.com/10.1029/2017JD027992">10.1029/2017JD027992</a>

Knutti, R. at al. (2017): A climate model projection weighting scheme accounting for performance and interdependence, _Geophys. Res. Lett._, 44, 1909–1918. DOI <a href="http://doi.wiley.com/10.1002/2016GL072012">10.1002/2016GL072012</a>


Requirements and Installation
-----------------------------

ClimWIP is written in Python and requires at least version 3.6. It can be cloned from this GitHub repository using

<code>git clone https://git.iac.ethz.ch/model_weighting/model_weighting.git</code>

To install dependencies change into the newly created directory (by default with <code>cd ClimWIP</code> (and at the moment also <code>git checkout paper</code>)) and run
<code>conda env create -f environment.yml</code>

Alternatively, create a new environment and install the required packages manually. This is easiest achieved by running the following:
<code>conda create -n ClimWIP python=3.7 xarray=0.14.1 regionmask python-cdo netCDF4</code>

Activate the environment:
<code>conda activate ClimWIP</code>

[Optional] To use some of the utility functions it is necessary to add the project to your PYTHONPATH environment variable. In the base directory type
<code>pwd</code>
to get the current path. To temporarily add this path type
<code>export PYTHONPATH=$PYTHONPATH:\<output of pwd\></code>
To permanently add it run
<code>echo 'export PYTHONPATH=$PYTHONPATH:\<output of pwd\>' >> ~/.bashrc</code>


Setup and Data Paths
--------------------

ClimWIP makes several assumptions about the folder structure and filename conventions when collection the models to weight. It is developed and tested on the ETH CMIP3/CMIP5/CMIP6 next generation archives which is similar to the ESGF structure, but slightly flatter. Basically the assumed structure is:
<code>BASE_PATH/varn/varn_mon_model_scenario_ensemble_g025.nc</code> (CMIP3, 5) or
<code>BASE_PATH/varn/mon/g025/varn_mon_model_scenario_ensemble_g025.nc</code> (CMIP6).

The filename conventions are constrained to core/get_filenames.py. Depending on the structure on your system it might be necessary to re-write parts of the functions there.

ClimWIP saves all calculated diagnostics to speed up repeated calls using the same diagnostics. The default path for this is <code>./data</code>, in which sub-folders for each variable will be created. The final results will also be save in <code>./data</code> as netCDF4 files. They will be named after the configuration name, existing files will be overwritten.


Usage and Testing
-----------------

Run
<code>cp configs/config_default.ini configs/config.ini</code>
to copy the default configuration file. Then update the required fields in the config.ini file to match your system (mainly that will be the 'data_path' field).

To see the call structure of the ClimWIP main file run
<code>./ClimWIP_main.py -h</code>

To run the DEFAULT configuration from the configs/config.ini file it is sufficient to run

<code>./ClimWIP_main.py</code>

which is equivalent to running

<code>./ClimWIP_main.py -f configs/config.ini DEFAULT</code>.

To run all configuration within one file run

<code>./run_all.py configs/config.ini</code>

this will also automatically set the logger to log to a file instead of Stdout.

If the 'plot' field in the configuration is set to True ClimWIP will create simple plots with intermediate results by default in <code>./plots/process_plots</code>.

The results will by default be saved as netCDF4 files in <code>./data</code> and will be named after their respective configuration (note that this means they can be overwritten if different configuration files have configuration with the exact same name!).

If you are using the ETH next generation archives with the standard settings you can run
<code>./run_all.py configs/config_default.ini</code>
to test several simple cases.

Updates
-------

Update January 2020
-------------------

This update introduces changes which are NOT backward compatible with old config files!

- Change some variable names in the config file:
  - predictor_* -> performance_*
  - performance_masko -> performance_masks
  - target_masko -> target_mask
- Add some variables to the config file:
  - performance_normalizers : None or float or string or list of float or string
  - performance_weights : None or float or list of float
  - independence_* : similar to performance_*
- Change some allowed variable values in the config file:
  - predictor_masks : bool -> string or bool
  - predictor_* : list -> single value or list

It is now possible to use different performance and independence diagnostics if the sigma values are set. Due to that change the normalization of diagnostics before combining them has changed: performance and independence diagnostics are now normalized separately (even if they are identical). This can lead to slightly different results!

It is now possible to mask either land or sea or neither.

It is now possible to assign weights to each diagnostic to define how much they contribute to the weights.
2
It is now possible to use user-defined values to normalize the predictors before combining them. This can be useful when working with bootstrapping, which exchanges model variants. In the past this could lead to random changes in the normalization, which could lead to surprising results.

Added the script 'search_potential_constraints.py' which takes a config file equivalent to the main script and calculates the correlation between diagnostics and the target as well as the correlation between each diagnostics pair (to exclude highly correlated diagnostics).


Example config file
-------------------

model_path : string or list of strings

    Example: /net/atmos/data/cmip5-ng/

    Description: Path(s) to the model archive(s).

model_id : string or list of strings

    Allowed values: CMIP3, CMIP5, CMIP6, LE

    Description: Unique identifier for each model archive.

model_scenario : string or list of strings

    Example: rcp85

    Description: Identifier of the experiment.

# - need to have the same lenght -
# observation input path(s): None or string or list of strings
obs_path = /net/tropo/climphys1/rlorenz/Datasets/ERAint/v0/processed/monthly/,
# observation id(s): None or string or list of strings
obs_id = ERA-Interim,
# inclusion of observational uncertainty: string {range, mean, median, center, none}
obs_uncertainty = center

# output data path: string
save_path = ../data/
# output path for plots: string
plot_path = ../plots/

# --- core settings ---
# overwrite existing diagnostics: bool
overwrite = False
# percentiles to use for the perfect model test: list of two floats (0, 1)
percentiles = .1, .9
# inside_ratio to use for the perfect model test: None or float (0, 1) or force
    # if None: calculate as percentiles[1] - percentiles[0]
    # if force: same as None but dynamicaly relaxes the test if it fails
inside_ratio = None
# subset of models to use: list of model identifiers strings of form '<model>_<ensemble>_<id>'
subset = None
# include all initial conditions ensemble members: bool
ensembles = True
# use the ensemble members to establish the independence sigma: bool
    # can only be True if ensembles is True
ensemble_independence = True
# how to estimate model performance: string {RMSE, <TODO>}
# - RMSE: root mean squared error
# performance_metric = RMSE
# plot some intermediate results (decreases performance): bool
plot = True

idx_lats = None
idx_lons = None

# --- sigmas settings ---
# sigma value handling: None or float > 0 or -99
# if None: calculation via perfect model test of ensembles
# if -99: set corresponding weights to 1

# independence: small: ~all models depend on each other; large: ~all models are independent)
sigma_i = None
# performance: smaller is more aggressive
# NOTE: if this is set to -99 the perfect model test will probably not yield
# meaning full results so sigma_i should also be set manually.
sigma_q = None

# --- target settings ---
# variable name: string
target_diagnostic = tas
# aggregation: string {CLIM, STD, TREND, ANOM-GLOABL, ANOM-LOCAL, CORR}
target_agg = CLIM
# season: string {ANN, JJA, SON, DJF, MAM}
target_season = JJA
# mask ocean: {None, land, sea}
target_mask = sea
# target region: string {GLOBAL, valid SREX region, <valid shapefiles/*.txt>}
target_region = EUR
# time period: integer yyyy
target_startyear = 2031
target_endyear = 2060
# reference time period: None or integer yyyy
# if not None: change from period_ref to period is the target!
target_startyear_ref = 1951
target_endyear_ref = 2005

# --- performance settings ---
# ! all performance_* parameters need to have same lenght !
# same as target: string or list of strings
performance_diagnostics = tas, pr, tas
performance_aggs = TREND, CLIM, STD

# for convenience these values will be expaned into a list of appropriate
# length if a single string is given
performance_seasons = JJA, DJF, ANN
performance_masks =  land, sea, False
performance_regions = EUR, CEU, NEU
performance_startyears = 1981, 1981, 1981
performance_endyears = 2010, 2010, 2010
performance_normalizers = 'median'
performance_weights = 1, 1, 1

# # --- predictors settings ---
# # same as performance_*
# # Setting these to other values than the corresponding performance values is
# # only allowed if the sigmas are also set!!!
# # NOTE: if these parameters are not set they will default to the same values
# # as the coresponding performance parameters. ('not set' means
# # commenting/deleting the lines below NOT just setting them to None!)
# independence_diagnostics = tas, pr, tas
# independence_aggs = TREND, CLIM, STD
# independence_seasons = JJA, DJF, ANN
# independence_masko =  True, True, True
# independence_regions = EUR, CEU, NEU
# independence_startyears = 1981, 1981, 1981
# independence_endyears = 2010, 2010, 2010
# independence_normalizers = 'median', 'median', 'median'
# independence_weights = 1, 1, 1




Contributors
------------

- Jan Sedlacek
- Lukas Brunner (lukas.brunner@env.ethz.ch)
- Ruth Lorenz (ruth.lorenz@env.ethz.ch)

Attribution
-----------

If you publish scientific work based on this code please consider citing some of our papers. If you want to acknowledge us consider putting this sentence in the Acknowledgments: "We thank Jan Sedlacek, Lukas Brunner, and Ruth Lorenz (ETH Zurich) for providing the ClimWIP model weighting package."


License
-------

ClimWIP is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
