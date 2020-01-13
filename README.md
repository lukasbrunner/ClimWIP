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

### Update January 2020

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

For an example file also look into model_weighting/configs/config_default.ini!

model_path : string or list of strings

    Example: /net/atmos/data/cmip5-ng/

    Description: Path(s) to the model archive(s).

model_id : string or list of strings

    Allowed values: CMIP3, CMIP5, CMIP6, LE

    Description: Unique identifier for each model archive. Needs to have same lenth as model_path.

model_scenario : string or list of strings

    Example: rcp85

    Description: Identifier of the experiment. Needs to have same lenth as model_path.

obs_path : None or string or list of strings

    Example: /net/tropo/climphys1/rlorenz/Datasets/ERAint/v0/processed/monthly/,

    Description: Path(s) to the observation archive(s).

obs_id : None or string or list of strings

    Example: ERA-Interim

    Description: Unique identifier for each observational data set. Needs to have same lenth as model_path.

obs_uncertainty : None or string

    Allowed values: range, mean, median, center, None

    Description: If more than one observational data set is used: how to deal with the observatinoal uncertainty.
    - range : All values within the full range of the observations get assigned zero distance.
    - mean: The mean of all observational data sets is used as reference.
    - median: The median of all observational data sets is used as reference.
    - center: The center [0.5*(max + min)] of all observational data sets is used as reference.
    - None: Only allowed if only one observational data set is used.

save_path : string

    Example: ../data/

    Description: Path to save temp files and final output file. Has to exist and be writable.

plot_path : string

    Example: ../plots/

    Description: Path to save plots.

overwrite : bool

    Example: False

    Description: Overwrite already existing diagnostics or calculate every time.

percentiles : list of two floats in (0, 1)

    Example: .1, .9

    Description: Percentiles to use in the perfect model test. It will be tested how often the perfect model lies between the two percentiles.

    # if None: calculate as percentiles[1] - percentiles[0]
    # if force: same as None but dynamicaly relaxes the test if it fails
inside_ratio : None or float or force

    Example: .8

    Description: It will be tested for which sigma value the perfect model lies between the two percentiles at least this often.
    - None: Value will be calculated on the fly as percentiles[1] - percentiles[0]
    - force: Same as None but if no sigma value can be found to fulfil the perfect model test this will be relaxed.

subset : None or list of strings

    Pattern: <model>_<ensemble>_<id>

    Description: If not None use onle the models specified here. If not all models specified here can be found a Value Error will be raised in order to make sure all models specified are used.

ensembles : bool

    Example: True

    Description: Use all ensemble members or onle the first one (using natsort, so r1* will be used in stead of r10*).

ensemble_independence : bool

    Example: True

    Description: Can only be True if ensemble is True. If True use an alternative approach to calculate the independence sigma based on ensemble member similarity (see appendix of Brunner et al. 2019)

performance_metric: string, optional

    Default: RMSE

    Allowed values: RMSE

    Description: Metric used to establish model performance and independence.

plot : bool

    Example: True

    Description: Plot some intermediate results (decreases performance).

idx_lats : None or float or list of float, optional

    Example: None

idx_lons : None or float or list of float, optional

    Example: None

sigma_i : None or float > 0 or -99

    Example: None

    Description: Independence sigma value handling
    - None: Calculate the sigma value via perfect model test.
    - float: Use given sigma value.
    - -99: Use no sigma value and set all independence weights to 1.

sigma_q : None or float or -99

    Example: None

    Description: Performance sigma value handling
    - None: Calculate the sigma value via perfect model test.
    - float: Use given sigma value.
    - -99: Use no sigma value and set all performance weights to 1.

target_diagnostic : string

    Example: tas

    Description: Variable identifyer of the target diagnostic. Has to be in the model archive.

target_gag : string

    Allowed values: CLIM, STD, TREND, ANOM-GLOABL, ANOM-LOCAL, CORR

    Description: Time aggregation of the target variable.

target_season : string

    Allowed values: ANN, JJA, SON, DJF, MAM

    Description: Season to use or annual.

target_mask : False or sea or land

    Allowed values: False, sea, land

    Description: Mask applied to the target variable.

target_region : string

    Example: GLOBAL

    Description: Region to use. Can either be GLOBAL or a valid SREX region or a region which is definde in the shapefiles folder as 'target_region.txt'.

target_startyear : integer

    Example 2080

    Description: Beginning of the time period to consider for the target variable.

target_endyear : integer

    Example:2099

    Description: End of the time period to consider of the target variable.

target_startyear_ref : None or integer

    Example: 1995

    Description: Beginning of the reference time period for the target variable. If None no reference period is used if not None the difference between the target period and the reference period will be used.

target_endyear_ref : None or integer

    Example: 2014

    Description: End of the reference time period for the target variable.

performance_diagnostics : string or list of strings

    Example: tas, pr, tas

performance_aggs : string or list of strings

    Example: TREND, CLIM, STD

    Description: Has to have same length as performance_diagnostics

performance_seasons : string or list of strings

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_masks : False or string or list of False/strings

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_regions : string or list of strings

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_startyears : integer or list of integers

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_endyears : integer or list of integers

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_normalizers : string of float or list of strings or floats

    Allowed values: median, mean, center, <list of floats>

    Description: Metric to normalize different diagnostics before combining them. Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_weights : None or float of list of floats

    Example: 1, 1.5, 0

    Description: Weights to use when combining different diagnostics them. A weight of 0 means that diagnostic is ignored. The difference between setting the weight to zero and not using a diagnostic at all is the model subset, which might be different as it is always choosen so that all diagnostics (also those with zero weight) are available from all models. Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

### If these variables are commented out they will be set to the same values as the corresponding performance_* variables.
### Setting these to other values than the corresponding performance values is only allowed if the sigmas are also set (this is for implementation reasons only since the independence matrix is used in the perfect model test)!

independence_diagnostics : string or list of strings

independence_aggs : string or list of strings

independence_seasons : string or list of strings

independence_masks : False or string or list of False/strings

independence_regions : string or list of strings

independence_startyears : integer or list of integer

independence_endyears : integer or list of integer

independence_normalizers : None or float or string or list of floats or strings

independence_weights : None or float list of floats


Contributors
------------

- Jan Sedlacek
- Lukas Brunner (lukas.brunner@env.ethz.ch)
- Ruth Lorenz (ruth.lorenz@env.ethz.ch)

Attribution
-----------

If you use our code please put this (or a similar) sentence in the acknowledgments: "We thank Lukas Brunner, Ruth Lorenz, and Jan Sedlacek (ETH Zurich) for providing the ClimWIP model weighting package."

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
