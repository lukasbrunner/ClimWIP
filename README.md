[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4073039.svg)](https://zenodo.org/badge/DOI/10.5281/zenodo.4073039.svg)

ClimWIP
=======

A collection of functions implementing the Climate model Weighting by Independence and Performance (ClimWIP) package. A simplified implementation of ClimWIP is now also available within the [ESMValTool](https://docs.esmvaltool.org/en/latest/recipes/recipe_climwip.html). 

Content
-------

* [Key Publications](#key-publications)
* [Requirements and Installation](#requirements-and-installation)
* [Setup and Data Paths](#setup-and-data-paths)
* [Usage and Testing](#usage-and-testing)
* [Contributors](#contributors)
* [Attribution](#attribution)
* [License](#license)
* [Bugfixes](#bugfixes)
* [Updates](#updates)
* [Example config file](#example-config-file)


Key Publications
---------------------

Brunner, L. et al. (2020): A weighting scheme to constrain global temperature change from CMIP6 accounting for model independence and performance _Earth Syst. Dynam. Diss._ DOI:  <a href="https://doi.org/10.5194/esd-2020-23">10.5194/esd-2020-23</a>

Merrifield, A. L. et al. (2020): A weighting scheme to incorporate large ensembles in multi-model ensemble projections. _Earth Syst. Dynam._, 11, 807-834, DOI: <a href="https://doi.org/10.5194/esd-11-807-2020">10.5194/esd-11-807-2020</a>

Brunner, L. et al. (2019): Quantifying uncertainty in European climate projections using combined performance-independence weighting. _Eniron. Res. Lett._ DOI: <a href="https://doi.org/10.1088/1748-9326/ab492f">10.1088/1748-9326/ab492f</a>

Lorenz, R. et al. (2018): Prospects and caveats of weighting climate models for summer maximum temperature projections over North America. _J. Geophys. Res.: Atmospheres_, 123, 4509–4526. DOI: <a href="http://doi.wiley.com/10.1029/2017JD027992">10.1029/2017JD027992</a>

Knutti, R. at al. (2017): A climate model projection weighting scheme accounting for performance and interdependence, _Geophys. Res. Lett._, 44, 1909–1918. DOI <a href="http://doi.wiley.com/10.1002/2016GL072012">10.1002/2016GL072012</a>


Requirements and Installation
-----------------------------

ClimWIP is written in Python and requires at least version 3.6. It is currently run and tested in 3.8.1.

To clone it from GitHub use

<code>git clone git@github.com:lukasbrunner/ClimWIP.git</code>

The easiest way to install all required packages is to run:
<code>conda create -n ClimWIP python=3.8.1 xarray=0.15.1 regionmask python-cdo netCDF4</code>

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

ClimWIP makes several assumptions about the folder structure and filename conventions when collection the models to weight. It is developed and tested on the ETH CMIP3/CMIP5/CMIP6 next generation archives (e.g., <a href="https://doi.org/10.5281/zenodo.3734128">Brunner et al. 2020b</a>) which is similar to the ESGF structure, but slightly flatter. Basically the assumed structure is:
<code>BASE_PATH/varn/varn_mon_model_scenario_ensemble_g025.nc</code> (CMIP3, 5) or
<code>BASE_PATH/varn/mon/g025/varn_mon_model_scenario_ensemble_g025.nc</code> (CMIP6).

The filename conventions are constrained to core/get_filenames.py. Depending on the structure on your system it might be necessary to re-write parts of the functions there.

ClimWIP saves calculated diagnostics to speed up repeated calls using the same diagnostics (this behaviour can be changed by the "overwrite" flag). The default path for this is <code>./data</code>, in which sub-folders for each variable will be created. The final results will also be save in <code>./data</code> as netCDF4 files. They will be named after the configuration name, existing files will be overwritten!


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

If the 'plot' flag in the configuration is set to True ClimWIP will create simple plots with intermediate results by default in <code>./plots/process_plots</code>.

The results will by default be saved as netCDF4 files in <code>./data</code> and will be named after their respective configuration (note that this means they can be overwritten if different configuration files have configuration with the exact same name!).

Contributors
------------

- Lukas Brunner (lukas.brunner@env.ethz.ch)
- Ruth Lorenz (ruth.lorenz@env.ethz.ch)
- Anna L. Merrifield (anna.merrifield@env.ethz.ch)
- Jan Sedlacek

Attribution
-----------

If you use our code please cite us and put this (or a similar) sentence in the acknowledgments: "We thank Lukas Brunner, Ruth Lorenz, and Jan Sedlacek (ETH Zurich) for providing the ClimWIP model weighting package."

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

Bugfixes
--------

Winter season variability and trend diagnostics used a annual mean function for time aggregation. This was wrong as for winter one needs to average December from year X together with January and February from year X+1 in order to get the correct seasonal mean. This has been corrected now in core/diagnostics.py


Updates
-------

### Update May 2020 | Version for Brunner et al. (2020)

This update introduces changes which are NOT backward compatible with old config files!

- Change some flag names in the config file:
  - predictor_* -> performance_*
  - performance_masko -> performance_masks
  - target_masko -> target_mask
  - ensembles -> variants_use
  - ensemble_independence -> variantes_independence
- Add some flags to the config file:
  - performance_normalizers : None or float or string or list of float or string
  - performance_weights : None or float or list of float
  - independence_* : similar to performance_*
  - variants_select : string
  - variants_combine : bool
- Change some allowed flag values in the config file:
  - predictor_masks : bool -> string or False
  - predictor_* : list -> single value or list
  - variants_use : bool -> integer > 0 or 'all'

*Main updates*

It is now possible to use different performance and independence diagnostics if the sigma values are set. Due to that change the normalization of diagnostics before combining them has changed: performance and independence diagnostics are now normalized separately (even if they are identical). This can lead to slightly different results!

It is now possible to mask either land or sea or neither.

It is now possible to assign weights to each diagnostic to define how much they contribute to the weights.

It is now possible to use user-defined values to normalize the predictors before combining them. This can be useful when working with bootstrapping, which exchanges model variants. In the past this could lead to random changes in the normalization, which could lead to surprising results.

It is now possible to give a maximum number of variants to use per model instead of a bool indicating all or only one variant. In addition the new parameter variants_select specifies how to sort variants before selection (the first xx will be used) or if random variants are selected (can be used, e.g., for bootstrapping).

It is now possible to combine ensemble variants of the same model within the method (see Brunner et al. 2020 for details).


Example config file
-------------------

For an example file also look into model_weighting/configs/config_default.ini! Flags which are marked as optional do not need to be set (not even in the DEFAULT section).

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

    Example: /net/h2o/climphys/lukbrunn/Data/InputData/ERA5/v1/

    Description: Path(s) to the observation archive(s). If None no observations will be used and weights will be calculted in a pure "perfect model" world with each model beeing used as "pseudo observation" once.

obs_id : None or string or list of strings

    Example: ERA5

    Description: Unique identifier for each observational data set. Needs to have same lenth as model_path.

obs_uncertainty : None or string

    Allowed values: range, mean, median, center, None

    Description: If more than one observational data set is used: how to deal with the observatinoal uncertainty.
    - range : All values within the full range of the observations get assigned zero distance.
    - mean: The mean of all observational data sets is used as reference.
    - median: The median of all observational data sets is used as reference.
    - center: The center [0.5*(max + min)] of all observational data sets is used as reference.
    - None: Only allowed if only one or no observational data set is used.

save_path : string

    Example: ../data/

    Description: Path to save temp files and final output file. Needs to exist and be writable.

plot_path : string

    Example: ../plots/

    Description: Path to save plots if plot is set to True. Needs to exist and be writable.

overwrite : bool

    Allowed values: True, False

    Description: If True overwrite already existing diagnostics otherwise re-use them. Re-using can lead to a considerably speedup depending on the mumber of models and diagnostics used.

percentiles : list of two floats in (0, 1)

    Example: .1, .9

    Description: Percentiles to use in the perfect model test. The test checks how often the perfect model lies between the two percentiles.

inside_ratio : None or float or force

    Example: force

    Description: Select strength of the weighing such that the perfect models are inside of the two percentiles (see above) at least as often as given by inside_ratio. If None value will be calculated on the fly as percentiles[1] - percentiles[0]. If force: similar to None but if no sigma value can be found to fulfil the perfect model test this will be relaxed until it can be fulfilled.

subset : None or list of strings

    Pattern: <model>_<ensemble>_<id>

    Description: If not None use only the models specified here. If not all models specified here can be found a Value Error will be raised in order to make sure that all models specified are used.

variants_use : integer > 0 or all

    Example: all

    Description: Use all ensemble members or only up to a maximum number given by this parameter.

variants_select: string

    Allowed values: sorted, natsorted, random

    Description: Specify the sorting strategy applied to model variante before selecting the frist xx (given by variants_use) variants.
    - sorted: Sort using the Python buildin sorted() function. This was the original sorting strategy but leads to potentially unexpected sorting: [r10i*, r11i*, r1i*, ...]
    - natsorted: Sort using the natsort.natsorted function: [r1i*, r10i*, r11i*, ...]
    - random: Do not sort but pick random members. This can, e.g., be used for bootstrapping of model variants: [r24i*, r7i*, r13i*, ...]

variants_independence : bool

    Allowed values: True, False

    Description: Can only be True if variants_use is not 1. If True use an alternative approach to calculate the independence sigma based on ensemble member similarity (see appendix of Brunner et al. 2019)

variants_combine : bool

    Allowed values: True, False

    Description: If False treat each ensemble member as individual model - this was the default behavior in the original version. If True average ensemble members of the same model before calculating the weights (see Merrifield et al. 2019 and Brunner et al. 2020 for details).

performance_metric: string, optional

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
    - None: Calculate the sigma value on the fly
    - float: Use given sigma value.
    - -99: Use no sigma value and set all independence weights to 1.

sigma_q : None or float or -99

    Example: None

    Description: Performance sigma value handling
    - None: Calculate the sigma value on the fly via a perfect model test.
    - float: Use given sigma value.
    - -99: Use no sigma value and set all performance weights to 1.

target_diagnostic : None or string

    Example: tas

    Description: Variable identifyer of the target diagnostic. Has to be in the model archive. Setting this to None is only allowed if sigma_q is given and therefore no perfect model test is needed.

target_agg : string

    Allowed values: CLIM, STD, TREND, ANOM-GLOABL, ANOM-LOCAL, CORR

    Description: Time aggregation of the target variable.
    - CLIM: Time mean over the given period
    - ANOM-GLOBAL: Same as CLIM but with the global mean removed
    - ANOM-LOCAL: Same as CLIM but with the mean of the region removed
    - STD: Standart deviation of the de-trended time series
    - TREND: Trend over the given period
    - CORR: Time correlation between two variables

target_season : string

    Allowed values: ANN, JJA, SON, DJF, MAM

    Description: Season to use or annual.

target_mask : False or sea or land

    Allowed values: False, sea, land

    Description: Mask applied to the target variable. The mask is based on the grid cell center (using the Python regionmask package).

target_region : string

    Example: GLOBAL

    Description: Region to use. Can either be GLOBAL or a valid SREX region or a region which is defined in the shapefiles folder as 'target_region.txt'.

target_startyear : integer

    Example 2080

    Description: Beginning of the time period to consider for the target variable.

target_endyear : integer

    Example: 2099

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

    Example: JJA

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_masks : False or string or list of False/strings

    Allowed values: False, sea, land

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_regions : string or list of strings

    Example: GLOBAL

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_startyears : integer or list of integers

    Example: 1995

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_endyears : integer or list of integers

    Example: 2014

    Description: Has to either have same length as performance_diagnostics or be a single value. If it is a single value this value will be used for each value in performance_diagnostics.

performance_normalizers : string or float or list of strings or floats

    Allowed values: median, mean, center, <float>, <list of floats>

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
