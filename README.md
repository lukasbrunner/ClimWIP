ClimWIP
=======

A collection of functions to perform Climate model Weighting by Independence and Performance (ClimWIP).

Content
-------

* [Idea and Publications](#idea)
* [Requirements and Installation](#requirements)
* [Setup and Data Paths](#setup)
* [Usage and Testing](#usage)
* [Contributors](#contributors)
* [Attribution](#attribution)
* [License](#license)


Idea and Publications
---------------------


Lorenz, R. et al. (2018): Prospects and caveats of weighting climate models for summer maximum temperature projections over North America. _Journal of Geophysical Research: Atmospheres_, 123, 4509–4526. DOI: <a href="http://doi.wiley.com/10.1029/2017JD027992">10.1029/2017JD027992</a>

Knutti, R. at al. (2017): A climate model projection weighting scheme accounting for performance and interdependence, _Geophys. Res. Lett._, 44, 1909–1918. DOI <a href="http://doi.wiley.com/10.1002/2016GL072012">10.1002/2016GL072012</a>


Requirements and Installation
-----------------------------

ClimWIP is written in Python and requires at least version 3.6. It can be cloned from this GitHub repository using

<code>git clone https://git.iac.ethz.ch/model_weighting/model_weighting.git</code>

To install dependencies change into the newly created directory (by default with <code>cd ClimWIP</code> (and at the moment also <code>git checkout paper</code>)) and run
<code>conda env create -f environment.yml</code>

Alternatively, create a new environment and install the required packages manually. This is easiest achieved by running the following:
<code>conda create -n ClimWIP python=3.7 xarray=0.12.2 regionmask python-cdo netCDF4</code>

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
