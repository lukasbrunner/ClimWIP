ClimWIP
=======

A collection of functions to do Climate model Weighting by Independence and Performance (ClimWIP).

Content
-------

* [Idea](#idea)
* [Requirements and Installation](#requirements)
* [Setup and Data Paths](#setup)
* [Usage](#usage)
* [License](#license)


Idea
----


Requirements and Installation
-----------------------------

ClimWIP is written in pure Python and requires at least Python version 3.6. It can be cloned from this GitHub repository using

<code>git clone ...</code>

To install dependencies it is easiest to use conda, running

<code>cd ClimWIP
conda env create -f environment.yml
conda activate ClimWIP
</code>

Alternatively, create a new environment and install the required packages manually. This is easiest achieved by running the following:

<code>conda create -n ClimWIP python=3.7 xarray regionmask python-cdo netCDF4
conda activate ClimWIP
</code>


Setup and Data Paths
--------------------

ClimWIP makes several assumptions about the folder structure and filename conventions when collection all available models. It is developed and tested on the ETH CMIP5/CMIP6 next generation archive which is similar to the ESGF structure, but slightly flatter. Basically the assumed structure is:

<code>BASE_PATH/varn/varn_mon_model_scenario_ensemble_g025.nc</code>

The filename conventions are constrained to core/get_filenames.py. Depending on the structure on your system it might be necessary to re-write parts of the functions there.

ClimWIP saves all calculated diagnostics to speed up repeated calls using the same diagnostics. The default path for this is <code>./data</code>, in which sub-folders for each variable will be created. The final results will also be save in <code>./data</code> as netCDF4 files. They will be named after the configuration name, existing files will be overwritten.


Usage
-----

Run <code>cp config_default.ini config.ini</code> to copy the default configuration file. Update the required fields in the config.ini file to match your system (mainly that will be the 'data_path' field).

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

The results will by default be saved as netCDF4 files in <code>./data</code> and will be named after their respective configuration file.


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
