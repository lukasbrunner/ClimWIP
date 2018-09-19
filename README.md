# Model weighting

## Function description

### calc_target

This function calculates the target diagnostic for the model weighting scheme. A diagnostic consists the evaluation of a (basic or derived) variable in a certain region (can be global, also included masking the ocean) and time period. The time period is aggregated [better word here?] over the timer period:
<ul>
    <li>CLIM (Climatology): Time mean of yearly mean (in case of annual data) or of seasonal mean in each year (in case of seasonal data). </li>
    <li>STD (Standard deviation): Time standard deviation of the detrended data. </li>
    <li>TREND (Trend): Time trend of the data </li>
</ul>

Each of the three aggregation types can be evaluated in the future period or as change from the historical to the future period.

**Input and output**

The function takes a list of filenames and options setting the diagnostic. Each filename represents a (model, ensemble) pair.

The function saves files containing the target diagnostic in a given base_path following the structure:
base_path/variable_name/time_resolution/ocean_mask_flag/
The filename are build from the input filenames by appending the diagnostics information. 4/8 files are saved if the region is not global, otherwise only the first 2 respective files are saved. 8 files are saved if the change of the aggregation is evaluated (historical period and future period)
<ul>
    <li>base_filename_StartYear-EndYear_SeasonORAnnual_GLOBAL.nc </li>
    <li>base_filename_StartYear-EndYear_SeasonORAnnual_AggregationType_GLOBAL.nc </li>
    <li>base_filename_StartYear-EndYear_SeasonORAnnual_Region.nc </li>
    <li>base_filename_StartYear-EndYear_SeasonORAnnual_AggregationType_Region.nc </li>
</ul>

The function also returns a n np.array of shape (nr_models, nr_lats, nr_lons) containing the contents of the last of the 4 (2) files. Except if the evaluation is the change of the aggregation then the returned array is the difference of historical to future.

### calc_predictors

This function calculates the predictor diagnostics for the model weighting scheme. The first step is equivalent to calc_target. Since the diagnostics have to be compared to observations the time period for predictors can only be historical and it might be different from the historical period of the target.
In addition the diagnostic for the predictor can also be a correlation between to variables. TODO

For each of the given diagnostics the RMSE between all model pairs as well as between all models and the observations is calculated. The mean of the normalized (TODO: I don't fully understand how the normalization is done) RMSE is then returned.

**Input and output**

The function takes a list of diagnostics and a list of filenames for each diagnostic (which should be the same except for the variable name).

The function saves files containing the predictor diagnostic equivalent to calc_target.

The function also returns delta_u and delta_q, the measures the model independence and model quality, respectively.

delta_u is a symmetric (nr_models, nr_models) matrix with the diagonal elements being 0 by definition. The larger the values of delta_u are, the more independent a model pair is from each other.

delta_q is either equivalent to delta_u (if there are no observations available) or it is an array of length (nr_models) giving the mean over all diagnostics of the RMSE of each model to the observations.


### calc_sigmas

This function calculates the sigmas to determine the relative weighting of model independence versus model quality as well as the shape of the performance functions for independence and quality itself.

- First, the independence weighting **wu** for each model is calculated as the sum of its distances to all other models. The larger the value, the grater the independence of the model

- Second, the model quality weighting **wq** for each model is calculated in one of two ways:
  - If observations exist it is the difference of each model to the observations
  - If no observations exist it is the difference of each model to each other model
