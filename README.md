# Walk Forward Validation using NN model

## Overview
The fast and powerful methods that we rely on in machine learning, such as using train-test splits and k-fold cross-validation, do not work in the case of time-series data. This is because they ignore the temporal components inherent in the problem. 


## This Section is divided into the following parts
  * Train-Test Split
  * Series as Supervised Learning
  * Walk-Forward Validation
  * Data Engineering Excel data and convert it into CSV format
  * Run the NN model using a CSV file
  
## Train-Test Split
The train_test_split() function will split the series taking the raw observations and the number of observations to use in the test set as arguments.

![Train test split](/images/train_test_split.png)

## Series as Supervised Learning

The output component will be the returned long/short value in the next minute because we are interested in developing a model to make one-step forecasts.
We can implement this using the shift() function on the pandas DataFrame. It allows us to shift a column down (forward in time) or back (backward in time). We can take the series as a column of data, then create multiple copies of the column, shifted forward or backward in time in order to create the samples with the input and output elements we require.


![Train test split](/images/series_to_sup.png)


## Walk Forward Validation
Walk-forward validation is an approach where the model makes a forecast for each observation in the test dataset one at a time. After each forecast is made for a time step in the test dataset, the true observation for the forecast is added to the test dataset and made available to the model.
I defined a generic model_fit() function to perform this operation that can be filled in for the given type of neural network that we are interested in. The function takes the training dataset and the model configuration and returns the fit model ready for making predictions.

![Train test split](/images/model_fit.png)


I define a simple model with one hidden layer and define five hyperparameters to tune in the future. They are:
n_input: The number of prior inputs to use as input for the model.
n_nodes: The number of nodes to use in the hidden layer.
n_epochs: The number of training epochs.
n_batch: The number of samples to include in each mini-batch.
n_diff: The difference order.

Each time step of the test dataset is enumerated. A prediction is made using the fit model.
Then, I defined a generic function named model_predict() that takes the fit model, the history, and the model configuration and makes a single one-step prediction.

![Train test split](/images/model_predict.png)

The prediction is added to a list of predictions and the true observation from the test set is added to a list of observations that were seeded with all observations from the training dataset. This list is built up during each step in the walk-forward validation, allowing the model to make a one-step prediction using the most recent history.
All of the predictions can then be compared to the true values in the test set and an error measure calculated.

I calculated the root mean squared error, or RMSE, between predictions and the true values.
RMSE is calculated as the square root of the average of the squared differences between the forecasts and the actual values. The measure_rmse() implements this below using the mean_squared_error() sci-kit-learn function to first calculate the mean squared error, or MSE, before calculating the square root.

![Train test split](/images/measure_rmse.png)

The complete walk_forward_validation() function that ties all of this together is listed below.
It takes the dataset, the number of observations to use as the test set, and the configuration for the model, and returns the RMSE for the model performance on the test set.


## Data Engineering
I defined two functions named long() and short() those are formulas given in requirements from the customer. 

![Train test split](/images/longshort.png)

I had to reformat excel files in order to make easy for my Neural Network. The function named excel_to_df() takes a data frame and cuts the first two rows and the last columns rename columns, resets indices and returns a new data frame. 

![Train test split](/images/excel_to_df.png)

Next two functions time_series_long() and time_series_short() create new columns using functions long() and short(). 

![Train test split](/images/times_series_long.png)

The complete short_to_csv() and long_to_csv functions that tie all of this together and create CSV files for modeling. 


![Train test split](/images/long_csv.png)

Finally, I took CSV file as series and take last 100 data entries and called walk_forward_validation() function and made predictions. 



![Train test split](/images/series.png)


Those are true values and predicted values. 


![Train test split](/images/true_values.png)


The function plot_it() creates chart true vs predicted values. 


![Train test split](/images/plot.png)

If you have anything to ask, please contact me clicking following link? 

www.fiverr.com/coderjs

Thank you