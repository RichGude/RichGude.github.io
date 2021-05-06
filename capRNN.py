'''
Author:     Rich Gude
Purpose:    To amend metal commodity pricing data for time-series analysis
Revision:   1, dated April 1, 2021

The following code follows and borrows extensively from the TensorFlow homepage tutorial for Time-Series Structured Data
  modeling ('https://www.tensorflow.org/tutorials/structured_data/time_series').  I have ammended as appropriate for the 
  price data within this project and commented to better understand the flow of the work.

Document Review: This file loads economic data procured via .xlsx format from the office of Amit Goyal at his personal
  site (http://www.hec.unil.ch/agoyal) published in conjunction with his paper, "A Comprehensive Look at the Empirical
  Performance of Equity Premium Prediction".  The paper reviews the lagged effect multiple economic factors have equity
  premiums.  Specifically, this paper found that the numerous variables had the most significant correlation and
  suitability for forecasting equity premium trends over a lagged time frame.  Wang et al. (2020) utilized eight of
  these variables in their paper with limited forecasting success.  The following variables will be used for predictive
  Recurring Neural Network (RNN) creation:

  The following fourteen (14) variables are collected monthly from January 1990 to December 2019, relatively matching
  the commodity price data collected from the IMF:
  - Dividends (D12): Dividends are twelve-month moving sums of dividends paid on the S&P 500 index; data obtained from
    the S&P Corporation.
  - Earnings (E12): Earnings are twelve-month moving sums of earnings on the S&P 500 index. Data obtained from Robert
    Shiller’s website for the period 1990 to June 2003. Earnings from June 2003 to 2020 are from Goyal estimates on
    interpolation of quarterly earnings provided by S&P Corporation.
  - Book to Market Ratio (b/m): the ratio of book value to market value for the Dow Jones Industrial Average. For the
    months of March to December, computed by dividing book value at the end of previous year by the price at the end of
    the current month. For the months of January to February, this is computed by dividing book value at the end of 2
    years ago by the price at the end of the current month.
  - Treasury Bills (tbl):  T-bill rates from 1990 to 2020 are the 3-Month Treasury Bill: Secondary Market Rate.
  - Corporate Bond Yields (AAA): Yields on AAA-rated for the period 1990 to 2020
  - Corporate Bond Yields (BAA): Yields on BAA-rated bonds for the period 1990 to 2020
  - Long Term Yield (lty): Long-term government bond yields for the period 1990 to 2020
  - Net Equity Expansion (ntis): the ratio of twelve-month moving sums of net issues by NYSE listed stocks divided by
    the total market capitalization of NYSE stocks. This dollar amount of net equity issuing activity (IPOs, SEOs, stock
    repurchases, less dividends) for NYSE listed stocks is computed from Center for Research in Security Prices data
  - Risk-free Rate (Rfree): The risk-free rate for the period 1990 to 2020 is the T-bill rate
  - Inflation (infl): the Consumer Price Index (All Urban Consumers) for the period 1990 to 2020 from the Bureau of
    Labor Statistics, lagged by one month to account for distribution lag.
  - Long Term Rate of Return (ltr): Long-term government bond returns for the period 1990 to 2020 are from Ibbotson’s
    Stocks, Bonds, Bills and Inflation Yearbook.
  - Corporate Bond Returns (corpr): Long-term corporate bond returns for the period 1990 to 2020 are from Ibbotson’s
    Stocks, Bonds, Bills and Inflation Yearbook.
  - Stock Variance (svar): Stock Variance is computed as sum of squared daily returns on S&P 500. Daily returns from
    1990 to 2020 are obtained from CRSP.
  - Stock Prices (SPvw): S&P 500 index prices from 1990 to 2020 from CRSP’s month-end values. Stock Returns are the
    continuously-compounded returns on the S&P 500 index.

'''

# %% Prep library and model constants

import os                   # for specifying working directory commands
import openpyxl             # for appending to excel files
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt     # for graph illustration
import numpy as np                  # for pandas value typing
import pandas as pd                 # for csv file reading and dataFrame manipulation
import seaborn as sns               # for specific graphics
import tensorflow as tf
# for splitting training and test data
from sklearn.model_selection import train_test_split

# %% Data Preprocessing and Loading

# Choose a metal to evaluate
metals = ['Nickel']

# Define figure constants:
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# Define other constants:
# set data working directory
dwd = os.path.join(os.getcwd(), 'EconData')
metal_list = ['Aluminum', 'Copper', 'IronOre',
              'Nickel', 'Zinc']      # set list of metal names
var_names = ['D12', 'E12', 'b/m', 'tbl', 'AAA', 'BAA', 'lty',
             'ntis', 'Rfree', 'infl', 'ltr', 'corpr', 'svar', 'SPvw']

# %# Load data and begin data preparation:
econData = pd.read_excel(os.path.join(dwd, 'PredictorData2019.xlsx'), sheet_name='Monthly',
                         usecols=['Date'] + var_names, index_col=0)
priceData = pd.read_excel(os.path.join(
    dwd, 'PriceData.xlsx'), sheet_name='1990Price', index_col=0)

# Review Data
# print("Economic Indicators:\n', econData.head())
# print("Commodity Price Values:\n', priceData.head())
# showCase = econData.plot(subplots=True)
# plt.show()
# All data looks acceptable and is otherwise able to proceed:

# %# Train/Test/Validate Split
# data_length = len(econData)
# train_econ = econData[0:int(data_length*0.7)]
# valid_econ = econData[int(data_length*0.7):int(data_length*0.9)]
# test_econ = econData[int(data_length*0.9):]
#
# train_price = priceData[0:int(data_length*0.7)]
# valid_price = priceData[int(data_length*0.7):int(data_length*0.9)]
# test_price = priceData[int(data_length*0.9):]
#
# # Normalize Economic data
# econ_mean = train_econ.mean()
# econ_stdv = train_econ.std()
# train_econ = (train_econ - econ_mean)/econ_stdv
# valid_econ = (valid_econ - econ_mean)/econ_stdv
# test_econ = (test_econ - econ_mean)/econ_stdv
#
# # Normalize Price data
# price_mean = train_price.mean()
# price_stdv = train_price.std()
# train_price = (train_price - price_mean)/price_stdv
# valid_price = (valid_price - price_mean)/price_stdv
# test_price = (test_price - price_mean)/price_stdv

# Review normalized data structure in a violin plot
# sample_std = (econData - econData.mean()) / econData.std()      # Define a new variable so as to not adjust current
# # Create a reverse-pivot table basically from a two-column dataframe for each violin-chart creation
# sample_std = sample_std.melt(var_name='Variable', value_name='Normalized')
# plt.figure(figsize=(14, 8))
# ax = sns.violinplot(x='Variable', y='Normalized', data=sample_std)
# _ = ax.set_xticklabels(econData.keys(), rotation=90)
# plt.show()
# All data looks acceptable and is otherwise able to proceed (despite large outliers on 'svar' and 'infl')

# Create a triangular correlation heat map
# (help from https://medium.com/@szabo.bibor/how-to-create-a-seaborn-correlation-heatmap-in-python-834c0686b88e)
# plt.figure(figsize=(19, 8))
# allData = pd.concat([econData, priceData['Aluminum']], axis=1)
# mask = np.triu(np.ones_like(allData.corr(), dtype=np.bool), k=1) # create a mask with starting correlations
# heatmap = sns.heatmap(allData.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap=sns.diverging_palette(20, 220, as_cmap=True))


# Create a window for reviewing past economic variable data for predicting current commodity prices
'''
In some cases, economic data may have a absolute and immediate impact on price data, such as inflation, discussed in the
Time Series section and quantitatively, but not qualitatively excluded from the real price data here, or treasury bill
rates, which are published by the government and can be immediately assessed for their return on investment over various
other investment opportunities.  In other cases, this data may have a delayed effect on price data, such as corporate
bond return or stock variance data being compiled and released for investor consumption and integration into buying and
selling behavior at a later date from their real-world calculation.

For this delayed consumption effect reason, a six-month window going back in time from the current day will be used to
predict commodity price for the next month  (e.g., using economic data from January through June, calendar months 1
through 6, to predict commodity prices in July, calendar month 7).

Define a class that takes in economic factors and price data dataframes
'''

# %% Define a class that takes in economic factors and price data dataframes


class SampleGenerator:
    def __init__(self, metal_label, input_width=6, label_width=6, shift=6,
                 econ_data=econData, comm_data=priceData):

        # Concatenate the metal label into the economic data to make one dataset from which to pull data
        if econ_data is not None:
            self.data = pd.concat([econ_data, comm_data[metal_label]], axis=1)
        else:
            self.data = comm_data[metal_label]

        # Split (70:20:10, train/validation/test):
        self.data_length = len(self.data)
        self.trn_data = self.data[0:int(self.data_length * 0.7)]
        self.val_data = self.data[int(
            self.data_length * 0.7):int(self.data_length * 0.9)]
        self.tst_data = self.data[int(self.data_length * 0.9):]

        # Normalize the data:
        # Must use mean and standard deviation of training data, for appropriate rigor
        self.data_mean = self.trn_data.mean()
        self.data_stdv = self.trn_data.std()
        self.trn_data = (self.trn_data - self.data_mean) / self.data_stdv
        self.val_data = (self.val_data - self.data_mean) / self.data_stdv
        self.tst_data = (self.tst_data - self.data_mean) / self.data_stdv

        # Work out the label column indices.
        # metal_label but be a list with string name(s) of metals(s) from list
        self.metal_label = metal_label
        self.column_indices = {name: i for i, name in
                               enumerate(self.trn_data.columns)}

        # Work out the window parameters (input and label widths are
        # standard is 6 (i.e., 6 months back of information)
        self.input_width = input_width
        # standard is 1 (i.e., 1 month of prediction)
        self.label_width = label_width
        # standard is 1 (i.e., 1 month forward in prediction)
        self.shift = shift

        # standard is 6 back + 1 forward = *7*
        self.total_window_size = self.input_width + self.shift

        # standard is 'slice(0, 6, None)'
        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]    # std is 'array([0, 1, 2, 3, 4, 5])'

        self.label_start = self.total_window_size - \
            self.label_width                # standard is 7 - 1 = *6*
        # standard is 'slice(6, None, None)'
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]   # standard is 'array([6])'

    # Define output for self-calling a SampleGenerator object
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name: {self.metal_label}'])

    # SampleGenerator instance has a single object with all feature and label data.  Create a function, 'split_window',
    #   to separate single instance into two objects of features and labels over the same time frame
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        labels = tf.stack([labels[:, :, self.column_indices[name]]
                           for name in self.metal_label], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the 'tf.data.Datasets' are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # Using a time-series DataFrame object, convert to TensorFlow data.Dataset object in feature and label window pairs
    def make_dataset(self, data, batch=6):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            # numpy array containing consecutive-time data points
            data=data,
            targets=None,                               # set to 'None' to only yield input data
            # number of time steps in output sequence (std is *7*)
            sequence_length=self.total_window_size,
            # How many time steps to skip between each batch
            sequence_stride=1,
            # shuffle output sequences to improve model rigor
            shuffle=True,
            batch_size=batch, )                         # set batch size of Dataset (std is *6*)

        # Automatically separate data into feature and label sets
        ds = ds.map(self.split_window)

        return ds

    # Define property values for training, validating, and testing data
    @property
    def train(self):
        return self.make_dataset(self.trn_data)

    @property
    def validate(self):
        return self.make_dataset(self.val_data)

    @property
    def test(self):
        return self.make_dataset(self.tst_data)

    @property
    def example(self):
        # Get and cache an example batch of `inputs, labels` for plotting and asset investigation
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the '.train' dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    # Construct a function for viewing model outputs:
    def plot(self, model=None, plot_col=metals[0], max_subplots=3):
        # Pull a batch (std is *6*) of window values and save the input and label tensors
        inputs, labels = self.example
        # Generate a standard figure
        plt.figure(figsize=(12, 8))
        # Store the value of the label in the input column index (std is *14*)
        plot_col_index = self.column_indices[plot_col]
        # Plot subplots for each element in the batch (*6*) or max_sub, whichever is smaller
        max_n = min(max_subplots, len(inputs))
        # For each subplot:
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            # Each subplot will show real metal price
            plt.ylabel(f'{plot_col} Price [norm]')
            # Plot the price values for each of the training time steps (i.e., non-forecasted)
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Train', marker='.', zorder=-10)

            # The label index for a single list metal_label name is always 0
            if self.metal_label:
                label_col_index = 0
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            # If there is a label for the window, plot the labels (the actual values for each forecast)
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Actual', c='#2ca02c', s=64)
            # 'plot' works without a model (will ust show input and label prices); if there is a model, plot
            #  the predicted values for comparison with the label values (which will share an x-axis value)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Forecast',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Month')


# %% Run Generic SampleGenerator (6 months back to predict one month forward)

# Define a standard model window (1-month-ahead prediction from data up to six months behind)
# std_window = SampleGenerator(metal_label=metals)

# Define a forecasting model window (6-month-ahead prediction from data up to six-months behind) for the RNN_1
#   model, as defined in report
econ_1List = ['D12', 'b/m', 'tbl', 'AAA', 'ntis',
              'infl', 'ltr', 'corpr', 'svar', 'SPvw']
RNN_1_window = SampleGenerator(
    metal_label=metals, econ_data=econData[econ_1List])
# Define a forecasting model window for the RNN_2 model, as defined in report
if metals[0] == 'Aluminum':
    econ_list = ['D12', 'tbl', 'lty']
elif (metals[0] == 'Copper' or metals[0] == 'IronOre'):
    econ_list = ['E12', 'b/m', 'tbl', 'AAA', 'ntis']
else:    # for 'Nickel' or 'Zinc'
    econ_list = ['E12', 'ntis']
RNN_2_window = SampleGenerator(
    metal_label=metals, econ_data=econData[econ_list])
# Define a forecasting model window for the RNN_3 model, as defined in report
RNN_3_window = SampleGenerator(metal_label=metals, econ_data=None)

# Display and confirm batch and input/label sizes
# for example_inputs, example_labels in std_window.train.take(1):
#     print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#     print(f'Labels shape (batch, time, features): {example_labels.shape}')
# column_names = pd.concat([econData, priceData[metals]], axis=1).columns
# column_indices = {name: i for i, name in enumerate(column_names)}


# For ease in testing models later, define a function for testing separate models on separate windows
MAX_EPOCHS = 50


def compile_and_fit(model, window, patience=5):
    # Stop the model compiling if the value-loss parameter doesn't decrease at least once over two consecutive cycles
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',   # monitor validation loss, vice training
                                                      patience=patience,
                                                      mode='min')
    # Compile model with standard loss and optimizer values
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # Save weights only if the val_loss improves (decreases) and load those weights after model creation
    checkpoint_filepath = '/tmp/check_weights.h5'
    model_weight_saving = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    # Output model fit data for storing and comparing
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.validate,
                        callbacks=[early_stopping, model_weight_saving])
    # load best weights
    model.load_weights(checkpoint_filepath)
    return history


'''
The Time-Series Analysis (ARIMA modeling) previously performed, despite not otherwise having robust prediction quality,
provides insight into the underlying trend of the price data; that is, for each commodity, the principal trend is a
random walk.  Essentially, the commodity price for any particular month is the price from the previous month, altered
by a random fluctuation in with an experimentally-derived mean (not statistically different from 0) and standard
deviation.  Below is a baseline prediction algorithm that models this prediction behavior by predicting that the next
month's commodity price will be the previous month's commodity price; this is the baseline model.
'''

# Define Baseline Model (need a special subclass of keras.Model)


class Baseline(tf.keras.Model):
    # '__call__' is a generic function for whenever you reference just the class name
    def call(self, inputs):
        # return last price output for the next 6 time steps
        return tf.tile(inputs[:, -1:, :], [1, 6, 1])


'''
Machine Learning models with multiple layers can be complex to the point where the interactions between variables and
their calculated weights are no longer understandable to even the experienced machine learning programmer.  This effect 
can be positive since it allows more robust (and, on the negative side, potentially overfit) predicted values better matching the actual values;
however, it does not aid in a simplistic understanding of each factor's role within the model.  The simplest model that
can be built is a linear regression model; this model is simple enough that extracting the weights for each factor at 
each time step shows the correlated effect that factor has on the predicted price value.
'''
# Define a simple Linear model (this is used for discussing the role of each factor on the predicted price w/ 1 ts)
# linear = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1)
# ])

'''
The documentation for the tf.keras.layer.dense class identifies that a 3-rank tensor fed into the layer will be 'shrunk',
in a sense, into a 2-rank tensor via the computation of the dot product between the inputs and the kernel along the last
axis.  This means that a (6,6,11) input tensor fed into the single-Dense-layer model will return the weight for all the
time layers agglomerated together.  In order to get a single 2-D matrix showing the weights for each input at each time 
a Flattened and reshaped model is needed.  The tf.keras Flatten() function lays out all time layers with their input 
factors sequentially from 6-months past to present.  The final output will be reshaped accordingly to place the correct 
numerical weight with each time and input.
'''
# Define a larger Linear model (this is used for discussing the role of each factor on the predicted price w/ 6 ts)
# linMulti = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(units=1),
#     tf.keras.layers.Reshape([1, -1])
# ])


# Define a Recurring Neural Network (RNN) model for predicting up to 6 months out for RNN_1 (all economic features)
RNN_1_model = tf.keras.models.Sequential([
    # Shape [6, 6, 11] => [6, 4], because return_sequences is true, model predicts for each time step
    tf.keras.layers.LSTM(4, return_sequences=False),
    # Shape => [6, 66]
    tf.keras.layers.Dense(
        units=6*11, kernel_initializer=tf.initializers.zeros()),
    # Shape => [6, 6, 11]
    tf.keras.layers.Reshape([6, 11])
])

# Define a Recurring Neural Network (RNN) model for predicting up to 6 months out for RNN_2 (all moderately+ correlated features)
# econ_num is number of features in model, which is different for each metal
econ_num = 0
if metals[0] == 'Aluminum':
    econ_num = 3
elif (metals[0] == 'Copper' or metals[0] == 'IronOre'):
    econ_num = 5
else:    # for 'Nickel' or 'Zinc'
    econ_num = 2
RNN_2_model = tf.keras.models.Sequential([
    # Shape [6, 6, econ_num+1] => [6, 4], because return_sequences is true, model predicts for each time step
    tf.keras.layers.LSTM(4, return_sequences=False),
    # Shape => [6, 66]
    tf.keras.layers.Dense(
        units=6*int(econ_num+1), kernel_initializer=tf.initializers.zeros()),
    # Shape => [6, 6, econ_num+1]
    tf.keras.layers.Reshape([6, int(econ_num+1)])
])

# Define a Recurring Neural Network (RNN) model for predicting up to 6 months out for RNN_3 (just price)
RNN_3_model = tf.keras.models.Sequential([
    # Shape [6, 6, 1] => [6, 2], because return_sequences is true, model predicts for each time step
    tf.keras.layers.LSTM(2, return_sequences=False),
    # Shape => [6, 6]
    tf.keras.layers.Dense(
        units=6, kernel_initializer=tf.initializers.zeros()),
    # Shape => [6, 6, 1]
    tf.keras.layers.Reshape([6, 1])
])

# Initiate value and performance dictionary to compare future models with baseline
performance = {}

# %% Create a Baseline model and store results

# Evaluate the Baseline models with TensorFlow/Keras performance indicators
basePricePred = Baseline()

basePricePred.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

# Save test performance value
performance['Baseline'] = basePricePred.evaluate(RNN_3_window.test)

# Show three snippets of forecasting windows
RNN_3_window.plot(basePricePred)

# %% Create a Linear model and store results
# history = compile_and_fit(linear, single_window)
# Save weight outputs as an Excel file (kernel is the weights matrix taken from the first/only layer)
# single_weights = pd.Series(linear.layers[0].kernel[:, 0].numpy(),
#                            index=column_names)
# single_weights.to_excel(os.path.join(dwd, metals[0] + '_sing_weights.xlsx'))
# Review correlation of factors
# single_window.data.corr().to_excel(os.path.join(dwd, 'corr_factors.xlsx'))

# # %% Create a Larger Linear model and store results
# history = compile_and_fit(linMulti, std_window)
# # Save weight outputs as an Excel file
# multi_weights = pd.DataFrame(linMulti.layers[1].kernel[:, 0].numpy().reshape((6, -1)),
#                              columns=column_names, index=[-6, -5, -4, -3, -2, -1])
# multi_weights.to_excel(os.path.join(dwd, metals[0] + '_multi_weights.xlsx'))

# %% Create a 6-Step-Ahead RNN Network Model for RNN_1 and store results
history = compile_and_fit(RNN_1_model, RNN_1_window)
print('RNN_1 Summary:\n', RNN_1_model.summary())
performance['RNN_1'] = RNN_1_model.evaluate(RNN_1_window.test)

# %% Create a 6-Step-Ahead RNN Network Model for RNN_2 and store results
history = compile_and_fit(RNN_2_model, RNN_2_window)
print('RNN_2 Summary:\n', RNN_2_model.summary())
performance['RNN_2'] = RNN_2_model.evaluate(RNN_2_window.test)

# %% Create a 6-Step-Ahead RNN Network Model for RNN_2 and store results
history = compile_and_fit(RNN_3_model, RNN_3_window)
print('RNN_3 Summary:\n', RNN_3_model.summary())
performance['RNN_3'] = RNN_3_model.evaluate(RNN_3_window.test)

# Show three snippets of forecasting windows
# RNN_3_window.plot(RNN_3_model)

# %% Develope a time-delayed correlation value between each economic factor and aluminum price
# allData = single_window.data
# col_time = ['Alum_1', 'Alum_2', 'Alum_3', 'Alum_4', 'Alum_5']
# for i in range(len(col_time)):
#     temp = allData['Aluminum'][i+1:].reset_index()
#     temp['Date'] = allData.index[:-(i+1)]
#     temp.set_index('Date', inplace=True)
#     allData[col_time[i]] = temp
# corr_win = allData.corr()
# corr_win.iloc[0:15, 15:20].to_excel(os.path.join(dwd, 'ahead_corr.xlsx'))

# %% Develope a grouped chart for model performances for each commodity
BLRW = [0.07854, 0.04054, 0.1340, 0.04199, 0.1822]
RNN_1 = [1.058, 0.2539, 0.4265, 0.7710, 1.161]
RNN_2 = [0.3203, 0.07312, 0.2304, 0.2869, 0.8314]
RNN_3 = [0.1701, 0.1342, 0.1885, 0.02972, 0.5507]

# Set label locals and column width
x = np.arange(len(metal_list))
width = 0.20

fig, ax = plt.subplots()
col1 = ax.bar(x - 3*width/2, BLRW, width, label='BLRW')
col2 = ax.bar(x - width/2, RNN_1, width, label='RNN_1')
col3 = ax.bar(x + width/2, RNN_2, width, label='RNN_2')
col4 = ax.bar(x + 3*width/2, RNN_3, width, label='RNN_3')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('MSE Value')
ax.set_title('Mean Square Error of Models for each Commodity')
ax.set_xticks(x)
ax.set_xticklabels(metal_list)
ax.legend()

# ax.bar_label(col1, padding=3)
# ax.bar_label(col2, padding=3)
# ax.bar_label(col3, padding=3)
# ax.bar_label(col4, padding=3)

fig.tight_layout()

plt.show()
# %% Create residuals

# Define 6-month window for each test value
windowTest = np.zeros((25, 6, 1), dtype=np.float32)
for i in range(25):
    windowTest[i] = RNN_3_window.tst_data[i:6+i]
# Define RNN_3 residuals
testResidRNN_3 = RNN_3_model.predict(RNN_3_window.test) - windowTest
# Define baseline residuals
testResidBase = basePricePred.predict(RNN_3_window.test) - windowTest

# Two-sample t-test equation for    H0: u1 = u2 -> t = (mean1 - mean2)/sqrt(s1^2/n1 + s2^2/n2)
#                                   Ha: u1 > u2         where u2 is the RNN_3 mean absolute error
#   n1 = n2 = 162 price predictions for 36 months (multiple price predictions based on windows)
n_samp = 162
s1 = testResidBase.std()
s2 = testResidRNN_3.std()
# Set mean absolute error values from one instance of model computation (can change based on model instance)
if metals[0] == 'Aluminum':
    u2 = np.sqrt(0.1701)        # RNN_3 mean absolute error
    u1 = np.sqrt(0.07854)       # BLRW mean absolute error
elif metals[0] == 'Copper':
    u2 = np.sqrt(0.1342)
    u1 = np.sqrt(0.04054)
elif metals[0] == 'IronOre':
    u2 = np.sqrt(0.1885)
    u1 = np.sqrt(0.1340)
elif metals[0] == 'Nickel':
    u2 = np.sqrt(0.02972)
    u1 = np.sqrt(0.04199)
else:
    u2 = np.sqrt(0.5507)
    u1 = np.sqrt(0.1822)
t_samp = (u1 - u2)/np.sqrt(s1**2/n_samp + s2**2/n_samp)

# For 162-1 = 161 degrees of freedom and a 95% confidence interval
t_crit = 1.645
