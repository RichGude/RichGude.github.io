'''
Author:     Rich Gude
Purpose:    To amend metal commodity pricing data for time-series analysis
Revision:   1, dated April 1, 2021
'''

# Import Libraries
import os                   # for specifying working directory commands
import pandas as pd         # for csv file reading and dataFrame manipulation
import numpy as np          # for pandas value typing
import openpyxl             # for appending to excel files
import csv                  # for csv file reading and writing
import seaborn as sns       # For heat mapping
import matplotlib.pyplot as plt         # For graph illustration

from statsmodels.tsa.stattools import acf, pacf         # For ACF calculations
from statsmodels.tsa.stattools import adfuller          # For Augmented Dickey-Fuller Test
from statsmodels.tsa.arima_model import ARIMA                 # For ARIMA Model development
from sklearn.model_selection import train_test_split    # for splitting training and test data

# Load data and define other constants:
dwd = os.path.join(os.getcwd(), 'PriceData')                        # set data working directory
metal_list = ['Aluminum', 'Copper', 'IronOre', 'Nickel', 'Zinc']    # set list of metal names

ogPriceData = pd.read_excel(os.path.join(dwd, 'adjLogPrice.xlsx'), sheet_name='metalPrices', header=0)
conPriceData = pd.read_excel(os.path.join(dwd, 'adjLogPrice.xlsx'), sheet_name='1990Price', header=0)
adjPriceData = pd.read_excel(os.path.join(dwd, 'adjLogPrice.xlsx'), sheet_name='1990dif', header=0)

lag_num = 15                            # for ACF calculations
lag_arr = np.arange(0, lag_num + 1)     # for ACF plotting


# %% Define a function and test for stationarity of test data:

# Define and run an Augmented Dickey-Fuller (ADF) output function
def adf_calc(x, title):
    out = adfuller(x)
    print('\n\tThe ADF Output of the', title, 'price values are:')
    print('\t\tADF Stat: %f' % out[0])
    print('\t\tp-value:  %f' % out[1])


# Calculate the Augmented Dickey-Fuller Results of each original metal pricing
# for metal in metal_list:
#     adf_calc(ogPriceData[metal], metal)

'''
Problem Explanation:
Monthly commodity price data for five metal commodities, aluminum, copper, iron ore, nickel, and zinc, from 1990 to 2021
  are recorded by the International Monetary Fund, and distributed in a downloadable, accessible capacity by the Federal
  Reserve Bank of St. Louis, Economic Research Division.  This price data is collected monthly and amended in no other
  way, namely it is not adjusted for seasonal changes in price or year-over-year inflation changes.  For the purposes of
  time-series analysis, the price will be predicted from constant 1990-US dollars.
'''

# Calculate the Augmented Dickey-Fuller Results of each 1990-dollar metal pricing
# for metal in metal_list:
#     adf_calc(conPriceData[metal], '1990 ' + metal)

# Calculate the Augmented Dickey-Fuller Results of each 1990-dollar first-difference metal pricing
# for metal in metal_list:
#     adf_calc(adjPriceData[metal], '1990 first-difference ' + metal)
# print('\n')

'''
The Augmented Dickey-Fuller (ADF) Test is a test to measure the stationarity of data: the null hypothesis (H0) for the
ADF test is the time-series data has a unit root, meaning it is non-stationary, and the alternative hypothesis is the
time-series data does not have a unit root, meaning it is stationary.

The original and 1990-dollar price values are not stationary with time for all metals; however with p-values much less
than 0.05, coinciding with a 95% confidence interval, for the first-difference values, this set of data is considered 
stationary and will be used for time-series model building.
'''

# %% Define a generalized partial auto-correlation (GPAC) array for determining the order of an appropriate ARMA model:
'''
Problem Explanation:
For the given stationary commodity price data, an auto-regressive, moving average 
  (ARMA) model may be considered for modeling and forecasting the monthly change in price.  An ARMA model is comprised
  of two parts: the auto-regressive and moving average elements, each with their own order, a_n and b_n, respectively.
  Determining the appropriate ARMA model involves evaluating the appropriate order and the significance of the 
  coefficients for each order, and this is principally done through evaluation of the auto-correlation and partial auto-
  correlation function (ACF and PACF, respectively) values.  If the order of the both the auto-regressive and moving 
  average models is determined to be zero, the underlying data is considered random noise or, in the case of first-
  difference data, a random walk (i.e., the data is not capable of being modeled or predicted via ARMA modeling).
  
Determining the order of the auto-regressive and moving average models will be done via the generalized partial auto-
  correlation (GPAC) array method.  This method constitutes constructing a matrix of the theta values determined from a 
  series of calculations from auto-correlation function matrix determinants.
  
Because the evaluated price data is a first-difference, the original price data can also be modeled with an ARIMA
  (Auto-Regressive Integrated Moving Average) with an integrated order of 1 (d=1) and the same auto-regressive and
  moving average order and model coefficients.
'''
# initialize train and test datasets
trainData = pd.DataFrame({})
testData = pd.DataFrame({})
acfData = pd.DataFrame({})
pacfData = pd.DataFrame({})
fDIFFacf = pd.DataFrame({})
fDIFFpacf = pd.DataFrame({})

# Define test and training data for each metal:
for metal in metal_list:
    # Iron Ore is relatively stagnant in price until 2008; save a separate file for comparing iron ore model
    if metal == 'IronOre':
        date_trn_iron, date_tst_iron, price_trn_iron, price_tst_iron = train_test_split(conPriceData['Date'][216:], conPriceData[metal][216:], shuffle=False, test_size=0.08)

    trainData['Date'], testData['Date'], trainData[metal], testData[metal] = train_test_split(conPriceData['Date'], conPriceData[metal], shuffle=False, test_size=0.08)

# Define an ACF and PACF of the first-difference price data
for metal in metal_list:
    acfData[metal] = acf(conPriceData[metal], nlags=abs(lag_num))
for metal in metal_list:
    pacfData[metal] = pacf(conPriceData[metal], nlags=abs(lag_num))
for metal in metal_list:
    fDIFFacf[metal] = acf(adjPriceData[metal], nlags=abs(lag_num))
for metal in metal_list:
    fDIFFpacf[metal] = pacf(adjPriceData[metal], nlags=abs(lag_num))

metal_plt = 'Aluminum'
'''
# ACF Plots
plt.stem(lag_arr, acfData[metal_plt], use_line_collection=True)
plt.xlabel('ACF')
plt.ylabel('Lag Values')
plt.title('AutoCorrelation Values of ' + metal_plt + ' Zero Difference Prices')
plt.show()

# PACF Plots
plt.stem(lag_arr, pacfData[metal_plt], use_line_collection=True)
plt.xlabel('PACF')
plt.ylabel('Lag Values')
plt.title('Partial AutoCorrelation Values of ' + metal_plt + ' Zero Difference Prices')
plt.show()

# First Diff ACF Plots
plt.stem(lag_arr, fDIFFacf[metal_plt], use_line_collection=True)
plt.xlabel('ACF')
plt.ylabel('Lag Values')
plt.title('AutoCorrelation Values of ' + metal_plt + ' First Difference Prices')
plt.show()

# First Diff PACF Plots
plt.stem(lag_arr, fDIFFpacf[metal_plt], use_line_collection=True)
plt.xlabel('PACF')
plt.ylabel('Lag Values')
plt.title('Partial AutoCorrelation Values of ' + metal_plt + ' First Difference Prices')
plt.show()

# # Also plot post-2008 Iron Ore values:
# plt.stem(lag_arr, acf(price_trn_iron, nlags=abs(lag_num)), use_line_collection=True)
# plt.xlabel('ACF')
# plt.ylabel('Lag Values')
# plt.title('AutoCorrelation Values of Iron Ore First Difference Prices (post-2008)')
# plt.show()
#
# plt.stem(lag_arr, pacf(price_trn_iron, nlags=abs(lag_num)), use_line_collection=True)
# plt.xlabel('PACF')
# plt.ylabel('Lag Values')
# plt.title('Partial AutoCorrelation Values of Iron Ore First Difference Prices (post-2008)')
# plt.show()
'''
# For Aluminum: Create DataFrame for ACF and PACF values as save as Excel:
alumACF = pd.DataFrame({'Lag': np.arange(16)})
alumACF['ACF'] = acfData['Aluminum']
alumACF['PACF'] = pacfData['Aluminum']
alumACF.to_excel('TimeSeries/alumZeroACF.xlsx', sheet_name='Zero Diff')
'''
# Develop ARIMA Model for Aluminum
ar_order = 0
diff_order = 1
ma_order = 1

# For Filtered Iron Ore model fitting
# modelFE = ARIMA(price_trn_iron, order=(ar_order, diff_order, ma_order), dates=date_trn_iron)
# resFE = modelFE.fit()
# print(resFE.summary())
#
# # Produce Model Fit data for shortened Iron Ore dates
# predValues = resFE.predict(1, 157 - diff_order)     # Predict out to 157 months (from 1990 to 2021) minus diff order
# if diff_order == 1:
#     forecast = conPriceData[metal_plt][216:].copy().reset_index(drop=True)
#     for i in range(len(predValues)):
#         forecast[i+1] = forecast[i] + predValues[i]
# else:
#     forecast = predValues
#
# # Predict out to 371 monthly points (from 1990 to 2021)
# plt.plot(conPriceData['Date'][216:], forecast, label="Forecast")
# plt.plot(conPriceData['Date'][216:], conPriceData[metal_plt][216:], label="OG")
# plt.title(f"({ar_order}, {diff_order}, {ma_order}) Model for {metal_plt}")
# plt.legend()
# plt.show()


modelAL = ARIMA(trainData[metal_plt], order=(ar_order, diff_order, ma_order), dates=trainData['Date'])
resAL = modelAL.fit()
print(resAL.summary())

# Produce Model Fit data
predValues = resAL.predict(1, 373 - diff_order)     # Predict out to 371 months (from 1990 to 2021) minus diff order
if diff_order == 1:
    forecast = conPriceData[metal_plt].copy()
    for i in range(len(predValues)):
        forecast[i+1] = forecast[i] + predValues[i]
else:
    forecast = predValues

# Predict out to 371 monthly points (from 1990 to 2021)
plt.plot(conPriceData['Date'], forecast, label="Forecast")
plt.plot(conPriceData['Date'], conPriceData[metal_plt], label="OG")
plt.title(f"({ar_order}, {diff_order}, {ma_order}) Model for {metal_plt}")
plt.legend()
plt.show()

# Save forecast Data
graphData = pd.DataFrame({'Date': conPriceData['Date'].copy()})
graphData['Real Price'] = conPriceData[metal_plt].copy()
graphData['Model Price'] = forecast
graphData.to_excel('TimeSeries/' + str(metal_plt) + '.xlsx', sheet_name=str(metal_plt))
'''





