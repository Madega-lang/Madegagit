import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import kpss, adfuller
import matplotlib.pyplot as plt
data = pd.read_excel("gold Analytic data (Historical data).xlsx", usecols=['Date', 'Close', 'rt'])
rt = data['rt']
print(rt.dtype)
#rt = pd.to_numeric(data['rt'], errors='coerce').dropna()
data['Date'] = pd.to_datetime(data['Date'])

# Set Date as index (important for time series)
data = data.set_index('Date')

print(data.head())
print(data.tail())
print(data.shape)
print(rt.describe()) # summary statistics
print(skew(rt))
print(kurtosis(rt, fisher=False)) #pearson kurtosis

# Plot Close price
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Gold Close Price')

plt.title("Gold Price Time Series")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)

#plt.show()

plt.figure(figsize=(12,6))
plt.plot(data['rt'], label='Returns (rt)', color='orange')

plt.title("Gold Returns Time Series")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


#Stationary tests
adf_result = adfuller(rt)

print("ADF Test Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])