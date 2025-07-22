import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

#Read data
datapath = 'PRSA_data_2010.1.1-2014.12.31.csv'
df = pd.read_csv(datapath)

#Preprocess Data
df['datetime'] = pd.to_datetime(df[['year','month','day','hour']])
df.set_index('datetime', inplace= True)
df['TEMP'] = df['TEMP'].ffill()

def createlag(series, lag=24):
    X,y = [], []
    for i in range(lag, len(series)):
        X.append(series[i-lag:i])
        y.append(series[i])
    return np.array(X), np.array(y)

#Split to training:test
X,y = createlag(df['TEMP'].values, lag=24)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, shuffle=False)

model = RandomForestRegressor(n_estimators = 100, random_state = 42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("mse:", mse)
print("mae:", mae)

plt.figure(figsize = (12,6))
plt.plot(y_test[:1000], label = "True Temp")
plt.plot(y_pred[:1000], label = "Predicted Temp")
plt.legend()
plt.title("True vs Predicted")
plt.show()