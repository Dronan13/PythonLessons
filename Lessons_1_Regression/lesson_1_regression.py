import pandas as pd
import quandl
import math
import numpy as np

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#loads dataframe from quandl
df = quandl.get('WIKI/GOOGL') 
#print(df.head())
#tcRqukPKR9otxWkDdMf6 api key


# save to csv
#file_name = 'GOOGL.csv'
#df.to_csv(file_name, index_label='Date') 

# reads from csv
#df = pd.read_csv(file_name) 
# use 'Date', aster saving to csv
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close']*100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open']*100

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#print(df.head()) 

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace = True) #replace non

#math.ceil - Return the ceiling of x as a float, 
#the smallest integer value greater than or equal to x.
forecast_out = int(math.ceil(0.01*len(df))) 
print('Prediction for {} days'.format(forecast_out)) 

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

x = np.array(df.drop(['label'],1))
y = np.array(df['label'])

x = preprocessing.scale(x)

#x = x[:-forecast_out+1]
df.dropna(inplace = True)

y = np.array(df['label'])

#print(len(x), len(y))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = LinearRegression() # n_jobs= x number of threads
clf.fit(x_train, y_train) # train on data
accuracy = clf.score(x_test, y_test) # test on data

print('Linear regression accuracy ', accuracy)

clf = svm.SVR()
clf.fit(x_train, y_train) # train on data
accuracy = clf.score(x_test, y_test) # test on data

print('Support vector regression accuracy ', accuracy)





