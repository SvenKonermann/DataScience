import math
import pandas_datareader as web
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import plotly.express as px
import os
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

#Daten einlesen
df = pd.read_csv("IndizesName.csv", delimiter=',', header=0, index_col=False)

#Data Normalization
data = df.filter(['Adj Close'])
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
training_data_len
    
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len ,:]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
    
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
    
#Model

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
    
model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=["accuracy"])
    
model.fit(x_train, y_train, batch_size=1, epochs=1)
    
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
        
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
    
rmse = np.sqrt(np.mean(((predictions - y_test)**2)))
    
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show
    
plt.savefig('Adj Close'+'.png')
    
valid.to_csv('Adj Close'+'_valid.csv')
    
score=(sum(abs(valid['Adj Close']-valid['Predictions'])/valid['Adj Close'])/len(valid['Adj Close']))*100
score1=[]
score1.append(score)
        
acc_score=(1-sum(abs(valid['Adj Close']-valid['Predictions'])/valid['Adj Close'])/len(valid['Adj Close']))*100
acc_score1=[]
acc_score1.append(acc_score)

with open('score.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(score1)  
        
with open('accscore.csv', 'w') as f: 
    write = csv.writer(f) 
    write.writerow(acc_score1) 