import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Generate synthetic Data
m = 1.37
b = 0.65
arr_length = 400


x = np.random.rand(arr_length)

y = m*x + b

# add some noise
y = y + np.random.rand(arr_length)

plt.scatter(x, y)

x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.4)
x_train, x_test = x_train.reshape(-1, 1), x_test.reshape(-1, 1)




# Linear Regression
reg = LinearRegression()
regression_model = reg.fit(x_train, y_train)
regression_predictions = regression_model.predict(x_test)

#To retrieve the intercept:
print(regression_model.intercept_)

#For retrieving the slope:
print(regression_model.coef_)




# Neural Network
ann_model = tf.keras.Sequential()
ann_model.add(tf.keras.layers.Dense(1, input_shape = (1,)))
ann_model.compile(optimizer=tf.keras.optimizers.SGD(0.001, 0.9), loss='mae')
ann_model.fit(x_train, y_train, epochs=200)
ann_predictions = ann_model.predict(x_test)



print(regression_predictions)
print(ann_predictions)

#print average difference between the result of ann prediction and the result of linear regression prediction
print(np.average(regression_predictions - ann_predictions))
