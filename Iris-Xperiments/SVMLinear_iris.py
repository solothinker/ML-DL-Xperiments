import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from getIris import loadIris

df = loadIris()
x,y = df[df.columns[:-1]],df[df.columns[-1]]
xTrain,xTest,yTrain,yTest = train_test_split(x,y,test_size=0.1,shuffle=True)

print(xTrain.shape,xTest.shape,yTrain.shape,yTest.shape)

lrSVC = LinearSVC(max_iter=10000)
lrSVC.fit(xTrain,yTrain)
predict = lrSVC.predict(xTest)

print("Mean Square Error:      {:.5f}".format(mean_squared_error(yTest,predict)))
print("Root Mean Square Error: {:.5f}".format(np.sqrt(mean_squared_error(yTest,predict))))
print("Mean Absolute Error:    {:.5f}".format(mean_absolute_error(yTest,predict)))

print("Test Value:    {}".format(yTest.values))
print("Predict Value: {}".format(predict))
