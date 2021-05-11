import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

data = datasets.load_iris()
xTrain,xTest,yTrain,yTest = train_test_split(data.data,data.target,
                                             test_size=0.1,shuffle=True)
print(xTrain.shape,xTest.shape,yTrain.shape,yTest.shape)

lr = LinearRegression()
lr.fit(xTrain,yTrain)
predict = lr.predict(xTest)

print("Mean Square Error:      {}".format(mean_squared_error(yTest,predict)))
print("Root Mean Square Error: {}".format(np.sqrt(mean_squared_error(yTest,predict))))
print("Mean Absolute Error:    {}".format(mean_absolute_error(yTest,predict)))




