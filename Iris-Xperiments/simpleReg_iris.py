import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

data = datasets.load_iris()
xTrain, xTest, yTrain, yTest = train_test_split(
    data.data, data.target, test_size=0.1, shuffle=True
)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

def linearRegression(Positive=False):
    lr = LinearRegression(positive=Positive)
    lr.fit(xTrain, yTrain)
    predict = lr.predict(xTest)

    print("Mean Square Error:{:.2f}".format(mean_squared_error(yTest, predict)))
    print("r2 Score:         {:.2f}".format(r2_score(yTest, predict)))
    plt.figure()
    plt.step(np.arange(len(yTest)),yTest,'o', label="test")
    plt.step(np.arange(len(yTest)),predict,'x', label="predict")
    plt.legend()
    plt.show()

print("Ordinary Least Square")
linearRegression()
print("---------------------")
print("Non-Negative linear least square")
linearRegression(Positive=True)
