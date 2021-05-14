import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from getIris import loadIris

np.random.seed(404)

df = loadIris()

x, y = df[df.columns[:-1]], df[df.columns[-1]]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, shuffle=True)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=1000, random_state=0)
kmeans.fit(xTrain, yTrain)
predict = kmeans.predict(xTest)

print("Mean Square Error:      {:.5f}".format(mean_squared_error(yTest, predict)))
print("Root Mean Square Error: {:.5f}".format(np.sqrt(mean_squared_error(yTest, predict))))
print("Mean Absolute Error:    {:.5f}".format(mean_absolute_error(yTest, predict)))

print("Test Value:    {}".format(yTest.values))
print("Predict Value: {}".format(predict))

plt.step(np.arange(len(yTest.values)),yTest.values,'o', label="test")
plt.step(np.arange(len(yTest.values)),predict,'x', label="predict")
plt.legend()
plt.show()
