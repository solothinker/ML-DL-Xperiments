#Implementing Ridge regression and classification
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge,RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

from getIris import loadIris

np.random.seed(404)

df = loadIris()
x, y = df[df.columns[:-1]], df[df.columns[-1]]
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.1, shuffle=True)
print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)

def modelCal(model):
    predict = model.predict(xTest)
    print("Coeff            : {}".format(model.coef_))
    print("intercept        : {}".format(model.intercept_))

    print("Mean Square Error: {:.2f}".format(mean_squared_error(yTest, predict)))
    print("r2 Score         : {:.2f}".format(r2_score(yTest, predict)))
    print("-------------------------")
    plt.figure()
    plt.step(np.arange(len(yTest)),yTest,'o', label="test")
    plt.step(np.arange(len(yTest)),predict,'x', label="predict")
    plt.legend()
    plt.show()
    
print("Ridge Regression ")    
model = Ridge(alpha=0.5)
model.fit(xTrain,yTrain)
modelCal(model)

print("Ridge Classification")
model = RidgeClassifier(alpha=0.5)
model.fit(xTrain,yTrain)
modelCal(model)

