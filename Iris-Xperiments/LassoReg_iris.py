#LASSO,ElastiNet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso,ElasticNet,LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from getIris import loadIris

np.random.seed(404)

df = loadIris()
train,target = df[df.columns[0:-1]],df[df.columns[-1]]
xTrain,xTest,yTrain,yTest = train_test_split(train,target,test_size=0.15,shuffle=True)
print(xTrain.shape,xTest.shape,yTrain.shape,yTest.shape)

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

alpha = 0.5
# Lasso Model
print("Lasso Model")
lModel = Lasso(alpha=alpha)
lModel.fit(xTrain,yTrain)
modelCal(lModel)

print("LassoCV Model")
lcvModel = LassoCV(cv=10, random_state=True)
lcvModel.fit(xTrain,yTrain)
modelCal(lcvModel)

print("ElastiNet Model")
mtlModel = ElasticNet(alpha=1.)
mtlModel.fit(xTrain,yTrain)
modelCal(mtlModel)

print("ElasticNetCV Model")
encvModel = LassoCV(cv=10, random_state=True)
encvModel.fit(xTrain,yTrain)
modelCal(encvModel)
