import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def loadIris():
    # creating panda dataframe
    data = load_iris()
    df = pd.DataFrame(data.data,columns=data.feature_names)
    df['target'] = data.target
    return df

if __name__ == "__main__":
    pass
