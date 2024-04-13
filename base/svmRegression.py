from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error 
import numpy as np

def run(filePath):
    df = pd.read_csv(filePath)

    x = df.iloc[:, 0:10].values  # the first 10 are independent variables
    y = df.iloc[:, 10:11].values  # the last 2 are depended variables


    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    y_train = np.reshape(y_train, (-1, y_train.shape[-1])).ravel()

    regressor = svm.SVR()

    regressor.fit(x_train, y_train)

    predicted = regressor.predict(x_test)
    print(mean_absolute_error(y_test,predicted))


run('../data/wine.csv')
