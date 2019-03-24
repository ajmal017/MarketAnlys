import numpy as np
import pandas as pd 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

master = pd.read_csv("test.csv")
master["AUD"].fillna(master["AUD"].mean(), inplace=True)
master["CAD"].fillna(master["CAD"].mean(), inplace=True)
master["CHF"].fillna(master["CHF"].mean(), inplace=True)
master["GBP"].fillna(master["GBP"].mean(), inplace=True)
master["JPY"].fillna(master["JPY"].mean(), inplace=True)
master["MXN"].fillna(master["MXN"].mean(), inplace=True)

X = master.drop(["Time","USD"], axis=1)
y = master["USD"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svr_rbf = SVR(kernel="rbf")
svr_rbf.fit(X_train, y_train)

pred = svr_rbf.predict(X_test)
print(svr_rbf.score(X_test, y_test))