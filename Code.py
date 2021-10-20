


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score

train = pd.read_csv(r'C:\Users\alex_\Desktop\Kaggle\Tabular playground - oct\train.csv')
test = pd.read_csv(r'C:\Users\alex_\Desktop\Kaggle\Tabular playground - oct\test.csv')


print(train.columns)
print(train.info())
print(train.describe())
print(train.isnull().sum())
print(train.nunique())
print(train.dtypes)
print(train['target'].value_counts())


X = train.drop('target', axis=1) 
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgboost = XGBClassifier(random_state=42)
xgboost.fit(X_train, y_train)

predictions = xgboost.predict(X_test)

roc_score = roc_auc_score(y_test, predictions)

print(roc_score)





import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
params = { 'max_depth': [3,6,10,12],
           'n_estimators': [50, 100, 500, 1000]}
xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='roc_auc', 
                   verbose=1)
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Best ROC: ", (-clf.best_score_))



# plot
pyplot.errorbar(max_depth, means, yerr=stds)
pyplot.title("XGBoost max_depth vs Log Loss")
pyplot.xlabel('max_depth')
pyplot.ylabel('Log Loss')
pyplot.savefig('max_depth.png')









