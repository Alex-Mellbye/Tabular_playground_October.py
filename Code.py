


import pandas as pd
import numpy as np

from xgboost import XGBClassifier

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate


train = pd.read_csv(r'C:\...\train.csv')
test = pd.read_csv(r'C:\...\test.csv')


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


# Performing gridsearch to find best parameters for max_depth

params = { 'max_depth': [3,6,10,12]}
xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr, 
                   param_grid=params,
                   scoring='roc_auc', 
                   verbose=1)
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
print("Best ROC: ", (-clf.best_score_))

# Best parameters: {'max_depth': 3}
# Best ROC:  -0.850952134051909


# Cross-validation
xgboost = XGBClassifier(random_state=42, max_depth=3)
scoring = "roc_auc"
xgboost_scores = cross_validate(xgboost, X_train, y_train, scoring=scoring, return_estimator=True)
print(xgboost_scores["test_score"].mean())

# Retrain the model and evaluate
xgboost = sklearn.base.clone(xgboost)
xgboost.fit(X_train, y_train)
print("Test set ROC:", roc_auc_score(y_test, xgboost.predict(X_test)))
print("Mean validation ROC:", xgboost_scores["test_score"].mean())


################## Going to test dataset #################

# Some quick tests to check that everything is in order
print(test.shape)
print(test.columns)
print(test.isnull().sum())
print(test.nunique())
print(test.dtypes)

# Running the predictions
predictions = xgboost.predict_proba(test)[:,1]
predictions = np.round(predictions, 1)

# Producing the results in the required format for the submission
subm = test[['id']]
subm['target'] = predictions

print(subm.head())

subm.to_csv(r'C:\...\PT_oct_submission.csv', index = False)

# Final result was a ROC of 0.84573







