'''
Created on 2 Jun 2019

@author: Shalev
'''

import numpy as np

from sklearn.datasets import load_svmlight_files
from sklearn.metrics import accuracy_score

from xgboost.sklearn import XGBClassifier

'''
We are going to use the same dataset as in previous lecture. 
The scikit-learn package provides a convenient function load_svmlight capable of reading many libsvm files at once and storing them as Scipy's sparse matrices.
'''

X_train, y_train, X_test, y_test = load_svmlight_files(('./data/agaricus.txt.train', './data/agaricus.txt.test'))

print("Train dataset contains {0} rows and {1} columns".format(X_train.shape[0], X_train.shape[1]))
print("Test dataset contains {0} rows and {1} columns".format(X_test.shape[0], X_test.shape[1]))

print("Train possible labels: ")
print(np.unique(y_train))

print("\nTest possible labels: ")
print(np.unique(y_test))

params = {
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 1.0,
    'silent': True,
    'n_estimators': 5
}

bst = XGBClassifier(**params).fit(X_train, y_train)

preds = bst.predict(X_test)
preds

correct = 0

for i in range(len(preds)):
    if (y_test[i] == preds[i]):
        correct += 1
        

        
acc = accuracy_score(y_test, preds)

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-acc))


'''
Train dataset contains 6513 rows and 126 columns
Test dataset contains 1611 rows and 126 columns
Train possible labels: 
[0. 1.]

Test possible labels: 
[0. 1.]
Predicted correctly: 1601/1611
Error: 0.0062
'''