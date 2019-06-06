'''
Created on 2 Jun 2019

@author: Shalev
'''

'''
Handle Imbalanced Dataset
-------------------------
There are plenty of examples in real-world problems that deals with imbalanced target classes. 
Imagine medical data where there are only a few positive instances out of thousands of negatie (normal) ones. 
Another example might be analyzing fraud transaction, in which the actual frauds represent only a fraction of all available data.

Imbalanced data refers to a classification problems where the classes are not equally distributed.

You can read good introduction about tackling imbalanced datasets here.

General advices
----------------
These are some common tactics when approaching imbalanced datasets:

* collect more data,
* use better evaluation metric (that notices mistakes - ie. AUC, F1, Kappa, ...),
* try oversampling minority class or undersampling majority class,
* generate artificial samples of minority class (ie. SMOTE algorithm)

In XGBoost you can try to:
--------------------------

* make sure that parameter min_child_weight is small (because leaf nodes can have smaller size groups), it is set to min_child_weight=1 by default,
* assign more weights to specific samples while initalizing DMatrix,
* control the balance of positive and negative weights using set_pos_weight parameter,
* use AUC for evaluation

'''
import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# reproducibility
seed = 123

'''
We'll use a function to generate dataset for binary classification. 
To assure that it's imbalanced use weights parameter. 
In this case there will be 200 samples each described by 5 features, but only 10% of them (about 20 samples) will be positive. That makes the problem harder.
'''

X, y = make_classification(
    n_samples=200,
    n_features=5,
    n_informative=3,
    n_classes=2,
    weights=[.9, .1],
    shuffle=True,
    random_state=seed
)

print('There are {} positive instances.'.format(y.sum()))
# There are 20 positive instances.

# Divide created data into train and test. Remember so that both datasets should be similiar in terms of distribution, so they need stratification.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y, random_state=seed)

print('Total number of postivie train instances: {}'.format(y_train.sum()))
print('Total number of positive test instances: {}'.format(y_test.sum()))
'''
Total number of postivie train instances: 13
Total number of positive test instances: 7
'''

'''
Baseline model
--------------
In this approach try to completely ignore the fact that classed are imbalanced and see how it will perform. Create DMatrix for train and test data.
'''

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

'''
Assume that we will create 15 decision tree stumps, solving binary classification problem, where each next one will be train very aggressively.
These parameters will also be used in consecutive examples.
'''

params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':True,
    'eta':1
}

num_rounds = 15

bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

# Let's see how the confusion matrix looks like.

z = pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)

print(z)
'''
Predicted   0  1  All
Actual               
0          58  1   59
1           5  2    7
All        63  3   66

'''

'''
We can also present the performance using 3 different evaluation metrics:

* accuracy,
* precision (the ability of the classifier not to label as positive a sample that is negative),
* recall (the ability of the classifier to find all the positive samples).
'''

print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))

'''
Accuracy: 0.91
Precision: 0.67
Recall: 0.29
'''

'''
Intuitively we know that the foucs should be on finding positive samples. 
First results are very promising (91% accuracy - wow), but deeper analysis show that the results are biased towards majority class - 
    we are very poor at predicting the actual label of positive instances
'''

'''
Custom weights
--------------
Try to explicitly tell the algorithm what important using relative instance weights. 
Let's specify that positive instances have 5x more weight and add this information while creating DMatrix.
'''

weights = np.zeros(len(y_train))
weights[y_train == 0] = 1
weights[y_train == 1] = 5

dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights) # weights added
dtest = xgb.DMatrix(X_test)

# Train the classifier and get predictions (same as in baseline):

bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

# Inspect the confusion matrix, and obtained evaluation metrics:

z = pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)

print(z)
'''
Predicted   0  1  All
Actual               
0          57  2   59
1           4  3    7
All        61  5   66
'''

print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))
'''
Accuracy: 0.91
Precision: 0.60
Recall: 0.43
'''

'''
You see that we made a trade-off here. 
We are now able to better classify the minority class, but the overall accuracy and precision decreased. 
Test multiple weights combinations and see which one works best.
'''

'''
Use scale_pos_weight parameter
-----------------------------
You can automate the process of assigning weights manually by calculating the proportion between negative and positive instances and setting it to scale_pos_weight parameter.

Let's reinitialize datasets.
'''

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)

# Calculate the ratio between both classes and assign it to a parameter.
train_labels = dtrain.get_label()

ratio = float(np.sum(train_labels == 0)) / np.sum(train_labels == 1)
params['scale_pos_weight'] = ratio
print('0 / 1 ratio ={}'.format(ratio))
# 0 / 1 ratio =9.307692307692308

bst = xgb.train(params, dtrain, num_rounds)
y_test_preds = (bst.predict(dtest) > 0.5).astype('int')

z = pd.crosstab(
    pd.Series(y_test, name='Actual'),
    pd.Series(y_test_preds, name='Predicted'),
    margins=True
)

print(z)

'''
Predicted   0  1  All
Actual               
0          56  3   59
1           4  3    7
All        60  6   66
'''

print('Accuracy: {0:.2f}'.format(accuracy_score(y_test, y_test_preds)))
print('Precision: {0:.2f}'.format(precision_score(y_test, y_test_preds)))
print('Recall: {0:.2f}'.format(recall_score(y_test, y_test_preds)))
'''
Accuracy: 0.89
Precision: 0.50
Recall: 0.43
'''






print('done')

