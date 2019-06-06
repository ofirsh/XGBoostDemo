'''
Created on 2 Jun 2019

@author: Shalev
'''

'''
pip3 install graphviz
brew link graphviz

video
https://www.youtube.com/watch?v=m_BUKRG1Hv0&list=PLZnYQQzkMilqTC12LmnN4WpQexB9raKQG&index=5
'''

import numpy as np
import subprocess

from IPython.display import Image

from collections import Counter

from sklearn.datasets import make_classification
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score

# classifiers
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# reproducibility
seed = 104


X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)


print("Train label distribution:")
print(Counter(y_train))

print("\nTest label distribution:")
print(Counter(y_test))

'''
Train label distribution:
Counter({1: 404, 0: 396})

Test label distribution:
Counter({0: 106, 1: 94})
'''


# Single Decision Tree 

decision_tree = DecisionTreeClassifier(random_state=seed)

# train classifier
decision_tree.fit(X_train, y_train)

# predict output
decision_tree_y_pred  = decision_tree.predict(X_test)
decision_tree_y_pred_prob  = decision_tree.predict_proba(X_test)

# evaluation
decision_tree_accuracy = accuracy_score(y_test, decision_tree_y_pred)
decision_tree_logloss = log_loss(y_test, decision_tree_y_pred_prob)

print("== Decision Tree ==")
print("Accuracy: {0:.2f}".format(decision_tree_accuracy))
print("Log loss: {0:.2f}".format(decision_tree_logloss))
print("Number of nodes created: {}".format(decision_tree.tree_.node_count))

print('True labels:')
print(y_test[:5,])
print('\nPredicted labels:')
print(decision_tree_y_pred[:5,])
print('\nPredicted probabilities:')
print(decision_tree_y_pred_prob[:5,])

dt_viz_file = './images/dt.dot'
dt_png_file = './images/dt.png'

# create visualization
export_graphviz(decision_tree, out_file=dt_viz_file)

# convert to PNG
command = ["dot", "-Tpng", dt_viz_file, "-o", dt_png_file]
'''
subprocess.check_call(command)

# display image
Image(filename=dt_png_file)
'''

'''
AdaBoost 
In the example below we are creating a AdaBoost classifier running on 1000 iterations (1000 trees created). 
Also we are growing decision node up to first split (they are called decision stumps). We are also going to use SAMME algorithm which is 
    inteneded to work with discrete data (output from base_estimator is 0 or 1). Please refer to the documentation and here for more details.
'''

adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    algorithm='SAMME',
    n_estimators=1000,
    random_state=seed)

# train classifier
adaboost.fit(X_train, y_train)

# calculate predictions
adaboost_y_pred = adaboost.predict(X_test)
adaboost_y_pred_prob = adaboost.predict_proba(X_test)

# evaluate
adaboost_accuracy = accuracy_score(y_test, adaboost_y_pred)
adaboost_logloss = log_loss(y_test, adaboost_y_pred_prob)

print("== AdaBoost ==")
print("Accuracy: {0:.2f}".format(adaboost_accuracy))
print("Log loss: {0:.2f}".format(adaboost_logloss))

'''
The log-loss metrics is much lower than in single decision tree (mainly to the fact that now we obtain probabilities output). 
The accuracy is the same, but notice that the structure of the tree is much simpler. We are creating 1000 decision tree stumps.

Also here a quick peek into predicted values show that now 4 out of 5 first test instances are classified correctly.
'''

print('True labels:')
print(y_test[:5,])
print('\nPredicted labels:')
print(adaboost_y_pred[:5,])
print('\nPredicted probabilities:')
print(adaboost_y_pred_prob[:5,])

'''
Just for clarity, let's check how the first tree looks like.
'''

'''
ada_t1 = adaboost.estimators_[0]
ada_t1_viz_file = '../images/ada-t1.dot'
ada_t1_png_file = '../images/ada-t1.png'

# create visualization
export_graphviz(ada_t1, out_file=ada_t1_viz_file)

# convert to PNG
command = ["dot", "-Tpng", ada_t1_viz_file, "-o", ada_t1_png_file]
subprocess.check_call(command)

# display image
Image(filename=ada_t1_png_file)
'''

print("Error: {0:.2f}".format(adaboost.estimator_errors_[0]))
print("Tree importance: {0:.2f}".format(adaboost.estimator_weights_[0]))


'''
Gradient Boosted Trees
Let's construct a gradient boosted tree consiting of 1000 trees where each successive one will be created with gradient optimization. 
Again we are going to leave most parameters with their default values, specifiy only maximum depth of the tree to 1 (again decision stumps), 
    and setting warm start for more intelligent computations. Please refer to the docs if something is not clear.
'''

gbc = GradientBoostingClassifier(
    max_depth=1,
    n_estimators=1000,
    warm_start=True,
    random_state=seed)

gbc.fit(X_train, y_train)

# make predictions
gbc_y_pred = gbc.predict(X_test)
gbc_y_pred_prob = gbc.predict_proba(X_test)

# calculate log loss
gbc_accuracy = accuracy_score(y_test, gbc_y_pred)
gbc_logloss = log_loss(y_test, gbc_y_pred_prob)

print("== Gradient Boosting ==")
print("Accuracy: {0:.2f}".format(gbc_accuracy))
print("Log loss: {0:.2f}".format(gbc_logloss))


print('True labels:')
print(y_test[:5,])
print('\nPredicted labels:')
print(gbc_y_pred[:5,])
print('\nPredicted probabilities:')
print(gbc_y_pred_prob[:5,])

