'''
Created on 2 Jun 2019

@author: Shalev
'''

'''
pip3 install xgboost

curl -O  https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train
curl -O  https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test
'''

import numpy as np
import xgboost as xgb

'''
Loading data
------------
We are going to use bundled Agaricus dataset which can be downloaded here.
https://github.com/dmlc/xgboost/tree/master/demo/data


This data set records biological attributes of different mushroom species, and the target is to predict whether it is poisonous

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family. Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom;

It consist of 8124 instances, characterized by 22 attributes (both numeric and categorical). The target class is either 0 or 1 which means binary classification problem.

Important: XGBoost handles only numeric variables.

Lucily all the data have alreay been pre-process for us. Categorical variables have been encoded, and all instances divided into train and test datasets. You will know how to do this on your own in later lectures.

Data needs to be stored in DMatrix object which is designed to handle sparse datasets. It can be populated in couple ways:

using libsvm format txt file,
using Numpy 2D array (most popular),
using XGBoost binary buffer file
In this case we'll use first option.

Libsvm files stores only non-zero elements in format

<label> <feature_a>:<value_a> <feature_c>:<value_c> ... <feature_z>:<value_z>

Any missing features indicate that it's corresponding value is 0.

'''


dtrain = xgb.DMatrix('./data/agaricus.txt.train')
dtest = xgb.DMatrix('./data/agaricus.txt.test')

print("Train dataset contains {0} rows and {1} columns".format(dtrain.num_row(), dtrain.num_col()))
print("Test dataset contains {0} rows and {1} columns".format(dtest.num_row(), dtest.num_col()))


print("Train possible labels: ")
print(np.unique(dtrain.get_label()))

print("\nTest possible labels: ")
print(np.unique(dtest.get_label()))

'''
Specify training parameters
Let's make the following assuptions and adjust algorithm parameters to it:

we are dealing with binary classification problem ('objective':'binary:logistic'),
we want shallow single trees with no more than 2 levels ('max_depth':2),
we don't any oupout ('silent':1),
we want algorithm to learn fast and aggressively ('eta':1),
we want to iterate only 5 rounds
'''

params = {
    'objective':'binary:logistic',
    'max_depth':2,
    'silent':1,
    'eta':1
}

num_rounds = 5

bst = xgb.train(params, dtrain, num_rounds)

watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only
bst = xgb.train(params, dtrain, num_rounds, watchlist)

preds_prob = bst.predict(dtest)
preds_prob

labels = dtest.get_label()
preds = preds_prob > 0.5 # threshold
correct = 0

for i in range(len(preds)):
    if (labels[i] == preds[i]):
        correct += 1

print('Predicted correctly: {0}/{1}'.format(correct, len(preds)))
print('Error: {0:.4f}'.format(1-correct/len(preds)))


'''
[21:01:13] 6513x127 matrix with 143286 entries loaded from ./data/agaricus.txt.train
[21:01:13] 1611x127 matrix with 35442 entries loaded from ./data/agaricus.txt.test
Train dataset contains 6513 rows and 127 columns
Test dataset contains 1611 rows and 127 columns
Train possible labels: 
[0. 1.]

Test possible labels: 
[0. 1.]
[0]    test-error:0.042831    train-error:0.046522
[1]    test-error:0.021726    train-error:0.022263
[2]    test-error:0.006207    train-error:0.007063
[3]    test-error:0.018001    train-error:0.0152
[4]    test-error:0.006207    train-error:0.007063
Predicted correctly: 1601/1611
Error: 0.0062
'''

