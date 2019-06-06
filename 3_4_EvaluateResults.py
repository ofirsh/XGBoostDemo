'''
Created on 2 Jun 2019

@author: Shalev
'''

import numpy as np
import xgboost as xgb

from pprint import pprint

# reproducibility
seed = 123
np.random.seed(seed)

dtrain = xgb.DMatrix('./data/agaricus.txt.train')
dtest = xgb.DMatrix('./data/agaricus.txt.test')

# specify general training parameters
params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5


watchlist  = [(dtest,'test'), (dtrain,'train')]

bst = xgb.train(params, dtrain, num_rounds, watchlist)
'''
[0]    test-error:0.11049    train-error:0.113926
[1]    test-error:0.11049    train-error:0.113926
[2]    test-error:0.03352    train-error:0.030401
[3]    test-error:0.027312    train-error:0.021495
[4]    test-error:0.031037    train-error:0.025487
'''

params['eval_metric'] = 'logloss'
bst = xgb.train(params, dtrain, num_rounds, watchlist)
'''
[0]    test-logloss:0.457893    train-logloss:0.460117
[1]    test-logloss:0.383914    train-logloss:0.378727
[2]    test-logloss:0.312679    train-logloss:0.308074
[3]    test-logloss:0.269116    train-logloss:0.261396
[4]    test-logloss:0.239746    train-logloss:0.232171
'''

params['eval_metric'] = ['logloss', 'auc']
bst = xgb.train(params, dtrain, num_rounds, watchlist)
'''
[0]    test-logloss:0.457893    test-auc:0.892138    train-logloss:0.460117    train-auc:0.888997
[1]    test-logloss:0.383914    test-auc:0.938901    train-logloss:0.378727    train-auc:0.942881
[2]    test-logloss:0.312679    test-auc:0.976157    train-logloss:0.308074    train-auc:0.981415
[3]    test-logloss:0.269116    test-auc:0.979685    train-logloss:0.261396    train-auc:0.985158
[4]    test-logloss:0.239746    test-auc:0.9785    train-logloss:0.232171    train-auc:0.983744
'''


'''
Creating custom evaluation metric
----------------------------------
In order to create our own evaluation metric, the only thing needed to do is to create a method taking two arguments - 
    predicted probabilities and DMatrix object holding training data.

In this example our classification metric will simply count the number of misclassified examples assuming that classes with  p>0.5 are positive. 
You can change this threshold if you want more certainty.

The algorithm is getting better when the number of misclassified examples is getting lower. Remember to also set the argument maximize=False while training.

'''
# custom evaluation metric
def misclassified(pred_probs, dtrain):
    labels = dtrain.get_label() # obtain true labels
    preds = pred_probs > 0.5 # obtain predicted values
    return 'misclassified', np.sum(labels != preds)

bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False)
'''
[0]    test-logloss:0.457893    test-auc:0.892138    train-logloss:0.460117    train-auc:0.888997    test-misclassified:178    train-misclassified:742
[1]    test-logloss:0.383914    test-auc:0.938901    train-logloss:0.378727    train-auc:0.942881    test-misclassified:178    train-misclassified:742
[2]    test-logloss:0.312679    test-auc:0.976157    train-logloss:0.308074    train-auc:0.981415    test-misclassified:54    train-misclassified:198
[3]    test-logloss:0.269116    test-auc:0.979685    train-logloss:0.261396    train-auc:0.985158    test-misclassified:44    train-misclassified:140
[4]    test-logloss:0.239746    test-auc:0.9785    train-logloss:0.232171    train-auc:0.983744    test-misclassified:50    train-misclassified:166
'''

'''
Extracting the evaluation results
---------------------------------
You can get evaluation scores by declaring a dictionary for holding values and passing it as a parameter for evals_result argument.
'''

evals_result = {}
bst = xgb.train(params, dtrain, num_rounds, watchlist, feval=misclassified, maximize=False, evals_result=evals_result)


pprint(evals_result)
'''
{'test': {'auc': [0.892138, 0.938901, 0.976157, 0.979685, 0.9785],
          'logloss': [0.457893, 0.383914, 0.312679, 0.269116, 0.239746],
          'misclassified': [178.0, 178.0, 54.0, 44.0, 50.0]},
 'train': {'auc': [0.888997, 0.942881, 0.981415, 0.985158, 0.983744],
           'logloss': [0.460117, 0.378727, 0.308074, 0.261396, 0.232171],
           'misclassified': [742.0, 742.0, 198.0, 140.0, 166.0]}}

'''

'''
Early stopping
--------------
There is a nice optimization trick when fitting multiple trees.

You can train the model until the validation score stops improving. 
Validation error needs to decrease at least every early_stopping_rounds to continue training. 
This approach results in simpler model, because the lowest number of trees will be found (simplicity).

In the following example a total number of 1500 trees is to be created, but we are telling it to stop if the validation score does not improve for last ten iterations.
'''

params['eval_metric'] = 'error'
num_rounds = 1500

bst = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=10)

'''
When using early_stopping_rounds parameter resulting model will have 3 additional fields - 
    bst.best_score 
    bst.best_iteration
    bst.best_ntree_limit.
'''

print("Booster best train score: {}".format(bst.best_score))
print("Booster best iteration: {}".format(bst.best_iteration))
print("Booster best number of trees limit: {}".format(bst.best_ntree_limit))

'''
Booster best train score: 0.001996
Booster best iteration: 13
Booster best number of trees limit: 14
'''




print('done')