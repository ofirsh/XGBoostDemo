'''
Created on 2 Jun 2019

@author: Shalev
'''

import numpy as np

from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

from scipy.stats import randint, uniform

# reproducibility
seed = 342
np.random.seed(seed)

X, y = make_classification(n_samples=1000, n_features=20, n_informative=8, n_redundant=3, n_repeated=2, random_state=seed)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cv = skf.get_n_splits(y)

# Grid-Search
params_grid = {
    'max_depth': [1, 2, 3],
    'n_estimators': [5, 10, 25, 50],
    'learning_rate': np.linspace(1e-16, 1, 3)
}

params_fixed = {
    'objective': 'binary:logistic',
    'silent': True}

bst_grid = GridSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_grid=params_grid,
    cv=cv,
    scoring='accuracy')

'''
Before running the calculations notice that  3∗4∗3∗10=3603∗4∗3∗10=360  models will be created to test all combinations. 
You should always have rough estimations about what is going to happen.
'''

bst_grid.fit(X, y)


# bst_grid.grid_scores_
print(bst_grid.cv_results_)

print("Best accuracy obtained: {0}".format(bst_grid.best_score_))
print("Parameters:")
for key, value in bst_grid.best_params_.items():
    print("\t{}: {}".format(key, value))
    
'''
Best accuracy obtained: 0.867
Parameters:
    learning_rate: 0.5
    max_depth: 3
    n_estimators: 50

'''

'''
Randomized Grid-Search
-----------------------
When the number of parameters and their values is getting big traditional grid-search approach quickly becomes ineffective. 
A possible solution might be to randomly pick certain parameters from their distribution. While it's not an exhaustive solution, it's worth giving a shot.

Create a parameters distribution dictionary:

'''

params_dist_grid = {
    'max_depth': [1, 2, 3, 4],
    'gamma': [0, 0.5, 1],
    'n_estimators': randint(1, 1001), # uniform discrete random distribution
    'learning_rate': uniform(), # gaussian distribution
    'subsample': uniform(), # gaussian distribution
    'colsample_bytree': uniform() # gaussian distribution
}

# Initialize RandomizedSearchCV to randomly pick 10 combinations of parameters. With this approach you can easily control the number of tested models.

rs_grid = RandomizedSearchCV(
    estimator=XGBClassifier(**params_fixed, seed=seed),
    param_distributions=params_dist_grid,
    n_iter=10,
    cv=cv,
    scoring='accuracy',
    random_state=seed
)


rs_grid.fit(X, y)

print(rs_grid.cv_results_)
print(rs_grid.best_estimator_)
'''
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.6052416274824561, gamma=0,
       learning_rate=0.6050492440098373, max_delta_step=0, max_depth=3,
       min_child_weight=1, missing=None, n_estimators=275, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=342,
       silent=True, subsample=0.7395162785706926, verbosity=1)
'''


print(rs_grid.best_params_)
# {'colsample_bytree': 0.6052416274824561, 'gamma': 0, 'learning_rate': 0.6050492440098373, 'max_depth': 3, 'n_estimators': 275, 'subsample': 0.7395162785706926}

print(rs_grid.best_score_)
# 0.872




print('done')