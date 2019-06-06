'''
Created on 2 Jun 2019

@author: Shalev
'''

import xgboost as xgb
import seaborn as sns
import pandas as pd

sns.set(font_scale = 1.5)

dtrain = xgb.DMatrix('./data/agaricus.txt.train')
dtest = xgb.DMatrix('./data/agaricus.txt.test')

params = {
    'objective':'binary:logistic',
    'max_depth':1,
    'silent':1,
    'eta':0.5
}

num_rounds = 5


# see how does it perform
watchlist  = [(dtest,'test'), (dtrain,'train')] # native interface only
bst = xgb.train(params, dtrain, num_rounds, watchlist)

'''
Representation of a tree
-------------------------
Before moving on it's good to understand the intuition about how trees are grown.

While building a tree is divided recursively several times (in this example only once) - this operation is called split. 
To perform a split the algorithm must figure out which is the best (one) feature to use.

After that, at the bottom of the we get groups of observations packed in the leaves.

In the final model, these leafs are supposed to be as pure as possible for each tree, meaning in our case that each leaf should be made of one label class.

Not all splits are equally important. Basically the first split of a tree will have more impact on the purity that, for instance, the deepest split. 
Intuitively, we understand that the first split makes most of the work, and the following splits focus on smaller parts of the dataset which 
    have been missclassified by the first tree.

In the same way, in Boosting we try to optimize the missclassification at each round (it is called the loss). 
So the first tree will do the big work and the following trees will focus on the remaining, on the parts not correctly learned by the previous trees.

The improvement brought by each split can be measured, it is the gain.

~ Quoted from the Kaggle Tianqi Chen's Kaggle notebook.
https://www.kaggle.com/tqchen/understanding-xgboost-model-on-otto-data

Let's investigate how trees look like on our case:
'''

# trees_dump = bst.get_dump(fmap='./data/featmap.txt', with_stats=True)
trees_dump = bst.get_dump(with_stats=True)

for tree in trees_dump:
    print(tree)
    
'''
0:[f29<-9.53674316e-07] yes=1,no=2,missing=1,gain=4000.53101,cover=1628.25
    1:leaf=0.647757947,cover=924.5
    2:leaf=-0.933309674,cover=703.75

0:[f27<-9.53674316e-07] yes=1,no=2,missing=1,gain=1377.22424,cover=1404.203
    1:leaf=-0.339609325,cover=1008.21417
    2:leaf=0.759690285,cover=395.988831

0:[f39<-9.53674316e-07] yes=1,no=2,missing=1,gain=1210.76575,cover=1232.64319
    1:leaf=0.673357666,cover=430.293335
    2:leaf=-0.36520344,cover=802.349915

0:[f64<-9.53674316e-07] yes=1,no=2,missing=1,gain=791.95874,cover=1111.84363
    1:leaf=-0.277528912,cover=765.906372
    2:leaf=0.632880688,cover=345.937195

0:[f29<-9.53674316e-07] yes=1,no=2,missing=1,gain=493.703644,cover=981.6828
    1:leaf=0.275961101,cover=638.372559
    2:leaf=-0.466680348,cover=343.310272

'''
    
'''
For each split we are getting the following details:

which feature was used to make split,
possible choices to make (branches)
gain which is the actual improvement in accuracy brough by that feature. 
The idea is that before adding a new split on a feature X to the branch there was some wrongly classified elements, 
    after adding the split on this feature, there are two new branches, and each of these branch is more accurate 
    (one branch saying if your observation is on this branch then it should be classified as 1, and the other branch saying the exact opposite),
cover measuring the relative quantity of observations concerned by that feature
'''


'''
Hopefully there are better ways to figure out which features really matter. 
We can use built-in function plot_importance that will create a plot presenting most important features due to some criterias. 
We will analyze the impact of each feature for all splits and all trees and visualize results.

See which feature provided the most gain:

'''

xgb.plot_importance(bst, importance_type='gain', xlabel='Gain')


xgb.plot_importance(bst)


# In case you want to visualize it another way, a created model enables convinient way of accessing the F-score.

importances = bst.get_fscore()
importances

# {'f27': 1, 'f29': 2, 'f39': 1, 'f64': 1}

# create df
importance_df = pd.DataFrame({
        'Splits': list(importances.values()),
        'Feature': list(importances.keys())
    })
importance_df.sort_values(by='Splits', inplace=True)
importance_df.plot(kind='barh', x='Feature', figsize=(8,6), color='orange')



print('done')