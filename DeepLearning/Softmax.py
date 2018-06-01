# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 06:52:46 2017

@author: vishay
"""
import numpy as np
scores = [3.0, 1.0, 0.2]
scores_2 = [1, 2, 3]

scores_3 = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

x1 = [1, 2, 3, 6]
x2 = np.array([[1, 2, 3, 6],  # sample 1
               [2, 4, 5, 6],  # sample 2
               [1, 2, 3, 6]]) # sample 1 again(!)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #logit_lst = [np.exp(logit) for logit in x]
    
    #print len(exp_np.shape)
    x = np.asarray(x)
    # subtract np.max(x) from exponent to prevent
    # numerical unstability cause by dividing large 
    # numbers, this is a normalization trick
    exp_np = np.exp(x-np.max(x))
    exp_np = exp_np/exp_np.sum(axis=0)
    return exp_np
   



print (softmax(scores_3))
print (softmax(scores_3/10))
print (softmax(x1))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
# score shape = 3*8
print "no",scores
print softmax(scores) # shape 3*8
print softmax(scores).T # shape 8*3
# take transpose so that columns are plotted
plt.plot(x, softmax(scores).T, linewidth=2)
# proabilities sum to 1 i.3 sum of red , green ,blue value =1
plt.show()