import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
from automat._methodical import counter
import pandas as pd
import random

class Support_Vector_Machine
    def __init__(self, visualization = True):
        self.visualization = visualization
        self.colors = {1: 'r', -1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    def fit(self, data): 
        self.data = data
        #{ ||w||: [w,b] }
        opt_dict = ()    
        
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]   
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        
        self.max_feature_value = max(all_data)        
        self.min_feature_value = max(all_data)  
        all_data = None
        
        # support vectors yi(xi.w+b) = 1
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      #point of expanse
                      self.max_feature_value * 0.001]
        
        # extremely expansive
        b_range_multiple = 5
        
        # we don't need take as small of steps
        # with b as we do w
        b_multiple = 5
        
        latest_optimum = self.max_feature_value*10
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            
            # we can do this because convex
            optimized = False
            
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple, 
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation 
                        found_option = True
                        
                        # weakest link in SVM fundamentally 
                        # SMO Attempts to fix this a bit
                        # yi(xi.w+b)>=1                      
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t,xi)+b) >=1:
                                    found_option = False
                        
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                        
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step')  
                else:
                    w = w - step         
                            
            norms = sorted([n for n in opt_dict])
            opt_choice = norms[0]
            
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2
        
    def predict(self):
        # sign(x.w+b)
        classification = np.sign(np.dot(np.array(features), self.w)+self.b)
        
        return classification

style.use('ggplot')
data_dict = {-1:np.array([[1,7]
                          [2,8]
                          [3,8]]),
            1:np.array([[5,1]
                        [6,-1]
                        [7,3]])}

