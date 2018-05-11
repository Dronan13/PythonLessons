from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style 
import random
style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype = np.float64)
#ys = np.array([5,4,6,5,6,7], dtype = np.float64)

def create_dataset(hm, variance, step=2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]    
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

def best_fit_scope_and_intercept(xs, ys):
    #scope
    m = ( ((mean(xs)*mean(ys)) - mean(xs*ys)) / 
          (mean(xs)**2 - mean(xs**2)) )   
    #intercept
    b = mean(ys) - m*mean(xs)    
    return m, b

def square_error(ys_orig, ys_line):        
    return sum( (ys_line - ys_orig)**2 )

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    square_error_regr = square_error(ys_orig, ys_line)
    square_error_y_mean = square_error(ys_orig, y_mean_line)    
    return 1 - (square_error_regr / square_error_y_mean)

xs, ys = create_dataset(40, 20, 2, correlation = 'pos')

m,b = best_fit_scope_and_intercept(xs, ys)

print('For Best Fit Slope method Scope = {} and Intercept = {}'.format(m, b))

regression_line = [(m*x)+b for x in xs]

predict_x = 8;
predinc_y = (m*predict_x) + b

r_squered = coefficient_of_determination(ys, regression_line)
print('Coefficient Of Determination R^2 = {}'.format(r_squered))

plt.scatter(xs, ys)
plt.scatter(predict_x, predinc_y, s = 100, color = 'g')
plt.plot(xs, regression_line, color = 'r')
plt.show()
