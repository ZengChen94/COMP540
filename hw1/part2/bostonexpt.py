### TODO: your Boston code here or in a separate notebook.
### More codes are in bostonexpt.py
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_utils, utils
from reg_linear_regressor_multi import RegularizedLinearReg_SquaredLoss
from sklearn.preprocessing import PolynomialFeatures
# from linear_regressor_multi import LinearRegressor_Multi, LinearReg_SquaredLoss

# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.

# %matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Some more magic so that the notebook will reload external python modules;
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

# %load_ext autoreload
# %autoreload 2

print 'Reading data ...'
bdata = load_boston()
df = pd.DataFrame(data = bdata.data, columns = bdata.feature_names)
X = df.values
y = bdata.target

from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=10)

X_train_norm, mu, sigma = utils.feature_normalize(X_train)
X_test_norm = (X_test - mu) / sigma
X_val_norm = (X_val - mu) / sigma
XX_train_norm = np.vstack([np.ones((X_train_norm.shape[0],)),X_train_norm.T]).T
XX_test_norm = np.vstack([np.ones((X_test_norm.shape[0],)),X_test_norm.T]).T
XX_val_norm = np.vstack([np.ones((X_val_norm.shape[0],)),X_val_norm.T]).T

# lambda = 0
reglinear_reg1 = RegularizedLinearReg_SquaredLoss()
theta_opt0 = reglinear_reg1.train(XX_train_norm,y_train,reg=0.0,num_iters=1000)
print 'Theta at lambda = 0 is ', theta_opt0
print 'Test error of the best linear model with lambda = 0 is: '+str(reglinear_reg1.loss(theta_opt0, XX_test_norm, y_test, 0))

# linear
reg_vec, error_train, error_val = utils.validation_curve(XX_train_norm,y_train,XX_val_norm,y_val)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('liner'+'.png')
plt.show()

reg_best = reg_vec[error_val.tolist().index(min(error_val))]
theta_opt1 = reglinear_reg1.train(XX_train_norm,y_train,reg=reg_best,num_iters=10000)
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~linear model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print 'the best lambda is: ', reg_best
print 'When lambda='+str(reg_best)+', test error of the linear model is: '+str(reglinear_reg1.loss(theta_opt1, XX_test_norm, y_test, 0))
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

# quadratic, poly = 2
p = 2
quadratic = PolynomialFeatures(degree=p,include_bias=False)

X_quadratic_train = quadratic.fit_transform(X_train)
X_quadratic_test = quadratic.fit_transform(X_test)
X_quadratic_val = quadratic.fit_transform(X_val)

X_quadratic_train_norm, mu, sigma = utils.feature_normalize(X_quadratic_train)
X_quadratic_test_norm = (X_quadratic_test - mu) / sigma
X_quadratic_val_norm = (X_quadratic_val - mu) / sigma

XX_quadratic_train = np.vstack([np.ones((X_quadratic_train_norm.shape[0],)),X_quadratic_train_norm.T]).T
XX_quadratic_test = np.vstack([np.ones((X_quadratic_test_norm.shape[0],)),X_quadratic_test_norm.T]).T
XX_quadratic_val = np.vstack([np.ones((X_quadratic_val_norm.shape[0],)),X_quadratic_val_norm.T]).T

reg_vec, error_train, error_val = utils.validation_curve(XX_quadratic_train,y_train,XX_quadratic_val,y_val)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('quadratic'+'.png')
plt.show()

reg_best = reg_vec[error_val.tolist().index(min(error_val))]
print 'the best lambda is: ', reg_best
theta_opt1 = reglinear_reg1.train(XX_quadratic_train,y_train,reg=reg_best,num_iters=10000)
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~quadratic model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print 'the best lambda is: ', reg_best
print 'When lambda='+str(reg_best)+', test error of the quadratic model is: '+str(reglinear_reg1.loss(theta_opt1, XX_quadratic_test, y_test, 0))
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

# cubic, poly = 3
p = 3
cubic = PolynomialFeatures(degree=p,include_bias=False)

X_cubic_train = cubic.fit_transform(X_train)
X_cubic_test = cubic.fit_transform(X_test)
X_cubic_val = cubic.fit_transform(X_val)

X_cubic_train_norm, mu, sigma = utils.feature_normalize(X_cubic_train)
X_cubic_test_norm = (X_cubic_test - mu) / sigma
X_cubic_val_norm = (X_cubic_val - mu) / sigma

XX_cubic_train = np.vstack([np.ones((X_cubic_train_norm.shape[0],)),X_cubic_train_norm.T]).T
XX_cubic_test = np.vstack([np.ones((X_cubic_test_norm.shape[0],)),X_cubic_test_norm.T]).T
XX_cubic_val = np.vstack([np.ones((X_cubic_val_norm.shape[0],)),X_cubic_val_norm.T]).T

reg_vec, error_train, error_val = utils.validation_curve(XX_cubic_train,y_train,XX_cubic_val,y_val)
plot_utils.plot_lambda_selection(reg_vec,error_train,error_val)
plt.savefig('cubic'+'.png')
plt.show()

reg_best = reg_vec[error_val.tolist().index(min(error_val))]
print 'the best lambda is: ', reg_best
theta_opt1 = reglinear_reg1.train(XX_cubic_train,y_train,reg=reg_best,num_iters=10000)
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~cubic model~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
print 'the best lambda is: ', reg_best
print 'When lambda='+str(reg_best)+', test error of the cubic model is: '+str(reglinear_reg1.loss(theta_opt1, XX_cubic_test, y_test, 0))
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'