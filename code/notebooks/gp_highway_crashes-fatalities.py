
# coding: utf-8

# # Highway Crashes: Fatal Crashes
# ---

# # Long-range Forecasting and Pattern Discovery given Limited Data

# In[2]:


# %load setup_bayes.py
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import pickle
# import theano
# import theano.tensor as tt
# import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
# from bqplot import pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
# from beakerx import *
sns.set_context('notebook', font_scale = 1.1)
np.random.seed(12345)
rc = {'xtick.labelsize': 40, 'ytick.labelsize': 40, 'axes.labelsize': 40, 'font.size': 40, 'lines.linewidth': 2.0, 
      'lines.markersize': 8, 'font.family': "serif", 'font.serif': "cm", 'savefig.dpi': 200,
      'text.usetex': False, 'legend.fontsize': 40.0, 'axes.titlesize': 40, "figure.figsize": [28, 18]}
sns.set(rc = rc)
sns.set_style("ticks")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import gpflow
from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.acquisition import ExpectedImprovement, MinValueEntropySearch
from gpflowopt.optim import StagedOptimizer, MCOptimizer, SciPyOptimizer  
from gpflowopt.design import LatinHyperCube
import random
random.seed(1234)
from warnings import filterwarnings
filterwarnings("ignore")


# ## Data Preparation

# In[5]:


monthly_crash_data = pd.read_csv("../../data/crashes/monthly_data.csv")
monthly_crash_data.head()


# In[6]:


# Check the data types of each variable
monthly_crash_data.info()


# Create a new datetime column labelled "ds" that combines original columns "Year" and "Month"
# 

# In[7]:


from datetime import datetime
monthly_crash_data["ds"] = monthly_crash_data["Year"].map(str) + "-" + monthly_crash_data["Month"].map(str)
monthly_crash_data["ds"] = pd.to_datetime(monthly_crash_data["ds"])
monthly_crash_data["ds"] = monthly_crash_data.ds.astype("O")
monthly_crash_data.tail()


# In[8]:


from datetime import datetime
monthly_crash_data["ds"] = monthly_crash_data.apply(lambda x: datetime.date(x["ds"]), axis = 1)


# In[9]:


# Check again the data types of each variable
monthly_crash_data.info()


# In[10]:


fatal_crash_data = monthly_crash_data[["ds", "Fatalities"]].rename(columns = {"Fatalities" : "y"})
fatal_crash_data.head()


# visualize fatal crashes

# In[11]:


fig, ax = plt.subplots()
ax.plot(fatal_crash_data.ds, fatal_crash_data.y, color = "b", marker = "o")
# ax.set_xticklabels(labels = fatal_crash_data.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Number of fatal crashes");
# ax.set_title("Number of Fatal Crashes");


# In[12]:


injury_crash_data = monthly_crash_data[["ds", "SeriousInjuries"]].rename(columns = {"SeriousInjuries" : "y"})
injury_crash_data.head()


# In[13]:


injury_crash_data[injury_crash_data.index == 192]


# In[14]:


fig, ax = plt.subplots()
ax.plot(injury_crash_data.ds, injury_crash_data.y, color = "b", marker = "o")
# ax.set_xticklabels(labels = injury_crash_data.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Number of serious injuries");
# ax.set_title("Number of Emails sent Monthly");


# Import Yearly crashes per 100 million vehicle miles traveled (VMT)

# In[15]:


yearly_data = pd.read_csv("../../data/crashes/yearly_data.csv", parse_dates = ["Years"])
yearly_data.tail(n = 10)


# In[17]:


groups = monthly_crash_data.groupby(["Year"]).sum().reset_index()
injury_data = groups[["Year", "SeriousInjuries"]].rename(columns = {"Year": "Years"})
vmt = pd.read_csv("../../data/crashes/yearly_data.csv")[["Years", "VMT(Billions)"]]
injury_data = injury_data.merge(vmt)
injury_data["y"] = injury_data["SeriousInjuries"]/(injury_data["VMT(Billions)"]*10)
injury_data["ds"] = pd.to_datetime(injury_data["Years"].astype(str))
injury_data = injury_data.drop(["Years", "VMT(Billions)", "SeriousInjuries"], axis = 1)
injury_data


# In[18]:


# fatal_data = injury_data


# In[19]:


my_date = pd.to_datetime(str(1994), format = "%Y")
my_date


# In[20]:


fatal_data = yearly_data[["Years", "VMT(Billions)", "Fatalities"]]
fatal_data["y"] = fatal_data["Fatalities"]/(fatal_data["VMT(Billions)"]*10)
fatal_data["ds"] = fatal_data.apply(lambda x: datetime.date(x["Years"]), axis = 1)
fatal_data = fatal_data.drop(["Years", "VMT(Billions)", "Fatalities"], axis = 1)
fatal_data = fatal_data.drop(fatal_data.index[:25]).reset_index().drop(["index"], axis = 1)
fatal_data


# In[21]:


fatal_data.describe()


# In[22]:


fig, ax = plt.subplots()
ax.plot(fatal_data.ds, fatal_data.y, color = "b", marker = "o")
# ax.set_xticklabels(labels = injury_crash_data.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Number of fatal crashes per 100 million VMT");
# ax.set_title("Number of Emails sent Monthly");


# In[202]:


test_size = 17
X_complete = np.array([fatal_data.index]).reshape((fatal_data.shape[0], 1)).astype('float64')
X_train = X_complete[0:test_size, ]
X_test = X_complete[test_size:fatal_data.shape[0], ]
Y_complete = np.array([fatal_data.y]).reshape((fatal_data.shape[0], 1)).astype('float64')
Y_train = Y_complete[0:test_size, ]
Y_test = Y_complete[test_size:fatal_data.shape[0], ]
D = Y_train.shape[1];


# In[203]:


D


# In[204]:


Y_train.shape; X_train.shape; Y_test.shape; X_test.shape; X_complete.shape


# In[205]:


Y_train; X_train; X_test; X_complete


# In[206]:


Y_train.dtype


# In[221]:


fig, ax = plt.subplots()
ax.plot(X_train.flatten(),Y_train.flatten(), c ='b', marker = "o", label = "Training data")
ax.plot(X_test.flatten(),Y_test.flatten(), c='r', marker = "o", label = 'Test data')
ax.set_xticklabels(labels = fatal_data.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Number of fatal crashes per 100 million VMT")
plt.legend(loc = "best");
plt.savefig("results/crashes/VMT/fatalities/data-fatalities.png");
# fig1 = plt.gcf()
# py.offline.iplot_mpl(fig1);


# ## Gaussian Process modeling

# In[208]:


# Trains a model with a spectral mixture kernel, given an ndarray of 
# 2Q frequencies and lengthscales

Q = 10 # nr of terms in the sum
max_iters = 1000

def create_model(hypers):
    f = np.clip(hypers[:Q], 0, 5)
    weights = np.ones(Q) / Q
    lengths = hypers[Q:]

    kterms = []
    for i in range(Q):
        rbf = gpflow.kernels.RBF(D, lengthscales=lengths[i], variance=1./Q)
        rbf.lengthscales.transform = gpflow.transforms.Exp()
        cos = gpflow.kernels.Cosine(D, lengthscales=f[i])
        kterms.append(rbf * cos)

    k = np.sum(kterms) + gpflow.kernels.Linear(D) + gpflow.kernels.Bias(D)
    m = gpflow.gpr.GPR(X_train, Y_train, kern=k)
    return m

m = create_model(np.ones((2*Q,)))


# In[209]:


get_ipython().run_cell_magic('time', '', 'm.optimize(maxiter = max_iters)')


# In[210]:


def plotprediction(m):
    # Perform prediction
    mu, var = m.predict_f(X_complete)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(labels = fatal_data.ds, rotation = 45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of fatal crashes per 100 million VMT");
    ax.plot(X_train.flatten(), Y_train.flatten(), c='r', marker = "o", label = 'Training data')
    ax.plot(X_test.flatten(), Y_test.flatten(), c='b', marker = "o", label = 'Test data')
    ax.plot(X_complete.flatten(), mu.flatten(), c='g', marker = "o", label = "Predicted mean function")
    lower = mu - 2*np.sqrt(var)
    upper = mu + 2*np.sqrt(var)
    ax.plot(X_complete, upper, 'g--', X_complete, lower, 'g--', lw=1.2)
    ax.fill_between(X_complete.flatten(), lower.flatten(), upper.flatten(),
                    color='g', alpha=.1, label = "95% Predicted Credible interval")
    plt.legend(loc = "best")
    plt.tight_layout()


# In[225]:


plotprediction(m)
plt.savefig("results/crashes/VMT/fatalities/model-fatalities.png");


# In[212]:


## Calculate the RMSE and MAPE
def calculate_rmse(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    rmse = np.sqrt(((mu - Y_test)**2).mean())
    return rmse

def calculate_mape(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    mape = (np.absolute(((mu - Y_test)/Y_test)*100)).mean()
    return mape


# In[213]:


calculate_mape(model=m, X_test = X_test, Y_test = Y_test)


# ## Perform hyperparameter tuning using Bayesian Optimization

# Let's use Bayesian Optimization to find the optimal model parameters of the GP model and then use then to estimate the model and prediction.

# In[47]:


from gpflowopt.objective import batch_apply

# Objective function for our optimization
# Input: N x 2Q ndarray, output: N x 1.
# returns the negative log likelihood obtained by training with given frequencies and rbf lengthscales
# Applies some tricks for stability similar to GPy's jitchol
@batch_apply
def objectivefx(freq):
    m = create_model(freq)
    for i in [0] + [10**exponent for exponent in range(6,1,-1)]:
        try:
            mean_diag = np.mean(np.diag(m.kern.compute_K_symm(X_train)))
            m.likelihood.variance = 1 + mean_diag * i
            m.optimize(maxiter=max_iters)
            return -m.compute_log_likelihood()
        except:
            pass
    raise RuntimeError("Frequency combination failed indefinately.")

# Setting up optimization domain.
lower = [0.]*Q
upper = [5.]*int(Q)
df = np.sum([ContinuousParameter('freq{0}'.format(i), l, u) for i, l, u in zip(range(Q), lower, upper)])

lower = [1e-5]*Q
upper = [300]*int(Q)
dl = np.sum([ContinuousParameter('l{0}'.format(i), l, u) for i, l, u in zip(range(Q), lower, upper)])
domain = df + dl
domain


# In[48]:


get_ipython().run_cell_magic('time', '', 'design = LatinHyperCube(6, domain)\nX = design.generate()')


# In[49]:


get_ipython().run_cell_magic('time', '', 'Y = objectivefx(X)')


# In[50]:


get_ipython().run_cell_magic('time', '', 'k_surrogate = gpflow.kernels.Matern52(input_dim = domain.size, ARD = False)')


# In[51]:


get_ipython().run_cell_magic('time', '', 'model_surrogate = gpflow.gpr.GPR(X, Y, kern = k_surrogate)')


# In[52]:


get_ipython().run_cell_magic('time', '', 'acq_fn = ExpectedImprovement(model_surrogate)\n# acq_fn = MinValueEntropySearch(model_surrogate, domain = domain)')


# In[53]:


get_ipython().run_cell_magic('time', '', 'acq_optimizer = StagedOptimizer([MCOptimizer(domain, nsamples = 5000), \n                                SciPyOptimizer(domain)])')


# In[54]:


get_ipython().run_cell_magic('time', '', 'optimizer = BayesianOptimizer(domain = domain, \n                              acquisition = acq_fn, \n                              optimizer = acq_optimizer)')


# In[55]:


get_ipython().run_cell_magic('time', '', 'with optimizer.silent():\n    result = optimizer.optimize(objectivefx = objectivefx, n_iter = 30)')


# In[56]:


print(result)


# In[57]:


get_ipython().run_cell_magic('time', '', 'm_opt = create_model(result.x[0,:])\nm_opt.optimize()')


# In[46]:


plotprediction(m_opt)
# plt.savefig("results/crashes/VMT/fatalities/model-opt-fatalities.png");


# In[47]:


## Inspect the evolution
f, axes = plt.subplots()
f = acq_fn.data[1][:,0]
axes.plot(np.arange(0, acq_fn.data[0].shape[0]), np.minimum.accumulate(f))
axes.set_ylabel('fmin')
axes.set_xlabel('Number of evaluated points');
# plt.savefig("results/crashes/VMT/fatalities/iterations-fatalities.png");


# In[ ]:


# save the model and results to the files 'model.pkl' model_optimized.pkl'
# and 'results.pkl' for later use
with open('results/crashes/VMT/injuries/model.pkl', 'wb') as mdl:
    pickle.dump(m, mdl, protocol = pickle.HIGHEST_PROTOCOL)
    
with open('results/crashes/VMT/injuries/model_optimized.pkl', 'wb') as mdl_opt:
    pickle.dump(m_opt, mdl_opt, protocol = pickle.HIGHEST_PROTOCOL)    
    
with open('results/crashes/VMT/injuries/result_optimized.pkl', 'wb') as res:
    pickle.dump(result, res, protocol = pickle.HIGHEST_PROTOCOL) 
    
with open('results/crashes/VMT/injuries/acq_fn.pkl', 'wb') as acq:
    pickle.dump(acq_fn, acq, protocol = pickle.HIGHEST_PROTOCOL)  


# In[3]:


# # load it at some future point
# with open('results/crashes/VMT/fatalities/model2.pkl', 'rb') as mdl:
#     m = pickle.load(mdl)

# with open('results/crashes/VMT/fatalities/model_optimized2.pkl', 'rb') as mdl_opt:
#     m_opt = pickle.load(mdl_opt)
    
# with open('results/crashes/VMT/fatalities/result_optimized2.pkl', 'rb') as res:
#     result = pickle.load(res)   
    
# with open('results/crashes/VMT/fatalities/acq_fn2.pkl', 'rb') as acq:
#     acq_fn = pickle.load(acq)     


# ## ARIMA

# In[214]:


import itertools
import numpy.ma as ma
import warnings
from statsmodels.tsa.arima_model import ARIMA
from numpy.linalg import LinAlgError


def get_ARIMA_param_values(y):
    """ Get best ARIMA values given data
    """
    warnings.filterwarnings('ignore')
    
    # Values to try
    p = [0, 1, 2, 3, 4, 5]
    d = [0, 1, 2]
    q = [0, 1, 2, 3, 4, 5]
    results = []

    for pi, di, qi in itertools.product(p, d, q):
        try:
            model = ARIMA(y, order=(pi, di, qi))
            model_fit = model.fit()
            aic = model_fit.aic
            if not np.isnan(aic):
                results.append(((pi,di,qi), aic, model_fit))
        except ValueError:
            pass
        except LinAlgError:
            pass
    warnings.filterwarnings('default')
    return sorted(results, key=lambda x: x[1])[0]


# In[215]:


# Make prediction
steps = X_test.shape[0]
params, aic, model_fit = get_ARIMA_param_values(y = Y_train)
mu, stderr, conf_int = model_fit.forecast(steps = steps, alpha=0.05)


# In[216]:


params; aic


# In[217]:


mu; stderr, conf_int


# In[223]:


def plotprediction_arima(m):
    # Perform prediction
#     mu, var = m.predict_f(X_complete)
    mu, stderr, conf_int = m.forecast(steps=steps, alpha=0.05)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(labels = fatal_data.ds, rotation = 45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Number of fatal crashes per 100 million VMT");
    ax.plot(X_train.flatten(), Y_train.flatten(), c='r', marker = "o", label = 'Training data')
    ax.plot(X_test.flatten(), Y_test.flatten(), c='b', marker = "o", label = 'Test data')
    ax.plot(X_test.flatten(), mu.flatten(), c='g', marker = "o", label = "Predicted value")
#     lower = mu - 2*np.sqrt(var)
#     upper = mu + 2*np.sqrt(var)
    lower = conf_int[:,0 ]
    upper = conf_int[:,1 ]
    ax.plot(X_test, upper, 'g--', X_test, lower, 'g--', lw=1.2)
    ax.fill_between(X_test.flatten(), lower.flatten(), upper.flatten(),
                    color='g', alpha=.1, label = "95% confidence interval")
    plt.legend(loc = "best")
    plt.tight_layout()

plotprediction_arima(m = model_fit)
plt.savefig("results/crashes/VMT/fatalities/model-fatalities-arima.png");


# In[219]:


## Calculate the RMSE and MAPE
def calculate_rmse_arima(mu, Y_test):
    rmse = np.sqrt(((mu - Y_test)**2).mean())
    return rmse

def calculate_mape_arima(mu, Y_test):
    mape = (np.absolute(((mu - Y_test)/Y_test)*100)).mean()
    return mape


# In[220]:


calculate_mape_arima(mu = mu, Y_test = Y_test)


# In[201]:


# improve quality of figures for journal paper
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


# In[81]:


# print system information/setup
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p numpy,pandas,gpflowopt,gpflow,tensorflow,matplotlib,ipywidgets,beakerx,seaborn -g')

