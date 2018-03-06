
# coding: utf-8

# # Insider Threat: Email Data
# ----

# # Long-range Forecasting and Pattern Discovery given Limited Data

# In[1]:


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
rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20, 'lines.linewidth': 2.0, 
      'lines.markersize': 8, 'font.family': "serif", 'font.serif': "cm", 'savefig.dpi': 200,
      'text.usetex': False, 'legend.fontsize': 20.0, 'axes.titlesize': 20, "figure.figsize": [20, 12]}
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

# In[2]:


email_filtered = pd.read_csv("../../data/email_filtered.csv", parse_dates=["date"])


# In[3]:


email_filtered.head()      


# In[4]:


email_filtered.info()


# Let's select one user in particular who is a known insider.
# 

# In[5]:


# The insider
df_insider = email_filtered[email_filtered["user"] == "CDE1846"]
df_insider.head()
df_insider.shape


# In[6]:


emails_per_month = df_insider.resample(rule = "1M", on = "date").sum().reset_index()
emails_per_month["date"] = pd.to_datetime(emails_per_month["date"], format = "%Y-%m-%d")
emails_per_month.columns = ["ds", "y"]
emails_per_month


# In[7]:


emails_per_month.info()


# In[8]:


fig, ax = plt.subplots()
sns.barplot(data = emails_per_month, x = "ds", y = "y", ax = ax)
ax.set_xticklabels(labels = emails_per_month["ds"], rotation = 90)
ax.set_xlabel("Months of the Year")
ax.set_ylabel("Number of Emails")
ax.set_title("Number of Emails sent Monthly");


# In[9]:


df_insider_non_org = df_insider[~df_insider['to'].str.contains('dtaa.com')]
df_insider_non_org


# In[10]:


df_insider_non_org.resample(rule = "1M", on = "date").sum().plot(kind = "bar", logy = "size");


# In[53]:


df = df_insider_non_org.resample(rule = "1M", on = "date").sum().reset_index()
df["date"] = pd.to_datetime(df["date"], format = "%Y-%m-%d")
df.columns = ["ds", "y"]
df = df.drop([14, 15, 16])
(df.y/1e6).describe()
df.y = df.y/1e6


# Explore Insider Threat Case

# In[54]:


# Here, we look at the case where the insider email IP to their home account
# The data is resampled per month and the anomalous behavior is clearly
# visible
df_insider_ewing = df_insider_non_org[df_insider_non_org['to'] == 'Ewing_Carlos@comcast.net']
df = df_insider_ewing.resample('1M', on='date').sum().reset_index()
df.columns = ["ds", "y"]
(df.y/1e6).describe()
df.y = df.y/1e6


# In[55]:


df.info()


# In[56]:


df


# In[57]:


from datetime import datetime
df["ds"] = df.apply(lambda x: datetime.date(x["ds"]), axis = 1)


# In[58]:


fig, ax = plt.subplots()
sns.barplot(data = df, x = "ds", y = "y")
ax.set_xticklabels(labels = df.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Number of Emails ($10^6$)");
# ax.set_title("Number of Emails sent Monthly");


# In[59]:


df = df.drop([14, 15, 16])


# In[60]:


df.info()


# In[61]:


df


# In[62]:


df.describe()


# In[63]:


fig, ax = plt.subplots()
ax.plot(df.ds, df.y, color = "b", marker = "o")
# ax.set_xticklabels(labels = injury_crash_data.ds, rotation = 45)
ax.set_xlabel("Time")
ax.set_ylabel("Total Size of Emails ($10^6$)");
# ax.set_title("Number of Emails sent Monthly");


# In[96]:


test_size = 11
X_complete = np.array([df.index]).reshape((df.shape[0], 1)).astype('float64')
X_train = X_complete[0:test_size, ]
X_test = X_complete[test_size:df.shape[0], ]
Y_complete = np.array([df.y]).reshape((df.shape[0], 1)).astype('float64')
Y_train = Y_complete[0:test_size, ]
Y_test = Y_complete[test_size:df.shape[0], ]
D = Y_train.shape[1];


# In[97]:


D


# In[98]:


Y_train.shape; X_train.shape; Y_test.shape; X_test.shape; X_complete.shape


# In[99]:


Y_train; X_train; Y_test; X_test; X_complete


# In[100]:


np.sort(Y_complete.flatten())


# In[101]:


Y_train.dtype


# In[118]:


fig, ax = plt.subplots()
ax.plot(X_train.flatten(),Y_train.flatten(), c ='b', marker = "o", label = "Training data")
ax.plot(X_test.flatten(),Y_test.flatten(), c='r', marker = "o", label = 'Test data')
ax.set_xticklabels(labels = df.ds, rotation = 45)
ax.set_xlabel('Time')
ax.set_ylabel('Total size of emails in GB')
plt.legend(loc = "best")
plt.savefig('results/emails/data-email.png');
# fig1 = plt.gcf()
# py.offline.iplot_mpl(fig1);


# ## Gaussian Process modeling

# This study used a Gaussian Process model with a Spectral Mixture (SM) kernel proposed by Wilson (2014). This is because the SM kernel is capable of capturing hidden structure with data without hard cording features in a kernel. Moreover, the SM kernel is capable of performing long-range extrapolation beyond available data.
# 

# In[103]:


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


# In[104]:


get_ipython().run_cell_magic('time', '', 'm.optimize(maxiter = max_iters)')


# In[119]:


def plotprediction(m):
    # Perform prediction
    mu, var = m.predict_f(X_complete)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(labels = df.ds, rotation = 45)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total size of emails in GB');
    ax.plot(X_train.flatten(),Y_train.flatten(), c='b', marker = "o", label = 'Training data')
    ax.plot(X_test.flatten(),Y_test.flatten(), c='r', marker = "o", label = 'Test data')
    ax.plot(X_complete.flatten(), mu.flatten(), c='g', marker = "o", label = "Predicted mean function")
    lower = mu - 2*np.sqrt(var)
    upper = mu + 2*np.sqrt(var)
    ax.plot(X_complete, upper, 'g--', X_complete, lower, 'g--', lw=1.2)
    ax.fill_between(X_complete.flatten(), lower.flatten(), upper.flatten(),
                    color='g', alpha=.1, label = "95% Predicted credible interval")
    plt.legend(loc = "best")
    plt.tight_layout()


# In[121]:


plotprediction(m)
plt.savefig('results/emails/model-emails.png');


# In[107]:


## Calculate the RMSE and MAPE
def calculate_rmse(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    rmse = np.sqrt(((mu - Y_test)**2).mean())
    return rmse

def calculate_mape(model, X_test, Y_test):
    mu, var = model.predict_y(X_test)
    mape = (np.absolute(((mu - Y_test)/Y_test)*100)).mean()
    return mape


# In[109]:


calculate_mape(model=m, X_test = X_test, Y_test = Y_test)


# ## Perform hyperparameter tuning using Bayesian Optimization

# Let's use Bayesian Optimization to find the optimal model parameters of the GP model and then use then to estimate the model and prediction.

# In[37]:


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


# In[38]:


get_ipython().run_cell_magic('time', '', 'design = LatinHyperCube(6, domain)\nX = design.generate()')


# In[39]:


get_ipython().run_cell_magic('time', '', 'Y = objectivefx(X)')


# In[40]:


get_ipython().run_cell_magic('time', '', 'k_surrogate = gpflow.kernels.Matern52(input_dim = domain.size, ARD = False)')


# In[41]:


get_ipython().run_cell_magic('time', '', 'model_surrogate = gpflow.gpr.GPR(X, Y, kern = k_surrogate)')


# In[42]:


get_ipython().run_cell_magic('time', '', 'acq_fn = ExpectedImprovement(model_surrogate)\n# acq_fn = MinValueEntropySearch(model_surrogate, domain = domain)')


# In[43]:


get_ipython().run_cell_magic('time', '', 'acq_optimizer = StagedOptimizer([MCOptimizer(domain, nsamples = 5000), \n                                SciPyOptimizer(domain)])')


# In[44]:


get_ipython().run_cell_magic('time', '', 'optimizer = BayesianOptimizer(domain = domain, \n                              acquisition = acq_fn, \n                              optimizer = acq_optimizer)')


# In[45]:


get_ipython().run_cell_magic('time', '', 'with optimizer.silent():\n    result = optimizer.optimize(objectivefx = objectivefx, n_iter = 30)')


# In[46]:


print(result)


# In[47]:


get_ipython().run_cell_magic('time', '', 'm_opt = create_model(result.x[0,:])\nm_opt.optimize()')


# In[74]:


plotprediction(m_opt)
# plt.savefig('results/emails/model-opt-emails.png');


# In[ ]:


## Inspect the evolution
f, axes = plt.subplots()
f = acq_fn.data[1][:,0]
axes.plot(np.arange(0, acq_fn.data[0].shape[0]), np.minimum.accumulate(f))
axes.set_ylabel('fmin')
axes.set_xlabel('Number of evaluated points')
plt.savefig('results/emails/iterations-email.png');


# In[50]:


# save the model and results to the files 'model.pkl' model_optimized.pkl'
# and 'results.pkl' for later use
with open('results/emails/model2.pkl', 'wb') as mdl:
    pickle.dump(m, mdl, protocol = pickle.HIGHEST_PROTOCOL)
    
with open('results/emails/model_optimized2.pkl', 'wb') as mdl_opt:
    pickle.dump(m_opt, mdl_opt, protocol = pickle.HIGHEST_PROTOCOL)    
    
with open('results/emails/result_optimized2.pkl', 'wb') as res:
    pickle.dump(result, res, protocol = pickle.HIGHEST_PROTOCOL) 
    
with open('results/emails/acq_fn2.pkl', 'wb') as acq:
    pickle.dump(acq_fn, acq, protocol = pickle.HIGHEST_PROTOCOL)     


# In[75]:


# # load it at some future point
# with open('results/emails/model3.pkl', 'rb') as mdl:
#     m = pickle.load(mdl)

# with open('results/emails/model_optimized3.pkl', 'rb') as mdl_opt:
#     m_opt = pickle.load(mdl_opt)
    
# with open('results/emails/result_optimized3.pkl', 'rb') as res:
#     result = pickle.load(res)   
    
# with open('results/emails/acq_fn3.pkl', 'rb') as acq:
#     acq_fn = pickle.load(acq)       


# ## ARIMA

# In[110]:


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
    p = [0, 1, 2, 3, 4, 5, 6]
    d = [0, 1, 2]
    q = [0, 1, 2, 3, 4, 5, 6]
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


# In[111]:


# Make prediction
steps = X_test.shape[0]
params, aic, model_fit = get_ARIMA_param_values(y = Y_train)
mu, stderr, conf_int = model_fit.forecast(steps = steps, alpha=0.05)


# In[112]:


steps


# In[113]:


params, aic, mu, stderr, conf_int


# In[114]:


def plotprediction_arima(m):
    # Perform prediction
#     mu, var = m.predict_f(X_complete)
    mu, stderr, conf_int = m.forecast(steps=steps, alpha=0.05)

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xticklabels(labels = df.ds, rotation = 45)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total size of emails in GB");
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
plt.savefig('results/emails/model-emails-arima.png');


# In[115]:


## Calculate the RMSE and MAPE
def calculate_rmse_arima(mu, Y_test):
    rmse = np.sqrt(((mu - Y_test)**2).mean())
    return rmse

def calculate_mape_arima(mu, Y_test):
    mape = (np.absolute(((mu - Y_test)/Y_test)*100)).mean()
    return mape


# In[116]:


calculate_mape_arima(mu = mu, Y_test = Y_test)


# In[30]:


# improve quality of figures for journal paper
get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "retina"')


# In[41]:


# print system information/setup
get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -m -p numpy,pandas,gpflowopt,gpflow,tensorflow,matplotlib,ipywidgets,beakerx,seaborn -g')

