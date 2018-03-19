The scientific field of insider-threat detection often lacks sufficient amounts of time-series training data for the purpose of scientific discovery. Moreover, the available limited data are quite noisy. For instance Greitzer and Ferryman (2013) state that ”ground truth” data on actual insider behavior is typically either not available or is limited. In some cases, one might acquire real data, but for privacy reasons, there is no attribution of any individuals relating to abuses or offenses i.e., there is no ground truth. The data may contain insider threats, but these are not identified or knowable to the researcher (Greitzer and Ferryman, 2013; Gheyas and Abdallah, 2016).

## The Problem
Having limited and quite noisy data for insider-threat detection presents a major challenge when estimating time-series models that are robust to overfitting and have well-calibrated uncertainty estimates. Most of the current literature in time-series modeling for insider-threat detection is associated with two major limitations.

First, the methods involve visualizing the time series for noticeable structure and patterns such as periodicity, smoothness, growing/decreasing trends and then hard-coding these patterns into the statistical models during formulation. This approach is suitable for large datasets where more data typically provides more information to learn expressive structure. Given limited amounts of data, such expressive structure may not be easily noticeable. For instance, the figure below shows monthly attachment size in emails (in Gigabytes) sent by an insider from their employee account to their home account. Trends such as periodicity, smoothness, growing/decreasing trends are not easily noticeable.

<img src="https://github.com/Emaasit/long-range-extrapolation/blob/dev/blog/data-emails.png?raw=true" width="600" height="200" />

Second, most of the current literature focuses on parametric models that impose strong restrictive assumptions by pre-specifying the functional form and number of parameters. Pre-specifying a functional form for a time-series model could lead to either overly complex model specifications or simplistic models. It is difficult to know *a priori* the most appropriate function to use for modeling sophisticated insider-threat behavior that involve complex hidden patterns and many other influencing factors.

### Source code
For the impatient reader, two options are provided below to access the source code used for empirical analyses:

1. The entire project (code, notebooks, data, and results) can be found [here on GitHub](https://github.com/Emaasit/long-range-extrapolation).

2. Click this icon [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Emaasit/long-range-extrapolation/master?urlpath=lab) to open the notebooks in a web browser and explore the entire project without downloading and installing any software.

## Data Science Questions
Given the above limitations in the current state-of-art, this study formulated the following three Data Science questions. Given limited and quite noisy time-series data for insider-threat detection, is it possible to perform:

1. pattern discovery without hard-coding trends into statistical models during formulation?

2. model estimation that precludes pre-specifying a functional form?

3. model estimation that is robust to overfitting and has well-calibrated uncertainty estimates? 

## Hypothesis
To answer these three Data Science questions and address the above-described limitations, this study formulated the following hypothesis:
<blockquote>This study hypothesizes that by leveraging current state-of-the-art innovations in Nonparametric Bayesian methods, such as Gaussian processes, it is possible to perform pattern discovery without prespecifying functional forms and hard-coding trends into statistical models.</blockquote>

## Methodology
To test the above hypothesis, a nonparametric Bayesian approach was proposed to implicitly capture hidden structure from time series having limited data. The proposed model, a Gaussian process with a spectral mixture kernel, precludes the need to pre-specify a functional form and hard code trends, is robust to overfitting and has well-calibrated uncertainty estimates.

Mathematical details of the proposed model formulation are described in a corresponding paper that can be found on arXiv through the link below:

* Emaasit, D. and Johnson, M. (2018). [Capturing Structure Implicitly from Noisy Time-Series having Limited Data](https://arxiv.org/abs/1803.05867). arXiv preprint arXiv:1803.05867.

A Brief description of the fundamental concepts of the proposed methodology are as follows. Consider for each data point, $latex i$, that $latex y_i$ represents the attachment size in emails sent by an insider to their home account and $latex x_i$ is a temporal covariate such as month. The task is to estimate a latent function $latex f$, which maps input data, $latex x_i$, to output data $latex y_i$ for $latex i$ = 1, 2, $latex \ldots{}$, $latex N$, where $latex N$ is the total number of data points. Each of the input data $latex x_i$ is of a single dimension $latex D = 1$, and $latex \textbf{X}$ is a $latex N$ x $latex D$ matrix with rows $latex x_i$.

<img class="size-medium wp-image-6429 aligncenter" src="http://haystax.com/wp-content/uploads/2018/03/gp-pgm-352x300.png" alt="" width="352" height="200" />

The observations are assumed to satisfy:
\begin{equation}\label{eqn:additivenoise}
y_i = f(x_i) + \varepsilon, \quad where \, \, \varepsilon \sim \mathcal{N}(0, \sigma_{\varepsilon}^2)
\end{equation}
The noise term, $latex \varepsilon$, is assumed to be normally distributed with a zero mean and variance, $latex \sigma_{\varepsilon}^2$. Latent function $latex f$ represents hidden underlying trends that produced the observed time-series data.

Given that it is difficult to know $latex \textit{a priori}$ the most appropriate functional form to use for $latex f$, a prior distribution, $latex p(\textbf{f})$, over an infinite number of possible functions of interest is formulated. A natural prior over an infinite space of functions is a Gaussian process prior (Williams and Rasmussen, 2006). A GP is fully parameterized by a mean function, $latex \textbf{m}$, and covariance function, $latex \textbf{K}_{N,N}$, denoted as:
\begin{equation}\label{eqn:gpsim}
\textbf{f} \sim \mathcal{GP}(\textbf{m}, \textbf{K}_{N,N}),
\end{equation}

The posterior distribution over the unknown function evaluations, $latex \textbf{f}$, at all data points, $latex x_i$, was estimated using Bayes theorem as follows:
\begin{equation}\label{eqn:bayesinfty}
\begin{aligned}
p(\textbf{f} \mid \textbf{y},\textbf{X}) &amp;= \frac{p(\textbf{y} \mid \textbf{f}, \textbf{X}) \, p(\textbf{f})}{p(\textbf{y} \mid \textbf{X})} = \frac{p(\textbf{y} \mid \textbf{f}, \textbf{X}) \, \mathcal{N}(\textbf{f} \mid \textbf{m}, \textbf{K}_{N,N})}{p(\textbf{y} \mid \textbf{X})},
\end{aligned}
\end{equation}
where:

$latex p(\textbf{f}\mid \textbf{y},\textbf{X})$ = the posterior distribution of functions that best explain the email-attachment size, given the covariates
$latex p(\textbf{y} \mid \textbf{f}, \textbf{X})$ = the likelihood of email-attachment size, given the functions and covariates
$latex p(\textbf{f})$ = the prior over all possible functions of email-attachment size
$latex p(\textbf{y} \mid \textbf{X})$ = the data (constant)

This posterior is a Gaussian process composed of a distribution of possible functions that best explain the time-series pattern.

## Experiments
### Raw data and sample formation
The insider-threat data used for empirical analysis in this study was provided by the computer emergency response team (CERT) division of the software engineering institute (SEI) at Carnegie Mellon University. The particular insider threat focused on is the case where a known insider sent information as email attachments from their work email to their home email. The `pydata` software stack including packages such as `pandas`, `numpy`, `seaborn`, and others, was used for data manipulation and visualization. The Figure below shows that the email-attachment size increased drastically in March and April 2011. 

<img src="https://github.com/Emaasit/long-range-extrapolation/blob/dev/blog/data-emails-barplot.png?raw=true" width="600" height="200" />

### Empirical analysis
In the Figure below, the first ten data points shown in black were used for training and the rest in blue for testing. The Figure below also shows that the Gaussian process model with a spectral mixture kernel is able to capture the structure implicitly both in regions of the training and testing data. The 95% predicted credible interval (CI) contains the "normal" size of email attachments for the duration of the measurements. The GP model was also able to detect both of the anomalous data points, shown in red, that fall outside of the 95% predicted credible interval. 

<img src="https://github.com/Emaasit/long-range-extrapolation/blob/dev/blog/model-emails.png?raw=true" width="600" height="200" />

An ARIMA model was estimated using the methodology in `statsmodels` python package for comparison. The Figure below shows that the ARIMA model is poor at capturing the structure within the region of testing data. This finding suggests that ARIMA models have poor performance for small data without noticeable structure. The 95% confidence interval for ARIMA is much wider than the GP model showing a high degree of uncertainty about the ARIMA predictions. The ARIMA model is able to detect only one anomalous data point in April 2011 and misses the second anomaly in March 2011.

<img src="https://github.com/Emaasit/long-range-extrapolation/blob/dev/blog/model-emails-arima.png?raw=true" width="600" height="200" />

## References
1. Emaasit, D. and Johnson, M. (2018). Capturing Structure Implicitly from Noisy Time-Series having Limited Data. arXiv preprint arXiv:1803.05867.

2. Williams, C. K. and Rasmussen, C. E. (2006). Gaussian processes for machine learning. the MIT Press, 2(3):4.

3. Knudde, N., van der Herten, J., Dhaene, T., &amp; Couckuyt, I. (2017). GPflowOpt: A Bayesian Optimization Library using TensorFlow. arXiv preprint arXiv:1711.03845.

4. Wilson, A. G. (2014). Covariance kernels for fast automatic pattern discovery and extrapolation with Gaussian processes. University of Cambridge.

5. Greitzer, F. L. and Ferryman, T. A. (2013). Methods and metrics for evaluating analytic insider threat tools. In Security and Privacy Workshops (SPW), 2013 IEEE, pages 90–97. IEEE.

6. Gheyas, I. A. and Abdallah, A. E. (2016). Detection and prediction of insider threats to cybersecurity: a systematic literature review and meta-analysis. Big Data Analytics, 1(1):6.