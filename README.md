Capturing Structure Implicitly from Limited and Noisy Time-Series Processes
===========================================================================

* Click this icon [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/haystax-technology/long-range-extrapolation/master?urlpath=lab) to open the notebooks in a web broser.

<img src="code/notebooks/results/emails/model-emails.png" width="75%" />

Purpose
-------

This repo provides the code and data used to generate results in our
paper "**Capturing Structure Implicitly from Limited and Noisy Time-Series Processes**"

Abstract
--------

Scientific fields such as insider-threat detection and highway-safety planning often lack sufficient amounts of time-series training data for the purpose of scientific discovery. Moreover, the available limited data are very noisy. This presents a major challenge when estimating statistical models to extract hidden patterns and perform accurate forecasting. Most of the current literature in insider-threat detection and highway-safety planning involve visualizing the time-series for noticeable structure, such as periodicity, and hard coding them into pre-specified parametric functions. This approach is associated with two limitations. First, given that such trends may not be noticeable in small data, it is difficult to explicitly incorporate expressive structure into the statistical models during formulation.  Second, it is difficult to know *a priori* the most appropriate functional form to use. To address these limitations, a nonparametric Bayesian approach was proposed to capture hidden structure from limited data and perform accurate long-range forecasting. The proposed model, a Gaussian process with a spectral mixture kernel, precludes the need to pre-specify a functional form and hard code trends. Bayesian modeling was adopted to account for uncertainty.

Citation
--------

If you do end up using our code, please cite our paper as follows.

    Emaasit, D. and Johnson, M.(2018). Capturing Structure Implicitly from Limited and Noisy Time-Series Processes. arXiv preprint arXiv:xxxx.xxxx

Or using BibTex as follows:

    @article{emaasit2018long,
      title={Capturing Structure Implicitly from Limited and Noisy Time-Series Processes},
      author={Emaasit, Daniel and Johnson, Matthew},
      journal={arXiv preprint arXiv:xxxx.xxxx},
      year={2018}
    }

Getting Help
------------

Incase you need help running this code or have general questions, don't hesitate to email us at <demaasit@haystax.com> or <daniel.emaasit@gmail.com>.
