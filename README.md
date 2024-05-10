
# BARMPy

![text ignored, basic tests](https://github.com/dvbuntu/barmpy/actions/workflows/test-barmpy.yml/badge.svg)

![text ignored, docs](https://github.com/dvbuntu/barmpy/actions/workflows/pages/pages-build-deployment/badge.svg)

![text ignored, download count](https://img.shields.io/github/downloads/dvbuntu/barmpy/total.svg)

![HitCount to repo page](https://hits.dwyl.com/dvbuntu/barmpy/start-here.svg)


## Intro

`barmpy` is the Python implementation of Baeysian Additive Regression Models, a generalization of BART, currently being researched [1].  We hope this library is useful for practictioners, enabling Bayesian architecture search and model ensembling.

Skeleton repo adapted from [BartPy](https://github.com/JakeColtman/bartpy).

Check out the [Tutorial](https://drive.google.com/file/d/1FgpCyEUqqnihkfm-6nuV5RdZwAJlSJq5/view?usp=drive_link)

## Quick Start

`barmpy` is on PyPi!  Install the latest released version with `pip install barmpy`.  `barmpy` also strives to be compatible with `sklearn` and easy-to-use.  If you have arrays of target data, `Y`, and input data, `X`, you can quickly train a model and make predictions using it.  `barmpy` currently supports ensembles of neural networks for both regression and binary classification.  See below for simple examples.

```python
from sklearn import datasets, metrics
from barmpy.barn import BARN, BARN_bin
import numpy as np

# Regression problem
db = datasets.load_diabetes()
model = BARN(num_nets=10,
          random_state=0,
          warm_start=True,
          solver='lbfgs',
		  l=1)
model.fit(db.data, db.target)
pred = model.predict(db.data)
print(metrics.r2_score(db.target, pred))

# Classification problem
bc = datasets.load_breast_cancer()
bmodel = BARN_bin(num_nets=10,
          random_state=0,
          warm_start=True,
          solver='lbfgs',
		  l=1)
bmodel.fit(bc.data, bc.target)
pred = bmodel.predict(bc.data)
print(metrics.classification_report(bc.target, np.round(pred)))

```

## References

 [1] https://arxiv.org/abs/2404.04425
