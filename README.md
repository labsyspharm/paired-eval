# Paired evaluation of machine learning models

An O(n log n) implementation of [paired evaluation](https://www.cell.com/patterns/fulltext/S2666-3899(23)00146-0).

## Installation

The package can be installed directly from GitHub with:

```
pip install git+https://github.com/labsyspharm/paired-eval
```

## Introduction

Paired evaluation accepts as input a set of predictions produced by a model and the corresponding true labels. An optional argument `min_dist` specifies the necessary separation of labels for a pair of samples to be considered rankable. The recommended value of this parameter is `0.5` for discrete predictions tasks (e.g., binary classification, ordinal regression), and the expected amount of measurement noise for real-valued tasks (e.g., linear regression).

The function returns a tuple containing a) the number of pairs that are rankable under the specified `min_dist`, and b) the number of pairs that were actually ranked correctly by the prediction scores. Area under the ROC curve (AUC) can be estimated by dividing the second number by the first. Alternatively, the numbers of correctly- and incorrectly-ranked pairs from two separate models can be aggregated into a 2x2 contingency table for assessing statistical significance.

## Example

To demonstrate application of paired evaluation, we will make use of the [diabetes toy dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) from `sklearn`.

``` python
import scipy

import sklearn.datasets
import sklearn.model_selection as msel
import sklearn.linear_model
import sklearn.ensemble

import pairedeval as pe

# Load the dataset (ds) and randomly partition it into 80% train, 20% test
ds = sklearn.datasets.load_diabetes()
xtr, xte, ytr, yte = msel.train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

# Consider two models: random forest (rf) and linear regression (lr)
rf = sklearn.ensemble.RandomForestRegressor(max_depth=5, random_state=42)
lr = sklearn.linear_model.LinearRegression()

# Train both models and use them to compute prediction scores on the test data
rf.fit(xtr, ytr)
lr.fit(xtr, ytr)
rf_scores = rf.predict(xte)
lr_scores = lr.predict(xte)

# Apply paired evaluation to compute the number of test pairs ranked correctly
#  by each method. Since the target variable is an integer, we leave min_dist
#  at the default value of 0.5
# nr - number of rankable pairs
# nc - number of correctly ranked pairs
rf_nr, rf_nc = pe.paired_eval(rf_scores, yte)
lr_nr, lr_nc = pe.paired_eval(lr_scores, yte)

# The number of rankable pairs is a function of true labels and does not depend
#  on the predictions made by each method
rf_nr == lr_nr    # True

# Estimates of AUC
rf_AUC = rf_nc / rf_nr    # 0.7292307692307692
lr_AUC = lr_nc / lr_nr    # 0.7407692307692307

# Is the above difference significant? To answer this, we construct a 2x2
#  contingency table (ctab) cataloging the number of correcty- and
#  incorrectly-ranked pairs for each method
ctab = [
    [rf_nc,         lr_nc],         # Number of correctly-ranked pairs
    [rf_nr - rf_nc, lr_nr - lr_nc]  # Number of incorrectly-ranked pairs
]

# Apply Fisher's exact test to obtain a p value for the comparison
scipy.stats.fisher_exact(ctab)
# SignificanceResult(statistic=0.9424738034551119, pvalue=0.2589553719125148)
```
