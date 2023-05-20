# LPOCV: Leave-pair-out cross validation
A Python module that implements leave-pair-out cross-validation to obtain a more accurate estimate of model performance in machine learning classification and regression problems. It does so by allowing the user to match the samples by a confounding batch variable and including errors on the labels (for regression problems).

### Requirements
Requires `scikit_learn` and `numpy`
### Installation
Use the package manager `pip` to  install the `lpocv`
```
pip install lpocv
```
### Usage
The `lpocv` module is compatible with machine learning algorithms in `scikit-learn`. One can call the `split()` function of `LeavePairOut` (analogous to `split()` for `KFold`, `ShuffleSplit`, etc.) as follows:

<pre><code>
from lpocv.lpocv import LeavePairOut
for train, test in LeavePairOut().split(X,y):
	BLOCK OF CODE FOR TRAINING
	AND EVALUATION OF THE PREDICTOR
</code></pre>
where`X` is the data matrix of shape  `(n_samples, n_features)` and `y` is an array consisting of the class labels (integers or strings) or regression labels (floats) of shape `(n_samples)`.

For categorical labels, only the pairs from opposing or dissimilar classes, <code>y[i]!=y[j]</code>, are allowed as test pairs. For regression problems, one should specify `erry` so that only those test pairs for which  <code>|y[i]-y[j]| > max(erry[i],erry[j])</code> are allowed. `erry` can be an array or a singleton. By default `erry=10e-4`, please change this if the difference in your prediction labels is smaller than this value.

The information about the known confounder can be supplied through the `groups` variable. `groups` can be categorical  or continuous and should be an array of shape `(n_samples)`. For categorical confounders only those pairs for which the `groups` is identical are allowed as test pairs. For continuous confounders one ought to specify a `match_window` such that only `|groups[i]-groups[j]| < math_window` are allowed as test pairs, or set `closest_match=True`  to select pairs with the lowest difference in their confounders.

Leave-pair-out method provides with flexibility in choosing the combinatorial complexity of the cross-validation folds. This can be specified through `num_pairs`. Setting `num_pairs=1` ensures that every sample gets paired with exactly one other allowed sample, giving us linear cross-validatory complexity with  <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(n)">  test pairs. When `num_pairs=n`, each sample gets paired with all other allowed samples giving quadratic complexity with <img src="https://render.githubusercontent.com/render/math?math=\mathcal{O}(n^2)">  test pairs.  In general, `num_pairs` can be any integer `1 <= num_pairs <= n`.
