import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import paireval as pe
import scipy

# Load relevant data
fnStem = 'https://raw.githubusercontent.com/labsyspharm/brca-profiling/main/data'
gl = pd.read_csv(f"{fnStem}/gene_list.txt", header=None)[0].tolist()
y  = (pd.read_csv(f"{fnStem}/grmetrics.csv")
    .loc[lambda z: z['generic_name'] == 'palbociclib']
    .set_index('cell_line')
    .filter(['GR_AOC']))
x  = (pd.read_csv(f"{fnStem}/rnaseq_log2rpkm.csv")
    .loc[lambda z: z['gene_name'].isin(gl)]
    .set_index('gene_name')
    .filter(y.index)
    .T)

# Train a model
def eval1(xtr, ytr, xte, yte):
    mdl = sklearn.ensemble.RandomForestRegressor(max_depth=5)
    mdl.fit(xtr, ytr['GR_AOC'])

    # Evaluate the predictions
    scores = mdl.predict(xte)
    nr, nc = pe.paired_eval(scores, yte['GR_AOC'], min_dist=0.1)
    return nc, (nr-nc)


pvals = []
for iter in range(100):
    xtr, xte, ytr, yte = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    if not all(xtr.index == ytr.index):
        raise Exception("Sample-label mismatch in the training data")
    if not all(xte.index == yte.index):
        raise Exception("Sample-label mismatch in the test data")

    print(f"Iteration {iter}")
    m1 = eval1(xtr, ytr, xte, yte)
    m2 = eval1(xtr, ytr, xte, yte)
    pvals.append(scipy.stats.fisher_exact([m1, m2]).pvalue)
