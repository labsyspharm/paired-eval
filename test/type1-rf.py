import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.model_selection
import pairedeval as pe
import scipy
import matplotlib.pyplot as plt

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

# Train and evaluate a random forest model with nest trees
def eval1(xtr, ytr, xte, yte, nest):
    mdl = sklearn.ensemble.RandomForestRegressor(n_estimators=nest)
    mdl.fit(xtr, ytr['GR_AOC'])

    # Evaluate the predictions
    scores = mdl.predict(xte)
    nr, nc = pe.paired_eval(scores, yte['GR_AOC'], min_dist=0.1)
    return nc, (nr-nc)

# Comparison of random forest and logistic regression models
def comparison(niter, nest):
    pvals = []
    for iter in range(niter):
        xtr, xte, ytr, yte = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

        if not all(xtr.index == ytr.index):
            raise Exception("Sample-label mismatch in the training data")
        if not all(xte.index == yte.index):
            raise Exception("Sample-label mismatch in the test data")

        print(f"Iteration {iter}")
        m0 = eval1(xtr, ytr, xte, yte, nest)
        m1 = eval1(xtr, ytr, xte, yte, nest)

        # Compare all to the reference
        pvals.append(scipy.stats.fisher_exact([m0, m1]).pvalue)
        
    return pvals

pvals1 = comparison(1000, 100)
pvals2 = comparison(1000, 22)
pvals3 = comparison(1000, 4)

# Collect statistics
xs = np.linspace(0.01, 0.5, num=50)
y1 = np.zeros_like(xs)
y2 = np.zeros_like(xs)
y3 = np.zeros_like(xs)
for i in range(len(xs)):
    y1[i] = np.sum(pvals1 <= xs[i]) / len(pvals1)
    y2[i] = np.sum(pvals2 <= xs[i]) / len(pvals2)
    y3[i] = np.sum(pvals3 <= xs[i]) / len(pvals3)

# Compose the plot
fig, ax = plt.subplots()
ax.set_facecolor("whitesmoke")
plt.grid(color = 'white', linestyle = '-', linewidth = 1)
ax.plot(xs, y1, color='red', label="RF (100 trees)")
ax.plot(xs, y2, color='black', label="RF (22 trees)")
ax.plot(xs, y3, color='blue', label="RF (4 trees)")
ax.plot([0, 0.5], [0, 0.5], '--', color='gray', label="Reference")
ax.legend(loc='upper left')
ax.set_xlabel('Significance threshold')
ax.set_ylabel('Fraction of p values below threshold')
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])

fig.savefig('type1-rf.pdf', bbox_inches='tight')
fig.savefig('type1-rf.png', bbox_inches='tight')
