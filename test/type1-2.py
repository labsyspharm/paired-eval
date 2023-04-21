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

# Train a model
def eval1(xtr, ytr, xte, yte):
    mdl = sklearn.ensemble.RandomForestRegressor(max_depth=5)
    mdl.fit(xtr, ytr['GR_AOC'])

    # Evaluate the predictions
    scores = mdl.predict(xte)
    nr, nc = pe.paired_eval(scores, yte['GR_AOC'], min_dist=0.1)
    return nc, (nr-nc)

# Train a linear regression model
def eval2(xtr, ytr, xte, yte):
    mdl = sklearn.linear_model.LinearRegression()
    mdl.fit(xtr, ytr['GR_AOC'])
    
    # Evaluate the predictions
    scores = mdl.predict(xte)
    nr, nc = pe.paired_eval(scores, yte['GR_AOC'], min_dist=0.1)
    return nc, (nr-nc)

pvals1 = []
pvals2 = []
for iter in range(1000):
    xtr, xte, ytr, yte = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

    if not all(xtr.index == ytr.index):
        raise Exception("Sample-label mismatch in the training data")
    if not all(xte.index == yte.index):
        raise Exception("Sample-label mismatch in the test data")

    print(f"Iteration {iter}")
    m0 = eval1(xtr, ytr, xte, yte)
    m1 = eval1(xtr, ytr, xte, yte)
    m2 = eval2(xtr, ytr, xte, yte)
    pvals1.append(scipy.stats.fisher_exact([m0, m1]).pvalue)
    pvals2.append(scipy.stats.fisher_exact([m0, m2]).pvalue)

# Collect statistics
xs = np.linspace(0.01, 0.5, num=50)
y1 = np.zeros_like(xs)
y2 = np.zeros_like(xs)
for i in range(len(xs)):
    y1[i] = np.sum(pvals1 <= xs[i]) / len(pvals1)
    y2[i] = np.sum(pvals2 <= xs[i]) / len(pvals2)

# Compose the plot
fig, ax = plt.subplots()
ax.set_facecolor("whitesmoke")
plt.grid(color = 'white', linestyle = '-', linewidth = 1)
ax.plot(xs, y1, color='red', label="Similar models")
ax.plot(xs, y2, color='blue', label="Different models")
ax.plot([0, 0.5], [0, 0.5], '--', color='gray', label="Reference")
ax.legend(loc='upper left')
#ax.axvline(0.05, linestyle=':', color='blue')
ax.set_xlabel('Significance threshold')
ax.set_ylabel('Fraction of p values below threshold')
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 0.5])

fig.savefig('type1-2.pdf', bbox_inches='tight')
fig.savefig('type1-2.png', bbox_inches='tight')
