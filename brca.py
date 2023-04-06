import numpy as np
import pandas as pd
import sklearn.ensemble
import paireval as pe

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

if not all(x.index == y.index):
    raise Exception("Sample-label mismatch")

# Train a model
mdl = sklearn.ensemble.RandomForestRegressor(max_depth=5)
mdl.fit(x, y['GR_AOC'])

# Evaluate the predictions
scores = mdl.predict(x)
pe.paired_eval(scores, y['GR_AOC'])
