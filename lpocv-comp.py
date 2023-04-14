import pandas as pd
import numpy as np
import paireval as pe
import matplotlib.pyplot as plt
import matplotlib.lines

def wrangle(df):
    return df.assign(
        ldiff = lambda x: np.abs(x['measured1'] - x['measured2']),
        rnkcr = lambda x: np.sign(x['measured1']  - x['measured2']) == 
                          np.sign(x['predicted1'] - x['predicted2'])
    )

# Estimate AUC directly from LPOCV
def count1(df, delta):
    dfsub = df.loc[lambda x: x['ldiff'] > delta]
    return np.sum(dfsub['rnkcr']) / dfsub.shape[0]

# Estimate AUC by averaging scores for each cell line
def count2(df, delta):
    dfave = (
        pd.concat([
            df.filter(like='1').rename(columns=lambda x: x[:-1]),
            df.filter(like='2').rename(columns=lambda x: x[:-1])
        ])
        .groupby('cell_line')
        .agg(np.mean)
    )
    nr, nc = pe.paired_eval(dfave['predicted'], dfave['measured'], min_dist=delta)
    return nc / nr

# Annotate relevant tables
dfold = wrangle(pd.read_csv('data/palbociclib_lpocv_old.csv'))
dfnew = wrangle(pd.read_csv('data/palbociclib_lpocv.csv'))

# Examine both methods in the context of both data frames
x     = np.linspace(0.01, 0.5, num=50)
y1old = [count1(dfold, i) for i in x]
y1new = [count1(dfnew, i) for i in x]
y2old = [count2(dfold, i) for i in x]
y2new = [count2(dfnew, i) for i in x]


lines = [matplotlib.lines.Line2D([0], [0], color=cl, linewidth=2) for
        cl in ['red', 'blue', 'darkred', 'darkblue']]
lbls  = ['LPOCV on old', 'PE-avg on old', 'LPOCV on new', 'PE-avg on new']

fig, ax = plt.subplots()
ax.plot(x, y1old, color='red')
ax.plot(x, y2old, color='blue')
ax.plot(x, y1new, color='darkred')
ax.plot(x, y2new, color='darkblue')
ax.legend(lines, lbls, loc='upper left')
ax.set_xlabel('Delta')
ax.set_ylabel('Estimate of AUC')
fig.savefig('lpocv-comp.png', bbox_inches='tight')
