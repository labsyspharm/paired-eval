import random
import numpy as np
import time
import pairedeval as pe
import matplotlib.pyplot as plt

# Worker function
def timeit(n):
    t0 = time.time()
    labels = np.random.uniform(size=n)
    scores = np.random.uniform(size=n)
    nr, nc = pe.paired_eval(scores, labels, 0.1)
    t1 = time.time()
    return (t1-t0)

# Set up the results structure
niter = 30
x = [10, 100, 1000, 10000, 100000, 1000000]
t = np.zeros([niter, len(x)])

# Evaluate all iterations
for i in range(niter):
    print(f"Iteration {i}")
    for j in range(len(x)):
        xj = x[j]
        print(f"Evaluating n={xj}")
        t[i, j] = timeit(xj)

# Compute relevant statistics
tm = np.mean(t, axis=0)
ts = np.std(t, axis=0)

# Compose the plot
fig, ax = plt.subplots()
ax.set_facecolor("whitesmoke")
plt.grid(color = 'white', linestyle = '-', linewidth = 1)

ax.plot(np.log10(x), np.log10(tm), color='red')
ax.fill_between(np.log10(x),
                np.log10(tm-ts),
                np.log10(tm+ts),
                alpha=0.3)

ax.set_xticks(np.log10(x))
ax.set_xticklabels(['10', '100', '1K', '10K', '100K', '1M'])
ax.set_xlabel('Number of samples')

ytk = [0.001, 0.01, 0.1, 1, 10]
ax.set_yticks(np.log10(ytk))
ax.set_yticklabels(ytk)
ax.set_ylabel('Execution time (s)')

fig.savefig('scaling.pdf', bbox_inches='tight')
fig.savefig('scaling.png', bbox_inches='tight')
