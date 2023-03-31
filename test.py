import random
import numpy as np

import paireval as pe

def test_bc(n, k):
    '''
    Testing binary classification
    n - number of samples
    k - number of tests
    '''
    
    for iter in range(k):
        print(f"Iteration {k+1}")
        labels = random.choices([0, 1], k=n)
        scores = np.random.uniform(size=n)
        
        nr1, nc1 = pe.paired_eval(scores, labels, 0.5)
        nr2, nc2 =  pe.naive_eval(scores, labels, 0.5)
        
        print(f"O(n log n) implementation: {nc1} of {nr1} pairs are ranked correctly")
        print(f"O(n^2)     implementation: {nc2} of {nr2} pairs are ranked correctly")
        print("---")

        if nr1 != nr2: raise Exception("Total number of rankable pairs doesn't match")
        if nc1 != nc2: raise Exception("Rankable number of pairs doesn't match")

test_bc(100, 5)
