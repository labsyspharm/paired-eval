import random
import numpy as np

import paireval as pe

def report(nln, n2):
    '''
    Shared reporting function
    nln - results from the O(n log n) implementation
    n2  - results from the O(n^2) implementation
    '''
    print(f"O(n log n) implementation: {nln[1]} of {nln[0]} pairs are ranked correctly")
    print(f"O(n^2)     implementation: {n2[1]} of {n2[0]} pairs are ranked correctly")

    if nln[0] != n2[0]: raise Exception("Total number of rankable pairs doesn't match")
    if nln[1] != n2[1]: raise Exception("Rankable number of pairs doesn't match")


def test_bc(n, k):
    '''
    n - number of samples
    k - number of tests
    '''
    print("Testing binary classification")
    for iter in range(k):
        print(f"---\nIteration {iter+1}")
        labels = random.choices([0, 1], k=n)
        scores = np.random.uniform(size=n)
        
        nln = pe.paired_eval(scores, labels, 0.5)
        n2  =  pe.naive_eval(scores, labels, 0.5)
        report(nln, n2)        

def test_or(n, m, k):
    '''
    n - number of samples
    m - number of classes
    k - number of tests
    '''
    print("Testing ordinal regression")
    for iter in range(k):
        print(f"Iteration {iter+1}")
        labels = random.choices(range(m), k=n)
        scores = np.random.uniform(size=n)

        nln = pe.paired_eval(scores, labels, 0.5)
        n2  =  pe.naive_eval(scores, labels, 0.5)
        report(nln, n2)        

test_bc(100, 20)
test_or(100, 5, 20)
