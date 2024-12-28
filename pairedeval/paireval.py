import numpy as np

def paired_eval(scores, labels, min_dist=0.5):
    '''
    An O(n log n) implementation of paired evaluation
    https://doi.org/10.1016/j.patter.2023.100824

    Parameters
    ----------
    scores   - predicted scores returned by a machine learning model
    labels   - true labels
    min_dist - two labels must be separated by this distance for the
                corresponding samples to be considered rankable
    
    Returns
    -------
    1. The total number of rankable pairs
    2. The number of rankable pairs that are ranked correctly
    
    To compute AUC divide the second value by the first one
    '''
    
    # Argument verification
    if len(scores) < 2:
        raise Exception("Paired eval requires at least one pair")
    if len(scores) != len(labels):
        raise Exception("The number of scores and labels must be the same")
        
    # Compute the total number of rankable pairs (nrp) by traversing
    #   the sorted labels (sl) with two pointers, left(l) and right(r)
    sl = np.sort(labels)
    l = 0; r = 0; nrp = 0

    # Start by separating the two pointers by at least min_dist
    while (sl[r] - sl[l]) < min_dist and r < len(sl) - 1:
        r += 1
    
    # Reaching the end of the array indicates there are no rankable pairs
    if (sl[r] - sl[l]) < min_dist:
        return 0, 0
        
    while r < len(sl):
    
        # Catch the left pointer up, keeping it at least min_dist away
        while (sl[r] - sl[l+1]) >= min_dist:     # +1 to look ahead
            l += 1
        
        # Everything from the beginning of the array up to (and including) l is
        #   a rankable pair relative to r
        nrp = nrp + (l + 1)      # +1 to account for 0-based indexing
        r += 1
    
    # Compute the number of misranked pairs through inversion counting
    arr = np.array(labels)[np.argsort(scores)]      # Array for merge sort
    tmp = np.zeros_like(arr)                        # Temporary work space
    
    # Computes an inversion count (cnt) on the [left, right] region of arr
    def _merge_sort(l, r):
        if l >= r: return 0
        
        # Recurse on [left, mid) and [mid, right]
        m = (l + r)//2
        cnt  = _merge_sort(l, m)
        cnt += _merge_sort(m+1, r)
        
        # i2 points to the element in l that is at least min_dist higher than j
        i = l; j = m+1; k = l; i2 = l
        while i2 <= m and (arr[i2] - arr[j]) < min_dist: i2 += 1

        # Count inversions from the merge
        while i <= m and j <= r:
            if arr[i] > arr[j]:

                # Everything from i2 to m is an inversion relative to j
                cnt += m - i2 + 1

                tmp[k] = arr[j]
                j += 1; k += 1

                # Push i2 away to keep it min_dist away
                while i2 <= m and j <= r and (arr[i2] - arr[j]) < min_dist:
                    i2 += 1

            else:
                tmp[k] = arr[i]
                i += 1; k += 1
        
        # Copy the remaining bits of left and right
        while i <= m: tmp[k] = arr[i]; k += 1; i += 1
        while j <= r: tmp[k] = arr[j]; k += 1; j += 1
            
        # Update the array chunk with its sorted version
        arr[l:(r+1)] = tmp[l:(r+1)]
        return cnt

    # Ranked correctly = total - misranked
    return nrp, nrp - _merge_sort(0, len(arr)-1)

def naive_eval(scores, labels, min_dist=0.5):
    '''
    The naive O(n^2) implementation of paired evaluation
    https://doi.org/10.1101/2022.09.07.507020
    
    Used for testing.
    
    Parameters and return values are as in paired_eval()
    '''
    
    # Argument verification
    if len(scores) < 2:
        raise Exception("Paired eval requires at least one rankable pair")
    if len(scores) != len(labels):
        raise Exception("The number of scores and labels must be the same")

    
    n = 0   # Total rankable
    m = 0   # Ranked correctly
    
    for j in range(1, len(scores)):
        for i in range(j):
            
            # Is (i, j) rankable?
            if np.abs(labels[i] - labels[j]) >= min_dist:
                n += 1
            else:
                continue
                
            # Is (i, j) ranked correctly?
            if np.sign(scores[i] - scores[j]) == np.sign(labels[i] - labels[j]):
                m += 1
                
    return n, m
