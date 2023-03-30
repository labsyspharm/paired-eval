import numpy as np

def paired_eval(scores, labels, min_dist=0.5):
    '''
    An O(n log n) implementation of paired evaluation
    https://doi.org/10.1101/2022.09.07.507020

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
        raise Exception("Paired eval requires at least one rankable pair")
    if len(scores) != len(labels):
        raise Exception("The number of scores and labels must be the same")
        
    # Compute the total number of rankable pairs (nrp) by traversing
    #   the sorted labels (sl) with two pointers, left(l) and right(r)
    sl = np.sort(labels)
    l = 0; r = 0; nrp = 0

    # Start by separating the two pointers by at least min_dist
    while (sl[r] - sl[l]) < min_dist and r < len(sl) - 1:
        r = r + 1
    
    # Reaching the end of the array indicates there are no rankable pairs
    if (sl[r] - sl[l]) < min_dist:
        raise Exception("Paired eval requires at least one rankable pair")
        
    while r < len(sl):
    
        # Catch the left pointer up, keeping it at least min_dist away
        while (sl[r] - sl[l+1]) >= min_dist:     # +1 to look ahead
            l = l + 1
        
        # Everything from the beginning of the array up to (and including) l is
        #   a rankable pair relative to r
        nrp = nrp + (l + 1)      # +1 to account for 0-based indexing
        r = r + 1
    
    # Compute the number of misranked pairs through inversion counting
    arr = np.array(labels)[np.argsort(scores)]      # Array for merge sort
    tmp = np.zeros_like(arr)                        # Temporary work space
    
    # Computes an inversion count (cnt) on the [left, right] region of arr
    def _merge_sort(l, r):
        cnt = 0
        if l >= r: return cnt
        
        # Recurse on [left, mid) and [mid, right]
        m = (l + r)//2
        cnt += _merge_sort(l, m)
        cnt += _merge_sort(m+1, r)
        
        # Count inversions from the merge
        i = l; j = m+1; k = l
        while i <= m and j <= r:
            
            # No inversions if (i, j) is ranked correctly
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1; k += 1
            
            # Otherwise, everything up to mid is an inversion relative to j
            else:
                cnt += m - i + 1
                tmp[k] = arr[j]
                j += 1; k += 1
        
        # Copy the remaining bits of left and right
        while i <= m: tmp[k] = arr[i]; k += 1; i += 1
        while j <= r: tmp[k] = arr[j]; k += 1; j += 1
            
        # Update the array chunk with its sorted version
        arr[l:(r+1)] = tmp[l:(r+1)]
        return cnt

    # Ranked correctly = total - misranked
    return nrp, nrp - _merge_sort(0, len(arr)-1)
