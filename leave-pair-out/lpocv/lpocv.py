import numpy as np
from numpy.random import choice, randint, RandomState
from sklearn.utils.validation import indexable, _num_samples
from itertools import combinations
import warnings

class LeavePairOut:
    '''
    Splits the data into train-test by choosing a pair of rankable samples as test 
    pair with several options such as O(N) pairs, group (or batch) matched pairs.
    
    Attributes:
    -----------
    split: generates the train-test split as per the specified conditions. 
        Parameters:
        -----------
        X: array-like, shape=(n_samples, n_features)
            data matrix
        y: array-like, shape=(n_samples)
            labels, allowed types: str, int, float, (bool, int)
        erry: float or array-like
            error in y for regression problems, must be non-negative real number(s)
        groups: array-like, shape=(n_samples)
            group labels
        match_window: float, default=np.inf
            the allowed window for matching the groups (when groups variable is 
            continuous)  
        num_pairs: int, default=None
            when specified each sample gets paired with num_pairs (<n_samples) samples provided 
            they satisfy rankable and group matching conditions
        closest_match: bool, default=False
            if True then pairs those test samples for which groups variable is closest
        random_state: int, default None
            seed for the random number generator when orderN=True and 
            match_closest=False.

    Returns:
    --------
    train and test indices
        '''
    def split(self, X, y, erry=10e-4, groups=None, match_window=np.inf, 
              num_pairs=None, closest_match=False, random_state=None):

        X, y, groups = indexable(X, y, groups)
        num_samples = _num_samples(X)
        if num_samples<2: 
            raise ValueError ('Number of samples must be greater than or equal to 2.') 
        return self.generate_train_test(X, y, erry, groups, match_window,
                                num_pairs, closest_match, random_state)

    def generate_train_test(self, X, y, erry, groups, match_window, 
                    num_pairs, closest_match, random_state):
        num_samples = _num_samples(X)
        indices = np.arange(num_samples)

        def rankable_match(a, b):
            '''
            Determines if the pair of indicies (a, b) are a "rankable" test pair
            which satisfy the
            rankable label conditions:
                y[a]!= y[b] for categorical
                abs(y[a] - y[b] < max(erry[a], erry[b])) for numerical
                y[a] = (True, A) and y[b] = (True, B) or
                y[a] = (True, A) and  y[b] = (False, B) and A<B or
                y[a] = (False, A) and y[b] = (True, B) and A>B for survival prediction
            group-matching conditions:
                groups[a]==groups[b] for categorical groups
                abs(groups[a]-groups[b]) < match_window for continuous groups
            '''

            label_str = y.dtype.type is np.str_ and y[a]!=y[b]
            label_numeric = ((y.dtype.type is np.float_ or y.dtype.type is np.int_) \
                                and a!=b and abs(y[a] - y[b]) > max(erry[a], erry[b]))
            try:
                label_surv = (a!=b and \
                              (y.dtype[0].type is np.bool_ and \
                                   (y.dtype[1].type is np.float_ or \
                                    y.dtype[1].type is np.int_)
                              ) and \
                              ((y[a][0]==True and y[b][0]==True) or \
                               (y[a][0]==True and y[b][0]==False and y[a][1]<y[b][1])or \
                               (y[a][0]==False and y[b][0]==True and y[a][1]>y[b][1])
                              )
                             )
            except KeyError:
                label_surv = False
            
            groups_str = groups.dtype.type is np.str_ and groups[a]==groups[b]
            groups_numeric = ((groups.dtype.type is np.int_ or \
                            groups.dtype.type is np.float_) and \
                            abs(groups[a]-groups[b]) < match_window)
            
            if ((label_str and groups_str) or \
                (label_str and groups_numeric) or \
                (label_numeric and groups_str) or \
                (label_numeric and groups_numeric) or \
                (label_surv and groups_str) or\
                (label_surv and groups_numeric)):
                return True
            else:
                return False
            
        def closest_rankable_match(a):
            '''
            Gets the closest rankable match for the index a for continuous groups
            rankable label conditions:
                y[a]!= y[b] for categorical
                abs(y[a] - y[b] < max(erry[a], erry[b])) for numerical
                y[a] = (True, A) and y[b] = (True, B) or
                y[a] = (True, A) and  y[b] = (False, B) and A<B or
                y[a] = (False, A) and y[b] = (True, B) and A>B for survival prediction
            closest-match condition:
                abs(groups[a]-groups[b]) is minimum
            '''
            delta = np.inf
            for b in indices:
                label_str = y.dtype.type is np.str_ and y[a]!=y[b]
                label_numeric = ((y.dtype.type is np.float_ or \
                                  y.dtype.type is np.int_) and \
                                  a!=b and abs(y[a] - y[b]) > max(erry[a], erry[b]))
                try:
                    label_surv = (a!=b and\
                                   (y.dtype[0].type is np.bool_ and \
                                        (y.dtype[1].type is np.float_ or \
                                        y.dtype[1].type is np.int_)
                                   ) and \
                                  ((y[a][0]==True and y[b][0]==True) or \
                                   (y[a][0]==True and y[b][0]==False and y[a][1]<y[b][1])or \
                                   (y[a][0]==False and y[b][0]==True and y[a][1]>y[b][1])
                                  )
                                 )
                except KeyError:
                    label_surv = False
                if (abs(groups[a]-groups[b])< delta and \
                    (label_str or label_numeric or label_surv)):
                    index = b
                    delta = abs(groups[a]-groups[b])
            return index

        def train_test_indices(a, b):
            '''
            Returns the train and test indices as numpy arrays
            '''
            test_indices = np.array([a, b])
            test_mask = np.zeros(num_samples, dtype=bool)
            test_mask[test_indices] = True
            train_indices = indices[np.logical_not(test_mask)]
            return train_indices, test_indices


        if erry is None:
            erry = np.zeros(num_samples, dtype=float)

        if type(erry) is float or type(erry) is int:
            erry = np.full_like(np.arange(num_samples, dtype=float), erry)
        
        if groups is None:
            groups = np.zeros(num_samples, dtype=int)
        
        assert num_pairs is None or num_pairs<num_samples, "'num_pairs' should be less than 'n_samples'"
            
            
        # All possible pairs ~ N**2 pairs
        if num_pairs is None and closest_match is False:
            for i, j in combinations(range(num_samples), 2):
                if rankable_match(i, j):
                    yield train_test_indices(i, j)
                else:
                    continue
        # num_pairs
        else:
            paired = np.zeros(shape=(num_samples, num_samples), dtype='bool')
            for i in indices:
                if closest_match is False:
                    if random_state is None:
                        r = RandomState(randint(2**32-1))
                    else:
                        r = RandomState(random_state+i)
                    #try:
                    rankable_list = [k for k in indices if rankable_match(i, k)]
                    if len(rankable_list)<num_pairs:
                        warnings.warn('Asked for %s pairs,  %s-th sample has %s rankable pairs.'% \
                                        (num_pairs, i, len(rankable_list)), UserWarning)
                    for l in range(num_pairs):
                        if len(rankable_list)>0:
                            j = r.choice(rankable_list)
                            if paired[i, j]==False and paired[j, i]==False:
                                yield train_test_indices(i, j)
                                paired[i, j] = paired[j, i] = True
                                #rankable_list.remove(j)
                    #except ValueError:
                        #pass
                else:
                    try:
                        j = closest_rankable_match(i)
                        yield train_test_indices(i, j)
                    except UnboundLocalError:
                        pass
