import numpy as np
import networkx as nx

def random_sample(distance_matrix, target_size, nsample=10000):
    """
    Sample random subsets and return the minimum separations within these

    Args:
        distance_matrix (array): 2D matrix of pairwise distances.
        target_size (int): Desired size of subset.
        nsample (int): Numer of random samples to draw.

    Returns:
        array: distribution of minumimum separations over the samples

    """
    min_pairwise_dist_arr = []
    n = distance_matrix.shape[0]
    for i in range(nsample):
        H = np.random.choice(n, target_size, replace=False)
        idx = [[h1,h2] for h1 in H for h2 in H if h1<h2]
        idx = tuple(np.array(idx).T)
        min_pairwise_dist_tmp = np.min(distance_matrix[idx])
        min_pairwise_dist_arr += [min_pairwise_dist_tmp]
    return min_pairwise_dist_arr

def greedy_search(distance_matrix, target_size):
    """Greedy algorithm to find dissimilar subsets

    Args:
        distance_matrix (array): 2D matrix of pairwise distances.
        target_size (int): Desired size of subset.

    Returns:
        tuple: (list of index values of subset in distance_matrix,
            minimum pairwise distance in this subset)

    """
    n = distance_matrix.shape[0]
    idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    idx = list(idx)
    tmp = distance_matrix[idx][:,idx]
    for n0 in range(3, target_size+1):
        iii = list(range(n))
        for idx0 in idx:
            iii.remove(idx0)
        ttt = []
        for i in iii:
            idx_tmp = idx + [i]
            tmp = distance_matrix[idx_tmp][:,idx_tmp]
            ttt += [np.min(tmp[np.triu_indices(n0, k=1)])]
        idx += [iii[np.argmax(ttt)]]
    tmp = distance_matrix[idx][:,idx]
    min_pairwise_dist = np.min(tmp[np.triu_indices(target_size, k=1)])
    return idx, min_pairwise_dist

def independent_set_search(distance_matrix,
                           target_size,
                           search_bounds_percentile=[99,1],
                           n_iter_binary_search=20,
                           n_trials_per_iter=1000):
    """Find a dissimilar subset by searching for independent sets in graphs

    Given a graph where with edges between two vertices iff they are closer than
    t, the independent sets of this graph will have all separations > t. This
    function searches for large independent sets, varying the threshold t till
    we recover a subset of the desired size. The threshold is varied via a
    binary search starting with bounds defined by percentiles of the separation
    distributuon. For each threshold, repeated trials are performed to stabilise
    the stochastic result of the independent set search algortihm.

    Args:
        distance_matrix (array): 2D matrix of pairwise distances.
        target_size (int): Desired size of subset.
        search_bounds_percentile: [hi,lo] percentile values which define
            initial distance threshold bounds for binary search. Decreasing the
            lower bound towards 0 can incur significant slowing.
        n_iter_binary_search (int): Number of iterations of binary search.
        n_trials_per_iter (int): Number of trials per iteration.

    Returns:
        tuple: (list of index values of subset in distance_matrix,
            minimum pairwise distance in this subset)

    Notes:
        This is an implementation of https://cs.stackexchange.com/a/22783 using
        a generic approximation algorithm for independent sets from  `networkx`.
    """
    def find_a_maximal_independent_set(distance_threshold):
        # Get graph G where vertices within `distance_threshold` of one another
        # are connected
        adjacency_matrix = distance_matrix<distance_threshold
        for i in range(adjacency_matrix.shape[0]):
            adjacency_matrix[i,i] = False
        G = nx.convert.to_networkx_graph(adjacency_matrix)
        # Search through maximal_independent_sets of G. These are subsets which
        # will have all pairwise distances > distance_threshold. Repeat for many
        # trials, updating the solution with the best one found so far, where
        # best is `largest subset with the largest minumim pairwise distance`
        max_idpt_set = []
        max_idpt_set_size = 0
        min_pairwise_dist = 0.
        for i in range(n_trials_per_iter):
            H = nx.algorithms.mis.maximal_independent_set(G)
            if len(H)>=max_idpt_set_size:
                min_pairwise_dist = 0.
                idx = [[h1,h2] for h1 in H for h2 in H if h1<h2]
                idx = tuple(np.array(idx).T)
                min_pairwise_dist_tmp = np.min(distance_matrix[idx])
                if min_pairwise_dist_tmp>min_pairwise_dist:
                    max_idpt_set = H
                    max_idpt_set_size = len(H)
                    min_pairwise_dist = min_pairwise_dist_tmp
        return max_idpt_set, max_idpt_set_size, min_pairwise_dist
    # get endpoints of distances for binary search
    n = distance_matrix.shape[0]
    flattened_distances = distance_matrix[np.triu_indices(n, k=1)]
    lo, hi = np.percentile(flattened_distances, search_bounds_percentile)
    # binary search
    soln_lo = find_a_maximal_independent_set(lo)
    soln_hi = find_a_maximal_independent_set(hi)
    soln_with_target_size = []
    if (soln_lo[1]<target_size<=soln_hi[1]):
        for i in range(n_iter_binary_search):
            med = np.mean([lo, hi])
            soln_med = find_a_maximal_independent_set(med)
            if soln_med[1]<target_size:
                lo, hi = med, hi
                soln_lo, soln_hi = soln_med, soln_hi
            else:
                lo, hi = lo, med
                soln_lo, soln_hi = soln_lo, soln_med
            if soln_lo[1]==target_size:
                soln_with_target_size += [soln_lo]
            if soln_hi[1]==target_size:
                soln_with_target_size += [soln_hi]
        if len(soln_with_target_size)>0:
            # of all solutions with target size, take the one with largest
            # minimum pairwise distance.
            min_pairwise_dists = [s[2] for s in soln_with_target_size]
            idx = np.argmax(min_pairwise_dists)
            soln = soln_with_target_size[idx]
        else:
            msg = f'No solutions with target_size={target_size} found. '
            msg += f'Current size bounds are [{soln_lo[1],soln_hi[1]}] - '
            msg += 'try increasing n_iter_binary_search to converge to target?'
            raise ValueError(msg)
    else:
        init_bounds = [soln_lo[1], soln_hi[1]]
        msg = f'target_size {target} is outside initial bounds {init_bounds}. '
        msg += 'Try extending search_bounds_percent towards [100,0].'
        raise ValueError(msg)
    soln = (soln[0], soln[2])
    return soln
