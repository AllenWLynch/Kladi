
import numpy as np
import networkx as nx
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import inv
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.sparse import csgraph
from scipy.stats import entropy, pearsonr, norm
from copy import deepcopy

## ___ TREE DATATYPE FUNCTIONS ###

class TreeException(Exception):
    pass

def get_root_state(G):
    G = G.astype(bool)

    incoming_connections = G.sum(0) #(incoming, outgoing)

    if (incoming_connections == 0).sum() > 1:
        raise TreeException('Graph has multiple possible sources, and is not a tree')

    return np.argwhere(incoming_connections == 0)[0,0]

def num_children(G, node_idx):
    G = G.astype(bool)
    return G[node_idx, :].sum()

def get_children(G, node_idx):
    G = G.astype(bool)

    return np.argwhere(G[node_idx, :])[:,0]

def is_leaf(G, node_idx):
    
    G = G.astype(bool)
    return num_children(G) == 0

def get_dendogram_levels(G):
    G = G.astype(bool)
    nx_graph = nx.convert_matrix.from_numpy_array(G)

    dfs_tree = list(nx.dfs_predecessors(nx_graph, get_root_state(G)))[::-1]

    centerlines = {}
    num_termini = 0

    def get_or_set_node_position(node):

        if not node in centerlines:
            if is_leaf(G, node):
                centerlines[node] = num_termini
                num_termini+=1
            else:
                centerlines[node] = np.mean([get_or_set_node_position(child) for child in get_children(G, node)])

        return centerlines[node]

    for node in dfs_tree:
        get_or_set_node_position(node)

    return centerlines

### ___ PALANTIR FUNCTIONS ____ ##

def get_pseudotime(n_waypoints = 3000, max_iterations = 25, n_neighbors = 30, n_jobs = 1,
        *, start_cell, diffmap, **terminal_cells):

    assert(isinstance(max_iterations, int) and max_iterations > 0)
    n_waypoints= min(len(diffmap), n_waypoints)
    lineage_names, termin_cells = list(zip(*sorted(terminal_cells.items(), key = lambda x : x[1])))

    waypoints = sample_waypoints(n_waypoints = n_waypoints, diffmap = diffmap)
    waypoints = np.array(list(set([*waypoints, start_cell, *terminal_cells])))

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs).fit(diffmap)
    distance_matrix = nbrs.kneighbors_graph(diffmap, mode="distance")

    ####


    
    # CHECK FOR GRAPH CONNECTION!


    ###

    # Distances
    dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(csgraph.dijkstra)(distance_matrix, False, cell)
        for cell in waypoints
    )

    # Convert to distance matrix
    D = np.vstack([d[np.newaxis, :] for d in dists])

    start_waypoint_idx = np.argwhere(waypoints == start_cell)[0]

    # Determine the perspective matrix
    # Waypoint weights
    sdv = np.std(np.ravel(D)) * 1.06 * len(np.ravel(D)) ** (-1 / 5)
    W = np.exp(-0.5 * np.power((D / sdv), 2))
    # Stochastize the matrix
    W = W / W.sum(0)

    # Initalize pseudotime to start cell distances
    pseudotime = D[start_waypoint_idx, :].reshape(-1)
    converged = False

    # Iteratively update perspective and determine pseudotime
    iteration = 1
    while not converged and iteration < max_iterations:
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for i, waypoint_idx in enumerate(waypoints):
            # Position of waypoints relative to start
            if waypoint_idx != start_cell:
                idx_val = pseudotime[waypoint_idx]

                # Convert all cells before starting point to the negative
                before_indices = pseudotime < idx_val
                P[i, before_indices] = -D[i, before_indices]

                # Align to start
                P[i, :] = P[i, :] + idx_val

        # Weighted pseudotime
        new_traj = np.multiply(P,W).sum(0)
        # Check for convergence
        corr = pearsonr(pseudotime, new_traj)[0]
        if corr > 0.9999:
            converged = True

        # If not converged, continue iteration
        pseudotime = new_traj
        iteration += 1

    pseudotime = pseudotime - pseudotime.min() #make 0 minimum
    waypoint_weights = W
    
    return dict(
        pseudotime = pseudotime, 
        waypoints = waypoints, 
        waypoint_weights = waypoint_weights, 
        lineage_names = lineage_names, 
        terminal_cells = terminal_cells, 
        start_cell = start_cell,
        n_neighbors = n_neighbors,
        n_jobs = n_jobs,
    )


def make_markov_matrix(affinity_matrix):
    inverse_rowsums = sparse.diags(1/np.array(affinity_matrix.sum(axis = 1)).reshape(-1)).tocsr()
    markov_matrix = inverse_rowsums.dot(affinity_matrix)
    return markov_matrix


def sample_waypoints(num_waypoints = 3000,*, diffmap):

    waypoint_set = list()
    no_iterations = int((num_waypoints) / diffmap.shape[1])

    # Sample along each component
    N = len(diffmap)
    for feature_col in diffmap.T:

        # Random initialzlation
        iter_set = [np.random.choice(N)]

        # Distances along the component
        dists = np.zeros([N, no_iterations])
        dists[:, 0] = np.abs(feature_col - feature_col[iter_set])
        for k in range(1, no_iterations):
            # Minimum distances across the current set
            min_dists = dists[:, 0:k].min(axis=1)

            # Point with the maximum of the minimum distances is the new waypoint
            new_wp = np.where(min_dists == min_dists.max())[0][0]
            iter_set.append(new_wp)

            # Update distances
            dists[:, k] = np.abs(feature_col - feature_col[new_wp])

        # Update global set
        waypoint_set = waypoint_set + iter_set

    # Unique waypoints
    waypoints = np.unique(waypoint_set)

    return waypoints


def get_directed_knn_graph(n_neighbors = 30, n_jobs = 1,*, diffmap, pseudotime):

    # kNN graph
    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors, metric="euclidean", n_jobs=n_jobs
    ).fit(diffmap)
    kNN = nbrs.kneighbors_graph(diffmap, mode="distance")
    dist, ind = nbrs.kneighbors(diffmap)

    # Standard deviation allowing for "back" edges
    adpative_k = np.min([int(np.floor(n_neighbors / 3)) - 1, 30])
    adaptive_std = np.ravel(dist[:, adpative_k])

    # Remove edges that move backwards in pseudotime except for edges that are within
    # the computed standard deviation
    delta_pseudotime = (pseudotime[:, np.newaxis] - pseudotime[ind])

    rem_edges = np.argwhere( delta_pseudotime > adaptive_std[:, np.newaxis] )
    x = rem_edges[:,0]
    y = rem_edges[:,1]

    # Update adjacecy matrix    
    kNN[x, ind[x,y]] = np.inf

    return kNN, adaptive_std

def construct_directed_chain(n_neighbors = 30, n_jobs = 1, *, diffmap, pseudotime):

    # Markov chain construction
    kNN, adaptive_std = get_directed_knn_graph(n_neighbors = n_neighbors, n_jobs = n_jobs,
        diffmap = diffmap, pseudotime = pseudotime)

    # Affinity matrix and markov chain
    i,j,d = sparse.find(kNN)
    
    delta_pseudotime = np.abs(pseudotime[j] - pseudotime[i])
    
    affinity = np.exp(
        -(d ** 2) / (adaptive_std[i] * delta_pseudotime) * 0.5
        -(d ** 2) / (adaptive_std[j] * delta_pseudotime) * 0.5
    )
    affinity_matrix = sparse.csr_matrix((affinity, (i, j)), [len(pseudotime), len(pseudotime)])

    return make_markov_matrix(affinity_matrix)


def _terminal_states_from_markov_chain(self, directed_chain):
    raise NotImplementedError()


def get_branch_probs(num_waypoints = 3000, *, directed_chain, **terminal_cells):


    absorbing_states = np.zeros(directed_chain.shape[0]).astype(bool)
    absorbing_states[np.array(absorbing_cells)] = True

    directed_chain = directed_chain.toarray()
    # Reset absorption state affinities by Removing neigbors
    directed_chain[absorbing_states, :] = 0
    # Diagnoals as 1s
    directed_chain[absorbing_states, absorbing_states] = 1

    # Fundamental matrix and absorption probabilities
    # Transition states
    trans_states = ~absorbing_states

    # Q matrix
    Q = directed_chain[trans_states, :][:, trans_states]

    print(Q, Q.shape, Q.sum())
    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q
    N = inv(mat)

    # Absorption probabilities
    branch_probs = np.dot(N, directed_chain[trans_states, :][:, absorbing_states])
    branch_probs[branch_probs < 0] = 0

    # Add back terminal states
    branch_probs_including_terminal = np.full((directed_chain.shape[0], absorbing_states.sum()), 0.0)
    branch_probs_including_terminal[trans_states] = branch_probs
    branch_probs_including_terminal[absorbing_states] = np.eye(absorbing_states.sum())

    #self.branch_probs = np.dot(self.waypoint_weights.T, branch_probs_including_terminal)

    return branch_probs_including_terminal


## __ TREE INFERENCE FUNCTIONS __ ##

def adaptive_threshold(pseudotime, start_prob, end_cell, stretch = 10, shift = 0.75):

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    strictness = np.sqrt((1 - min(start_prob, 0.5) - 0.2) / start_prob)
    adaptive_scale = strictness*sigmoid(stretch*(pseudotime/pseudotime[end_cell] - shift)) - 0.05
        
    return start_prob*adaptive_scale + start_prob


def get_lineages(self, stretch = 100., shift=0.99,*, branch_probs, start_cell, pseudotime, terminal_states):
        
    assert(isinstance(stretch, float) and stretch > 0)
    assert(isinstance(shift, (float, int)) and shift > 0 and shift < 1)

    
    lineages = []
    for lineage_num, lineage_probs in enumerate(branch_probs.T):

        threshold = adaptive_threshold(pseudotime, lineage_probs[start_cell], 
                terminal_states[lineage_num], stretch = stretch, shift = shift)
        
        lineage_mask = (lineage_probs >= threshold)[:, np.newaxis]
        lineage_mask[terminal_states[lineage_num]] = True

        lineages.append(lineage_mask)

    lineages = np.hstack(lineages)

    return lineages


def get_lineage_branch_time(lineage1, lineage2, pseudotime, earliness_shift = 0.0):

    assert(-1 < earliness_shift < 1)

    part_of_lineage1 = lineage2[lineage1]
    time = pseudotime[lineage1]

    lr_model = LogisticRegression(class_weight = 'balanced')\
        .fit(time[:,np.newaxis], part_of_lineage1, (1-part_of_lineage1)+(1-earliness_shift))

    branch_time = -1*lr_model.intercept_/lr_model.coef_[0][0]

    return branch_time


def greedy_node_merge(self, earliness_shift = 0.0,*, lineages, pseudotime):

    num_cells, num_lineages = lineages.shape
    all_merged = False

    lineage_tree = np.full((num_lineages, num_lineages), 0)

    while not all_merged:
        
        split_time_matrix = np.full((num_lineages, num_lineages), -1.0)
        for i in range(0,num_lineages-1):
            for j in range(i+1, num_lineages):

                l1 = lineages[:, i].copy()
                l2 = lineages[:, j].copy()
                
                branch_time = get_lineage_branch_time(l1, l2, pseudotime, earliness_shift = earliness_shift) +\
                    get_lineage_branch_time(l2, l1, pseudotime, earliness_shift = earliness_shift)
                branch_time/=2

                split_time_matrix[i,j] = branch_time

        latest_split_event = np.where(split_time_matrix == split_time_matrix.max())
        merge1, merge2 = latest_split_event[0][0], latest_split_event[1][0]

        #new_branch_name = (lineage_names[merge1], lineage_names[merge2])
        #lineage_tree.add_split(new_branch_name, split_time_matrix.max())

        #lineage_names = [new_branch_name] + [lin for i, lin in enumerate(lineage_names) if not i in [merge1, merge2]]

        lineage_tree[merge1, merge2 ]
        lineages = np.hstack([
            np.logical_and(lineages[:, merge1], lineages[:, merge2])[:, np.newaxis], #merge two lineages into superlineage
            lineages[:, ~np.isin(np.arange(num_lineages), [merge1, merge2])].reshape((num_cells, -1))
        ])

        num_lineages = lineages.shape[-1]

        if num_lineages == 1:
            all_merged = True

    return lineage_tree


def get_cell_tree_states(self, earliness_shift = 0.0):

    lineage_tree = self._greedy_node_merge(earliness_shift)

    cell_states = np.zeros(self.lineages.shape[0]) - 1
    state_ratios = np.zeros(self.lineages.shape[0]) - 1

    states_assigned = 1
    states = { 0 : ("Root", lineage_tree.get_root())}
    min_split_time=np.inf

    for split, split_time in lineage_tree:

        min_split_time = min(split_time, min_split_time)
        for i in [0,1]:
            
            downstream_lineages = lineage_tree.get_all_leaves_from_node(split[i])
            downstream_cells = self.lineages[:, downstream_lineages].sum(-1)
            
            lineage_probability_thresholds = np.hstack([
                self.adaptive_threshold(self.pseudotime, self.branch_probs[self.start_cell, lin], self.terminal_states[lin],
                    stretch=self.stretch, shift=self.shift)[:, np.newaxis]
                for lin in downstream_lineages
            ])

            prob_lineage_ratio = (self.branch_probs[:, downstream_lineages]/lineage_probability_thresholds).max(-1)

            candidate_cells = np.logical_and(downstream_cells , self.pseudotime >= split_time)
            assign_cells = np.logical_and(candidate_cells,  prob_lineage_ratio > state_ratios)

            if assign_cells.sum() > 0:
                states_assigned+=1
                cell_states[assign_cells] = states_assigned
                state_ratios[assign_cells] = prob_lineage_ratio[assign_cells]
                states[states_assigned] = (split, split[i])

    cell_states[np.logical_and(cell_states == -1, self.pseudotime < min_split_time)] = 0

    self.cell_states, self.state_nodes, self.lineage_tree = cell_states.astype(np.int32), states, lineage_tree
    return self.cell_states