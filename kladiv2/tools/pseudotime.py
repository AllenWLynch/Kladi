
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
from scipy.sparse.linalg import eigs
import logging
import tqdm

logger = logging.getLogger(__name__)

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


def get_pseudotime(max_iterations = 25, n_jobs = -1,*, start_cell, distance_matrix):

    assert(isinstance(max_iterations, int) and max_iterations > 0)
    N = distance_matrix.shape[0]
    cells = np.arange(N)

    ####


    
    # CHECK FOR GRAPH CONNECTION!


    ###

    # Distances
    dists = Parallel(n_jobs=n_jobs, max_nbytes=None)(
        delayed(csgraph.dijkstra)(distance_matrix, False, cell)
        for cell in cells
    )

    # Convert to distance matrix
    D = np.vstack([d[np.newaxis, :] for d in dists])

    start_waypoint_idx = np.argwhere(cells == start_cell)[0]

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
    for iteration in range(max_iterations):
        # Perspective matrix by alinging to start distances
        P = deepcopy(D)
        for i, waypoint_idx in enumerate(cells):
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

        if converged:
            break

    pseudotime = pseudotime - pseudotime.min() #make 0 minimum
    waypoint_weights = W
    
    return pseudotime


def make_markov_matrix(affinity_matrix):
    inverse_rowsums = sparse.diags(1/np.array(affinity_matrix.sum(axis = 1)).reshape(-1)).tocsr()
    markov_matrix = inverse_rowsums.dot(affinity_matrix)
    return markov_matrix


def get_adaptive_affinity_matrix(ka = 10,*, distance_matrix, pseudotime):

    N = distance_matrix.shape[0]
    distance_matrix = distance_matrix.tocsr()
    indptr = distance_matrix.indptr
    k = indptr[1] - indptr[0]
    assert(np.all(indptr[1:] - indptr[0:-1] == k)), 'distance matrix is not a valid Knn matrix. Different numbers of neighbors for each cell.'

    j, i, d = sparse.find(distance_matrix.T)
    distances = d.reshape(N, k)
    
    sorted_dist = np.sort(distances, axis = 1)
    #ka_index = np.minimum(np.argmin(~np.isinf(sorted_dist), axis = 1) - 1, ka)
    kernel_width = sorted_dist[:, ka]

    #finite_mean = kernel_width[np.isfinite(kernel_width)].mean()
    #kernel_width = np.where(np.isinf(kernel_width), finite_mean, kernel_width)
    kernel_width = kernel_width
    delta_pseudotime = np.abs(pseudotime[i] - pseudotime[j])
    
    affinity = np.exp(
        -(d ** 2) / (kernel_width[i] * delta_pseudotime) * 0.5
        -(d ** 2) / (kernel_width[j] * delta_pseudotime) * 0.5
    )
    affinity_matrix = sparse.csr_matrix((affinity, (i, j)), [N,N])

    return affinity_matrix, kernel_width#, ka_index


def prune_backwards_affinities(*, distance_matrix, affinity_matrix, kernel_width, pseudotime):

    distance_matrix = distance_matrix.tocsr()
    N = distance_matrix.shape[0]
    j, i, d = sparse.find(distance_matrix.T)

    delta_pseudotime = pseudotime[i] - pseudotime[j]

    rem_edges = delta_pseudotime > kernel_width[i]
    d[rem_edges] = 0

    affinity_matrix = sparse.coo_matrix((d, (i,j)), (N,N)).tocsr()
    affinity_matrix.eliminate_zeros()

    return affinity_matrix


def get_transport_map(ka = 10, n_jobs = -1,*, start_cell, distance_matrix):

    logger.info('Calculating diffusion pseudotime ...')
    pseudotime = get_pseudotime(n_jobs = n_jobs, start_cell= start_cell, 
        distance_matrix = distance_matrix)

    logger.info('Calculating transport map ...')
    affinity_matrix, kernel_width = get_adaptive_affinity_matrix(ka = ka, 
        distance_matrix = distance_matrix, pseudotime = pseudotime)

    affinity_matrix = prune_backwards_affinities(
        distance_matrix = distance_matrix, affinity_matrix = affinity_matrix,
        kernel_width = kernel_width, pseudotime = pseudotime
    )

    transport_map = make_markov_matrix(affinity_matrix)

    return pseudotime, transport_map


def get_terminal_states(iterations = 1, max_termini = 10, *, transport_map):

    assert(transport_map.shape[0] == transport_map.shape[1])
    assert(len(transport_map.shape) == 2)

    def _get_stationary_points():

        vals, vectors = eigs(transport_map.T, k = max_termini)

        stationary_vecs = np.isclose(np.real(vals), 1., 1e-3)

        return list(np.real(vectors)[:, stationary_vecs].argmax(0))

    terminal_points = set()
    for i in range(iterations):
        terminal_points = terminal_points.union(_get_stationary_points())

    logger.info('Found {} terminal states from stationary distribution.'.format(str(len(terminal_points))))

    return np.array(list(terminal_points))



def get_branch_probabilities(*, transport_map, **terminal_cells):

    lineage_names, absorbing_cells = list(zip(*terminal_cells.items()))
    absorbing_states_idx = np.array(absorbing_cells)
    num_absorbing_cells = len(absorbing_cells)
    
    absorbing_states = np.zeros(transport_map.shape[0]).astype(bool)
    absorbing_states[absorbing_states_idx] = True

    transport_map = transport_map.toarray()
    # Reset absorption state affinities by Removing neigbors
    transport_map[absorbing_states, :] = 0
    # Diagnoals as 1s
    transport_map[absorbing_states, absorbing_states] = 1

    # Fundamental matrix and absorption probabilities
    # Transition states
    trans_states = ~absorbing_states

    # Q matrix
    Q = transport_map[trans_states, :][:, trans_states]

    # Fundamental matrix
    mat = np.eye(Q.shape[0]) - Q
    N = inv(mat)

    # Absorption probabilities
    branch_probs = np.dot(N, transport_map[trans_states, :][:, absorbing_states_idx])
    branch_probs[branch_probs < 0] = 0.

    # Add back terminal states
    branch_probs_including_terminal = np.full((transport_map.shape[0], num_absorbing_cells), 0.)
    branch_probs_including_terminal[trans_states] = branch_probs
    branch_probs_including_terminal[absorbing_states_idx, np.arange(num_absorbing_cells)] = 1.
    #terminal_probs = np.full((num_absorbing_cells, num_absorbing_cells), 0.)
    #terminal_probs[np.arange(num_absorbing_cells), np.argsort(absorbing_states_idx)] = 1.
    #branch_probs_including_terminal[absorbing_states] = terminal_probs

    #self.branch_probs = np.dot(self.waypoint_weights.T, branch_probs_including_terminal)
    return branch_probs_including_terminal


## __ TREE INFERENCE FUNCTIONS __ ##

def get_lineages(*,branch_probs, start_cell):

    return get_lineage_prob_fc(branch_probs = branch_probs, start_cell = start_cell) >= 0


def get_lineage_prob_fc(*, branch_probs, start_cell):

    ep = 0.01
    lineage_prob_fc = np.hstack([
        (np.log2(lineage_prob + ep) - np.log2(lineage_prob[start_cell] + ep) )[:, np.newaxis]
        for lineage_prob in branch_probs.T
    ])

    return lineage_prob_fc

def get_lineage_branch_time(lineage1, lineage2, pseudotime, prob_fc, threshold = 0.5):

    lin_mask = np.logical_or(prob_fc[:, lineage1] > 0, prob_fc[:, lineage2] > 0)
    divergence = prob_fc[lin_mask, lineage1] - prob_fc[lin_mask, lineage2]

    state_1 = pseudotime[lin_mask][divergence > threshold]
    state_2 = pseudotime[lin_mask][divergence < -threshold]
    
    
    if len(state_1) == 0:
        return state_2.min()
    elif len(state_2) == 0:
        return state_1.min()
    else:
        return max(state_1.min(), state_2.min())


def get_tree_structure(threshold = 0.5,*, lineage_names, branch_probs, pseudotime, start_cell):
    
    def get_all_leaves_from_node(edge):

        if isinstance(edge, tuple):
            return [*get_all_leaves_from_node(edge[0]), *get_all_leaves_from_node(edge[1])]
        else:
            return [edge]

    def get_node_name(self, node):
        return ', '.join(map(str, self.lineage_names[self.get_all_leaves_from_node(node)]))

    def merge_rows(x, col1, col2):
        return np.hstack([
            (x[:,col1] + x[:, col2])[:, np.newaxis], #merge two lineages into superlineage
            x[:, ~np.isin(np.arange(x.shape[-1]), [col1, col2])]
        ])


    num_cells, num_lineages = branch_probs.shape
    all_merged = False

    tree_states = np.zeros(num_cells)

    lineages = get_lineages(branch_probs = branch_probs, start_cell = start_cell)
    branch_probs = branch_probs.copy()
    lineage_names = list(lineage_names)

    lineage_tree = nx.DiGraph()
    states_assigned = 1

    while not all_merged:

        prob_fc = get_lineage_prob_fc(branch_probs = branch_probs, start_cell = start_cell)
        
        split_time_matrix = np.full((num_lineages, num_lineages), -1.0)
        for i in range(0,num_lineages-1):
            for j in range(i+1, num_lineages):

                branch_time = get_lineage_branch_time(i, j, pseudotime, prob_fc, threshold)
                split_time_matrix[i,j] = branch_time
        
        branch_time = split_time_matrix.max()
        latest_split_event = np.where(split_time_matrix == branch_time)
        merge1, merge2 = latest_split_event[0][0], latest_split_event[1][0]

        new_branch_name = (lineage_names[merge1], lineage_names[merge2])

        assign_cells_mask = np.logical_and(pseudotime >= branch_time, np.logical_or(lineages[:,merge1], lineages[:, merge2]))
        assign_cells_mask = np.logical_and(assign_cells_mask, ~tree_states.astype(bool))
        
        divergence = prob_fc[assign_cells_mask, merge1] - prob_fc[assign_cells_mask, merge2]
        get_assign_indices = lambda y : np.argwhere(assign_cells_mask)[:,0][y * divergence > 0]

        tree_states[get_assign_indices(1)] = states_assigned
        lineage_tree.add_edge(new_branch_name, lineage_names[merge1], branch_time = branch_time, state = states_assigned)
        states_assigned+=1

        tree_states[get_assign_indices(-1)] = states_assigned
        lineage_tree.add_edge(new_branch_name, lineage_names[merge2], branch_time = branch_time, state = states_assigned)
        states_assigned+=1

        lineages = merge_rows(lineages, merge1, merge2).astype(bool)
        branch_probs = merge_rows(branch_probs, merge1, merge2)
        lineage_names = [new_branch_name] + [lin for i, lin in enumerate(lineage_names) if not i in [merge1, merge2]]

        num_lineages = lineages.shape[-1]
        
        if num_lineages == 1:
            all_merged = True
            
    state_names = {
        edge[2]['state'] : ', '.join(set(get_all_leaves_from_node(edge[1])))
        for edge in lineage_tree.edges(data = True)
    }
    
    state_names[0] = 'Root'
    
    return {
        'tree_states' : [state_names[s] for s in tree_states.astype(int)],
        'tree' : nx.to_numpy_array(lineage_tree, weight='branch_time'),
        'state_names' : list(state_names.values()),
    }