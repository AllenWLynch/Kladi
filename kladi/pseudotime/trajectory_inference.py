

import numpy as np
from kladi.core.diffusion import calc_markov_diffusion_matrix, get_matrix_params, build_affinity_matrix, make_markov_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse import csgraph
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.stats import entropy, pearsonr, norm
from numpy.linalg import inv
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
from umap import UMAP
import logging

from kladi.pseudotime.plot_utils import Beeswarm
from kladi.pseudotime.lineage_tree import LineageTree
from matplotlib import cm
from matplotlib.colors import Normalize

logging.basicConfig(level = logging.INFO)
logger = logging.Logger('kladi.palantir')
logger.setLevel(logging.INFO)

class PalantirTrajectoryInference:

    @classmethod
    def run(cls, cell_features, early_cell, terminal_waypoints = None, n_neighbors = 30, distance_metric = 'euclidean', n_waypoints = 1200, 
        use_early_cell_as_start = False, n_jobs = -1, kth_neighbor_std = None, num_components = None, 
        scale_components = True, diffusion_time = 4):

        palantir = cls(cell_features, n_neighbors = n_neighbors, n_jobs = n_jobs, kth_neighbor_std = kth_neighbor_std)

        palantir.fit_diffusion_map(diffusion_time, num_components, distance_metric, scale_components)

        palantir.get_pseudotime(early_cell, use_early_cell_as_start = use_early_cell_as_start, n_waypoints = n_waypoints, 
            terminal_states = terminal_states)

        palantir.get_branch_probs()

        palantir.get_lineages()

        return palantir
        

    def __init__(self, cell_features, n_neighbors = 30, n_jobs = -1, kth_neighbor_std = None):
        assert(isinstance(cell_features, np.ndarray))
        assert(isinstance(n_jobs, int) and n_jobs >= -1 and n_jobs != 0)
        assert(isinstance(n_neighbors, int) and n_neighbors > 0)
        assert(isinstance(kth_neighbor_std, int) or kth_neighbor_std is None)

        if kth_neighbor_std is None:
            kth_neighbor_std = max(n_neighbors//5, 3)

        self.cell_features = cell_features
        self.n_jobs = n_jobs
        self.kth_neighbor_std = kth_neighbor_std
        self.neighborhood_size = max(n_neighbors//self.kth_neighbor_std, 3)
        self.n_neighbors = n_neighbors

    @staticmethod
    def _get_diffusion_eigenvectors(markov_matrix, num_eigs):

        W, V = eigs(markov_matrix, num_eigs, tol = 1e-4, maxiter = 1000)

        W,V = np.real(W), np.real(V)

        eigvalue_order = np.argsort(W)[::-1]
        W = W[eigvalue_order]
        V = V[:, eigvalue_order]

        V = V/np.linalg.norm(V, axis = 0)[np.newaxis, :]

        return W,V

    @staticmethod
    def _determine_eigspace(W, V, n_eigs = None):

        if n_eigs is None:
            n_eigs = np.argsort(W[: (len(W) - 1)] - W[1:])[-1] + 1
            if n_eigs < 3:
                n_eigs = np.argsort(W[: (len(W) - 1)] - W[1:])[-2] + 1

        # Scale the data
        eig_vals = W[1:n_eigs]
        scaled_vectors = V[:, 1:n_eigs] * (eig_vals / (1 - eig_vals))[np.newaxis, :]
        
        return scaled_vectors

    def fit_diffusion_map(self, diffusion_time = 3, num_components = None, distance_metric = 'euclidean', scale_components = True):

        assert(isinstance(diffusion_time, int) and diffusion_time > 0)
        assert(num_components is None or (isinstance(num_components, int) and num_components > 0))
        if not num_components is None and num_components <=2:
            logging.warn('We recommend using >2 diffusion components to capture more complex trends in data.')
    
        self.markov_matrix = calc_markov_diffusion_matrix(self.cell_features, diffusion_time = diffusion_time, metric = distance_metric, ka = self.kth_neighbor_std,
                                leave_self_out = False, neighborhood_size = self.neighborhood_size, n_jobs=self.n_jobs)

        self.eigvalues, self.eigvectors = self._get_diffusion_eigenvectors(self.markov_matrix, 10)

        self.eigspace = self._determine_eigspace(self.eigvalues, self.eigvectors, n_eigs=num_components)
        
        if scale_components:
            self.eigspace = minmax_scale(self.eigspace)

        return self.eigspace

    def _get_waypoints(self, early_cell, n_waypoints = 1200, 
        terminal_states = None, use_early_cell_as_start = False):

        assert(isinstance(early_cell, int) and early_cell >= 0 and early_cell < len(self.cell_features))
        assert(isinstance(n_waypoints, int) and n_waypoints > 0)
        n_waypoints = min(n_waypoints, len(self.cell_features))

        assert(isinstance(terminal_states, (list, np.ndarray)) or terminal_states is None)

        if not terminal_states is None:
            if isinstance(terminal_states, list):
                terminal_states = np.array(terminal_states)
            terminal_states = np.ravel(terminal_states)

        assert(isinstance(use_early_cell_as_start, bool))

        boundary_cells = np.union1d(np.argmax(self.cell_features, 0), np.argmin(self.cell_features, 0))

        if use_early_cell_as_start:
            start_cell = early_cell
        
        else:
            dists = pairwise_distances(
                self.cell_features[boundary_cells, :], self.cell_features[early_cell, :][np.newaxis, :]
            )
            start_cell = np.argmin(dists.reshape(-1))

        waypoints = self._max_min_sampling(n_waypoints)

        waypoints = np.unique(np.concatenate([waypoints, [start_cell], terminal_states if not terminal_states is None else []])).astype(np.int64)

        return waypoints, start_cell

    def _max_min_sampling(self, num_waypoints):

        waypoint_set = list()
        no_iterations = int((num_waypoints) / self.cell_features.shape[1])

        # Sample along each component
        N = self.cell_features.shape[0]
        for feature_col in self.cell_features.T:

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

    def get_pseudotime(self, early_cell, max_iterations=25, use_early_cell_as_start = False,
        terminal_states = None, terminal_state_names = None, n_waypoints = 1200):

        assert(isinstance(max_iterations, int) and max_iterations > 0)
       
        try:
            self.eigspace
        except AttributeError:
            raise Exception('Diffusion space not yet calculated! Run "fit_diffusion_map" function first!')

        self.waypoints, self.start_cell = self._get_waypoints(early_cell, n_waypoints = n_waypoints, 
            terminal_states = terminal_states, use_early_cell_as_start=use_early_cell_as_start)

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean", n_jobs=self.n_jobs).fit(
            self.eigspace
        )
        distance_matrix = nbrs.kneighbors_graph(self.eigspace, mode="distance")

        ####


        
        # CHECK FOR GRAPH CONNECTION!


        ###

        # Distances
        dists = Parallel(n_jobs=self.n_jobs, max_nbytes=None)(
            delayed(csgraph.dijkstra)(distance_matrix, False, cell)
            for cell in self.waypoints
        )

        # Convert to distance matrix
        D = np.vstack([d[np.newaxis, :] for d in dists])

        start_waypoint_idx = np.argwhere(self.waypoints == self.start_cell)[0]

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
            for i, waypoint_idx in enumerate(self.waypoints):
                # Position of waypoints relative to start
                if waypoint_idx != self.start_cell:
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

        self.pseudotime = pseudotime
        self.waypoint_weights = W

        if not terminal_states is None:
            self.terminal_states = np.ravel(np.array(terminal_states))
            
            if not terminal_state_names is None:
                self.lineage_names = np.ravel(np.array(terminal_state_names))
                assert(len(self.terminal_states) == len(self.lineage_names))
            else:
                self.lineage_names = list(range(len(self.terminal_states)))

        return pseudotime

    def _construct_directed_chain(self):

        # Markov chain construction
        try:
            self.pseudotime
        except AttributeError:
            raise Exception('Must compute pseudotime before running this function!')

        waypoint_matrix = self.eigspace[self.waypoints, :]

        # kNN graph
        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric="euclidean", n_jobs=self.n_jobs
        ).fit(waypoint_matrix)
        kNN = nbrs.kneighbors_graph(waypoint_matrix, mode="distance")
        dist, ind = nbrs.kneighbors(waypoint_matrix)

        waypoint_pseudotime = self.pseudotime[self.waypoints]
        # Standard deviation allowing for "back" edges
        adpative_k = np.min([int(np.floor(self.n_neighbors / 3)) - 1, 30])
        adaptive_std = np.ravel(dist[:, adpative_k])

        # Remove edges that move backwards in pseudotime except for edges that are within
        # the computed standard deviation
        rem_edges = np.argwhere((waypoint_pseudotime[:, np.newaxis] - waypoint_pseudotime[ind]) > adaptive_std[:, np.newaxis])
        x = rem_edges[:,0]
        y = rem_edges[:,1]

        # Update adjacecy matrix    
        kNN[x, ind[x,y]] = 0

        # Affinity matrix and markov chain
        i,j,d = sparse.find(kNN)
        affinity = np.exp(
            -(d ** 2) / (adaptive_std[i] ** 2) * 0.5
            - (d ** 2) / (adaptive_std[j] ** 2) * 0.5
        )
        affinity_matrix = sparse.csr_matrix((affinity, (i, j)), [len(self.waypoints), len(self.waypoints)])

        return make_markov_matrix(affinity_matrix)

    def _terminal_states_from_markov_chain(self, directed_chain):
        raise NotImplementedError()

    def get_branch_probs(self):

        directed_chain = np.array(self._construct_directed_chain().todense())

        try:
            self.terminal_states
        except AttributeError:
            logging.warn('User did not specify terminal states, will try to infer some from markov chain.')
            self.terminal_states = _terminal_states_from_markov_chain(directed_chain)

        absorbing_states = np.isin(self.waypoints, self.terminal_states)

        # Reset absorption state affinities by Removing neigbors
        directed_chain[absorbing_states, :] = 0
        # Diagnoals as 1s
        directed_chain[absorbing_states, absorbing_states] = 1

        # Fundamental matrix and absorption probabilities
        # Transition states
        trans_states = ~absorbing_states

        # Q matrix
        Q = directed_chain[trans_states, :][:, trans_states]
        # Fundamental matrix
        mat = np.eye(Q.shape[0]) - Q
        N = inv(mat)

        # Absorption probabilities
        branch_probs = np.dot(N, directed_chain[trans_states, :][:, absorbing_states])
        branch_probs[branch_probs < 0] = 0

        # Add back terminal states
        branch_probs_including_terminal = np.full((len(self.waypoints), absorbing_states.sum()), 0.0)
        branch_probs_including_terminal[trans_states] = branch_probs
        branch_probs_including_terminal[absorbing_states] = np.eye(absorbing_states.sum())

        self.branch_probs = np.dot(self.waypoint_weights.T, branch_probs_including_terminal)

        return self.branch_probs

    
    @staticmethod
    def adaptive_threshold(pseudotime, start_prob, end_cell, stretch = 10, shift = 0.75):

        def sigmoid(x):
            return 1/(1+np.exp(-x))

        strictness = np.sqrt((1 - start_prob - 0.2) / start_prob)
        adaptive_scale = strictness*sigmoid(stretch*(pseudotime/pseudotime[end_cell] - shift)) - 0.05
        
        return start_prob*adaptive_scale + start_prob

    def get_lineages(self, stretch = 10):

        try:
            self.branch_probs
        except AttributeError:
            self.branch_probs = self.get_branch_probs()
        
        lineages = []
        for lineage_num, lineage_probs in enumerate(self.branch_probs.T):

            threshold = self.adaptive_threshold(self.pseudotime, lineage_probs[self.start_cell], 
                    self.terminal_states[lineage_num], stretch = stretch)

            lineages.append((lineage_probs >= threshold)[:, np.newaxis])

        self.lineages = np.hstack(lineages)

        return self.lineages


    @staticmethod
    def _get_lineage_branch_time(lineage1, lineage2, pseudotime, earliness_shift = 0.33):

        assert(0 < earliness_shift <= 1)

        part_of_lineage1 = lineage2[lineage1]
        time = pseudotime[lineage1]

        lr_model = LogisticRegression(class_weight = 'balanced').fit(time[:,np.newaxis], part_of_lineage1, (1-part_of_lineage1)+(1-earliness_shift))

        branch_time = -1*lr_model.intercept_/lr_model.coef_[0][0]

        return branch_time


    def _greedy_node_merge(self, earliness_shift = 0.33):

        try:
            self.lineages
        except AttributeError:
            raise Exception('User must run "get_lineages" before making tree structure!')

        lineages = np.copy(self.lineages)

        num_cells, num_lineages = lineages.shape
        all_merged = False
        
        lineage_tree = LineageTree(self.lineage_names)
        lineage_names = list(range(num_lineages))

        while not all_merged:
            
            split_time_matrix = np.full((num_lineages, num_lineages), -1.0)
            for i in range(0,num_lineages-1):
                for j in range(i+1, num_lineages):
                    branch_time = min(self._get_lineage_branch_time(lineages[:, i], lineages[:,j], 
                        self.pseudotime, earliness_shift = earliness_shift),
                        self._get_lineage_branch_time(lineages[:, j], lineages[:,i], 
                            self.pseudotime, earliness_shift = earliness_shift))

                    split_time_matrix[i,j] = branch_time

            latest_split_event = np.where(split_time_matrix == split_time_matrix.max())
            merge1, merge2 = latest_split_event[0][0], latest_split_event[1][0]

            new_branch_name = (lineage_names[merge1], lineage_names[merge2])
            lineage_tree.add_split(new_branch_name, split_time_matrix.max())

            lineage_names = [new_branch_name] + [lin for i, lin in enumerate(lineage_names) if not i in [merge1, merge2]]

            lineages = np.hstack([
                np.logical_and(lineages[:, merge1], lineages[:, merge2])[:, np.newaxis], #merge two lineages into superlineage
                lineages[:, ~np.isin(np.arange(num_lineages), [merge1, merge2])].reshape((num_cells, -1))
            ])

            num_lineages = lineages.shape[-1]

            if num_lineages == 1:
                all_merged = True

        return lineage_tree


    def get_cell_tree_states(self, earliness_shift = 0.33):

        lineage_tree = self._greedy_node_merge(earliness_shift)

        cell_states = np.zeros(self.lineages.shape[0])
        state_ratios = np.zeros(self.lineages.shape[0]) - 1

        states_assigned = 0
        states = { 0 : ("Root", lineage_tree.get_root())}
        for split, split_time in lineage_tree:

            for i in [0,1]:
                
                downstream_lineages = lineage_tree.get_all_leaves_from_node(split[i])
                downstream_cells = self.lineages[:, downstream_lineages].sum(-1)
                
                lineage_probability_thresholds = np.hstack([
                    self.adaptive_threshold(self.pseudotime, self.branch_probs[self.start_cell, lin], self.terminal_states[lin])[:, np.newaxis]
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

        return cell_states, states

    def get_graphviz_tree(self, earliness_shift = 0.33):
        
        lineage_tree = self._greedy_node_merge(earliness_shift)
        
        return lineage_tree.get_graphviz_tree()

    
    def get_visual_representation(self, n_dims = 2, continuity = 1, n_neighbors = 15, min_dist = 0.1, **umap_kwargs):
        if continuity > 100:
            logger.warn('Setting continuity too low will cause UMAP to produce strange results. We suggest somewhere between 0.2 and 20 for best results.')

        if 'negative_sample_rate' in umap_kwargs:
            logger.warn('"negative_sample_rate" param has been replaced by 1/continuity for clarity. Adjust continuity parameter instead.')
            del umap_kwargs['negative_sample_rate']

        self.umap_model = UMAP(n_neighbors = n_neighbors, n_components = n_dims, min_dist = min_dist, negative_sample_rate = 1/continuity, **umap_kwargs)
        self.representation = self.umap_model.fit_transform(self.cell_features)

        return self.representation

    def plot_lineages(self, data_representation = None, ax = None):
        pass

    def plot_trajectory_velocity(self, data_representation = None, ax = None):
        pass

    def plot_swarm_tree(self, cell_colors = None, label = None, palette = None, show_cbar = True,
        size = 1, vmin = None, vmax = None, log_pseudotime = True, figsize = (20,10)):

        fig, ax = plt.subplots(1,1,figsize=figsize)

        cell_states, state_nodes = self.get_cell_tree_states()

        tree_layout = self._greedy_node_merge().get_tree_layout()

        if log_pseudotime:
            pseudotime = np.log2(self.pseudotime+1)
        else:
            pseudotime = self.pseudotime

        if cell_colors is None:
            cell_colors = cm.get_cmap('Set3')(cell_states.astype(int))
        else:
            assert(isinstance(cell_colors, np.ndarray))

            if np.issubdtype(cell_colors.dtype, np.number):

                vmin = cell_colors.min()
                vmax = cell_colors.max()

                #sort so higher values are on top
                color_order = np.argsort(cell_colors)
                cell_states, cell_colors, pseudotime = cell_states[color_order], cell_colors[color_order], pseudotime[color_order]

                if show_cbar:
                    fig.colorbar(cm.ScalarMappable(Normalize(vmin, vmax), cmap=palette), ax=ax, location = 'left', pad = 0.01, shrink = 0.25)

        if size is None:
            size = np.ones_like(cell_states)
        elif isinstance(size, int):
            size = np.ones_like(cell_states) * size
        
        for state_idx, (start_node, end_node) in state_nodes.items():

            centerline = tree_layout[end_node][0]
            cell_mask = cell_states==state_idx

            x = pseudotime[cell_mask]

            points = ax.scatter(x, np.ones_like(x) * centerline, s = size[cell_mask], cmap = palette, vmin = vmin, vmax = vmax, c = cell_colors[cell_mask])
            
            Beeswarm(orient='h')(points, centerline)
            
            if not start_node is "Root":
                #print(tree_layout[start_node][0], centerline)
                ax.plot([np.log2(tree_layout[start_node][1] + 1)]*2,
                        (tree_layout[start_node][0],centerline), color = 'black')
                
            plt.axis('off')

            if isinstance(end_node, int):
                ax.text(x.max()*1.05, centerline, str(self.lineage_names[end_node]), fontsize='large')

        plt.tight_layout()

        if not label is None:
            ax.set(title = str(label))

        return fig

    def plot_pseudotime(self, data_representation = None, ax = None):
        pass

    def plot_states(self, data_representation = None, ax = None):
        pass