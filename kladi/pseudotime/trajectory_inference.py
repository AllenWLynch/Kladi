

import numpy as np
from kladi.core.diffusion import calc_markov_diffusion_matrix, get_matrix_params, build_affinity_matrix, make_markov_matrix
from scipy.sparse.linalg import eigs
from scipy.sparse import csgraph
from scipy import sparse
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, scale
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from scipy.stats import entropy, pearsonr, norm
from numpy.linalg import inv
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pltcolors
from umap import UMAP
from matplotlib.patches import Patch
import logging
import warnings

from kladi.core.plot_utils import Beeswarm, map_colors, plot_umap
from kladi.pseudotime.lineage_tree import LineageTree
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from math import ceil

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
        terminal_states = None, use_early_cell_as_start = False, seed = 2556):

        np.random.seed(seed=seed)

        assert(isinstance(early_cell, int) and early_cell >= 0 and early_cell < len(self.cell_features))
        assert(isinstance(n_waypoints, int) and n_waypoints > 0)
        n_waypoints = min(n_waypoints, len(self.cell_features))

        assert(isinstance(terminal_states, (list, np.ndarray)))
        terminal_states = np.ravel(np.array(terminal_states))
        assert(np.issubdtype(terminal_states.dtype, np.number))

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

    def get_pseudotime(self, early_cell, max_iterations=25, use_early_cell_as_start = True, 
        n_waypoints = 300, **terminal_states):

        assert(isinstance(max_iterations, int) and max_iterations > 0)
        try:
            self.eigspace
        except AttributeError:
            raise Exception('Diffusion space not yet calculated! Run "fit_diffusion_map" function first!')
        n_waypoints= min(len(self.cell_features), n_waypoints)
        terminal_state_names, terminal_states = list(terminal_states.keys()), list(terminal_states.values())

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

        self.pseudotime = minmax_scale(pseudotime) #make 0 minimum
        self.waypoint_weights = W

        if len(terminal_states) > 0:
            
            terminal_states = np.ravel(np.array(terminal_states))
            terminal_state_order = np.argsort(terminal_states)

            self.terminal_states = terminal_states[terminal_state_order]
            
            self.lineage_names = np.ravel(np.array(terminal_state_names))[terminal_state_order]
            assert(len(self.terminal_states) == len(self.lineage_names))
        
        else:
            raise NotImplementedError()

        return pseudotime

    def get_directed_knn_graph(self):

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
        kNN[x, ind[x,y]] = np.inf

        return kNN, adaptive_std

    def _construct_directed_chain(self):

        # Markov chain construction
        kNN, adaptive_std = self.get_directed_knn_graph()

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

        strictness = np.sqrt((1 - min(start_prob, 0.5) - 0.2) / start_prob)
        adaptive_scale = strictness*sigmoid(stretch*(pseudotime/pseudotime[end_cell] - shift)) - 0.05
        
        return start_prob*adaptive_scale + start_prob

    def get_lineages(self, stretch = 100., shift=0.99):
        
        assert(isinstance(stretch, float) and stretch > 0)
        assert(isinstance(shift, (float, int)) and shift > 0 and shift < 1)

        self.stretch = stretch
        self.shift = shift

        try:
            self.branch_probs
        except AttributeError:
            self.branch_probs = self.get_branch_probs()
        
        lineages = []
        for lineage_num, lineage_probs in enumerate(self.branch_probs.T):

            threshold = self.adaptive_threshold(self.pseudotime, lineage_probs[self.start_cell], 
                    self.terminal_states[lineage_num], stretch = stretch, shift = shift)
            
            lineage_mask = (lineage_probs >= threshold)[:, np.newaxis]
            lineage_mask[self.terminal_states[lineage_num]] = True

            lineages.append(lineage_mask)

        self.lineages = np.hstack(lineages)

        return self.lineages


    @staticmethod
    def _get_lineage_branch_time(lineage1, lineage2, pseudotime, earliness_shift = 0.0):

        assert(-1 < earliness_shift < 1)

        part_of_lineage1 = lineage2[lineage1]
        time = pseudotime[lineage1]

        lr_model = LogisticRegression(class_weight = 'balanced')\
            .fit(time[:,np.newaxis], part_of_lineage1, (1-part_of_lineage1)+(1-earliness_shift))

        branch_time = -1*lr_model.intercept_/lr_model.coef_[0][0]

        return branch_time


    def _greedy_node_merge(self, earliness_shift = 0.0):

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

                    l1 = lineages[:, i].copy()
                    l2 = lineages[:, j].copy()
                    
                    branch_time = max(self._get_lineage_branch_time(l1, l2, 
                        self.pseudotime, earliness_shift = earliness_shift),
                        self._get_lineage_branch_time(l2, l1, 
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

    def get_graphviz_tree(self):
        
        return self.lineage_tree.get_graphviz_tree()

    
    def get_visual_representation(self, n_dims = 2, continuity = 1, n_neighbors = 15, min_dist = 0.1, 
        metric = 'euclidean', **umap_kwargs):
        if continuity > 100:
            logger.warn('Setting continuity too low will cause UMAP to produce strange results. We suggest somewhere between 0.2 and 20 for best results.')

        if 'negative_sample_rate' in umap_kwargs:
            logger.warn('"negative_sample_rate" param has been replaced by 1/continuity for clarity. Adjust continuity parameter instead.')
            del umap_kwargs['negative_sample_rate']

        self.umap_model = UMAP(n_neighbors = n_neighbors, n_components = n_dims, min_dist = min_dist, 
            negative_sample_rate = 1/continuity, metric = metric, **umap_kwargs)
        self.representation = self.umap_model.fit_transform(self.cell_features)

        return self.representation

    def get_node_names(self):
        try:
            self.cell_states
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before plotting states')

        legend_labels = {}
        for state_idx, (start_node, end_node) in self.state_nodes.items():
            legend_labels[state_idx] = self.lineage_tree.get_node_name(end_node) 

        return legend_labels
        
    def get_path_to(self, lineage_name):

        try:
            self.state_nodes, self.cell_states
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before plotting states')

        path = self.lineage_tree.get_path_to(lineage_name)

        inv_state_nodes = {v: k for k, v in self.state_nodes.items()}

        cell_state_order = np.hstack([
            (self.cell_states == inv_state_nodes[node])[:,np.newaxis]
            for node in path
        ])

        return cell_state_order


    def check_representation(self, data_representation):

        if data_representation is None:
            try:
                data_representation = self.representation
            except AttributeError:
                raise Exception('User must use "get_visual_representation" to get 2D representation of data for plotting. Adjust the "continuity" parameter for smoother trajectories. Or, you can pass your own (num cells x 2) representation.')

        else:
            assert(isinstance(data_representation, np.ndarray))
            assert(len(data_representation.shape) == 2)
            assert(data_representation.shape[-1] >= 2)

        return data_representation

    def plot_lineages(self, data_representation = None, lineages_per_row = 4, aspect = 2, 
        height = 2, return_fig = False, size = 2, palette = 'viridis'):
        try:
            self.lineages
        except AttributeError:
            raise Exception('User must use "get_lineages" to calculate lineages before plotting them!')

        data_representation = self.check_representation(data_representation)
        
        assert(isinstance(size, int))

        num_lineages = self.lineages.shape[-1]
        num_rows = ceil(num_lineages/lineages_per_row)

        lineages_per_row = min(lineages_per_row, num_lineages)

        fig, ax = plt.subplots(num_rows, lineages_per_row, figsize = (height*aspect*lineages_per_row, height*num_rows))
        if num_rows==1:
            ax = ax[np.newaxis, :]

        for i, ax_i in enumerate(ax.ravel()):

            if i >= self.lineages.shape[-1]:
                ax_i.axis('off')

            else:
                lineage = self.lineages[:,i]
                name = self.lineage_names[i]
                
                ax_i.scatter(x = data_representation[lineage,0], y = data_representation[lineage,1], 
                    c = self.branch_probs[lineage, i], vmin = 0, vmax = 1, cmap = palette, s = size)

                ax_i.scatter(x = data_representation[~lineage,0], y = data_representation[~lineage,1], 
                    c = 'lightgrey', s = size)

                ax_i.set_title(str(self.lineage_names[i]), fontdict= dict(fontsize = 'x-large'))
                ax_i.axis('off')

        plt.tight_layout()

        cbar = plt.colorbar(cm.ScalarMappable(Normalize(0, 1), cmap=palette), orientation = 'vertical', ax = ax[0,-1], 
            panchor = (1.05, 0.5), aspect = 15, shrink = 0.5, label = '')

        cbar.ax.tick_params(labelsize='x-large')
        cbar.set_label('P( Terminal State )',size='x-large')

        if return_fig:
            return fig, ax


    def plot_swarm_tree(self, cell_colors = None, title = None, palette = 'viridis', show_legend = True,
        size = 4, log_pseudotime = True, figsize = (20,10), max_swarm_size = 1500, ax = None, hue_order = None):

        try:
            cell_states = self.cell_states
            state_nodes = self.state_nodes
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before plotting states')

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

        tree_layout = self.lineage_tree.get_tree_layout()

        if log_pseudotime:
            pseudotime = np.log2(self.pseudotime+1)
        else:
            pseudotime = self.pseudotime

        if cell_colors is None:
            cell_colors = cm.get_cmap('Set3')(cell_states.astype(int))
        else:
            
            assert(isinstance(cell_colors, np.ndarray))
            cell_colors = np.ravel(cell_colors)
            assert(len(cell_colors) == len(pseudotime))

            cell_colors = map_colors(ax, cell_colors, palette, 
                add_legend = show_legend, hue_order = hue_order, 
                cbar_kwargs = dict(location = 'left', pad = 0.01, shrink = 0.25, aspect = 15, anchor = (0, 0.5)),
                legend_kwargs = dict(loc="upper left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                            bbox_to_anchor=(1.05, 0.5)))

        if size is None:
            size = np.ones_like(cell_states)
        elif isinstance(size, int):
            size = np.ones_like(cell_states) * size
        
        highest_centerline = -1

        for state_idx, (start_node, end_node) in list(state_nodes.items())[::-1]:

            segment_time_start = tree_layout[start_node][1] if not start_node is "Root" else 0
            if log_pseudotime:
                segment_time_start = np.log2(segment_time_start + 1)

            centerline = tree_layout[end_node][0]
            highest_centerline = max(centerline, highest_centerline)

            cell_mask = cell_states==state_idx

            if cell_mask.sum() > max_swarm_size:
                cell_mask = np.random.choice(np.argwhere(cell_mask)[:,0], max_swarm_size, replace = False)

            x = pseudotime[cell_mask]
            points = ax.scatter(x, np.ones_like(x) * centerline, s = size[cell_mask], c = cell_colors[cell_mask])
            
            Beeswarm(orient='h', width = 0.9)(points, centerline)
            
            if not start_node is "Root":
                ax.plot([segment_time_start]*2,
                        (tree_layout[start_node][0],centerline), color = 'black')

            if isinstance(end_node, int):
                ax.text(x.max()*1.05, centerline, str(self.lineage_names[end_node]), fontsize='large')

        ax.axis('off')
        ax.set(ylim = (-0.5, highest_centerline + 0.75))

        if not title is None:
            ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

        return ax

    def plot_pseudotime(self, data_representation = None, figsize = (10,7), palette = 'viridis', size = 2, ax = None):
        try:
            self.pseudotime
        except AttributeError:
            raise Exception('User must run "get_pseudotime" before this function.')

        data_representation = self.check_representation(data_representation)

        if ax is None:
            fig,ax = plt.subplots(1,1,figsize=figsize)

        ax.scatter(data_representation[:,0], data_representation[:,1], c = self.pseudotime, s = size, cmap = palette)
        fig.colorbar(cm.ScalarMappable(Normalize(self.pseudotime.min(), self.pseudotime.max()), cmap=palette), ax=ax, 
            location = 'left', pad = 0.01, shrink = 0.25, aspect = 15, label = 'Pseudotime')

        ax.scatter(data_representation[self.terminal_states,0], data_representation[self.terminal_states,1], c = 'black', 
            s = 25 * size, label = 'Terminal State')
        ax.scatter(data_representation[self.start_cell,0],
            data_representation[self.start_cell,1], label = 'Root', c = 'red', s = 25 * size)

        legend_kwargs = dict(loc="upper left", markerscale = 1, frameon = False, fontsize='large', bbox_to_anchor=(-0.1, 1.05))

        ax.legend(**legend_kwargs)
        ax.axis('off')

        return ax


    def plot_states(self, data_representation = None, figsize = (10,7), palette = 'Set3', size = 2, ax = None):
        
        try:
            cell_states = self.cell_states
            state_nodes = self.state_nodes
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before plotting states')

        data_representation = self.check_representation(data_representation)
        assert(isinstance(size, int))

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

        legend_labels = []
        for state_idx, (start_node, end_node) in state_nodes.items():
            legend_labels.append(str(state_idx) + ': ' + self.lineage_tree.get_node_name(end_node))

        sorted_by_multipotency = sorted(zip(legend_labels, state_nodes.items()), 
            key = lambda x : -len(x[0].split(',')))

        for i, (label, (state, node)) in enumerate(sorted_by_multipotency):
            ax.scatter(x = data_representation[cell_states == state,0], y = data_representation[cell_states == state,1], color = cm.get_cmap(palette)(i), 
                    s = size, label = label)

        ax.axis('off')
        ax.legend(loc="center left", title="Possible Terminal States", markerscale = 5, frameon = False, title_fontsize='x-large', fontsize='large',
                bbox_to_anchor=(1.05, 0.5))

        return ax

    def _iterate_all_bins(self, bin_size = 75):
        try:
            self.state_nodes
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before getting bins.')

        tree_layout = self.lineage_tree.get_tree_layout()

        for state_idx, (start_node, end_node) in list(self.state_nodes.items())[::-1]:
            
            centerline = tree_layout[end_node][0]

            for bin_mask, bin_time in self._get_trajectory_bins(state_idx, bin_size = bin_size):
                yield state_idx, (bin_time, centerline), bin_mask
                

    def _get_trajectory_bins(self, state_idx, bin_size = 75):

        try:
            self.state_nodes
            self.cell_states
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before plotting states')

        tree_layout = self.lineage_tree.get_tree_layout()

        start_node, end_node = self.state_nodes[state_idx]

        segment_time_start = tree_layout[start_node][1] if not start_node is "Root" else 0

        cell_mask = self.cell_states==state_idx

        x = self.pseudotime[cell_mask]
        #bin x on # of cells
        cell_order = np.argsort(x)
        x = x[cell_order]
        bin_num = np.arange(len(x))//bin_size

        for _bin in range(bin_num.max()+1):
            bin_mask = np.zeros_like(self.pseudotime).astype(np.bool)
            bin_mask[np.arange(len(bin_mask))[cell_mask][cell_order][bin_num == _bin]] = True
            
            if _bin == 0:
                bin_time = segment_time_start
            elif _bin == bin_num.max():
                bin_time = x.max()
            else:
                bin_time = x[_bin * bin_size]

            yield bin_mask, bin_time

    def plot_feature_stream(self, features, labels = None, color = 'black', log_pseudotime = True, figsize = (20,10),
        scale_features = False, center_baseline = True, bin_size = 25, palette = 'Set3', ax=None, title = None, show_legend = True,
        max_bar_height = 0.5, hide_feature_threshold = 0.03, linecolor = 'grey', linewidth = 0.1, annotate_streams = False, clip = 5,
        min_pseudotime = 0.05):

        assert(isinstance(max_bar_height, float) and max_bar_height > 0 and max_bar_height < 1)
        assert(isinstance(features, np.ndarray))
        assert(len(features) == len(self.cell_features))
        if len(features.shape) == 1:
            features = features[:,np.newaxis]
        assert(len(features.shape) == 2)
        assert(np.issubdtype(features.dtype, np.number))
        num_features = features.shape[-1]

        if labels is None:
            labels = list(range(num_features))
        else:
            assert(len(labels) == num_features)

        try:
            cell_states, state_nodes = self.cell_states, self.state_nodes
        except AttributeError:
            raise Exception('User must run "get_cell_tree_states" before plotting states')

        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)
        tree_layout = self.lineage_tree.get_tree_layout()
        num_colors = len(cm.get_cmap(palette).colors)
        highest_centerline = -1

        means, stds = features.mean(0, keepdims = True), features.std(0, keepdims = True)
        clip_min, clip_max = means - clip*stds, means + clip*stds
        features = np.clip(features, clip_min, clip_max)
        features_min, features_max = features.min(0, keepdims = True), features.max(0, keepdims = True)
        
        if scale_features:
            features = (features - features_min)/(features_max - features_min) #scale relative heights of features
        else:
            features = features-features_min 
        features = np.maximum(features, 0) #just make sure no vals are negative

        features = features/(features.sum(-1).max()) * max_bar_height
        
        #tracking bin stats for on-stream annotations
        feature_maxes = np.zeros(num_features)
        feature_max_times = np.zeros(num_features)
        feature_max_pos = np.zeros(num_features)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            for state_idx, (start_node, end_node) in list(state_nodes.items())[::-1]:

                centerline = tree_layout[end_node][0]
                highest_centerline = max(centerline, highest_centerline)

                bin_means, bin_times = [],[]
                for bin_mask, bin_time in self._get_trajectory_bins(state_idx, bin_size = bin_size):
                    bin_means.append(features[bin_mask,:].mean(0, keepdims = True))
                    bin_times.append(bin_time)

                bin_means = np.vstack(bin_means)
                bin_times = np.array(bin_times)

                if isinstance(end_node, int):
                    state_time_len = bin_times[-1] - bin_times[0]
                    if state_time_len == 0:
                        bin_times = np.repeat(bin_times, 2)
                        bin_times[-1] = bin_times[0] + min_pseudotime
                        bin_means = np.repeat(bin_means, 2, 0)
                    elif state_time_len < min_pseudotime:
                        bin_times = bin_times[-1] + (bin_times - bin_times[-1]) * min_pseudotime/state_time_len
                else:
                    if bin_times[-1] < tree_layout[end_node][1]:
                        bin_times = bin_times * tree_layout[end_node][1]/bin_times[-1]

                if log_pseudotime:
                    bin_times = np.log2(bin_times + 1)
                
                bin_means = np.where(bin_means/bin_means.sum(-1, keepdims = True) < hide_feature_threshold, 0, bin_means)
                
                max_bins = np.argmax(bin_means, 0)
                max_mask = bin_means.max(0) > feature_maxes
                
                bin_means = np.cumsum(bin_means, axis=-1)
                
                if center_baseline:
                    baseline_adjustment = (bin_means[:,-1]/4)[:, np.newaxis]
                else:
                    baseline_adjustment = np.zeros((bin_means.shape[0], 1))

                feature_fill_positions = bin_means - baseline_adjustment + centerline
                down_shifted_positions = np.hstack([centerline - baseline_adjustment, feature_fill_positions[:,:-1]])
                fill_center_position = down_shifted_positions + (feature_fill_positions - down_shifted_positions)/2

                feature_max_pos = np.where(max_mask, fill_center_position[max_bins, np.arange(len(max_bins))], feature_max_pos)
                feature_max_times = np.where(max_mask, bin_times[np.minimum(max_bins, len(bin_times) - 2)], feature_max_times)
                feature_maxes = np.where(max_mask, bin_means.max(0), feature_maxes)
                
                for i in np.arange(num_features)[::-1]:
                    color = cm.get_cmap(palette)(i % num_colors) if num_features > 1 else color

                    ax.fill_between(bin_times, feature_fill_positions[:, i], 
                        centerline - baseline_adjustment.reshape(-1), color = color)

                    if not linecolor is None:
                        ax.plot(bin_times, feature_fill_positions[:, i], color = linecolor, linewidth = linewidth)
                        if i == 0:
                            ax.plot(bin_times, centerline - baseline_adjustment.reshape(-1), color = linecolor, linewidth = linewidth)

                if not start_node is "Root":
                    if centerline > tree_layout[start_node][0]:
                        end_connection_line = bin_means[0,-1] - baseline_adjustment[0] + centerline
                    else:
                        end_connection_line = centerline - baseline_adjustment[0]
                    ax.plot([bin_times[0]]*2,
                            (tree_layout[start_node][0], end_connection_line), color = linecolor if not linecolor is None else 'black')

                if isinstance(end_node, int):
                    ax.text(bin_times[-1]*1.02, centerline, str(self.lineage_names[end_node]), fontsize='x-large', ha = 'left')

        ax.axis('off')
        plt.tight_layout()
        ax.set(ylim = (-0.5, highest_centerline + 0.75))

        if not title is None:
            ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

        if show_legend and num_features > 1:
            legend_params = dict(loc="center left", markerscale = 1, frameon = False, title_fontsize='x-large', fontsize='large',
                                bbox_to_anchor=(1.05, 0.5))
            ax.legend(handles = [
                            Patch(color = cm.get_cmap(palette)(i % num_colors), label = str(label)) for i,label in enumerate(labels) if feature_maxes[i] > hide_feature_threshold
                        ], **legend_params)

        if annotate_streams:
            for i in range(num_features):
                if feature_maxes[i] > hide_feature_threshold:
                    ax.text(feature_max_times[i], feature_max_pos[i], str(labels[i]), fontsize='large')

        return ax


    def plot_umap(self, hue, data_representation = None, palette = 'viridis', projection = '2d', ax = None, figsize= (10,7),
        show_legend = True, hue_order = None, size = 2, title = None):

        data_representation = self.check_representation(data_representation)

        plot_umap(data_representation, hue, palette = palette, projection = projection, ax = ax, figsize = figsize, 
            add_legend = show_legend, hue_order = hue_order, size = size, title = None)