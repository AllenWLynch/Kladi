
import warnings
import numpy as np
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

def map_colors(ax, c, palette, add_legend = True, hue_order = None, legend_kwargs = {}, cbar_kwargs = {}):

    assert(isinstance(c, np.ndarray))

    if np.issubdtype(c.dtype, np.number):

        colormapper=cm.ScalarMappable(Normalize(c.min(), c.max()), cmap=palette)
        c = colormapper.to_rgba(c)

        if add_legend:
            plt.colorbar(colormapper, ax=ax, **cbar_kwargs)

        return c

    else:
        
        classes = set(c)
        num_colors = len(cm.get_cmap(palette).colors)
        color_wheel = cm.get_cmap(palette)(np.arange(len(classes)) % num_colors)
        
        if hue_order is None:
            class_colors = dict(zip(classes, color_wheel))
        else:
            assert(len(hue_order) == len(classes))
            class_colors = dict(zip(hue_order, color_wheel))

        c = np.array([class_colors[c_class] for c_class in c])

        if add_legend:
            ax.legend(handles = [
                Patch(color = color, label = str(c_class)) for c_class, color in class_colors.items()
            ], **legend_kwargs)

        return c

def plot_umap(X, hue, palette = 'viridis', projection = '2d', ax = None, figsize= (10,5),
        add_legend = True, hue_order = None, size = 2, title = None):

    assert(isinstance(hue, np.ndarray))
    hue = hue.ravel()
    assert(len(hue) == len(X))

    if projection == '3d':
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise Exception('Must install plotly to use this feature. Run conda install plotly or pip install plotly.')
        
        color = map_colors(None, hue, palette, add_legend=False, hue_order = hue_order)

        if X.shape[-1] == 2:
            z_dim = np.zeros(len(X))
        else:
            z_dim = X[:,2]

        fig = go.Figure(data=[go.Scatter3d(
            x=X[:,0],
            y=X[:,1],
            z=z_dim,
            mode='markers',
            marker=dict(
                size=size,
                color = color,
                opacity=0.8
            ),
            hovertext=np.arange(len(X))
        )])

        fig.update_layout(scene=dict(xaxis = dict(showgrid=False, showticklabels = False, backgroundcolor="white", visible = False), 
                                    yaxis = dict(showgrid=False, showticklabels = False, backgroundcolor="white", visible = False), 
                                    zaxis = dict(showgrid = False, showticklabels = False, backgroundcolor="white", visible = False)),
                        width=figsize[0]*100, height=figsize[1]*100,
                        margin=dict(
                            r=0, l=0,
                            b=0, t=0)
                        )
        fig.show()
        return fig

    else:
        
        if ax is None:
            fig, ax = plt.subplots(1,1,figsize=figsize)

        colors = map_colors(ax, hue, palette, add_legend=add_legend, hue_order = hue_order,
                cbar_kwargs = dict(orientation = 'vertical', pad = 0.01, shrink = 0.5, aspect = 15, anchor = (1.05, 0.5)),
                legend_kwargs = dict(loc="center left", markerscale = 4, frameon = False, title_fontsize='x-large', fontsize='large',
                            bbox_to_anchor=(1.05, 0.5)))

        ax.scatter(X[:,0], X[:,1], c = colors, s= size)
        ax.axis('off')

        if not title is None:
            ax.set_title(str(title), fontdict= dict(fontsize = 'x-large'))

        return ax

class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""
    def __init__(self, orient="v", width=0.8, warn_thresh=.05):

        # XXX should we keep the orient parameterization or specify the swarm axis?

        self.orient = orient
        self.width = width
        self.warn_thresh = warn_thresh

    def __call__(self, points, center):
        """Swarm `points`, a PathCollection, around the `center` position."""
        # Convert from point size (area) to diameter

        ax = points.axes
        dpi = ax.figure.dpi

        # Get the original positions of the points
        orig_xy_data = points.get_offsets()

        # Reset the categorical positions to the center line
        cat_idx = 1 if self.orient == "h" else 0
        orig_xy_data[:, cat_idx] = center

        # Transform the data coordinates to point coordinates.
        # We'll figure out the swarm positions in the latter
        # and then convert back to data coordinates and replot
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)

        # Order the variables so that x is the categorical axis
        if self.orient == "h":
            orig_xy = orig_xy[:, [1, 0]]

        # Add a column with each point's radius
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth().item()
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]

        # Sort along the value axis to facilitate the beeswarm
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]

        # Adjust points along the categorical axis to prevent overlaps
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)

        # Transform the point coordinates back to data coordinates
        if self.orient == "h":
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T

        swarm_axis = {"h": "y", "v": "x"}[self.orient]
        log_scale = getattr(ax, f"get_{swarm_axis}scale")() == "log"

        # Add gutters
        if self.orient == "h":
            self.add_gutters(new_y_data, center, log_scale=log_scale)
        else:
            self.add_gutters(new_x_data, center, log_scale=log_scale)

        # Reposition the points so they do not overlap
        if self.orient == "h":
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        # In this method, `x` is always the categorical axis
        # Center of the swarm, in point coordinates
        midline = orig_xyr[0, 0]

        # Start the swarm with the first point
        swarm = np.atleast_2d(orig_xyr[0])

        # Loop over the remaining points
        for xyr_i in orig_xyr[1:]:

            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = self.could_overlap(xyr_i, swarm)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = self.position_candidates(xyr_i, neighbors)

            # Sort candidates by their centrality
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]

            # Find the first candidate that does not overlap any neighbors
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)

            # Place it into the swarm
            swarm = np.vstack([swarm, new_xyr_i])

        return swarm

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        # Because we work backwards through the swarm and can short-circuit,
        # the for-loop is faster than vectorization
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if (y_i - y_j) < (r_i + r_j):
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = (x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i)
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors):
        """Find the first candidate that does not overlap with the swarm."""

        # If we have no neighbors, all candidates are good.
        if len(neighbors) == 0:
            return candidates[0]

        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]

        for xyr_i in candidates:

            x_i, y_i, r_i = xyr_i

            dx = neighbors_x - x_i
            dy = neighbors_y - y_i
            sq_distances = np.square(dx) + np.square(dy)

            sep_needed = np.square(neighbors_r + r_i)

            # Good candidate does not overlap any of neighbors which means that
            # squared distance between candidate and any of the neighbors has
            # to be at least square of the summed radii
            good_candidate = np.all(sq_distances >= sep_needed)

            if good_candidate:
                return xyr_i

        raise RuntimeError(
            "No non-overlapping candidates found. This should not happen."
        )

    def add_gutters(self, points, center, log_scale=False):
        """Stop points from extending beyond their territory."""
        half_width = self.width / 2
        if log_scale:
            low_gutter = 10 ** (np.log10(center) - half_width)
        else:
            low_gutter = center - half_width
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        if log_scale:
            high_gutter = 10 ** (np.log10(center) + half_width)
        else:
            high_gutter = center + half_width
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter

        gutter_prop = (off_high + off_low).sum() / len(points)
        if gutter_prop > self.warn_thresh:
            msg = (
                "{:.1%} of the points cannot be placed; you may want "
                "to decrease the size of the markers or use stripplot."
            ).format(gutter_prop)
            warnings.warn(msg, UserWarning)

        return points