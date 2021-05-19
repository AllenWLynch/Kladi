

import numpy as np
import networkx as nx

class LineageTree:

    def __init__(self, lineage_names):
        self.node_splits = []
        self.split_times = []
        self.lineage_names = np.array(lineage_names)

    
    def add_split(self, nodes, split_time):
        self.node_splits.append(nodes)
        self.split_times.append(split_time)


    def get_all_leaves_from_node(self, node):

        if isinstance(node, (int, np.int32, np.int64)):
            return [node]

        return [*self.get_all_leaves_from_node(node[0]), *self.get_all_leaves_from_node(node[1])]

    def get_node_name(self, node):
        return ', '.join(map(str, self.lineage_names[self.get_all_leaves_from_node(node)]))

    def get_tree_layout(self):

        G = nx.DiGraph()

        for node, time in self:
            G.add_node(node)
            G.add_edge(node, node[0])
            G.add_edge(node, node[1])

        dfs_tree = list(nx.dfs_predecessors(G, self.get_root()))[::-1] + [self.get_root()]

        node_positions = {}
        num_termini = [0]
        branch_times = dict(list(iter(self)))

        def get_or_set_node_position(node):

            if not node in node_positions:
                if isinstance(node, (int, np.int32, np.int64)):
                    node_positions[node] = (num_termini[0], np.inf)
                    num_termini[0]+=1
                else:
                    node_positions[node] = ((get_or_set_node_position(node[0]) + get_or_set_node_position(node[1]))/2, branch_times[node])

            return node_positions[node][0]

        for node in dfs_tree:
            get_or_set_node_position(node)

        node_positions[("Root", node)] = (node_positions[node][0], 0)

        return node_positions

    def get_root(self):
        return self.node_splits[-1]

    def __iter__(self):

        for node, time in zip(self.node_splits, self.split_times):
            yield node, time


    def get_graphviz_tree(self):

        try:
            import pygraphviz as pgv
        except ModuleNotFoundError:
            raise Exception('Pygraphviz is required for this function. Run "pip install pygraphviz" to install')

        G = pgv.AGraph(directed = True)

        for split, pseudotime in zip(self.node_splits, self.split_times):
            
            node_1, node_2 = split

            node_1, node_2, split = map(lambda x : x + ' ', [self.get_node_name(node_1), self.get_node_name(node_2), self.get_node_name(split)])
            #convert all names to strings + ' ' so that pygraphviz doesn't break for some reason
            #node_1, node_2, split = list(map(lambda x : (str(x) + ' ').replace('(','').replace('\'', '').replace(')', ''), [node_1, node_2, split]))

            G.add_node(split)
            G.add_edge(split, node_1)
            G.add_edge(split, node_2)

        G.layout(prog = 'dot')
        return G

    