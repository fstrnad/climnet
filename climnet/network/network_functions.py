import multiprocessing as mpi
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import climnet.community_detection.graph_tool.gt_functions as gtf
import scipy.cluster.hierarchy as sch
import numpy as np
import xarray as xr
import networkx as nx
from importlib import reload
import geoutils.utils.general_utils as gut

reload(gut)


def nx_graph(el, n, verbose=False):
    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(range(n))

    # Add edges to the graph
    G.add_edges_from(el)
    if verbose:
        print(G)
    return G


def nx_to_adjacency(graph):
    return np.where(nx.to_numpy_array(graph) > 0, 1, 0)


def get_node_attr_dict(graph, attr):
    return nx.get_node_attributes(graph, attr)


def get_edge_attr_dict(graph, attr):
    return nx.get_edge_attributes(graph, attr)


def get_sparsity(M=None, verbose=True):
    """Obtain sparsity of adjacency matrix."""
    sparsity = (
        np.count_nonzero(M.flatten())
        / M.shape[0]**2
    )
    gut.myprint(f"Sparsity of adjacency:{sparsity}", verbose=verbose)
    return sparsity


def get_threshold(adjacency, corr):
    if corr is not None:
        min_val = np.nanmin(
            np.where(adjacency == 1, np.abs(corr), np.nan))
    else:
        min_val = 0
        gut.myprint('WARNING: No treshold defined. Set to default = 0.')
    return min_val


def get_isolated_nodes(adjacency):
    """Returns the isolated nodes of an adjacency matrix as indices.

    Args:
        adjacency (np.ndarray, optional): 2d array of indices of adjacency.

    Returns:
        list: list of indices of isolated node ids.
    """
    is_n_ids_o = np.where(~adjacency.any(axis=0))[0]  # columns = outgoing
    is_n_ids_i = np.where(~adjacency.any(axis=1))[0]  # rows = incoming
    is_nodes = np.union1d(is_n_ids_i, is_n_ids_o)
    if len(is_nodes) == 0:
        gut.myprint("No isolated nodes!")

    return is_nodes


def make_network_undirected(adjacency, corr=None, dense=True):
    # Makes sure that adjacency is symmetric (ie. in-degree = out-degree)
    gut.myprint(f"Make network undirected with dense={dense}")
    if dense:  # every link is counted
        adj_new = np.where(
            adjacency > adjacency.T, adjacency, adjacency.T
        )
        if adj_new.shape[0] == adj_new.shape[1]:
            adjacency = adj_new
        else:
            raise ValueError(f"Resulting adj not symmetric {adj_new.shape}!")
        if corr is not None:
            corr = np.where(corr > corr.T, corr, corr.T)
    else:
        adjacency = adjacency * adjacency.transpose()

    return adjacency


def get_adj_from_edge_list(self, edge_list, len_adj=None):
    """Gets the adjacency for a given edge list of size len_adj

    Args:
        edge_list (list): list of tuples u, values
        len_adj (int, optional): length of adjacency. Defaults to None.

    Returns:
        np.ndarray: 2d array of adjacency
    """
    if len_adj is None:
        len_adj = max(edge_list)
    adj = np.zeros((len_adj, len_adj))

    for u, v in edge_list:
        adj[u, v] = 1

    get_sparsity(M=adj)

    return adj


def get_intersect_2el(el1, el2):
    """Gets the intersection of two edge lists. Regardless if the edge list is provided
    as (i,j) or (j,i)

    Args:
        el1 (np.ndarray): 2d array of (u,v) entries
        el2 (np.ndarray): 2d array of (u,v) entries

    Returns:
        el: 2d array of elements in both edge lists.
    """
    # To account that links can be stored as i,j or j,i
    sorted_1 = map(sorted, el1)
    sorted_2 = map(sorted, el2)
    tuple_1 = map(tuple, sorted_1)
    tuple_2 = map(tuple, sorted_2)
    el = np.array(list(map(list, set(tuple_1).intersection(tuple_2))))

    return el


def get_edgelist(net, weighted=False, sort=False):
    if weighted:
        # edge_list = nx.get_edge_attributes(self.cnx, 'weight')
        edge_list = get_edge_attr_dict("weight")
    else:
        edge_list = net.edges()

    edge_list = remove_dublicates_el(np.array(list(edge_list)))
    if sort:
        edge_list, _ = sort_el_lon_lat(el=edge_list, graph=net)

    return edge_list


def remove_dublicates_el(el):
    return get_intersect_2el(el1=el, el2=el)


def sort_el_lon_lat(el, graph):
    data = []
    el_sort = []
    for e in el:
        u, v = e
        u_lon = graph.nodes[u]['lon']
        u_lat = graph.nodes[u]['lat']
        v_lon = graph.nodes[v]['lon']
        v_lat = graph.nodes[v]['lat']
        if u_lon <= v_lon:
            el_sort.append([u, v])
            data.append([u_lon, u_lat, v_lon, v_lat])
        else:
            el_sort.append([v, u])
            data.append([v_lon, v_lat, u_lon, u_lat])

    return np.array(el_sort), np.array(data)


def get_sel_ids_el(el, ids):
    """Returns for a given edge list only these edges that contain the ids.

    Args:
        el (list): 2d list of source-target ids.
        ids (list): list of int ids.
    """
    list_edges = []
    for e in el:
        if e[0] in ids or e[1] in ids:
            list_edges.append(e)

    return np.array(list_edges)


def get_lon_lat_el(el, graph):
    data = []
    for e in el:
        u, v = e
        u_lon = graph.nodes[u]['lon']
        u_lat = graph.nodes[u]['lat']
        v_lon = graph.nodes[v]['lon']
        v_lat = graph.nodes[v]['lat']

        data.append([v_lon, v_lat, u_lon, u_lat])

    return np.array(data)


def degree(graph, return_vals=False, weighted=False, verbose=True):
    gut.myprint(f'Compute degree, weighted={weighted}!', verbose=verbose)
    if isinstance(graph, nx.classes.digraph.DiGraph) or isinstance(graph, nx.classes.graph.Graph):
        degs = {node: int(val) for (node, val) in graph.degree()}
    elif isinstance(graph, gtf.gt.Graph):
        degs = {}
        for v in graph.vertices():
            degs[int(v)] = v.out_degree()
    else:
        gut.myprint('Graph object not known!')
        raise ValueError(f'Graph object not known!')
    if weighted:
        graph = weighted_degree(graph=graph)
    else:
        if return_vals:
            return np.array(list(degs.values()))
        else:
            nx.set_node_attributes(graph, degs, "degree")

    return graph


def in_degree(graph):
    degs = {node: val for (node, val) in graph.out_degree()}  # TODO check!
    nx.set_node_attributes(graph, degs, "in_degree")
    return graph


def out_degree(graph):
    degs = {node: val for (node, val) in graph.in_degree()}
    nx.set_node_attributes(graph, degs, "out_degree")
    return graph


def divergence(graph, return_vals=False, verbose=True,
               ):
    gut.myprint('compute divergence...', verbose=verbose)
    graph = in_degree(graph=graph)
    graph = out_degree(graph=graph)
    in_deg = nx.get_node_attributes(graph, 'in_degree')
    out_deg = nx.get_node_attributes(graph, 'out_degree')

    div = {}
    for (node, val) in in_deg.items():
        div[node] = val - out_deg[node]

    if return_vals:
        return np.array(list(div.values()))
    else:
        nx.set_node_attributes(graph, div, "divergence")
        return graph


def weighted_degree(graph):
    gut.myprint('Compute Weighted Node degree...')
    # Node attbs have to be dict like
    degs = {node: val for (node, val)
            in graph.degree(weight='weight')}
    nx.set_node_attributes(graph, degs, "weighted_degree")

    return graph


def betweenness(graph, return_vals=False, verbose=True):
    gut.myprint('Compute Betweenness...', verbose=verbose)

    if isinstance(graph, nx.classes.digraph.DiGraph) or isinstance(graph, nx.classes.graph.Graph):
        btn = nx.betweenness_centrality(graph)
    elif isinstance(graph, gtf.gt.Graph):
        v_btn, e_btn = gtf.gt.betweenness(graph)
        btn = {node: val for (node, val) in enumerate(
            list(v_btn))}  # v_btn = vertex betweenness
    if return_vals:
        return np.array(list(dict(btn).values()))
    else:
        nx.set_node_attributes(graph, btn, "betweenness")

        # Computes as well edge values
        btn = nx.edge_betweenness_centrality(graph, normalized=True)
        nx.set_edge_attributes(graph, btn, "betweenness")

        return graph


def clustering_coeff(graph, return_vals=False, verbose=True):
    gut.myprint('Compute Clustering Coefficient...', verbose=verbose)
    if isinstance(graph, nx.classes.digraph.DiGraph) or isinstance(graph, nx.classes.graph.Graph):
        clust_coff = nx.clustering(graph)
    elif isinstance(graph, gtf.gt.Graph):
        clust_coff = {node: val for (node, val) in enumerate(
            list(gtf.gt.local_clustering(graph)))}
    if return_vals:
        return np.array(list(dict(clust_coff).values()))
    else:
        nx.set_node_attributes(graph, clust_coff, "clustering")

        return graph


def get_node_edge_attr_q(
    graph,
    edge_attr,
    norm=True,
    q=0.95,
):
    """
    Calculate the 95th percentile of edge betweenness centrality values.

    Args:
        edge_betweenness_dict (dict): A dictionary where keys are (u, v) tuples representing edges,
                                      and values are the respective betweenness centrality values.

    """

    gut.myprint(q)
    q_val = np.quantile(list(edge_attr.values()), q=q)
    gut.myprint(f"Get values {q} <= {q_val}")
    nodes = list(graph.nodes)
    nodes_dict = gut.mk_dict_2_lists(
        key_lst=nodes, val_lst=np.zeros(len(nodes)))
    nodes_dict_cnt = gut.mk_dict_2_lists(
        key_lst=nodes, val_lst=np.zeros(len(nodes)))

    for (u, v), e in edge_attr.items():
        if (q < 0.5 and e <= q_val) | (
            q >= 0.5 and e >= q_val
        ):
            nodes_dict[u] += e
            nodes_dict[v] += v
            nodes_dict_cnt[u] += 1
            nodes_dict_cnt[v] += 1

    for ne in nodes:
        node_cnt = nodes_dict_cnt[ne]
        node_sum = nodes_dict[ne]
        # Normalized attribute by degree
        if node_cnt != 0:
            if norm is True:
                # Devide by local node degree, alternative:/self.cnx.degree(ne)
                nodes_dict[ne] = node_sum / node_cnt
            else:
                # Only sum up
                nodes_dict[ne] = node_sum
        else:
            # This needs to be later adjusted
            nodes_dict[ne] = np.nan


def curvature(graph, return_vals=False, verbose=True, c_type="forman"):
    """Creates Networkx with Forman or Ollivier curvatures

        Args:
            c_type (str, optional): curvature type (Forman or Ollivier). Defaults to 'forman'.

        Raises:
            ValueError: If wrong curvature type is given.py

        Returns:
            nx network: nx file of network that contains previous properties of cnx
        """
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    from GraphRicciCurvature.FormanRicci import FormanRicci

    # compute the Forman Ricci curvature of the given graph cnx

    if c_type == "forman":
        gut.myprint(
            "\n===== Compute the Forman Ricci curvature of the given graph =====",
            verbose=verbose
        )
        rc = FormanRicci(graph, verbose="INFO")
    elif c_type == "ollivier":
        gut.myprint(
            "\n===== Compute the Ollivier Ricci curvature of the given graph =====",
            verbose=verbose
        )
        rc = OllivierRicci(
            graph,
            alpha=0.5,
            verbose="TRACE",
            proc=mpi.cpu_count(),  # Is as well default value in OllivierRicci mix
            method="OTD",  # Sinkhorn does not work very well
        )
    else:
        raise ValueError(f"Curvature {c_type} does not exist!")

    rc.compute_ricci_curvature()
    graph = rc.G  # sets the node and edge attributes!

    node_curvatures = get_node_attr_dict(
        graph=graph, attr=f'{c_type}Curvature')
    edge_curvatures = get_edge_attr_dict(
        graph=graph, attr=f'{c_type}Curvature')
    return node_curvatures, edge_curvatures


def curvature_q(graph, q, return_vals=False, verbose=True, c_type="forman"):
    return


def triangles(graph):
    print('Compute Triangles...', flush=True)
    triangles = nx.triangles(graph)
    nx.set_node_attributes(graph, triangles, "triangles")

    return graph


def transitivity(graph):
    gut.myprint('Compute Transitivity Coefficient...')
    transitivity = nx.transitivity(graph)
    nx.set_node_attributes(graph, transitivity, "transitivity")

    return graph


def set_node_attr_array(graph, array, attr_name):
    if not isinstance(attr_name, str):
        raise ValueError(f"Attribute name {attr_name} is not a string!")
    num_nodes = len(graph.nodes)
    if len(array) != num_nodes:
        raise ValueError(
            f"Array {len(array)} has not the same length as number of nodes {num_nodes}!")
    gut.myprint('Set Node Attribute {attr_name}...')
    nx.set_node_attributes(graph, array, attr_name)

    return graph


def clustering_coeff_edges(graph):
    graph = clustering_coeff(graph=graph)
    for u, v, e in graph.edges(data=True):
        graph.edges[u, v]["clustering"] = graph.nodes[u]['clustering'] + \
            graph.nodes[v]['clustering']

    return graph


def inv_clustering_coeff_edges(graph):
    graph = clustering_coeff(graph=graph)
    for u, v, e in graph.edges(data=True):
        graph.edges[u, v]["inv_clustering"] = 1 / \
            graph.nodes[u]['clustering'] + 1/graph.nodes[v]['clustering']

    return graph


def cluster_deg_ratio(graph):
    """Get the ratio of Clustering/Degree per edge.

    Args:
        graph (networkx.graph): graph of graph
    """
    graph = degree(graph=graph)
    graph = triangles(graph=graph)
    for u, v, e in graph.edges(data=True):
        e["cdr"] = 1/2 * (graph.nodes[u]['triangles']/graph.nodes[u]['degree'] +
                          graph.nodes[v]['triangles']/graph.nodes[v]['degree'])

    return graph


def set_node_attr(G, attr, norm=False):
    for ne in G.nodes:
        node_sum = 0.0
        node_cnt = 0
        for u, v, e in G.edges(ne, data=True):
            node_sum += e[attr]
            node_cnt += 1
        # Normalized attribute by degree
        if node_cnt > 0:
            if norm is True:
                # /G.degree(ne)
                G.nodes[ne][attr] = node_sum / node_cnt
            else:
                G.nodes[ne][attr] = node_sum
        else:
            G.nodes[ne][attr] = np.nan

    return G

# ################## Clustering


def apply_K_means_el(data, n, el=None):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    cluster_numbers = kmeans.predict(data)
    coord_centers = kmeans.cluster_centers_
    num_clusters = np.max(cluster_numbers)
    cluster_dict = dict()
    if el is not None:
        min_num_elem = 2
        clcnt = 0
        for cn in range(num_clusters+1):
            el_ind_cl = np.where(cluster_numbers == cn)[0]
            if len(el_ind_cl) > min_num_elem:
                el_cl = el[el_ind_cl]
                cluster_dict[clcnt] = el_cl
                clcnt += 1
    return {'number': cluster_numbers,
            'center': coord_centers,
            'cluster': cluster_dict
            }


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    sch.dendrogram(linkage_matrix, **kwargs)


def apply_complete_linkage(data, el=None,
                           method='ward',
                           metric='euclidean',
                           n=None):

    dist_th = 0 if n is None else None
    cluster = AgglomerativeClustering(n_clusters=n,
                                      distance_threshold=dist_th,
                                      affinity=metric,
                                      linkage=method)
    if n is None:
        plot_dendrogram(cluster.fit(data))

    cluster_numbers = cluster.fit_predict(data)
    num_clusters = np.max(cluster_numbers)
    cluster_dict = dict()
    if el is not None:
        min_num_elem = 2
        clcnt = 0
        for cn in range(num_clusters+1):
            el_ind_cl = np.where(cluster_numbers == cn)[0]
            if len(el_ind_cl) > min_num_elem:
                el_cl = el[el_ind_cl]
                cluster_dict[clcnt] = el_cl
            clcnt += 1
    return {'number': cluster_numbers,
            'cluster': cluster_dict,
            }


def remove_node_attr(G: nx.Graph, attr_name: str) -> None:
    """Removes a node attribute from all nodes in the given networkx graph.

    Args:
    G (nx.Graph): A networkx graph
    attr_name (str): The name of the attribute to remove from all nodes
    """
    for node in G.nodes():
        if attr_name in G.nodes[node]:
            del G.nodes[node][attr_name]


def random_rewire(graph: nx.Graph) -> nx.Graph:
    """
    Randomly rewires the edges of a NetworkX graph while preserving the same number of links.

    Parameters:
    graph (nx.Graph): The input graph to be rewired.

    Returns:
    nx.Graph: A randomly rewired graph with the same number of links.

    Example:
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
    >>> rewired_graph = random_rewire(G)
    >>> print(rewired_graph.edges())
    [(1, 3), (1, 5), (2, 4), (3, 5), (4, 5)]
    """
    rewired_graph = nx.double_edge_swap(graph, nswap=len(
        graph.edges()), max_tries=len(graph.edges()) * 10)
    return rewired_graph
