import geoutils.utils.file_utils as fut
from tqdm import tqdm
import graph_tool.all as gt
import numpy as np
import networkx as nx
import geoutils.utils.general_utils as gut
import climnet.network.network_functions as nwf
import climnet.community_detection.cd_functions as cdf
import multiprocessing as mpi
from joblib import Parallel, delayed

from importlib import reload
reload(gut)
reload(nwf)


def construct_graph_from_network(net, weighted=False,
                                 savepath=None, verbose=False):
    # Preprocessing

    # ensure square matrix
    adj_matrix = nwf.nx_to_adjacency(graph=net)
    M, N = adj_matrix.shape
    if M != N:
        raise ValueError("Adjacency must be square!")

    # We start with an empty, directed graph
    g = gt.Graph()
    # Add N nodes
    g.add_vertex(N)

    edge_list = nwf.get_edgelist(
        net=net, weighted=weighted,
    )

    B = len(edge_list)
    gut.myprint(f"Graph N {N}, B {B}", verbose=verbose)
    if weighted:
        gut.myprint("Attention! Create weighted graph!", verbose=verbose)
        eweight = g.new_ep("double")
        g.properties[("e", "weight")] = eweight
        g.add_edge_list(edge_list, eprops=[eweight])
    else:
        g.add_edge_list(edge_list)
    gut.myprint("Finished creating gt-graph! Summary:", verbose=verbose)
    gut.myprint(g, verbose=verbose)
    if savepath is not None:
        g.save(savepath)
        gut.myprint(f"Graph File saved to {savepath}!", verbose=verbose)
    return g


def numpy_array_to_gt(adjacency_matrix):
    """
    Convert a NumPy array representing an adjacency matrix into a graph object using graph_tool.

    Args:
        adjacency_matrix (numpy.ndarray): The adjacency matrix as a square NumPy array.

    Returns:
        graph_tool.Graph: A graph object representing the graph corresponding to the adjacency matrix.

    """
    # Create an empty graph object
    graph = gt.Graph(directed=False)

    # Get the size of the adjacency matrix
    n = adjacency_matrix.shape[0]

    # Add vertices to the graph
    graph.add_vertex(n)

    # Iterate through the adjacency matrix and add edges to the graph
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] != 0:
                graph.add_edge(i, j)

    return graph


def collect_node_membership(nodes):
    """Gives for every node the group idx

    Args:
        nodes (list): array of nodes to

    Returns:
        list: list of groups for node
    """
    group_membership_nodes = []
    for node in nodes:
        group_membership_nodes.append(node)

    return group_membership_nodes


def get_sbm_matrix(e):
    """
    This functions stores the Matrix of edge counts between groups for the SBM

    Parameters
    ----------
    e : graph
        graph of the hierarchical level.

    Returns
    -------
    np.array
        Array of 2d-Matrices for each level.

    """

    matrix = e.todense()
    M, N = matrix.shape
    if M != N:
        gut.myprint(f"Shape SB Matrix: {M},{N}")
        raise ValueError(f"ERROR! SB matrix is not square! {M}, {N}")

    return np.array(matrix)


"""################### Groupings of nodes and clusters ############"""


def node_level_dict(node_levels):
    """
    Gives for each level, for each group number which leaf nodes are in it.
    """
    node_level_dict = dict()
    for lid, level_ids in enumerate(node_levels):
        group_ids = np.unique(level_ids)
        this_level = []
        for idx, gid in enumerate(group_ids):
            node_idx = np.where(level_ids == gid)[0]
            if idx != int(gid):
                raise ValueError(
                    f"Attention group ID missing: {gid} for idx {idx}!"
                )
            this_level.append(node_idx)
        node_level_dict[lid] = this_level

    return node_level_dict


def reduce_node_levels(node_levels):
    """
    Graph_tool with MCMC search does sometimes skip certain group numbers.
    This function brings back the ordering to numbers from 0 to len(level).
    """
    red_hierach_data = []
    trans_dict = dict()
    level_dict = cdf.level_dict(node_levels)
    node_red_dict = dict()
    for l_id, this_level_dict in enumerate(level_dict.values()):
        this_trans_dict = dict()
        for i, (group_id, group_count) in enumerate(this_level_dict.items()):
            this_trans_dict[group_id] = i
        trans_dict[l_id] = this_trans_dict

    for l_id, level_ids in enumerate(node_levels):
        this_level = []
        for level_id in level_ids:
            this_level.append(trans_dict[l_id][level_id])
            if l_id == 0:
                node_red_dict[level_id] = trans_dict[l_id][level_id]
        red_hierach_data.append(this_level)

    return np.array(red_hierach_data)


def get_upper_level_group_number(arr):
    """
    Returns for one level which groups belong to which group number
    in the upper level.
    """
    unique_sorted = np.unique(arr)
    orig_dict = dict()
    for i in unique_sorted:
        occ_in_arr = np.where(arr == i)
        orig_dict[i] = occ_in_arr[0]
    return orig_dict


def get_hierarchical_data_from_nodes(node_through_levels):
    new_hierarchical_data = []
    node_group_dict = dict()
    for l_id, level in enumerate(node_through_levels):
        upper_level_groups = get_upper_level_group_number(level)
        this_level_arr = np.zeros(len(level), dtype=int)
        if l_id == 0:
            for (group_id, group_count) in upper_level_groups.items():
                for i in group_count:
                    this_level_arr[i] = group_id
                    node_group_dict[i] = group_id
        else:
            lower_level = get_upper_level_group_number(
                node_through_levels[l_id - 1]
            )
            this_level_arr = np.zeros(len(lower_level), dtype=int)

            for (group_id, group_count) in upper_level_groups.items():
                for i in group_count:
                    this_group = node_group_dict[i]
                    this_level_arr[this_group] = group_id
                    node_group_dict[i] = group_id
        new_hierarchical_data.append(this_level_arr)

    return np.array(new_hierarchical_data, dtype=object)


def reduce_group_levels(group_levels):
    node_levels, _ = cdf.node_level_arr(group_levels)
    new_node_levels = reduce_node_levels(node_levels)
    red_group_levels = get_hierarchical_data_from_nodes(new_node_levels)

    return red_group_levels


def get_sorted_loc_gid(group_level_ids, lid, ds=None):
    mean_lat_arr = []
    mean_lon_arr = []
    result_dict = dict()
    for gid, node_ids in enumerate(group_level_ids):
        lon_arr = []
        lat_arr = []
        for nid in node_ids:
            map_idx = ds.get_map_index(nid)
            lon_arr.append(map_idx["lon"])
            lat_arr.append(map_idx["lat"])
        # loc_dict[gid]=[np.mean(lat_arr), np.mean(lon_arr)]
        mean_lat_arr.append(np.mean(lat_arr))
        mean_lon_arr.append(np.mean(lon_arr))
    # sorted_ids=np.argsort(mean_lat_arr)  # sort by arg
    sorted_ids = [
        sorted(mean_lat_arr).index(i) for i in mean_lat_arr
    ]  # relative sorting
    if len(sorted_ids) != len(mean_lat_arr):
        raise ValueError("Error! two lats with the exact same mean!")
    sorted_lat = np.sort(mean_lat_arr)  # sort by latitude
    sorted_lon = np.array(mean_lon_arr)[sorted_ids]
    result_dict["lid"] = lid
    result_dict["sorted_ids"] = sorted_ids
    result_dict["sorted_lat"] = sorted_lat
    result_dict["sorted_lon"] = sorted_lon
    return result_dict


def parallel_ordered_nl_loc(node_levels, ds=None):

    gut.myprint(f'Start Ordering communities according to location!')
    nl_dict = node_level_dict(node_levels)
    new_node_levels = np.empty_like(node_levels)

    # For parallel Programming
    num_cpus_avail = mpi.cpu_count()
    backend = "multiprocessing"
    parallelSortedLoc = Parallel(n_jobs=num_cpus_avail, backend=backend)(
        delayed(get_sorted_loc_gid)(group_level_ids, lid, ds)
        for lid, group_level_ids in tqdm(nl_dict.items())
    )
    loc_dict = dict()

    for result_dict in parallelSortedLoc:
        lid = result_dict["lid"]
        sorted_ids = result_dict["sorted_ids"]
        loc_dict[lid] = {
            "lat": np.array(result_dict["sorted_lat"]),
            "lon": np.array(result_dict["sorted_lon"]),
            "ids": np.array(result_dict["sorted_ids"]),
        }
        for gid, node_ids in enumerate(nl_dict[lid]):
            new_node_levels[lid][node_ids] = sorted_ids[gid]

    return new_node_levels, loc_dict


def apply_SBM(
    g,
    ds=None,
    weighted=False,
    B_min=None,
    B_max=None,
    epsilon=None,
    equilibrate=False,
    multi_level=True,
    savepath=None,
    add_sbm=False,
    order_nl=False,
    verbose=True,
):
    import time
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    graph = construct_graph_from_network(net=g, weighted=weighted,
                                         verbose=verbose)

    start = time.time()

    gut.myprint("Start computing SBM on graph...", verbose=verbose)
    multilevel_mcmc_args = dict()
    if B_min is not None:
        multilevel_mcmc_args['B_min'] = B_min
    if B_max is not None:
        multilevel_mcmc_args['B_max'] = B_max
    # if epsilon is not None:
    #     multilevel_mcmc_args["epsilon"] = epsilon

    state_args = {}
    if weighted:
        # Compare
        # https://graph-tool.skewed.de/static/doc/demos/inference/inference.html#edge-weights-and-covariates
        weight_map = graph.edge_properties["weight"]
        state_args = dict(recs=[weight_map], rec_types=["real-normal"])
    if multi_level:
        gut.myprint("Compute Nested Blockmodel with mulit levels...")
        state = gt.minimize_nested_blockmodel_dl(
            graph,
            multilevel_mcmc_args=multilevel_mcmc_args,
            # deg_corr=True,
            # state_args=state_args,
        )
        gut.myprint("Finished minimize Nested Blockmodel!")
        levels = state.get_levels()

        state.print_summary()
    else:
        gut.myprint("Compute Blockmodel on single level...", verbose=verbose)
        state = gt.minimize_blockmodel_dl(
            graph,
            # deg_corr=True,  # Changed in version 2.45
            multilevel_mcmc_args=multilevel_mcmc_args,
            # mcmc_args=mcmc_args,
            # mcmc_equilibrate_args=mcmc_equilibrate_args,
            # state_args=state_args,
        )
        levels = [state]
        gut.myprint("Finished minimize 1 - level Blockmodel!")
    S1 = state.entropy()

    # we will pad the hierarchy with another xx empty levels, to give it room to potentially increase
    # TODO maybe try out first equilibration and then minimize nested blockmodel!
    if equilibrate is True:
        state = state.copy(bs=state.get_bs() +
                           [np.zeros(1)] * 4, sampling=True)
        state = state.copy(bs=state.get_bs(), sampling=True)
        gut.myprint("Do further MCMC sweeps...")
        for i in range(2000):
            # ret = state.multiflip_mcmc_sweep(niter=20, beta=np.inf)
            state.multiflip_mcmc_sweep(niter=20, beta=np.inf)

        # Now equilibrate
        gut.myprint("Now equilibrate results...", verbose=verbose)
        # gt.mcmc_equilibrate(state, nbreaks=2, wait=100, mcmc_args=dict(niter=10))

    S2 = state.entropy()
    gut.myprint(
        f" Finished MCMC search! Improvement: {S2-S1}", verbose=verbose)

    # Now we collect the marginal distribution for exactly 100,000 sweeps
    niter = 1000
    N = graph.num_vertices() + 1
    # E = graph.num_edges()

    def collect_num_groups(s):
        num_levels = len(state.get_levels())
        h = np.zeros((num_levels, N))
        for li, sl in enumerate(s.get_levels()):
            B = sl.get_nonempty_B()
            h[li][B] += 1

    gut.myprint(
        f"Sample from the posterior in {niter} samples!", verbose=verbose)
    # gt.mcmc_equilibrate(state, force_niter=niter, mcmc_args=dict(niter=10),
    #                 callback=collect_num_groups)
    gut.myprint("Finished sampling from the posterior.", verbose=verbose)
    # gt.mcmc_anneal(state, beta_range=(1, 10), niter=1000, mcmc_equilibrate_args=dict(force_niter=10))

    # The hierarchical levels themselves are represented by individual BlockState instances
    # obtained via the get_levels() method:
    group_levels = []
    entropy_arr = []
    num_groups_arr = []
    sbm_matrix_arr = []
    num_groups_new = 0
    num_groups_old = N
    for s in levels:
        e_here = s.get_matrix()
        nodes = s.get_blocks()
        group_membership_nodes = collect_node_membership(nodes)
        num_groups_new = s.get_nonempty_B()

        if num_groups_new < num_groups_old:
            # Group numbers
            gut.myprint(
                f"New number of groups: {num_groups_new} < previous: {num_groups_old}", verbose=verbose
            )
            group_levels.append(group_membership_nodes)
            # SBM Matrx
            sbm_matrix = get_sbm_matrix(e_here)
            sbm_matrix_arr.append(sbm_matrix)
            # Total number of groups
            num_groups_arr.append(num_groups_new)
            # Minimum description length
            entropy = s.entropy()
            entropy_arr.append(entropy)

            num_groups_old = num_groups_new

    group_levels = np.array(group_levels, dtype=object)
    group_levels = reduce_group_levels(group_levels)
    ng_last_level = gut.count_elements(group_levels[-1])
    ng = len(ng_last_level)
    print(group_levels, ng)
    # To conclude always with one cluster!
    # if ng > 1:
    #     group_levels = np.array(group_levels.tolist() + [np.zeros(ng)])

    entropy_arr = np.array(entropy_arr)
    sbm_matrix_arr = np.array(sbm_matrix_arr, dtype=object)
    num_groups_arr = np.array(num_groups_arr)

    end = time.time()
    run_time = end - start
    gut.myprint(f"Elapsed time for SBM: {run_time:.2f}", verbose=verbose)

    node_levels, _ = cdf.node_level_arr(group_levels)

    if order_nl and ds is not None:
        node_levels, _ = parallel_ordered_nl_loc(node_levels, ds=ds)
    if add_sbm:
        result_dict = dict(
            node_levels=node_levels,
            sbm=sbm_matrix_arr,
            group_levels=group_levels,
            entropy=entropy_arr,
        )
    else:
        result_dict = dict(
            node_levels=node_levels,
            group_levels=group_levels,
            entropy=entropy_arr,
        )

    if savepath is not None:
        fut.save_np_dict(arr_dict=result_dict,
                         sp=savepath,
                         )

    return result_dict


def networkx_to_graph_tool(nx_graph, verbose=True):
    """
    Convert a NetworkX graph into a graph_tool graph.

    Args:
        nx_graph (networkx.Graph): The NetworkX graph to be converted.

    Returns:
        graph_tool.Graph: The corresponding graph_tool graph.

    """
    if isinstance(nx_graph, gt.Graph):
        gut.myprint(f'graph is already gt-graph!')
        return nx_graph
    else:
        # Create an empty graph_tool graph
        # gt_graph = gt.Graph(directed=nx_graph.is_directed())
        A = nwf.nx_to_adjacency(graph=nx_graph)
        gt_graph = numpy_array_to_gt(adjacency_matrix=A)
        gut.myprint("Finished creating graph tool graph! Summary:",
                    verbose=verbose)
        if verbose:
            gut.myprint(gt_graph)
        return gt_graph


def el_to_graph(N, edge_list, weights=None, verbose=True):
    # We start with an empty, directed graph
    g = gt.Graph(directed=False)
    # Add N nodes
    g.add_vertex(N)

    if weights is not None:
        gut.myprint("Attention! Create weighted graph!")
        eweight = g.new_ep("double")
        g.properties[("e", "weight")] = weights
        g.add_edge_list(edge_list, eprops=[eweight])
    else:
        g.add_edge_list(edge_list)
    gut.myprint("Finished creating graph! Summary:", verbose=verbose)
    if verbose:
        gut.myprint(g)

    return g


def node_degree(graph):
    """
    Compute the node degrees of a graph using graph_tool.

    Args:
        graph (graph_tool.Graph): The graph for which to compute the node degrees.

    Returns:
        dict: A dictionary mapping node indices to their corresponding degrees.

    """

    node_degrees = {}
    for v in graph.vertices():
        node_degrees[int(v)] = v.out_degree()

    return node_degrees
