# The functions in this form are mainly adapted from https://github.com/Rheinwalt/spatial-effects-networks
# See as well Rheinwalt et al. (2012), https://iopscience.iop.org/article/10.1209/0295-5075/100/28002

import geoutils.utils.general_utils as gut
from tqdm import tqdm
from scipy.stats import percentileofscore
import climnet.network.network_functions as nwf
import climnet.community_detection.graph_tool.gt_functions as gtf
import networkx as nx
import graph_tool.all as gt
import numpy as np
from scipy.spatial.distance import pdist
from math import radians, sqrt, sin, cos, atan2
from importlib import reload
reload(gtf)
reload(nwf)


def AdjacencyMatrix(ids, links):
    n = len(ids)
    a = np.zeros((n, n), dtype='int')
    for i in range(links.shape[0]):
        u = links[i, 0] == ids
        v = links[i, 1] == ids
        a[u, v] = a[v, u] = 1

    np.fill_diagonal(a, 0)
    b = a.sum(axis=0) > 0
    return (a[b, :][:, b], b)


def GreatCircleDistance(u, v):
    """Great circle distance from (lat, lon) in degree in kilometers."""
    lat1 = radians(u[0])
    lon1 = radians(u[1])
    lat2 = radians(v[0])
    lon2 = radians(v[1])
    dlon = lon1 - lon2
    EARTH_R = 6372.8

    y = sqrt(
        (cos(lat2) * sin(dlon)) ** 2
        + (cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)) ** 2
    )
    x = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(dlon)
    d = atan2(y, x)
    return EARTH_R * d


def IntegerDistances(lat, lon, scale=50.0):
    """Link distances as indices for the binned statistics of link probabilities."""

    # triu distances in km
    D = pdist(np.transpose((lat, lon)), GreatCircleDistance)
    Dm = D.min()

    # optional rescaling
    D = np.log10(D-Dm+1)

    # binning by rounding
    D = (scale * D).astype('int')

    # x axis for p, gives the lower distance of each bin
    x = 10**((np.arange(D.max() + 1) / scale) - 1)
    return (D, x)


def LinkProbability(A, D):
    m = D.max() + 1
    p = np.zeros(m)
    q = np.zeros(m)
    for i in range(len(D)):
        k = D[i]
        q[k] += 1
        if A[i]:
            p[k] += 1

    q[q == 0] = np.nan
    p /= q
    p[p == np.nan] = 0
    return p


def SernEdges(D, p, n):
    assert len(D) == n*(n-1)/2, 'n does not fit to D'
    a = np.zeros(D.shape, dtype='int')
    a[np.random.random(len(D)) <= p[D]] = 1
    A = np.zeros((n, n), dtype='int')
    A[np.triu_indices(n, 1)] = a
    edges = np.transpose(A.nonzero())
    return edges


def boundary_correction(cnx, measure='degree',
                        use_gt=True,
                        samples=10,
                        verbose=False):

    if measure == 'degree':
        mf = nwf.degree
    elif measure == 'clustering_coeff':
        mf = nwf.clustering_coeff
    elif measure == 'betweenness':
        mf = nwf.betweenness
    else:
        raise ValueError(f'This measure does not exist: {measure}!')

    A = cnx.adjacency
    n = A.shape[0]
    A[np.tril_indices(n)] = 0
    A = A[np.triu_indices(n, 1)]
    lons = cnx.get_node_attr('lon')
    lats = cnx.get_node_attr('lat')

    D, x = IntegerDistances(lats, lons, scale=50)
    p = LinkProbability(A, D)

    if measure != 'curvature' and use_gt:
        gut.myprint('Use graph-tool library', verbose=verbose)
        graph = gtf.networkx_to_graph_tool(cnx.cnx)
    else:
        gut.myprint('use networkx library', verbose=verbose)
        use_gt = False
        graph = cnx.cnx

    gut.myprint(f'Compute uncorrected {measure}')
    var_0 = mf(graph=graph, return_vals=True, verbose=verbose)

    gut.myprint('Sample network')
    var = sample_networks(samples=samples,
                          n=n,
                          D=D,
                          p=p,
                          measure=measure,
                          use_gt=use_gt,
                          verbose=verbose)

    gut.myprint(f'Correct {measure}')
    bc_dict = correct_measure(var_0=var_0,
                              var=var,
                              cnx=cnx
                              )

    return bc_dict


def sample_networks(samples, n, D, p, measure='degree',
                    use_gt=True, verbose=False):

    if measure == 'degree':
        mf = nwf.degree
    elif measure == 'clustering_coeff':
        mf = nwf.clustering_coeff
    elif measure == 'betweenness':
        mf = nwf.betweenness
    elif measure == 'all':
        mf = nwf.all_measures
    else:
        raise ValueError(f'This measure does not exist: {measure}!')
    var = np.zeros((samples, n))

    for i in tqdm(range(samples)):
        e = SernEdges(D, p, n)
        g = gtf.el_to_graph(edge_list=e, N=n, verbose=verbose) if use_gt else nwf.nx_graph(
            el=e, n=n, verbose=verbose)
        v = mf(graph=g, return_vals=True, verbose=False)
        v = np.array(v)
        var[i, :] = v

    return var


def correct_measure(var_0, var, cnx):
    assert len(var_0) == var.shape[1]

    # uncorrected values
    uc = cnx.ds.get_map(var_0)
    # sern ensemble mean
    c = var.mean(axis=0)
    c = cnx.ds.get_map(c)
    # corrected measure
    bc = var_0 - var.mean(axis=0)
    bc = cnx.ds.get_map(bc)

    # percentiles
    pc = np.array([percentileofscore(var[:, i], var_0[i])
                  for i in range(len(var_0))])
    pc = cnx.ds.get_map(pc)

    return {
        'uc': uc,
        'c': c,
        'bc': bc,
        'pc': pc}


def Graph(e, n):

    g = gt.Graph(directed=False)
    g.add_vertex(n)
    g.add_edge_list(e)
    return g
