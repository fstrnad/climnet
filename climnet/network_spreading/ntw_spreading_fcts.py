import numpy as np
import xarray as xr
import networkx as nx
import climnet.tsa.event_synchronization as es
import climnet.utils.general_utils as gut
from importlib import reload
import multiprocessing as mpi
from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.stats as st
import climnet.tsa.time_series_analysis as tsa
from itertools import product


def sync_es_ds(cnx, el, comb_e12, taumax=10):
    reload(tsa)
    reload(es)

    backend = 'multiprocessing'
    # backend='loky'
    # backend='threading'
    num_cpus_avail = mpi.cpu_count()
    print(f"Number of available CPUs: {num_cpus_avail}")
    parallelArray = (Parallel(n_jobs=num_cpus_avail, backend=backend)
                     (delayed(es.event_sync_reg)
                      (e1,  e2, idx, taumax, 2*taumax)
                      for idx, (e1, e2) in enumerate(tqdm(comb_e12))
                      )
                     )
    zero_ds = np.zeros_like(cnx.ds.ds['evs'].data, dtype=float)
    tau_ds = np.zeros_like(cnx.ds.ds['evs'].data, dtype=float)

    print('Finished computing synchronizity, now compute array!', flush=True)
    for ret_dict in tqdm(parallelArray):
        t_e = ret_dict['t']
        t12_e = ret_dict['t12']
        t21_e = ret_dict['t21']
        dyn_delay_12 = ret_dict['dyn_delay_12']
        edge = el[ret_dict['idx']]
        points = cnx.ds.get_points_for_idx(edge)
        if len(t12_e) > 0:
            zero_ds[np.ix_(t12_e, list(points))] += 1
            # vstack for correctly broadcasting to np.ix_
            tau_ds[np.ix_(t12_e, [points[1]])] += -1.*np.vstack(dyn_delay_12)
    sync_evs = xr.DataArray(
        data=zero_ds,  # might be zero_ds.T
        dims=cnx.ds.ds.dims,
        coords=cnx.ds.ds.coords,
        name='sync_evs'
    )
    mask_sync_evs = np.where(sync_evs > 0, sync_evs, np.nan)
    tau_evs = xr.DataArray(
        data=tau_ds/mask_sync_evs,  # devide by degree
        dims=cnx.ds.ds.dims,
        coords=cnx.ds.ds.coords,
        name='tau'
    )

    merge_ds = xr.merge([sync_evs, tau_evs])
    return merge_ds


def es_spread_reg(cnx,
                  idx_1,
                  idx_2=None,
                  taumax=10,
                  use_adj=True,
                  ):
    """
    Function that computes the time series of synchronous events for
    a given Network (via adjacency) and certain nodes (nodes can be the same!)

    """
    reload(es)
    if idx_2 is None:
        idx_2 = idx_1

    if use_adj:
        # compare only events along existing links
        el = cnx.get_edges_between_nodes(ids1=idx_1, ids2=idx_2)
        comb_e12 = tsa.get_el_evs_idx(cnx, el)
    else:
        ts_1 = tsa.get_evs_idx_ds(ds=cnx.ds, ids=idx_1)
        ts_2 = tsa.get_evs_idx_ds(ds=cnx.ds, ids=idx_2)
        comb_e12 = np.array(list(product(ts_1, ts_2)), dtype=object)
        el = np.array(list(product(idx_1, idx_2)))
    print("prepared time series!")

    return sync_es_ds(cnx=cnx, el=el, comb_e12=comb_e12, taumax=taumax)


def es_spread_el(cnx, el, taumax=10):

    comb_e12 = tsa.get_el_evs_idx(cnx, el)

    return sync_es_ds(cnx=cnx, el=el, comb_e12=comb_e12, taumax=taumax)


def prob_sync_evs(cnx, sync_evs, sids, method='max'):
    """Computes a probabilit map that an event in sids is followed by an event
    somewhere else.

    Args:
        cnx (climNetworkX): contains the edges and the dataset.py
        sync_evs (xr.Dataarray): Array that contains the synchrouns events for sids.
        sids (int): list of source ids.
        method (str, optional): For different sids the probability might differ,
            method sets the choice. Defaults to 'max'.

    Returns:
        xr.Dataarray: The probability map
    """
    num_tps_ds = len(cnx.ds.ds.time.data)
    num_tps_sevs = len(sync_evs.time.data)
    if num_tps_ds != num_tps_sevs:
        raise ValueError(
            f'Time series not of same length: {num_tps_ds} != {num_tps_sevs}')

    coll_prob_maps = []
    # times = cnx.ds.ds.time
    for sid in tqdm(sids):
        # Init zero - prob map
        zero_ds = np.zeros_like(cnx.ds.mask.data, dtype=float)

        # Get number of sync events
        pid = cnx.ds.get_points_for_idx([sid])
        tot_evs_sid = cnx.ds.ds['evs'].sel(points=pid)
        sid_evs_indices = es.get_evs_index(tot_evs_sid)
        # sync_evs_sid = sync_evs.sel(points=pid)   # Attention this might contain more events than tot_evs_sid, because of time delayed events
        num_evs_sid = len(sid_evs_indices)

        # Get target ids
        tids, _ = cnx.get_target_ids_for_node_ids(sid)
        ptids = cnx.ds.get_points_for_idx(tids)
        for ptid in ptids:
            tid_evs = sync_evs.sel(points=ptid)
            # Synchronous events in source and target are counted as 1
            # TODO use same tp as for sid_evs_indices
            # only events when an event happened in source
            sync_evs_st = xr.where(tid_evs > 0, tot_evs_sid, 0)

            num_sync_evs = sync_evs_st.sum(dim='time')
            if num_sync_evs > num_evs_sid:
                raise ValueError(
                    f'More sync evs than source events s {pid} - t {ptid}')
            if num_sync_evs < 1:
                continue
            else:
                # Quotient of sync events source -target / tot number ERE in source
                zero_ds[ptid] = float(num_sync_evs / num_evs_sid)
                # zero_ds[ptid] = 1

        coll_prob_maps.append(zero_ds)

    if method == 'max':
        prob_map = np.max(np.array(coll_prob_maps), axis=0)
    elif method == 'mean':
        prob_map = np.mean(np.array(coll_prob_maps), axis=0)
    elif method == 'med':
        prob_map = np.median(np.array(coll_prob_maps), axis=0)
    else:
        raise ValueError(f'Method {method} does not exist!')

    prob_evs = xr.DataArray(
        data=prob_map,  # might be zero_ds.T
        dims=cnx.ds.mask.dims,
        coords=cnx.ds.mask.coords,
        name="prob_evs",
    )

    return prob_evs


