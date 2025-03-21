import climnet.community_detection.cd_functions as cdf
import geoutils.utils.general_utils as gut
import geoutils.utils.spatial_utils as sput
import xarray as xr
from importlib import reload
import geoutils.utils.time_utils as tu
import geoutils.tsa.time_series_analysis as tsa
import numpy as np
reload(cdf)


def get_indices_prob_map(probs_idx, sig_th=0.95):
    indices = np.where(probs_idx > sig_th)[0]
    return indices


def get_sign_prob_map(ds, probs_idx, sig_th=0.95, exclude_outlayers=False,
                      rad=None):
    indices = get_indices_prob_map(probs_idx=probs_idx, sig_th=sig_th)
    link_map = ds.get_map_for_idx(idx_lst=indices)
    prob_map = ds.get_map(probs_idx)

    if exclude_outlayers:
        gut.myprint(f'Exclude as well outlayers!')
        kde_map = sput.get_kde_map(ds=ds, data=link_map, coord_rad=rad,
                                   bandwidth=None)
        link_map = xr.where(kde_map > np.nanmean(kde_map), link_map, 0)
        prob_map = xr.where(kde_map > np.nanmean(kde_map), prob_map, 0)
    return link_map, prob_map


def get_idx_region(ds, def_map,
                   exclude_outlayers=False,
                   rad=None):
    """
    Gets the indices for a specific dictionary that has lon/lat_range as keys.
    E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
    """
    reload(sput)
    if def_map is None:
        def_map = ds.mask
        if def_map is None:
            raise ValueError(
                "ERROR mask is None! Check if mask is computed properly!")

    points = np.where(def_map > 0)[0]
    ids = ds.get_idx_point_lst(point_lst=points)
    mmap = def_map

    return ids, mmap


def get_data_prob_map(
    ds, probs_idx,
    sig_th=0.95,
    exclude_outlayers=False,
    rad=None,
    an=False,
):
    """Gets for a specific monsoon regions these indices that
    belong to this region

    Args:
        c_indices (list): list of indices
        mname (str): Name of the monsoon region
        sm (str, optional): start month for time Series. Defaults to 'Jan'.
        em (str, optional): end month for time Series. Defaults to 'Dec'.

    Returns:
        dict: dictonary of monsoon containing the time series, the ids and the respective
        maps
    """
    reload(tsa)
    reload(tu)
    data_evs = ds.ds["evs"]
    data = ds.ds["pr"]
    data_an = ds.ds["an"] if an else None

    this_map, prob_map = get_sign_prob_map(ds=ds, probs_idx=probs_idx,
                                           sig_th=sig_th,
                                           exclude_outlayers=exclude_outlayers,
                                           rad=rad,)
    region_indices, region_map = get_idx_region(
        ds, def_map=this_map,
        exclude_outlayers=exclude_outlayers,
        rad=rad,
    )
    points = ds.get_points_for_idx(region_indices)  # This is important!
    if an:
        xr_ts = xr.merge(
            [
                data_evs.sel(points=points),
                data.sel(points=points),
                data_an.sel(points=points),
            ]
        )
    else:
        xr_ts = xr.merge(
            [data_evs.sel(points=points), data.sel(points=points)])

    return {
        "data": xr_ts,
        "map": region_map,
        "prob_map": prob_map,
        "ids": region_indices,
        "pids": points,
    }


def get_prob_maps_community(ds, theta_arr=None,
                            thetas=None, arr_names=[],
                            sig_th=0.9, an=False,
                            exclude_outlayers=False,
                            rad=None
                            ):
    
    if thetas is None:
        thetas = [theta['theta'] for theta in theta_arr]
    hc, probs = cdf.label_probabilities(thetas)

    prob_dict = dict(hard_cluster=hc)
    for idx, prob in enumerate(probs):
        new_data = get_data_prob_map(
            ds=ds,
            probs_idx=prob,
            sig_th=sig_th,
            an=an,
            exclude_outlayers=exclude_outlayers,
            rad=rad,
        )
        prob_dict[idx] = new_data

    if len(arr_names) > 0:
        prob_dict = {k: v for k, v in prob_dict.items() if k in arr_names}

    return prob_dict


def get_regions_data_prob_map(
    ds,
    theta_arr,
    all_region_dict,
    arr_names=[],
    sig_th=0.95,
    an=False,
    exclude_outlayers=False,  # So far only based on spatial location
):
    new_dict = dict()
    if exclude_outlayers:
        # Already computed here to not compute it for each region again and again
        deg, rad, idx_map = ds.get_coordinates_flatten()
        gut.myprint(f'Computed radian coordinates!')

    for region_name, region in all_region_dict.items():
        if region_name in arr_names or len(arr_names) == 0:
            gut.myprint(f'{region_name}')
            this_m_dict = dict()
            this_m_dict.update(region)

            prob_map, prob_map_std, gr_maps, all_comm = compute_prob_map_spatial(
                ds=ds,
                theta_arr=theta_arr,
                all_region_dict=all_region_dict, arr_regions=[
                    region_name]
            )

            new_data = get_data_prob_map(
                ds=ds,
                prob_map=prob_map,
                region_dict=region,
                sig_th=sig_th,
                an=an,
                exclude_outlayers=exclude_outlayers,
                rad=rad,
            )
            this_m_dict['prob_map_std'] = ds.get_map(prob_map_std)
            this_m_dict['gr_maps'] = gr_maps
            this_m_dict['all_comm'] = all_comm

            this_m_dict.update(new_data)
            new_dict[region_name] = this_m_dict

    return new_dict


def compute_prob_map_spatial(ds, theta_arr, all_region_dict, arr_regions,
                             level=0):

    collect_res = []
    gr_maps = []
    all_comm = []
    for run, theta_dict in enumerate(theta_arr):
        theta = theta_dict['theta']
        hard_cluster = theta_dict['hard_cluster']
        gr_num, hard_cluster = cdf.get_main_gr(
            region_dict=all_region_dict,
            names=arr_regions,
            theta=theta,
            hard_cluster=hard_cluster)

        this_prob_map = cdf.get_hc_for_gr_num(gr_num, theta=theta)
        collect_res.append(this_prob_map)

        gr_map = ds.get_map(this_prob_map)
        gr_maps.append(gr_map)

        all_hc = ds.get_map(hard_cluster)
        all_comm.append(all_hc)

    if len(all_comm) > 0:

        prob_map = np.mean(collect_res, axis=0)
        prob_map_std = np.std(collect_res, axis=0)
    else:
        raise ValueError(f'No file was existing!')

    return prob_map, prob_map_std, gr_maps, all_comm


def get_yearly_sum(
    dict, sy, ey,
):
    sum_arr = []
    for year in range(sy, ey + 1, 1):
        y_sum = np.sum(dict[year]["pr"])
        sum_arr.append(y_sum)

    return np.array(sum_arr)


def get_yearly_av(
    dict, sy, ey,
):
    av_arr = []
    for year in range(sy, ey + 1, 1):
        y_av = np.mean(dict[year])
        av_arr.append(y_av)

    return np.array(av_arr)


def get_yearly_av_var(dict, sy, ey, var="pr"):
    av_arr = []
    for year in range(sy, ey + 1, 1):
        y_av = np.mean(dict[year][var])
        av_arr.append(y_av)

    return np.array(av_arr)


def get_av_region(
    dict, var="pr",
):
    val_ts = dict[var]
    mean_ts = np.mean(val_ts, axis=0)
    if len(mean_ts) != len(dict["times"]):
        raise ValueError("Times and value ts not of same length")

    return mean_ts
