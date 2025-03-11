#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
from climnet.community_detection.dendrograms import Dendrogram_ES
from climnet.monsoon.monsoon_region import Monsoon_Region
import sys
import os
import numpy as np
import pandas as pd

import xarray as xr
import matplotlib.pyplot as plt

PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory


# %%
"""Class of monsoon regions, containing the
monsoon definitions, the node ids and the regional monsoons."""


class Monsoon_Region_ES(Monsoon_Region):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(self,
                 var_name=None,
                 data_nc=None,
                 load_nc=None,
                 time_range=None,
                 month_range=None,
                 lon_range=[-180, 180],
                 lat_range=[-90, 90],
                 grid_step=1,
                 grid_type='gaussian',
                 lsm=False,
                 large_ds=False,
                 name='',
                 network_file=None,
                 graph_path_file=None,
                 abs_th_wang=2, abs_th_ee=50, rel_th=0.55,
                 full_year=False,
                 group_levels=None,
                 **kwargs):
        super().__init__(data_nc=data_nc, load_nc=load_nc,
                         var_name=var_name, time_range=time_range,
                         lon_range=lon_range, lat_range=lat_range,
                         grid_step=grid_step, grid_type=grid_type,
                         month_range=month_range, large_ds=large_ds,
                         name=name, lsm=lsm,
                         network_file=network_file,
                         graph_path_file=graph_path_file,
                         abs_th_wang=abs_th_wang, abs_th_ee=abs_th_ee, rel_th=rel_th,
                         full_year=full_year,
                         **kwargs)

        if group_levels is not None:
            self.d_es = Dendrogram_ES(group_levels,)
        else:
            import warnings
            warnings.warn("Attention, no own dendrogram was created!")

    def get_Z_ids_monsoon(self, mname='India', defn='ee', th=2, plot=True):
        m_node_ids = self.get_m_ids(mname=mname, defn=defn)

        # Z_ids_hist, frequencies,  Z_most_occ=self.d_es.d_tree.get_hist_sel_nodes(m_node_ids, th=th)
        if plot is True:
            self.d_es.d_tree.plot_hist_is_nodes(
                m_node_ids, th=th, title=self.monsoon_dictionary[mname]['name'])

        # top_Z=max(self.d_es.d_tree.node_dict)
        # _, leaf_nodes=self.d_es.d_tree.backward_path(node_id=top_Z)

        Z_ids = self.d_es.d_tree.get_all_Z_ids_leaf_ids(m_node_ids)

        return Z_ids

    def get_Z_ids_monsoon_hist(self, mname='India', defn='ee', th=2, plot=True):
        m_node_ids = self.get_m_ids(mname=mname, defn=defn)

        Z_ids_hist, frequencies,  Z_most_occ = self.d_es.d_tree.get_hist_sel_nodes(
            m_node_ids, th=th)
        if plot is True:
            self.d_es.d_tree.plot_hist_is_nodes(
                m_node_ids, th=th, title=self.monsoon_dictionary[mname]['name'])

        return Z_ids_hist

    def centroids_by_ind(self, idx_list):
        ordered_ind_list = np.unique(idx_list)
        len_idx_lst = len(ordered_ind_list)
        rep_idx = ordered_ind_list[int(len_idx_lst/2)]
        av_idx = np.mean(ordered_ind_list)
        lon_arr = []
        lat_arr = []
        for idx in idx_list:
            map_idx = self.get_map_index(idx)
            lon_arr.append(map_idx['lon'])
            lat_arr.append(map_idx['lat'])
        mean_lat = np.mean(lat_arr)
        mean_lon = np.mean(lon_arr)

        centroid_dict = {'num_idx': [len_idx_lst], 'rep_idx': [
            rep_idx], 'av_idx': [av_idx], 'lat': mean_lat, 'lon': mean_lon}

        return centroid_dict

    def find_similar_Z_id(self, compare_centroid, Z_ids):
        from sklearn.metrics import pairwise_distances
        from sklearn.metrics.pairwise import cosine_similarity
        cols = ['num_idx', 'rep_idx', 'av_idx', 'lat', 'lon']
        centroid_pd = pd.DataFrame(columns=cols)

        for Z_id in Z_ids:
            leaf_nodes = self.d_es.d_tree.get_leaf_nodes_of_is_node(Z_id)
            centroids = self.centroids_by_ind(leaf_nodes)
            centroid_pd = pd.concat(
                [centroid_pd, pd.DataFrame.from_dict(centroids)], ignore_index=True)

        if len(centroid_pd.iloc[0]) != len(compare_centroid):
            raise ValueError(
                f"The similarity matrix len : {len(centroid_pd.iloc[0])} and  len {len(compare_centroid)} are not of the same rows length!")

        pair_dist = pairwise_distances(
            centroid_pd, [compare_centroid],  metric='manhattan')
        sim_idx = np.argmin(pair_dist)
        print(f"Similar: sim_idx {sim_idx}, {Z_ids[sim_idx]}", )
        print(pair_dist)
        print(centroid_pd.iloc[sim_idx])
        print(compare_centroid)
        return Z_ids[sim_idx]

    def get_sim_centroid_Z_id(self, mname, compare_centroid):

        Z_ids = self.get_Z_ids_monsoon(mname=mname, plot=True, th=40)
        sim_Z_id = self.find_similar_Z_id(compare_centroid, Z_ids)
        return sim_Z_id

    def get_all_g_2_mregions(self):
        from itertools import combinations
        mnames = [name for name in self.monsoon_dictionary]
        all_comb = list(combinations(mnames, 2))
        g_Zid_dict = dict()
        for (ms1, ms2) in all_comb:
            mids1 = self.monsoon_dictionary[ms1]['rep_ids']
            mids2 = self.monsoon_dictionary[ms2]['rep_ids']
            g_Zid = self.d_es.d_tree.get_split_2_groups(mids1, mids2)
            g_Zid_dict[(ms1, ms2)] = g_Zid

        return g_Zid_dict

    def get_mscluster_sel(self, d_es=None, sel='NH', key_ids='rep_ids', abs_th=0, rel_th=1):

        if sel == 'NH':
            m_arr = ['North America', 'North Africa',
                     'India South Asia', 'East Asia', 'NWP']
        elif sel == 'SH':
            m_arr = ['South America', 'South Africa', 'Australia', 'CP']
        elif sel == 'Pacific':
            m_arr = ['CP', 'NWP']
        elif sel == 'land_NH':
            m_arr = ['North America', 'North Africa',
                     'India South Asia', 'East Asia', 'NWP']
        elif sel == 'land_SH':
            m_arr = ['South America', 'South Africa', 'Australia']
        else:
            raise ValueError(f"This selection {sel} does not exist!")

        g_Zid, _ = self.get_ms_cluster_ids(
            m_arr, d_es=d_es, key_ids=key_ids, abs_th=abs_th, rel_th=rel_th)

        return g_Zid, m_arr

    def get_ms_cluster_ids(self, m_arr, d_es=None, key_ids='rep_ids', abs_th=0, rel_th=1):
        mids = []
        for m_key in m_arr:
            mids.append(self.monsoon_dictionary[m_key][key_ids])
        if d_es is None:
            d_es = self.d_es
        g_Zid = d_es.d_tree.get_split_groups(
            mids, abs_th=abs_th, rel_th=rel_th)

        return g_Zid, np.concatenate(mids)

    ######### Plotting Routines####################
    # Z Id

    def plot_Z_id(self,  Z_id, d_es=None, title='', projection='Mollweide', savepath=None):
        if d_es is None:
            d_es = self.d_es

        leaf_nodes_is = d_es.d_tree.get_leaf_nodes_of_is_node(Z_id)
        full_arr = self.flat_idx_array(leaf_nodes_is)
        map = self.get_map(full_arr)
        ax = self.plot_map(map, dcbar=False, color='Reds',
                           title=title, bar=False, projection=projection)
        self.plot_contour_map(self.ee_def, color='Blues', ax=ax,
                              bar=False, fill_out=False, projection=projection)
        if savepath is not None:
            plt.savefig(savepath)
        return ax

    # Visualize monsoon paths
    def plot_monsoon_paths(self, m_arr=None, li=0, monsoon_dict=None, title=None, key_ids='rep_ids'):

        label_arr = []
        node_ids = []
        color_arr = []
        if monsoon_dict is None:
            monsoon_dict = self.monsoon_dictionary

        if m_arr is not None:
            monsoon_dict = self.get_sel_mdict(m_arr=m_arr)

        for name, monsoon in monsoon_dict.items():
            monsoon_name = monsoon['name']
            color = monsoon['color']
            m_node_ids = monsoon[key_ids]
            print(f"Plot Monsoon Path {name} in color {color}.")
            for i, m_id in enumerate(m_node_ids):
                if li == 0:
                    node_id = m_id
                else:
                    idx_throuh_levels = self.d_es.foreward_node_levels(
                        self.d_es.node_levels, node_id=m_id)
                    node_id = idx_throuh_levels[li]
                node_ids.append(node_id)
                if i == 0:
                    label_arr.append(monsoon_name)
                else:
                    label_arr.append(None)
                color_arr.append(color)

        sel_group_levels = self.d_es.sel_group_levels(node_ids)
        sel_Z = self.d_es.create_Z_linkage(sel_group_levels)
        sel_node_ids = np.arange(0, len(node_ids))
        ddata = self.d_es.plot_dendrogram(Z=sel_Z, group_levels=sel_group_levels, node_ids=sel_node_ids,
                                          colors=color_arr, labels=label_arr, title=title,  color_branch=False)
        # ddata= self.d_es.plot_dendrogram(Z=sel_Z, group_levels=sel_group_levels,)
        return sel_Z, node_ids


# %%
if __name__ == "__main__":
    cluster = True
    grid_step = 2.5
    vname = 'pr'
    num_cpus = 64
    num_jobs = 16
    job_id = 8
    if os.getenv("HOME") == '/home/goswami/fstrnad80':
        dirname = "/home/goswami/fstrnad80/data/GPCP/"
    else:
        dirname = "/home/strnad/data/GPCP/"
        # dirname = "/home/jakob/climate_data/local/"
    network_file = PATH + \
        f"/../outputs/{vname}_link_bundle_ES_net_{grid_step}.npz"
    fname = dirname + "gpcp_daily_1996_2020_2p5_new.nc4"
    compare_centroids = None

    # %%
    job_id = 0
    sbm_filepath = PATH + f"/graphs/{job_id}_{vname}_graph_tool_ES_{grid_step}"
    group_levels = np.load(
        sbm_filepath+'_group_levels.npy',  allow_pickle=True)
    ds = Monsoon_Region_ES(fname,
                           network_file=network_file,
                           var_name=vname,
                           group_levels=group_levels,
                           time_range=['1998-01-01', '2019-01-01'],
                           grid_step=grid_step)
    monsoon_dict = ds.monsoon_dictionary

    # Rep Ids visualization
    n_idx_list = []
    for name, mregion in ds.monsoon_dictionary.items():
        n_idx_list.append(mregion['rep_ids'])
    loc_map = ds.get_map(ds.flat_idx_array(np.array(n_idx_list).flatten()))
    ax = ds.plot_map(loc_map, color='Reds', title=None, bar=False)
    ds.plot_contour_map(ds.ee_def, color='Blues', ax=ax,
                        bar=False, fill_out=False)

    sel_Z, node_ids = ds.plot_monsoon_paths()

    # %%

    # Common grid Ids
    mids_india = monsoon_dict['India South Asia']['rep_ids']
    mids_nafrica = monsoon_dict['North Africa']['rep_ids']
    mids_namerica = monsoon_dict['North America']['rep_ids']
    mids_samerica = monsoon_dict['South America']['rep_ids']
    mids_easia = monsoon_dict['East Asia']['rep_ids']
    mids = np.array([mids_namerica, mids_india, mids_nafrica, mids_easia])
    # c_Zid=ds.d_es.d_tree.get_split_groups(mids)

    c_Zid, m_ids = ds.get_ms_cluster_ids(
        ['India South Asia', 'North Africa', 'North America'], abs_th=5, rel_th=0.6)
    ds.plot_Z_id(c_Zid)

    hem = 'NH'
    c_Zid = ds.get_mscluster_sel(
        sel=hem, key_ids='rep_ids', abs_th=5, rel_th=0.6)
    ds.plot_Z_id(c_Zid)

    # %%
    i_dict = ds.d_es.d_tree.get_intersect_node_ids(sel_Z)
    ds.d_es.d_tree.plot_hist_is_nodes(n_idx_list)

    # %%
    hem = 'SH'
    for idx, job_id in enumerate(range(0, 30)):

        sbm_filepath = PATH + \
            f"/graphs/{job_id}_{vname}_graph_tool_ES_{grid_step}"
        if not os.path.exists(sbm_filepath + '_group_levels.npy'):
            print(
                f"WARNING file {sbm_filepath +'_group_levels.npy'} does not exist!")
            continue
        group_levels = np.load(
            sbm_filepath+'_group_levels.npy',  allow_pickle=True)

        print('Loading Data')
        ng_last_level = np.unique(group_levels[-1])
        ng = len(ng_last_level)
        if ng > 1:
            print(f"WARNING! Last level group levels not correct! NG: {ng}!")
            continue

        ds = Monsoon_Region_ES(fname, var_name=vname,
                               network_file=network_file,
                               group_levels=group_levels,
                               time_range=['1997-01-01', '2019-01-01'],
                               grid_step=grid_step)

        monsoon_dict = ds.get_monsoon_dict()
        c_Zid = ds.get_mscluster_hem(
            hem=hem, key_ids='rep_ids', abs_th=5, rel_th=0.6)

        savepath = PATH + \
            f"/../plots/monsoon_cluster/{job_id}_cluster_monsoons_{hem}.png"
        ds.plot_Z_id(c_Zid, title=f"{hem} #{job_id}", savepath=savepath)
