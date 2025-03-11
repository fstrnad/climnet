#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 08:53:08 2020
Class for network of rainfall events
@author: Felix Strnad
"""
# %%
from climnet.datasets.dataset import EvsDataset
import sys
import os
import numpy as np
import xarray as xr
import climnet.utils.time_utils as tu
import climnet.utils.general_utils as gut
PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PATH + "/../")  # Adds higher directory


# %%
"""Class of monsoon regions, containing the
monsoon definitions, the node ids and the regional monsoons."""


class Monsoon_Region(EvsDataset):
    """ Dataset for surface pressure.

    Args:
    ----------
    nc_file: str
        filename
    var_name: str
        Variable name of interest
    """

    def __init__(
        self,
        var_name=None,
        data_nc=None,
        load_nc=None,
        time_range=None,
        month_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="gaussian",
        lsm=False,
        large_ds=False,
        abs_th_wang=2,
        abs_th_ee=50,
        rel_th=0.55,
        nn_rep_ids=4,
        can=False,
        **kwargs,
    ):

        super().__init__(
            data_nc=data_nc,
            load_nc=load_nc,
            var_name=var_name,
            time_range=time_range,
            lon_range=lon_range,
            lat_range=lat_range,
            grid_step=grid_step,
            grid_type=grid_type,
            month_range=month_range,
            large_ds=large_ds,
            lsm=lsm,
            can=can,
            **kwargs,
        )

        # Define certain monsoon region roughly based on their lon-lat range
        self.boreal_winter = ("Dec", "Mar")
        self.boreal_summer = ("Jun", "Sep")
        self.monsoon_south_africa = [
            r"South Africa",
            (-30, -5),
            (0, 60),
            "tab:blue",
            "South_Africa",
        ]
        self.monsoon_north_africa = [
            r"North Africa",
            (0, 30),
            (-20, 60),
            "tab:orange",
            "North_Africa",
        ]
        self.monsoon_east_asia = [
            r"East Asia",
            (20, 35),
            (112, 120),
            "tab:green",
            "East_Asia",
        ]
        self.monsoon_india_south_asia = [
            r"India South Asia",
            (10, 20),
            (70, 120),
            "y",
            "India_South_Asia",
        ]
        self.monsoon_india = [r"India", (10, 20), (70, 90), "tab:red", "India"]
        self.monsoon_south_india = [
            r"South India",
            (10, 20),
            (70, 85),
            "tab:blue",
            "South_India",
        ]
        self.monsoon_north_india = [
            r"North India",
            (20, 30),
            (72, 85),
            "cyan",
            "North_India",
        ]
        self.monsoon_south_asia = [
            r"South Asia",
            (10, 35),
            (70, 120),
            "m",
            "South_Asia",
        ]
        self.monsoon_south_america = [
            r"South America",
            (-0, -30),
            (-80, -40),
            "magenta",
            "South_America",
        ]
        self.monsoon_nourth_america = [
            r"North America",
            (5, 30),
            (-120, -80),
            "tab:cyan",
            "North_America",
        ]
        self.monsoon_australia = [
            r"Australia",
            (-5, -30),
            (100, 140),
            "tab:olive",
            "Australia",
        ]
        self.monsoon_central_west_pacific = [
            r"CWP",
            (20, 30),
            (120, 130),
            "blue",
            "Central_West_Pacific",
        ]
        self.monsoon_central_pacific = [
            r"CP",
            (-5, -30),
            (-120, 160),
            "green",
            "Central_Pacific",
        ]
        self.itcz_tropics = [r"Tropics", (-5, 5), (10, 40), "tab:brown", "Tropics"]

        self.init_monsoon_dict(
            abs_th_wang=abs_th_wang,
            abs_th_ee=abs_th_ee,
            rel_th=rel_th,
            nn_rep_ids=nn_rep_ids,
        )

    def re_init(self):
        self.init_monsoon_dict()
        return None

    def init_monsoon_dict(self, abs_th_wang=2, abs_th_ee=50, rel_th=0.55, nn_rep_ids=4):
        full_year = tu.is_full_year(self.ds)
        if full_year:
            self.wang_def, _, _, _ = self.monsoon_regions(
                dtype="pr", abs_th=abs_th_wang, rel_th=rel_th
            )
            self.ee_def, _, _, _ = self.monsoon_regions(
                dtype="evs", abs_th=abs_th_ee, rel_th=rel_th
            )
        self.monsoon_dictionary = self.get_monsoon_dict(
            full_year=full_year, nn_rep_ids=nn_rep_ids
        )

    def get_sel_mdict(self, hem=None, m_arr=None):

        m_dict = dict()
        if m_arr is None:
            if hem == "NH":
                m_arr = ["North America", "North Africa", "India", "East Asia"]
            elif hem == "SH":
                m_arr = ["South America", "South Africa", "Australia"]
            elif hem == "Pacific":
                m_arr = ["CP", "CWP"]
            else:
                raise ValueError(f"This hemisphere {hem} does not exist!")
        for key in m_arr:
            m_dict[key] = self.monsoon_dictionary[key]
        return m_dict

    def summer_data(self, data):
        NH_data = self.get_month_range_data(
            data, start_month=self.boreal_summer[0], end_month=self.boreal_summer[1]
        )
        SH_data = self.get_month_range_data(
            data, start_month=self.boreal_winter[0], end_month=self.boreal_winter[1]
        )
        return NH_data, SH_data

    def winter_data(self, data):
        NH_data = self.get_month_range_data(
            data, start_month=self.boreal_winter[0], end_month=self.boreal_winter[1]
        )
        SH_data = self.get_month_range_data(
            data, start_month=self.boreal_summer[0], end_month=self.boreal_summer[1]
        )
        return NH_data, SH_data

    def NH_SH_data(self, data, season="summer"):
        if season == "summer":
            NH_data, SH_data = self.summer_data(data)
        elif season == "winter":
            NH_data, SH_data = self.winter_data(data)
        else:
            raise ValueError("The season {season} does not exist!")
        return NH_data, SH_data

    def compute_yearly_sum(self, dataarray):
        time_resample = "1Y"
        data_sum = dataarray.resample(time=time_resample).sum()
        data_sum_mean = data_sum.mean(dim="time")
        return data_sum_mean

    def rel_fraction(self, data, full_year):

        av_year_sum = self.compute_yearly_sum(full_year)
        av_data_sum = self.compute_yearly_sum(data)

        rel_map = av_data_sum / av_year_sum

        return rel_map

    def get_diff_rel_maps(self, data, season="summer", dtype="pr"):
        full_year = data
        NH_data, SH_data = self.NH_SH_data(data, season=season)

        if dtype == "pr":
            NH_type = NH_data.mean(dim="time")
            SH_type = SH_data.mean(dim="time")
        elif dtype == "evs":
            NH_type = NH_data.where(NH_data > 0).count(dim="time")
            SH_type = SH_data.where(SH_data > 0).count(dim="time")
        else:
            raise ValueError(f"Data type {dtype} not known!")

        # Get Difference between Summer and Winter
        diff_map = NH_type - SH_type

        # Get relative difference for 55%
        rel_map_NH = self.rel_fraction(NH_data, full_year)
        rel_map_SH = self.rel_fraction(SH_data, full_year)

        rel_map_combined = xr.where(
            abs(rel_map_NH) > abs(rel_map_SH), abs(rel_map_NH), rel_map_SH
        )

        return diff_map, rel_map_combined, rel_map_NH, rel_map_SH

    def monsoon_regions(self, dtype="pr", abs_th=2, rel_th=0.55):
        if dtype == "pr":
            data = self.ds[dtype]
        elif dtype == "evs":
            data = self.ds[dtype]
            print(f"Use abs th as {abs_th}")
        else:
            raise ValueError(f"Data type {dtype} not known!")

        diff_map, rel_map, _, _ = self.get_diff_rel_maps(
            data, season="summer", dtype=dtype
        )

        # Get difference above absolute threshold
        diff_map = xr.where(abs(diff_map) > abs_th, (diff_map), np.nan)
        # Get relative difference for 55%
        rel_map = xr.where(abs(rel_map) > rel_th, abs(rel_map), np.nan)

        # Now combine diff_map and rel_map
        m_def = xr.where((rel_map > 0) & (abs(diff_map) > 0), 1, np.nan)
        diff_rel_map = xr.where((rel_map > 0) & (abs(diff_map) > 0), diff_map, 0)

        return m_def, diff_rel_map, diff_map, rel_map

    def vis_annomalous_regions(self, data1, data2):
        map0 = data1 * data2
        map1 = (1 - data1) * data2
        map2 = (1 - data2) * data1
        map3 = (1 - data1) * (1 - data2)

        labels = [
            "RF_pr*RF_ee",
            "(1-RF_pr)*RF_ee",
            "(1-RF_ee)*RF_pr",
            "(1-RF_pr)*(1-RF_ee)",
        ]

        return [map0, map1, map2, map3], labels

    def get_monsoon_dict(self, full_year=True, nn_rep_ids=4):
        print("Compute Monsoon Dictionary!")
        monsoon_dictionary = dict()
        for monsoon in [
            self.monsoon_north_africa,
            self.monsoon_south_africa,
            self.monsoon_nourth_america,
            self.monsoon_south_america,
            self.monsoon_south_asia,
            self.monsoon_australia,
            # self.monsoon_india_south_asia, self.monsoon_india,
            # self.monsoon_north_india, self.monsoon_south_india,
            # self.monsoon_east_asia,
            # self.monsoon_central_pacific, self.monsoon_central_west_pacific
        ]:

            this_monsoon_dictionary = dict()
            for idx, item in enumerate(
                ["name", "lat_range", "lon_range", "color", "sname"]
            ):
                this_monsoon_dictionary[item] = monsoon[idx]

            monsoon_dictionary[monsoon[0]] = this_monsoon_dictionary

        for name, monsoon in monsoon_dictionary.copy().items():
            lat_range = monsoon["lat_range"]
            lon_range = monsoon["lon_range"]

            if (gut.check_range(lon_range, self.lon_range)) and (
                gut.check_range(lat_range, self.lat_range)
            ):

                if full_year is True:
                    wang_monsoon_regions = self.wang_def
                    ee_monsoon_regions = self.ee_def
                    monsoon["node_ids_wang"], _ = self.get_idx_region(
                        monsoon, wang_monsoon_regions
                    )
                    monsoon["node_ids_ee"], _ = self.get_idx_region(
                        monsoon, ee_monsoon_regions
                    )
                    m_ids_lst = monsoon["node_ids_ee"]
                else:  # Choose only based on lat lon range!
                    m_ids_lst, _ = self.get_idx_region(monsoon, def_map=None)

                if len(m_ids_lst) > 0:
                    # Representative Ids for every location
                    mean_loc = self.get_mean_loc(m_ids_lst)
                    print(f"Name: {name}, loc{mean_loc}")
                    slon, slat = mean_loc
                    idx_loc = self.get_index_for_coord(lon=slon, lat=slat)
                    if np.isnan(idx_loc):
                        print(
                            f"WARNING! Monsoon regions Rep IDs for {name} not defined!"
                        )
                    else:
                        rep_ids = self.get_n_ids(mean_loc, num_nn=nn_rep_ids)
                        monsoon["rep_ids"] = np.array(rep_ids)
                        monsoon["loc"] = mean_loc
                else:
                    print(f"WARNING! Monsoon regions for {name} not defined!")
            else:
                del monsoon_dictionary[name]

        if not monsoon_dictionary:
            raise ValueError("ERROR! The monsoon dictionary is empty!")

        return monsoon_dictionary

    def get_m_ids(self, mname, defn="ee"):
        if defn == "ee":
            m_node_ids = self.monsoon_dictionary[mname]["node_ids_ee"]
        elif defn == "wang":
            m_node_ids = self.monsoon_dictionary[mname]["node_ids_wang"]
        else:
            raise ValueError(f"This definition for ids does not exist: {defn}!")
        return m_node_ids
