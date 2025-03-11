import numpy as np
import xarray as xr
import os
from importlib import reload
import climnet.datasets.dataset as cds
import geoutils.utils.general_utils as gut
import geoutils.utils.time_utils as tu
import copy
reload(cds)


class EvsDataset(cds.BaseDataset):
    def __init__(
        self,
        data=None,
        data_nc=None,
        load_nc=None,
        var_name=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="gaussian",
        large_ds=False,
        q=0.95,
        min_evs=1,
        min_treshold=1,
        th_eev=15,
        rrevs=False,
        can=False,
        timemean=None,
        month_range=None,
        **kwargs,
    ):

        super().__init__(
            data=data,
            data_nc=data_nc,
            load_nc=load_nc,
            var_name=var_name,
            time_range=time_range,
            lon_range=lon_range,
            lat_range=lat_range,
            grid_step=grid_step,
            grid_type=grid_type,
            large_ds=large_ds,
            timemean=timemean,
            can=can,
            **kwargs,
        )

        self.q = q
        self.min_evs = min_evs
        self.min_treshold = min_treshold
        self.th_eev = th_eev

        # Event synchronization
        rrevs = kwargs.pop("rrevs", False)
        # compute event synch if not given in nc file
        if "evs" in self.vars:
            gut.myprint("Evs datat is stored in dataset.")
        elif self.var_name is None:
            raise ValueError("Specify varname to compute event sync.")
        else:
            gut.myprint(
                f"Compute Event synchronization for variable {self.var_name}.",
            )
            rrevs = True
        if rrevs is True:
            if self.var_name is None:
                var_name = self.var_name

            self.ds = self.create_evs_ds(
                var_name=var_name,
                min_threshold=self.min_treshold,
                q=self.q,
                min_evs=self.min_evs,
                th_eev=self.th_eev,
                month_range=month_range
            )
        else:
            self.mask = self.get_es_mask(self.ds["evs"], min_evs=self.min_evs)
            self.init_map_indices()
        self.vars = self.get_vars()
        self.var_name = 'evs'

    def create_evs_var_q(self, var_name=None, *qs):
        """Creates event time series for a list of passed q values.

        Raises:
            ValueError: if q is not float

        Returns:
            dataset: xr.DataSet with new q time series stored as variables!
        """
        for q in qs:
            if not isinstance(q, float):
                raise ValueError(f"{q} has to be float")
            self.q = q
            da_es, _ = self.compute_event_time_series(
                var_name=var_name,
                min_threshold=self.min_treshold,
                q=q,
                min_evs=self.min_evs,
                th_eev=self.th_eev,
            )
            self.set_ds_attrs_evs(ds=da_es)
            self.ds[f"evs_q{q}"] = da_es
            self.set_ds_attrs_evs(ds=self.ds)  # set also to dataset object the attrs.

        return self.ds

    def get_es_mask(self, data_evs, min_evs):
        gut.myprint(f'Init spatial evs-mask for EVS data of shape: {data_evs.shape}')
        num_non_nan_occurence = data_evs.where(data_evs == 1).count(dim="time")
        mask = xr.where(num_non_nan_occurence > min_evs, 1, 0)
        self.min_evs = min_evs
        self.mask = xr.DataArray(
                data=mask,
                dims=mask.dims,
                coords=mask.coords,
                name="mask",
            )
        gut.myprint(f'... Finished Initialization EVS-spatial mask')

        return self.mask

    def randomize_spatio_temporal_data_yearly(
        self,
        data,
        var=None,
        start_year=None,
        end_year=None,
        sm_arr=["Jan"],
        em_arr=["Dec"],
        set_rest_zero=False,
        full_rnd=False,
        seed=0,
    ):
        """
        Permutates randomly time series for every grid location.
        Keeps the year in which the events did occur.
        """

        if len(sm_arr) != len(em_arr):
            raise ValueError(
                "ERROR! Start month array and end month array not of the same length!"
            )
        if len(sm_arr) > 1 and set_rest_zero is True:
            raise ValueError(
                "Set Time Zeros 0 and shuffle for to time periods is not possible!"
            )
        times = self.ds["time"]
        if start_year is None:
            start_year = int(times[0].time.dt.year)
        if end_year is None:
            end_year = int(times[-1].time.dt.year) + 1
        if var is None:
            var = self.var

        with gut.temp_seed():
            if full_rnd is True:
                gut.myprint(f"WARNING! Fully randomized time Series of {var}!")

                start_date = times.data[0]
                end_date = times.data[-1]
                arr_data = data.sel(time=slice(start_date, end_date))
                arr_rnd = self.randomize_spatio_temporal_data_full(
                    arr_data.data, axis=0
                )
                data.loc[dict(time=slice(start_date, end_date))] = arr_rnd
            else:
                for idx in range(len(sm_arr)):
                    sm = sm_arr[idx]
                    em = em_arr[idx]
                    gut.myprint(
                        f"WARNING! Time Series of {var} are for {sm} to {em} randomized!"
                    )

                    for idx, year in enumerate(np.arange(start_year, end_year)):

                        gut.myprint(
                            f"Shuffle Year {year} for months {sm}, {em}")
                        smi = self._get_index_of_month(sm) + 1
                        emi = self._get_index_of_month(em) + 1
                        start_date = f"{smi}-01-{year}"
                        if em == "Feb":
                            end_day = 28
                        elif em in ["Jan", "Mar", "May", "Jul", "Aug", "Oct", "Dec"]:
                            end_day = 31
                        else:
                            end_day = 30

                        ey = copy.deepcopy(year)
                        if emi < smi:
                            ey = year + 1
                        end_date = f"{emi}-{end_day}-{ey}"
                        if emi < 10:
                            end_date = f"0{emi}-{end_day}-{ey}"

                        arr_1_year = data.sel(time=slice(start_date, end_date))
                        # arr_1_year_rnd=np.random.permutation(arr_1_year.data)
                        arr_1_year_rnd = self.randomize_spatio_temporal_data_full(
                            arr_1_year.data, axis=0
                        )

                        arr_1_year.data = arr_1_year_rnd
                        # if idx == 0:
                        #     all_year = arr_1_year
                        # else:
                        #     all_year = xr.merge([all_year, arr_1_year])
                        data.loc[
                            dict(time=slice(start_date, end_date))
                        ] = arr_1_year_rnd

                        if set_rest_zero is True:
                            gut.myprint("Warning: Set Rest to Zero!")
                            if emi >= smi:  # E.g. for Jun-Sep
                                start_date_before = f"01-01-{year}"
                                end_data_after = f"12-31-{year}"
                                rest_data_before = data.sel(
                                    time=slice(start_date_before, start_date)
                                )
                                rest_data_after = data.sel(
                                    time=slice(end_date, end_data_after)
                                )
                                data_before = xr.zeros_like(rest_data_before)
                                data_after = xr.zeros_like(rest_data_after)

                                data.loc[
                                    dict(time=slice(start_date_before, start_date))
                                ] = data_before
                                data.loc[
                                    dict(time=slice(end_date, end_data_after))
                                ] = data_after
                            else:
                                # Year not ey!
                                end_date = f"{emi}-{end_day}-{year}"
                                if emi < 10:
                                    end_date = f"0{emi}-{end_day}-{year}"
                                rest_data_between = data.sel(
                                    time=slice(end_date, start_date)
                                )
                                data_between = xr.zeros_like(rest_data_between)
                                data.loc[
                                    dict(time=slice(end_date, start_date))
                                ] = data_between

                # only_data = all_year[var]
        return data

    def randomize_spatio_temporal_data_full(self, a, axis=0, seed=None):
        """
        Shuffle `a` in-place along the given axis.
        Code mainly from
        https://stackoverflow.com/questions/26310346/quickly-calculate-randomized-3d-numpy-array-from-2d-numpy-array/
        Apply numpy.random.shuffle to the given axis of `a`.
        Each one-dimensional slice is shuffled independently.
        """
        if seed is not None:
            np.random.seed(seed)
        b = a.swapaxes(axis, -1)
        # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
        # so `a` is shuffled in place, too.
        shp = b.shape[:-1]
        for ndx in np.ndindex(shp):
            np.random.shuffle(b[ndx])
        return a
