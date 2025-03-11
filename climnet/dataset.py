# %%
"""
Base class for the different dataset classes of the multilayer climate network.
"""

import os
import numpy as np
import scipy.interpolate as interp
import xarray as xr
import copy
import climnet.grid.grid as grid
import climnet.utils.general_utils as gut
import climnet.utils.time_utils as tu
import climnet.utils.spatial_utils as sput
from importlib import reload
from tqdm import tqdm


class BaseDataset:
    """ Base Dataset.
    Args:

    ----------
    data_nc: str
        Filename of input data, Default: None
    load_nc: str
        Already processed data file. Default: None
        If specified all other attritbutes are loaded.
    time_range: list
        List of time range, e.g. ['1997-01-01', '2019-01-01']. Default: None
    lon_range: list
        Default: [-180, 180],
    lat_range: list
        Default: [-90,90],
    grid_step: int
        Default: 1
    grid_type: str
        Default: 'gaussian',
    lsm: bool
        Default:False,
    **kwargs
    """

    def __init__(
        self,
        var_name=None,
        data_nc=None,
        load_nc=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="fekete",
        lsm=False,
        large_ds=False,
        timemean=None,
        detrend=False,
        **kwargs,
    ):

        if data_nc is not None and load_nc is not None:
            raise ValueError("Specify either data or load file.")

        # initialized dataset
        elif data_nc is not None:
            # check if file exists
            if not os.path.exists(data_nc):
                PATH = os.path.dirname(os.path.abspath(__file__))
                gut.myprint(f"You are here: {PATH}!")
                raise ValueError(f"File does not exist {data_nc}!")

            if large_ds is True:
                ds = self.open_large_ds(
                    var_name=var_name,
                    data_nc=data_nc,
                    time_range=time_range,
                    grid_step=grid_step,
                    **kwargs,
                )
            else:
                ds = xr.open_dataset(data_nc)

            ds = self.check_dimensions(ds, **kwargs)  # check dimensions
            ds = self.rename_var(ds)  # rename specific variable names
            self.grid_step = grid_step
            self.grid_type = grid_type
            self.lsm = lsm
            self.info_dict = kwargs

            # choose time range
            if time_range is not None:
                ds = self.get_data_timerange(ds, time_range)

            if timemean is not None:
                ds = tu.apply_timemean(ds, timemean=timemean)

            # regridding
            self.GridClass = self.create_grid(
                grid_type=self.grid_type, **kwargs,)
            if lon_range != [-180, 180] and lat_range != [-90, 90]:
                ds = self.cut_map(ds=ds, lon_range=lon_range,
                                  lat_range=lat_range)

            self.grid = self.GridClass.cut_grid(
                lat_range=[ds["lat"].min().data, ds["lat"].max().data],
                lon_range=[ds["lon"].min().data, ds["lon"].max().data],
            )

            da = ds[var_name]
            # Bring always in the form (time, lat, lon)
            # much less memory consuming than for dataset!
            gut.myprint("transpose data!")
            da = da.transpose("time", "lat", "lon")
            da = self.interp_grid(da, self.grid)

            if large_ds is True:
                ds.unify_chunks()

            if self.lsm is True:
                self.mask, da = self.get_land_sea_mask_data(da)
            else:
                # self.mask = xr.DataArray(
                #     data=np.ones_like(da[0].data),
                #     dims=da.sel(time=da.time[0]).dims,
                #     coords=da.sel(time=da.time[0]).coords,
                #     name="mask",
                # )
                self.mask = self.init_mask(da=da)
            self.ds = da.to_dataset(name=var_name)
            (
                self.time_range,
                self.lon_range,
                self.lat_range,
            ) = self.get_spatio_temp_range(ds)

        # load dataset object from file
        elif load_nc is not None:
            self.load(load_nc)
            if timemean is not None:
                self.ds = self.apply_timemean(timemean=timemean)
            if time_range is not None:
                self.ds = self.get_data_timerange(
                    self.ds, time_range=time_range)
                self.time_range = time_range

        # select a main var name
        self.vars = list(self.ds.keys())
        self.var_name = var_name if var_name is not None else self.vars[0]

        # detrending
        if detrend is True:
            detrend_from = kwargs.pop('detrend_from', None)
            self.detrend(dim="time", startyear=detrend_from)

        # Flatten index in map
        # Predefined variables set to None
        init_indices = kwargs.pop('init_indices', True)
        if init_indices:
            self.indices_flat, self.idx_map = self.init_map_indices()

        if load_nc is None:
            self.ds = self.ds.assign_coords(
                idx_flat=("points", self.idx_map.data))

        self.loc_dict = dict()

    def re_init(self):
        return None

    def open_large_ds(self, var_name, data_nc, time_range, grid_step, **kwargs):
        sp_large_ds = kwargs.pop("sp_large_ds", None)
        if not os.path.exists(sp_large_ds):
            ds = self.preprocess_large_ds(
                nc_file=data_nc,
                var_name=var_name,
                time_range=time_range,
                grid_step=grid_step,
                sp_large_ds=sp_large_ds,
                **kwargs,
            )
            gut.myprint("Finished preprocessing large dataset...")
        else:
            gut.myprint(
                f"Compressed file {sp_large_ds} already exists! Read now!"
            )
        ds = xr.open_dataset(sp_large_ds)

        return ds

    def load(self, load_nc):
        """Load dataset object from file.

        Parameters:
        ----------
        ds: xr.Dataset
            Dataset object containing the preprocessed dataset

        """
        # check if file exists
        gut.myprint("Loading Data...")

        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            gut.myprint(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        gut.myprint(f"Load Dataset: {load_nc}")
        # set lock to false to allow running in parallel
        with xr.open_dataset(load_nc, lock=False) as ds:
            (
                self.time_range,
                self.lon_range,
                self.lat_range,
            ) = self.get_spatio_temp_range(ds)

            self.grid_step = ds.attrs["grid_step"]
            self.grid_type = ds.attrs["grid_type"]
            self.lsm = bool(ds.attrs["lsm"])
            self.info_dict = ds.attrs  # TODO
            # Read and create grid class
            self.grid = dict(lat=ds.lat.data, lon=ds.lon.data)
            for name, da in ds.data_vars.items():
                gut.myprint(f"Variables in dataset: {name}")

            # points which are always NaN will be NaNs in mask
            mask = np.ones_like(ds[name][0].data, dtype=bool)
            for idx, t in enumerate(ds.time):
                mask *= np.isnan(ds[name].sel(time=t).data)

            self.mask = xr.DataArray(
                data=xr.where(mask == 0, 1, np.NaN),  # or mask == False
                dims=da.sel(time=da.time[0]).dims,
                coords=da.sel(time=da.time[0]).coords,
                name="lsm",
            )

            ds = self.check_time(ds)

            self.ds = ds

        return self.ds

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        if os.path.exists(filepath):
            gut.myprint("File" + filepath + " already exists!")
            os.rename(filepath, filepath + "_backup")

        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            **self.info_dict,
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class
        ds_temp.to_netcdf(filepath)
        gut.myprint(f"File {filepath} written!")
        return None

    def check_dimensions(self, ds, **kwargs):
        """
        Checks whether the dimensions are the correct ones for xarray!
        """
        sort = kwargs.pop('sort', True)
        ds = gut.check_dimensions(ds=ds, sort=sort)
        # Set time series to days
        ds = self.check_time(ds, **kwargs)

        return ds

    def check_time(self, ds, **kwargs):
        ts_days = kwargs.pop("ts_days", True)
        if ts_days:
            if not gut.is_datetime360(time=ds.time.data[0]):
                ds = ds.assign_coords(
                    time=ds.time.data.astype("datetime64[D]"))
                self.calender360 = False
            else:
                gut.myprint('WARNING: 360 day calender is used!')
                self.calender360 = True

        return ds

    def preprocess_large_ds(
        self,
        nc_file,
        var_name,
        time_range=None,
        grid_step=1,
        sp_large_ds=None,
        **kwargs,
    ):
        gut.myprint("Start preprocessing data!")

        ds_large = xr.open_dataset(nc_file, chunks={"time": 100})
        ds_large = self.check_dimensions(ds_large, **kwargs)
        ds_large = self.rename_var(ds_large)
        da = ds_large[var_name]
        da = da.transpose("time", "lat", "lon")
        da = self.get_data_timerange(da, time_range)
        if max(da.lon) > 180:
            gut.myprint("Shift longitude in Preprocessing!")
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = self.common_grid(dataarray=da, grid_step=grid_step)
        ds_large.unify_chunks()

        ds = da.to_dataset(name=var_name)
        gut.myprint("Finished preprocessing data")

        if sp_large_ds is not None:
            dirname = os.path.dirname(sp_large_ds)
            if not os.path.exists(dirname):
                os.mkdir(dirname)
            ds.to_netcdf(sp_large_ds)
            gut.myprint(f"Save processed data to file {sp_large_ds}!")

        return ds

    def rename_var(self, ds):
        names = []
        for name, da in ds.data_vars.items():
            names.append(name)

        if "precipitation" in names:
            ds = ds.rename({"precipitation": "pr"})
            gut.myprint("Rename precipitation: pr!")
        if "tp" in names:
            ds = ds.rename({"tp": "pr"})
            gut.myprint("Rename tp: pr!")

        if "p86.162" in names:
            ds = ds.rename({"p86.162": "vidtef"})
            gut.myprint(
                "Rename vertical integral of divergence of total energy flux to: vidtef!"
            )
        if "p71.162" in names:
            ds = ds.rename({"p71.162": "ewvf"})
            gut.myprint(
                "Rename vertical integral of eastward water vapour flux to: ewvf!")

        if "p72.162" in names:
            ds = ds.rename({"p72.162": "nwvf"})
            gut.myprint(
                "Rename vertical integral of northward water vapour flux to: ewvf!")

        if "ttr" in names:
            ds = ds.rename({"ttr": "olr"})
            gut.myprint(
                "Rename top net thermal radiation (ttr) to: olr!\n Multiply by -1!")
            ds['olr'] *= -1
        return ds

    def create_grid(self, grid_type="fibonacci", num_iter=1000, **kwargs):
        """Common grid for all datasets.

        ReturnL
        -------
        Grid: grid.BaseGrid
        """
        reload(grid)
        dist_equator = grid.degree2distance_equator(self.grid_step)
        sp_grid = kwargs.pop("sp_grid", None)
        gut.myprint(f"Start create grid {grid_type}...")
        if grid_type == "gaussian":
            Grid = grid.GaussianGrid(self.grid_step, self.grid_step)
        elif grid_type == "fibonacci":
            Grid = grid.FibonacciGrid(dist_equator)
        elif grid_type == "fekete":
            num_points = grid.get_num_points(dist_equator)
            Grid = grid.FeketeGrid(
                num_points=num_points,
                num_iter=num_iter,
                pre_proccess_type=None,
                load_grid=sp_grid,
            )
        else:
            raise ValueError(f"Grid type {grid_type} does not exist.")

        return Grid

    def interp_grid(self, dataarray, new_grid):
        """Interpolate dataarray on new grid.
        dataarray: xr.DataArray
            Dataarray to interpolate.
        new_grid: dict
            Grid we want to interpolate on.
        """
        new_points = np.array([new_grid["lon"], new_grid["lat"]]).T
        lon_mesh, lat_mesh = np.meshgrid(dataarray.lon, dataarray.lat)
        origin_points = np.array([lon_mesh.flatten(), lat_mesh.flatten()]).T
        # for one timestep
        if len(dataarray.data.shape) < 3:
            origin_values = dataarray.data.flatten()
            assert len(origin_values) == origin_points.shape[0]
            new_values = interp.griddata(
                origin_points, origin_values, new_points, method="nearest"
            )
            new_values = np.array(new_values).T
            coordinates = dict(
                points=np.arange(0, len(new_points), 1),
                lon=("points", new_points[:, 0]),
                lat=("points", new_points[:, 1]),
            )
            dims = ["points"]
        else:
            new_values = []
            pb_fmt = "{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}"
            pb_desc = "Intpol Grid points in time..."
            for idx, t in enumerate(
                tqdm(dataarray.time, bar_format=pb_fmt, desc=pb_desc)
            ):
                origin_values = dataarray.sel(time=t.data).data.flatten()
                assert len(origin_values) == origin_points.shape[0]
                new_values.append(
                    interp.griddata(
                        origin_points, origin_values, new_points, method="nearest"
                    )
                )
            # coordinates = dict(time=dataarray.time.data,
            #                    points=np.arange(0, len(new_points), 1),
            #                    lon=("points", new_points[:, 0]),
            #                    lat=("points", new_points[:, 1]))
            coordinates = sput.create_new_coordinates(
                times=dataarray.time.data,
                lons=new_points[:, 0], lats=new_points[:, 1]
            )
            dims = ["time", "points"]
            new_values = np.array(new_values)

        new_dataarray = xr.DataArray(
            data=new_values, dims=dims, coords=coordinates, name=dataarray.name
        )

        return new_dataarray

    def common_grid(self, dataarray, grid_step=1):
        """Common grid for all datasets.
        """
        # min_lon = min(lon_range)
        # min_lat = min(lat_range)
        # Use minimum of original dataset because other lower variables aren't defined
        min_lat = float(np.min(dataarray["lat"]))  # without np. much slower!
        min_lon = float(np.min(dataarray["lon"]))

        max_lat = float(np.max(dataarray["lat"]))
        max_lon = float(np.max(dataarray["lon"]))
        if np.abs(180 - max_lon)-0.01 > grid_step:  # To avoid scenarios with big gap
            gut.myprint(f'WARNING: Max lon smaller than 180-{grid_step}!')
        if max_lon < 179 and max_lon > 175:  # To avoid scenarios with big gap
            gut.myprint(f'WARNING! Set max lon from {max_lon} to 179.75!')
            max_lon = 179.75
        if min_lon == -180 and max_lon == 180:  # To avoid scenarios with big gap
            gut.myprint(f'WARNING! Set max lon from {max_lon} to 179.75')
            max_lon = 179.75

        # init_lat = np.arange(min_lat, max_lat, grid_step, dtype=float)
        # init_lon = np.arange(min_lon, max_lon, grid_step, dtype=float)
        init_lat = gut.crange(min_lat, max_lat, grid_step)
        init_lon = gut.crange(min_lon, max_lon, grid_step)

        nlat = len(init_lat)
        if nlat % 2:
            # Odd number of latitudes includes the poles.
            gut.myprint(
                f"WARNING: Poles might be included: {min_lat} and {min_lat}!"
            )

        grid = {"lat": init_lat, "lon": init_lon}

        gut.myprint(
            f"Interpolte grid from {float(min_lon)} to {float(max_lon)},{float(min_lat)} to {float(max_lat)}!",
        )
        da = dataarray.interp(grid, method="nearest",
                              kwargs={"fill_value": "extrapolate"}
                              )  # Extrapolate if outside of the range

        return da

    def cut_map(
        self, ds=None, lon_range=[-180, 180], lat_range=[-90, 90], dateline=False,
        set_ds=False,
    ):
        """Cut an area in the map. Use always smallest range as default.
        It lon ranges accounts for regions (eg. Pacific) that are around the -180/180 region.

        Args:
        ----------
        lon_range: list [min, max]
            range of longitudes
        lat_range: list [min, max]
            range of latitudes
        dateline: boolean
            use dateline range in longitude (eg. -170, 170 range) contains all points from
            170-180, -180- -170, not all between -170 and 170. Default is True.
        Return:
        -------
        ds_area: xr.dataset
            Dataset cut to range
        """
        if ds is None:
            ds = self.ds
        ds_cut = sput.cut_map(
            ds=ds, lon_range=lon_range, lat_range=lat_range, dateline=dateline
        )
        if set_ds:
            self.ds = ds_cut
        return ds_cut

    def get_spatio_temp_range(self, ds):
        time_range = [ds.time.data[0], ds.time.data[-1]]
        lon_range = [float(ds.lon.min()), float(ds.lon.max())]
        lat_range = [float(ds.lat.min()), float(ds.lat.max())]

        return time_range, lon_range, lat_range

    def get_land_sea_mask_data(self, dataarray):
        """
        Compute a land-sea-mask for the dataarray,
        based on an input file for the land-sea-mask.
        """
        PATH = os.path.dirname(os.path.abspath(
            __file__))  # Adds higher directory
        lsm_mask_ds = xr.open_dataset(PATH + "/../input/land-sea-mask_era5.nc")
        lsm_mask = self.interp_grid(lsm_mask_ds["lsm"], self.grid)

        land_dataarray = xr.where(np.array([lsm_mask]) == 1, dataarray, np.nan)
        return lsm_mask, land_dataarray

    def flatten_array(self, time=True, check=False):
        """Flatten and remove NaNs.
        """
        dataarray = self.ds[self.var_name]

        data = sput.flatten_array(dataarray=dataarray, mask=self.mask,
                                  time=time, check=check)

        return data

    def get_dims(self, ds=None):
        if ds is None:
            ds = self.ds
        return list(ds.dims.keys())

    def init_mask(self, da):
        num_non_nans = xr.where(~np.isnan(da), 1, 0).sum(dim='time')
        mask = xr.where(num_non_nans == len(da.time), 1, 0)
        self.mask = xr.DataArray(
            data=mask,
            dims=da.sel(time=da.time[0]).dims,
            coords=da.sel(time=da.time[0]).coords,
            name="mask",
        )

        return self.mask

    def init_map_indices(self):
        """
        Initializes the flat indices of the map.
        Usefule if get_map_index is called multiple times.
        Also defined spatial lon, lat locations are initialized.
        """
        reload(gut)
        gut.myprint('Init the point-idx dictionaries')
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        if np.count_nonzero(mask_arr) == 0:
            raise ValueError('ERROR! Mask is the whole dataset!')
        self.indices_flat = np.arange(
            0, np.count_nonzero(mask_arr), 1, dtype=int)
        self.idx_map = self.get_map(self.indices_flat, name="idx_flat")
        idx_lst_map = self.get_map_for_idx(idx_lst=self.indices_flat)
        if 'points' in self.get_dims():
            point_lst = idx_lst_map.where(
                idx_lst_map == 1, drop=True).points.data
            lons = self.idx_map.lon
            lats = self.idx_map.lat
            self.def_locs = gut.zip_2_lists(list1=lons, list2=lats)[point_lst]
        else:
            point_lst = np.where(idx_lst_map.data.flatten() == 1)[0]
            lons = self.ds.lon
            lats = self.ds.lat
            lons, lats = np.meshgrid(lons, lats)
            self.def_locs = gut.zip_2_lists(
                lons.flatten(), lats.flatten())[point_lst]

        self.key_val_idx_point_dict = gut.mk_dict_2_lists(
            self.indices_flat, point_lst)

        # This takes longer
        # def_locs = []
        # for idx in self.indices_flat:
        #     slon, slat = self.get_coord_for_idx(idx)
        #     def_locs.append([slon, slat])
        # self.def_locs = np.array(def_locs)

        return self.indices_flat, self.idx_map

    def mask_point_ids(self, points):
        """In the index list the indices are delivered as eg. nodes of a network.
        This is not yet the point number! In the mask the corresponding point numbers
        are set to 0 and the new mask is reinitialized

        Args:
            points (list): list of points.
        """
        if len(points) > 0:
            self.mask[points] = int(0)
            self.init_map_indices()
        return

    def mask_node_ids(self, idx_list):
        """In the index list the indices are delivered as eg. nodes of a network.
        This is not yet the point number! In the mask the corresponding point numbers
        are set to 0 and the new mask is reinitialized

        Args:
            idx_list (list): list of indices.
        """
        if len(idx_list) > 0:
            points = self.get_points_for_idx(idx_list)
            self.mask_point_ids(points=points)
        return

    def get_map(self, data, name=None):
        """Restore dataarray map from flattened array.

        TODO: So far only a map at one time works, extend to more than one time

        This also includes adding NaNs which have been removed.
        Args:
        -----
        data: np.ndarray (n,0)
            flatten datapoints without NaNs
        mask_nan: xr.dataarray
            Mask of original dataarray containing True for position of NaNs
        name: str
            naming of xr.DataArray

        Return:
        -------
        dmap: xr.dataArray
            Map of data
        """
        if name is None:
            name = self.var_name
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        non_zero_ds = np.count_nonzero(mask_arr)
        # Number of non-NaNs should be equal to length of data
        if np.count_nonzero(mask_arr) != len(data):
            raise ValueError(
                f"Number of defined ds points {non_zero_ds} != # datapoints {len(data)}"
            )

        # create array with NaNs
        data_map = np.empty(len(mask_arr))
        data_map[:] = np.NaN

        # fill array with sample
        data_map[mask_arr] = data

        # dmap = xr.DataArray(
        #     data=data_map,
        #     dims=['points'],
        #     coords=dict(points=self.ds.points.data,
        #                 lon=("points", self.ds.lon.data),
        #                 lat=("points", self.ds.lat.data)),
        #     name=name)

        dmap = xr.DataArray(
            data=np.reshape(data_map, self.mask.data.shape),
            dims=self.mask.dims,
            coords=self.mask.coords,
            name=name,
        )

        return dmap

    def get_map_index(self, idx_flat):
        """Get lat, lon and index of map from index of flatten array
           without Nans.

        # Attention: Mask has to be initialised

        Args:
        -----
        idx_flat: int, list
            index or list of indices of the flatten array with removed NaNs

        Return:
        idx_map: dict
            Corresponding indices of the map as well as lat and lon coordinates
        """

        indices_flat = self.indices_flat

        idx_map = self.idx_map

        buff = idx_map.where(idx_map == idx_flat, drop=True)
        if idx_flat > len(indices_flat):
            raise ValueError("Index doesn't exist.")
        map_idx = {
            "lat": float(buff.lat.data),
            "lon": float(buff.lon.data),
            "point": int(np.argwhere(idx_map.data == idx_flat)),
        }
        return map_idx

    def get_points_for_idx(self, idx_lst):
        """Returns the point number of the map for a given index list.
        Important eg. to transform node ids to points of the network
        Args:
            idx_lst (list): list of indices of the network.

        Returns:
            np.array: array of the points of the index list
        """
        point_lst = []

        for idx in idx_lst:
            # map_idx = self.get_map_index(idx)
            # point = int(map_idx["point"])
            # point_lst.append(point)
            point_lst.append(self.key_val_idx_point_dict[idx])

        return np.array(point_lst, dtype=int)

    def get_idx_for_point(self, point):
        """Gets for a point its corresponding indices

        Args:
            point (int): point number (is a dimension of)

        Raises:
            ValueError:

        Returns:
            int: index number, must be <= point
        """
        flat_idx = self.idx_map.sel(points=point).data
        if np.isnan(flat_idx) is True:
            raise ValueError(f"Error the point {point} is not defined!")

        return int(flat_idx)

    def get_idx_point_lst(self, point_lst):
        idx_lst = []
        for point in point_lst:
            idx_lst.append(self.get_idx_for_point(point=point))

        return np.array(idx_lst)

    def get_coordinates_flatten(self):
        """Get coordinates of flatten array with removed NaNs.

        Return:
        -------
        coord_deg:
        coord_rad:
        map_idx:
        """
        # length of the flatten array with NaNs removed
        # length = self.flatten_array().shape[1]
        length = len(self.indices_flat)
        coord_deg = []
        map_idx = []
        for i in range(length):
            buff = self.get_map_index(i)
            coord_deg.append([buff["lat"], buff["lon"]])
            map_idx.append(buff["point"])

        coord_rad = np.radians(coord_deg)  # transforms to np.array

        return np.array(coord_deg), coord_rad, np.array(map_idx)

    def flat_idx_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        if len(idx_list) == 0:
            raise ValueError('Error no indices in idx_lst')
        max_idx = np.max(idx_list)
        if max_idx > len_index:
            raise ValueError(
                f"Error: index {max_idx} higher than #nodes {len_index}!")
        full_idx_lst = np.zeros(len_index)
        full_idx_lst[idx_list] = 1

        return full_idx_lst

    def get_map_for_idx(self, idx_lst):
        flat_idx_arr = self.flat_idx_array(idx_list=idx_lst)
        idx_lst_map = self.get_map(flat_idx_arr)
        return idx_lst_map

    def count_indices_to_array(self, idx_list):
        """
        Returns a flattened list of indices where the idx_list is at the correct position.
        It counts the occurrence of each index in the idx_lst.
        """
        mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
        len_index = np.count_nonzero(mask_arr)
        full_idx_lst = np.zeros(len_index)
        # counts as number of occurences
        u_index, counts = np.unique(idx_list, return_counts=True)
        full_idx_lst[u_index] = counts

        return full_idx_lst

    def find_nearest(self, a, a0):
        """
        Element in nd array `a` closest to the scalar value `a0`
        ----
        Args a: nd array
             a0: scalar value
        Return
            idx, value
        """
        idx = np.abs(a - a0).argmin()
        return idx, a.flat[idx]

    def interp_times(self, dataset, time_range):
        """Interpolate time in time range in steps of days.
        TODO: So far only days works.
        """
        time_grid = np.arange(
            time_range[0], time_range[1], dtype="datetime64[D]")
        ds = dataset.interp(time=time_grid, method="nearest")
        return ds

    def get_data_timerange(self, data, time_range=None):
        """Gets data in a certain time range.
        Checks as well if time range exists in file!

        Args:
            data (xr.Dataarray): xarray dataarray
            time_range (list, optional): List dim 2 that contains the time interval. Defaults to None.

        Raises:
            ValueError: If time range is not in time range of data

        Returns:
            xr.Dataarray: xr.Dataarray in seleced time range.
        """

        td = data.time.data
        if time_range is not None:
            if (td[0] > np.datetime64(time_range[0])) or (
                td[-1] < np.datetime64(time_range[1])
            ):
                raise ValueError(
                    f"Please select time array within {td[0]} - {td[-1]}!")
            else:
                gut.myprint(f"Time steps within {time_range} selected!")
            # da = data.interp(time=t, method='nearest')
            da = data.sel(time=slice(time_range[0], time_range[1]))

            gut.myprint("Time steps selected!")
        else:
            da = data
        return da

    def get_month_range_data(
        self, dataarray=None, start_month="Jan", end_month="Dec", set_zero=False,
    ):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.
        set_zero : Sets all values outside the month range to zero, but remains the days.
                   Might be useful for event time series!

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        reload(tu)
        if dataarray is None:
            dataarray = self.ds[self.var_name]
        if set_zero:
            seasonal_data = tu.get_month_range_zero(
                dataarray=dataarray, start_month=start_month, end_month=end_month
            )
        else:
            seasonal_data = tu.get_month_range_data(
                dataset=dataarray, start_month=start_month, end_month=end_month
            )

        return seasonal_data

    def set_month_range_data(
        self, dataset=None, start_month="Jan", end_month="Dec", set_zero=False
    ):
        if dataset is None:
            dataset = self.ds
        ds_all = []
        for name, da in dataset.data_vars.items():
            gut.myprint(f"Month range for: {name}")
            ds_all.append(
                self.get_month_range_data(
                    dataarray=da,
                    start_month=start_month,
                    end_month=end_month,
                    set_zero=set_zero,
                )
            )
        self.ds = xr.merge(ds_all)

    # ####################### Spatial Location functions #####################

    def get_coord_for_idx(self, idx):
        map_dict = self.get_map_index(idx)
        slon = float(map_dict["lon"])
        slat = float(map_dict["lat"])

        return slon, slat

    def get_idx_for_loc(self, locs):
        """This is just a wrapper for self.get_index_for_coord.

        Args:
            locs (tuples): tuples of (lon, lat) pairs

        Returns:
            int: idx
        """
        locs = np.array(locs)
        if not isinstance(locs, np.ndarray):
            locs = [locs]

        if len(locs) == 0:
            raise ValueError('No locations given!')

        idx_lst = []
        for loc in locs:
            lon, lat = loc
            idx = self.get_index_for_coord(lon=lon, lat=lat)
            idx_lst.append(idx)
        if len(locs) > 1:
            idx_lst = np.sort(np.unique(np.array(idx_lst)))
            return idx_lst
        else:
            return idx

    def get_index_for_coord(self, lon, lat):
        """Get index of flatten array for specific lat, lon."""
        lons = self.def_locs[:, 0]
        lats = self.def_locs[:, 1]

        # Here we reduce the range in which the location can be
        if lon < 180 - 2*self.grid_step and lon > -180 + 2*self.grid_step:
            idx_true = ((lats > lat-2*self.grid_step) &
                        (lats < lat+2*self.grid_step) &
                        (lons > lon-2*self.grid_step) &
                        (lons < lon+2*self.grid_step)
                        )
        else:  # Close to dateline you only use lats not to run into problems
            idx_true = ((lats > lat-2*self.grid_step) &
                        (lats < lat+2*self.grid_step)
                        )
        # This is important to find the old index again!
        idx_red = np.where(idx_true)[0]

        lon, lat, idx_all_red = grid.find_nearest_lat_lon(
            lon=lon, lat=lat,
            lon_arr=lons[idx_true],
            lat_arr=lats[idx_true]
        )
        idx = idx_red[idx_all_red]
        # idx = self.idx_map.sel(points=idx_all)   # wrong

        if np.isnan(idx) is True:
            raise ValueError(f"Error the lon {lon} lat {lat} is not defined!")

        return int(idx)

    # def get_index_for_coord(self, lon, lat):
    #     """Get index of flatten array for specific lat, lon."""
    #     mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
    #     indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

    #     idx_map = self.get_map(indices_flat, name="idx_flat")

    #     # idx = idx_map.sel(lat=lat, lon=lon, method='nearest')
    #     lon, lat, idx_all = grid.find_nearest_lat_lon(
    #         lon=lon, lat=lat, lon_arr=idx_map["lon"], lat_arr=idx_map["lat"]
    #     )

    #     idx = self.idx_map.sel(points=idx_all)  # Because idx_map['lon'] contains all values also non-defined!
    #     if np.isnan(idx.data) is True:
    #         raise ValueError(f"Error the lon {lon} lat {lat} is not defined!")

    #     return int(idx)

    def get_map_for_locs(self, locations):
        """Gives a map for a list of [lon, lat] locations.

        Args:
            locations (list): 2d list of locations

        Returns:
            xr.DataArray: Dataarray with the locations as 1 in the map
        """
        index_locs_lst = []

        index_locs_lst = self.get_idx_for_loc(locs=locations)
        loc_map = self.get_map_for_idx(idx_lst=index_locs_lst)

        return loc_map

    def add_loc_dict(
        self, name, lon_range, lat_range,
        slon=None, slat=None,
        n_rep_ids=1, color=None,
        sname=None,
        lname=None,
        reset_loc_dict=False,
    ):
        if reset_loc_dict:
            self.loc_dict = dict()
        this_loc_dict = dict(
            name=name,
            lon_range=lon_range,
            lat_range=lat_range,
            color=color,
            sname=sname,  # short name
            lname=lname   # long name
        )
        if (gut.check_range(lon_range, self.lon_range)) and (
            gut.check_range(lat_range, self.lat_range)
        ):
            # Choose first all ids in lon-lat range!
            ids_lst, _ = self.get_idx_region(
                this_loc_dict, def_map=None)  # these are ids, not points!

            if len(ids_lst) > 0:
                # Representative Ids for every location
                mean_loc = self.get_mean_loc(ids_lst)
                gut.myprint(f"Name: {name}, loc{mean_loc}")
                if slon is None or slat is None:
                    slon, slat = mean_loc
                    loc = mean_loc
                if gut.check_range_val(slon, lon_range) and gut.check_range_val(slat, lat_range):
                    slon = slon
                    slat = slat
                else:
                    raise ValueError(f'ERROR {slon} or {slat} not in range!')
                idx_loc = self.get_index_for_coord(lon=slon, lat=slat)
                loc = self.get_loc_for_idx(idx_loc)
                if np.isnan(idx_loc):
                    gut.myprint(
                        f"WARNING! Rep IDs {idx_loc} for {name} not defined!")
                else:
                    rep_ids = self.get_n_ids(loc=mean_loc, num_nn=n_rep_ids)
                    pids = self.get_points_for_idx(ids_lst)
                    this_loc_dict["rep_ids"] = np.array(rep_ids)
                    this_loc_dict["loc"] = loc
                    this_loc_dict['ids'] = ids_lst
                    this_loc_dict['pids'] = pids
                    this_loc_dict['data'] = self.ds.sel(points=pids)
                    this_loc_dict["map"] = self.get_map(
                        self.flat_idx_array(ids_lst))
                    this_loc_dict['ds'] = self.ds.sel(
                        points=this_loc_dict['pids'])
            else:
                raise ValueError(
                    f"ERROR! This region {name} does not contain any data points!"
                )
        else:
            raise ValueError(
                f"ERROR! This region {name} does not fit into {lon_range}, {lat_range}!"
            )

        self.loc_dict[name] = this_loc_dict

    def get_mean_loc(self, idx_lst):
        """
        Gets a mean location for a list of indices
        that is defined!
        """
        lon_arr = []
        lat_arr = []
        if len(idx_lst) == 0:
            raise ValueError("ERROR! List of points is empty!")

        for idx in idx_lst:
            map_idx = self.get_map_index(idx)
            lon_arr.append(map_idx["lon"])
            lat_arr.append(map_idx["lat"])
        mean_lat = np.mean(lat_arr)

        if max(lon_arr) - min(lon_arr) > 180:
            lon_arr = np.array(lon_arr)
            lon_arr[lon_arr < 0] = lon_arr[lon_arr < 0] + 360

        mean_lon = np.mean(lon_arr)
        if mean_lon > 180:
            mean_lon -= 360

        nearest_locs = grid.haversine(
            mean_lon, mean_lat, self.def_locs[:,
                                              0], self.def_locs[:, 1], radius=1
        )
        idx_min = np.argmin(nearest_locs)
        mean_lon = self.def_locs[idx_min, 0]
        mean_lat = self.def_locs[idx_min, 1]

        return (mean_lon, mean_lat)

    def get_mean_loc_idx(self, idx_lst):
        mean_loc = self.get_mean_loc(idx_lst=idx_lst)
        mean_idx = self.get_idx_for_loc(locs=mean_loc)
        return mean_idx

    def get_locations_in_range(
        self, def_map=None, lon_range=None, lat_range=None, rect_grid=False,
        dateline=False
    ):
        """
        Returns a map with the location within certain range.

        Parameters:
        -----------
        lon_range: list
            Range of longitudes, i.e. [min_lon, max_lon]
        lat_range: list
            Range of latitudes, i.e. [min_lat, max_lat]
        def_map: xr.Dataarray
            Map of data, i.e. the mask.

        Returns:
        --------
        idx_lst: np.array
            List of indices of the flattened map.
        mmap: xr.Dataarray
            Dataarray including ones at the location and NaNs everywhere else
        """
        if lon_range is None:
            lon_range = self.lon_range
        if lat_range is None:
            lat_range = self.lat_range
        if def_map is None:
            def_map = self.mask
        mmap = sput.get_locations_in_range(
            def_map=def_map, lon_range=lon_range, lat_range=lat_range,
            dateline=dateline
        )

        # Return these indices (NOT points!!!) that are defined
        if rect_grid:
            idx_lst = np.where(sput.flatten_array(dataarray=mmap,
                                                  mask=self.mask,
                                                  time=False,
                                                  check=False) > 0)[0]  # TODO check for better solution!
        else:
            defined_points = np.nonzero(self.mask.data)[0]
            point_lst_range = np.where(~np.isnan(mmap.data))[0]
            def_point_list = np.intersect1d(defined_points, point_lst_range)
            idx_lst = self.get_idx_point_lst(point_lst=def_point_list)

        return idx_lst, mmap

    def get_loc_for_idx(self, idx):
        """Returns a (lon,lat) location for an int index.

        Args:
            idx (int): list of indices integers

        Returns:
            tuple: tuple of (lon, lat)
        """
        val = sput.get_val_array(self.idx_map, idx)
        lon = float(val['lon'])
        lat = float(val['lat'])

        # point = self.get_points_for_idx([idx])[0]
        # lon = float(self.idx_map[point]['lon'])
        # lat = float(self.idx_map[point]['lat'])
        return lon, lat

    def get_locs_for_indices(self, idx_lst):
        """Returns a list of (lon,lat) locations for an int index.

        Args:
            idx_lst (list): list of indices integers

        Returns:
            np.array: array of tuples of (lon, lat)
        """
        # points = self.get_points_for_idx(idx_lst=idx_lst)
        # loc_array = []
        # for point in points:
        #     lon = float(self.idx_map[point]['lon'])
        #     lat = float(self.idx_map[point]['lat'])
        #     loc_array.append([lon, lat])

        loc_array = []
        for idx in idx_lst:
            lon, lat = self.get_loc_for_idx(idx=idx)
            loc_array.append([lon, lat])

        return np.array(loc_array)

    def get_n_ids(self, loc=None, nid=None, num_nn=3):
        """
        Gets for a specific location, the neighboring lats and lons ids.
        ----
        Args:
        loc: (float, float) provided as lon, lat values
        """
        if loc is None and nid is None:
            raise ValueError('provide either loc or nid!')
        if loc is not None and nid is not None:
            raise ValueError(
                f'Both loc {loc} and nid {nid} are given! Provide either loc or nid!')

        if nid is not None:
            slon, slat = self.get_loc_for_idx(nid)
        else:
            slon, slat = loc
        # lon = self.grid['lon']
        # lat = self.grid['lat']
        # sidx = self.get_index_for_coord(lon=slon, lat=slat)
        # sidx_r = self.get_index_for_coord(lon=slon + self.grid_step, lat=slat)
        # sidx_t = self.get_index_for_coord(lon=slon, lat=slat + self.grid_step)

        nearest_locs = grid.haversine(
            slon, slat, self.def_locs[:, 0], self.def_locs[:, 1], radius=1
        )
        idx_sort = np.argsort(nearest_locs)
        n_idx = []
        for idx in range(num_nn):
            sidx_t = self.get_index_for_coord(
                lon=self.def_locs[idx_sort[idx], 0],
                lat=self.def_locs[idx_sort[idx], 1]
            )
            n_idx.append(sidx_t)
        return np.array(n_idx)

    # ################## Time analyis functions #########################################

    def get_data_for_indices(self, idx_lst, var=None):
        if var is None:
            var = self.var_name

        dims = self.get_dims()
        if 'points' in dims:
            data_arr = self.ds[var].sel(
                points=self.get_points_for_idx(idx_lst))
        elif 'lon' in dims and 'lat' in dims:
            locs = self.get_locs_for_indices(idx_lst=idx_lst)
            data_arr = []
            for loc in locs:
                lon, lat = loc
                data_arr.append(self.ds.sel(lon=lon, lat=lat)[var].data)
            data_arr = np.array(data_arr)

            if 'plevel' in dims:
                data_arr = gut.create_xr_ds(
                    data=data_arr,
                    dims=['ids', 'plevel', 'time'],
                    coords={'time': self.ds.time,
                            'ids': idx_lst,
                            'plevel': self.ds.plevel},
                    name=var
                )
            else:
                data_arr = gut.create_xr_ds(
                    data=data_arr,
                    dims=['ids', 'time'],
                    coords={'time': self.ds.time,
                            'ids': idx_lst, },
                    name=var
                )
        return data_arr

    def get_data_for_coord(self, lon, lat, var=None):
        if var is None:
            var = self.var_name
        idx = self.get_index_for_coord(lon=lon, lat=lat)
        ts = self.get_data_for_idx(idx=idx, var=var)
        return ts

    def get_data_for_locs(self, locs, var=None):
        if var is None:
            var = self.var_name

        idx_lst = self.get_idx_for_loc(locs=locs)
        ts = self.get_data_for_idx(idx=idx_lst, var=var)
        return ts

    def apply_timemean(self, timemean=None):
        self.ds = tu.apply_timemean(ds=self.ds, timemean=timemean)
        return self.ds

    def get_max(self, var_name=None):
        if var_name is None:
            var_name = self.var_name
        maxval = (
            self.ds[var_name]
            .where(self.ds[var_name] == self.ds[var_name].max(), drop=True)
            .squeeze()
        )
        lon = float(maxval.lon)
        lat = float(maxval.lat)
        tp = maxval.time

        return {"lon": lon, "lat": lat, "tp": tp}

    def get_min(self, var_name=None):
        if var_name is None:
            var_name = self.var_name
        maxval = (
            self.ds[var_name]
            .where(self.ds[var_name] == self.ds[var_name].min(), drop=True)
            .squeeze()
        )
        lon = float(maxval.lon)
        lat = float(maxval.lat)
        tp = maxval.time

        return {"lon": lon, "lat": lat, "tp": tp}

    def compute_anomalies(self, dataarray=None, group="dayofyear"):
        """Calculate anomalies.

        Parameters:
        -----
        dataarray: xr.DataArray
            Dataarray to compute anomalies from.
        group: str
            time group the anomalies are calculated over, i.e. 'month', 'day', 'dayofyear'

        Return:
        -------
        anomalies: xr.dataarray
        """
        reload(tu)
        if dataarray is None:
            dataarray = self.ds[self.var_name]
        anomalies = tu.compute_anomalies(dataarray=dataarray, group=group)

        return anomalies

    def get_idx_region(self, region_dict, def_map=None, dateline=False):
        """
        Gets the indices for a specific dictionary that has lon/lat_range as keys.
        E.g. can be applied to get all indices of the South American monsoon defined by Wang/EE.
        """
        if def_map is None:
            def_map = self.mask
            if def_map is None:
                raise ValueError(
                    "ERROR mask is None! Check if mask is computed properly!"
                )

        lon_range = region_dict["lon_range"]
        lat_range = region_dict["lat_range"]
        ids, mmap = self.get_locations_in_range(
            lon_range=lon_range, lat_range=lat_range, def_map=def_map,
            dateline=dateline,
        )
        return ids, mmap

    def set_sel_tps_ds(self, tps):
        ds_sel = tu.get_sel_tps_ds(ds=self.ds, tps=tps)
        self.ds = ds_sel
        return

    def select_time_periods(self, time_snippets):
        """Cut time snippets from dataset and concatenate them.

        Args:
            time_snippets (np.array): Array of n time snippets
                with dimension (n,2).

        Returns:
            None
        """
        self.ds = tu.select_time_snippets(self.ds, time_snippets)
        return self.ds

    def detrend(self, dim="time", deg=1, startyear=None):
        """Detrend dataarray.
        Args:
            dim (str, optional): [description]. Defaults to 'time'.
            deg (int, optional): [description]. Defaults to 1.
        """
        reload(tu)
        gut.myprint("Detrending data...")
        da_detrend = tu.detrend_dim(self.ds[self.var_name], dim=dim, deg=deg,
                                    startyear=startyear)
        self.ds[self.var_name] = da_detrend
        self.ds.attrs["detrended"] = "True"
        gut.myprint("... finished!")
        return

    def get_vars(self, ds=None):
        # vars = []
        # for name, da in self.ds.data_vars.items():
        #     vars.append(name)
        if ds is None:
            ds = self.ds
        vars = list(ds.keys())
        return vars


class AnomalyDataset(BaseDataset):
    """Anomaly Dataset.

    Parameters:
    ----------
    data_nc: str
        Filename of input data, Default: None
    load_nc: str
        Already processed data file. Default: None
        If specified all other attritbutes are loaded.
    time_range: list
        Default: ['1997-01-01', '2019-01-01'],
    lon_range: list
        Default: [-180, 180],
    lat_range: list
        Default: [-90,90],
    grid_step: int
        Default: 1
    grid_type: str
        Default: 'gaussian',
    start_month: str
        Default: 'Jan'
    end_month: str
        Default: 'Dec'
    lsm: bool
        Default:False
    detrend: bool
        Default: True
    climatology: str
        Specified climatology the anomalies are computed over. Default: "dayofyear"
    **kwargs
    """

    def __init__(
        self,
        data_nc=None,
        load_nc=None,
        var_name=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="gaussian",
        timemean=None,
        lsm=False,
        climatology="dayofyear",
        detrend=False,
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
            timemean=timemean,
            lsm=lsm,
            detrend=detrend,
            **kwargs,
        )

        # compute anomalies if not given in nc file
        if "anomalies" in self.vars:
            gut.myprint("Anomalies are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute anomalies.")
        else:
            gut.myprint(f"Compute anomalies for variable {self.var_name}.")
            da = self.ds[self.var_name]
            da = self.compute_anomalies(da, group=climatology)
            da.attrs = {"var_name": self.var_name}
            self.ds["anomalies"] = da

        # set var name to "anomalies" in order to run network on anomalies
        self.var_name = "anomalies"


class EvsDataset(BaseDataset):
    def __init__(
        self,
        data_nc=None,
        load_nc=None,
        var_name=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        grid_type="gaussian",
        large_ds=False,
        lsm=False,
        q=0.95,
        min_evs=20,
        min_treshold=1,
        th_eev=15,
        rrevs=False,
        can=False,
        timemean=None,
        month_range=None,
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
            large_ds=large_ds,
            timemean=timemean,
            lsm=lsm,
            init_indices=False,  # Indices are later initialized with evs mask
            **kwargs,
        )

        # Event synchronization
        if data_nc is not None or rrevs:

            self.q = q
            self.min_evs = min_evs
            self.min_treshold = min_treshold
            self.th_eev = th_eev
        else:
            self.load_evs_attrs()
        self.can = can
        if "rrevs" in kwargs:
            rrevs = kwargs.pop("rrevs")
        # compute event synch if not given in nc file
        if "evs" in self.vars:
            gut.myprint("Evs are already stored in dataset.")
        elif var_name is None:
            raise ValueError("Specify varname to compute event sync.")
        else:
            gut.myprint(
                f"Compute Event synchronization for variable {self.var_name}.",
            )
            rrevs = True
        if self.can is True:
            an_types = kwargs.pop("an_types", ["dayofyear"])
            for an_type in an_types:
                self.ds[f"an_{an_type}"] = self.compute_anomalies(
                    self.ds[self.var_name], group=an_type
                )
        if rrevs is True:
            if var_name is None:
                var_name = self.var_name

            self.ds = self.create_evs_ds(
                var_name=var_name,
                th=self.min_treshold,
                q=self.q,
                min_evs=self.min_evs,
                th_eev=self.th_eev,
                month_range=month_range
            )
        else:
            self.mask = self.get_es_mask(self.ds["evs"], min_evs=self.min_evs)
            self.init_map_indices()
        self.vars = self.get_vars()

    def create_evs_ds(
        self, var_name, q=0.95, th=1, th_eev=15, min_evs=20, month_range=None
    ):
        """Genereates an event time series of the variable of the dataset.
        Attention, if month range is provided all values not in the month range are
        set to 0, not deleted, therefore the number of dates is retained

        Args:
            var_name (str): variable name
            q (float, optional): Quantile that defines extreme events. Defaults to 0.95.
            th (float, optional): threshold of minimum value in a time series. Defaults to 1.
            th_eev (float, optional): Threshold of minimum value of an extreme event. Defaults to 15.
            min_evs (int, optional): Minimum number of extreme events in the whole time Series. Defaults to 20.
            month_range (list, optional): list of strings as [start_month, end_month]. Defaults to None.

        Returns:
            xr.Dataset: Dataset with new values of variable and event series
        """
        self.q = q
        self.th = th
        self.th_eev = th_eev
        self.min_evs = min_evs
        gut.myprint(f'Create EVS with EE defined by q > {q}')
        if month_range is None:
            da_es, self.mask = self.compute_event_time_series(
                var_name=var_name,)
        else:
            da_es, self.mask = self.compute_event_time_series_month_range(
                start_month=month_range[0], end_month=month_range[1]
            )
        da_es.attrs = {"var_name": var_name}

        self.set_ds_attrs(ds=da_es)
        self.ds["evs"] = da_es

        return self.ds

    def get_q_maps(self, var_name):

        if var_name is None:
            var_name = self.var_name
        gut.myprint(f"Apply Event Series on variable {var_name}")

        dataarray = self.ds[var_name]

        q_val_map, ee_map, data_above_quantile, rel_frac_q_map = tu.get_ee_ds(
            dataarray=dataarray, q=self.q, th=self.th, th_eev=self.th_eev
        )

        return q_val_map, ee_map, data_above_quantile

    def compute_event_time_series(
        self, var_name=None,
    ):
        reload(tu)
        if var_name is None:
            var_name = self.var_name
        gut.myprint(f"Apply Event Series on variable {var_name}")

        dataarray = self.ds[var_name]

        event_series, mask = tu.compute_evs(
            dataarray=dataarray,
            q=self.q,
            th=self.th,
            th_eev=self.th_eev,
            min_evs=self.min_evs,
        )

        return event_series, mask

    def compute_event_time_series_month_range(
        self, start_month="Jan", end_month="Dec",
    ):
        """
        This function generates data within a given month range.
        It can be from smaller month to higher (eg. Jul-Sep) but as well from higher month
        to smaller month (eg. Dec-Feb)

        Parameters
        ----------
        start_month : string, optional
            Start month. The default is 'Jan'.
        end_month : string, optional
            End Month. The default is 'Dec'.

        Returns
        -------
        seasonal_data : xr.dataarray
            array that contains only data within month-range.

        """
        reload(tu)
        times = self.ds["time"]
        start_year, end_year = tu.get_sy_ey_time(times, sy=None, ey=None)
        gut.myprint(
            f"Get month range data from year {start_year} to {end_year}!")

        da = self.ds[self.var_name]
        # Sets the data outside the month range to 0, but retains the dates
        da_mr = self.get_month_range_data(
            dataarray=da, start_month=start_month, end_month=end_month, set_zero=True
        )
        # Computes the Event Series
        evs_mr, mask = tu.compute_evs(
            dataarray=da_mr,
            q=self.q,
            th=self.th,
            th_eev=self.th_eev,
            min_evs=self.min_evs,
        )

        return evs_mr, mask

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
                th=self.min_treshold,
                q=q,
                min_evs=self.min_evs,
                th_eev=self.th_eev,
            )
            self.set_ds_attrs(ds=da_es)
            self.ds[f"evs_q{q}"] = da_es

        return self.ds

    def get_es_mask(self, data_evs, min_evs):
        num_non_nan_occurence = data_evs.where(data_evs == 1).count(dim="time")
        self.mask = xr.where(num_non_nan_occurence > min_evs, 1, 0)
        self.min_evs = min_evs
        return self.mask

    def set_ds_attrs(self, ds):
        param_class = {
            "grid_step": self.grid_step,
            "grid_type": self.grid_type,
            "lsm": int(self.lsm),
            "q": self.q,
            "min_evs": self.min_evs,
            "min_threshold": self.min_treshold,
            "th_eev": self.th_eev,
            "th": self.th,
            "an": int(self.can),
        }
        ds.attrs = param_class
        return ds

    def save(self, file):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        filepath = os.path.dirname(file)

        if os.path.exists(file):
            gut.myprint("File" + file + " already exists!")
            os.rename(file, file + "_backup")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        gut.myprint(f"Save file {file}")
        ds_temp = self.set_ds_attrs(self.ds)
        ds_temp.to_netcdf(file)

        return None

    def load_evs_attrs(self):
        self.q = self.ds.attrs["q"]
        self.min_evs = self.ds.attrs["min_evs"]
        self.min_treshold = self.ds.attrs["min_threshold"]
        self.th_eev = self.ds.attrs["th_eev"]
        if "an" in self.ds.attrs:  # To account for old version saved files!
            self.can = bool(self.ds.attrs["an"])
        else:
            self.can = False

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


class BaseRectDataset(BaseDataset):
    """Class that defines a classical rectangular dataset, that is stored as as classical
    nc file. It has however the same functions of BaseDataset but is not defined of the
    grid of BaseDataset, but on the standard lon, lat grid that is by default used in nc-files.
    i.e. for different longitude and latitude boxes

    Args:
        BaseDataset (class): Base Dataset

    """

    def __init__(
        self,
        var_name=None,
        data_nc=None,
        load_nc=None,
        time_range=None,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        grid_step=1,
        large_ds=False,
        can=False,
        detrend=False,
        month_range=None,
        **kwargs,
    ):

        self.grid_type = 'rectangular'
        if data_nc is not None and load_nc is not None:
            raise ValueError("Specify either data or load file.")
        # initialize dataset
        elif data_nc is not None:
            # check if file exists
            if not os.path.exists(data_nc):
                PATH = os.path.dirname(os.path.abspath(__file__))
                gut.myprint(f"You are here: {PATH}!")
                gut.myprint(f'And this file is not here {data_nc}!')
                raise ValueError(f"File does not exist {data_nc}!")
            self.var_name = var_name
            self.grid_step = grid_step
            ds = self.open_ds(
                nc_file=data_nc,
                var_name=var_name,
                lat_range=lat_range,
                lon_range=lon_range,
                time_range=time_range,
                grid_step=grid_step,
                large_ds=large_ds,
                **kwargs,
            )
            (
                self.time_range,
                self.lon_range,
                self.lat_range,
            ) = self.get_spatio_temp_range(ds)

            self.ds = ds
        else:
            self.load(load_nc=load_nc,
                      lon_range=lon_range,
                      lat_range=lat_range)
        # select a main var name
        self.vars = self.get_vars()
        self.var_name = var_name if var_name is not None else self.vars[0]

        # detrending
        if detrend is True:
            detrend_from = kwargs.pop('detrend_from', None)
            self.detrend(dim="time", startyear=detrend_from)

        # Compute Anomalies if needed
        self.can = can
        if self.can is True:
            if "an" not in self.vars:
                self.an_types = kwargs.pop("an_types", ["dayofyear"])
                for an_type in self.an_types:
                    self.ds[f"an_{an_type}"] = self.compute_anomalies(
                        self.ds[self.var_name], group=an_type
                    )

        if month_range is not None:
            self.ds = tu.get_month_range_data(dataset=self.ds,
                                              start_month=month_range[0],
                                              end_month=month_range[1])
        # Init Mask
        self.init_mask(da=self.ds[self.var_name])
        self.indices_flat, self.idx_map = self.init_map_indices()

    def open_ds(
        self,
        nc_file,
        var_name,
        time_range=None,
        grid_step=1,
        large_ds=True,
        lon_range=[-180, 180],
        lat_range=[-90, 90],
        **kwargs,
    ):
        gut.myprint("Start processing data!")
        if large_ds:
            ds = xr.open_dataset(nc_file, chunks={"time": 100})
        else:
            ds = xr.open_dataset(nc_file)
        ds = self.check_dimensions(ds, **kwargs)
        ds = self.rename_var(ds)
        da = ds[var_name]
        da = da.transpose("time", "lat", "lon")
        da = self.get_data_timerange(da, time_range)
        if max(da.lon) > 180:
            gut.myprint("Shift longitude in Preprocessing!")
            da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180))
        da = self.common_grid(dataarray=da, grid_step=grid_step)

        ds.unify_chunks()
        if lon_range != [-180, 180] and lat_range != [-90, 90]:
            ds = self.cut_map(ds, lon_range, lat_range)

        ds = da.to_dataset(name=var_name)

        # For the moment, all values are defined! TODO implement lsm
        self.mask = xr.DataArray(
            data=np.ones_like(da[0].data),
            dims=da.sel(time=da.time[0]).dims,
            coords=da.sel(time=da.time[0]).coords,
            name="mask",
        )

        gut.myprint("Finished processing data")

        return ds

    def load(self, load_nc, lon_range=[-180, 180], lat_range=[-90, 90]):
        """Load dataset object from file.

        Parameters:
        ----------
        ds: xr.Dataset
            Dataset object containing the preprocessed dataset

        """
        # check if file exists
        if not os.path.exists(load_nc):
            PATH = os.path.dirname(os.path.abspath(__file__))
            gut.myprint(f"You are here: {PATH}!")
            raise ValueError(f"File does not exist {load_nc}!")

        gut.myprint(f"Load Dataset: {load_nc}")
        ds = xr.open_dataset(load_nc)
        ds = self.cut_map(ds=ds, lon_range=lon_range, lat_range=lat_range)
        self.time_range, self.lon_range, self.lat_range = self.get_spatio_temp_range(
            ds)
        ds_attrs = list(ds.attrs.keys())
        if "grid_step" in ds_attrs:
            self.grid_step = ds.attrs["grid_step"]
        self.info_dict = ds.attrs  # TODO
        # Read and create grid class
        ds = self.rename_var(ds)

        for name, da in ds.data_vars.items():
            gut.myprint(f"Variables in dataset: {name}")

        mask = np.ones_like(ds[name][0].data, dtype=bool)
        for idx, t in enumerate(ds.time):
            mask *= np.isnan(ds[name].sel(time=t).data)

        self.mask = xr.DataArray(
            data=xr.where(mask == 0, 1, np.NaN),
            dims=da.sel(time=da.time[0]).dims,
            coords=da.sel(time=da.time[0]).coords,
            name="lsm",
        )

        self.ds = self.check_time(ds)

        return None

    def save(self, filepath):
        """Save the dataset class object to file.
        Args:
        ----
        filepath: str
        """
        param_class = {
            "grid_step": self.grid_step,
        }
        ds_temp = self.ds
        ds_temp.attrs = param_class

        gut.save_ds(ds=ds_temp, filepath=filepath)

        return None

    # def get_index_for_coord(self, lon, lat):
    #     mask_arr = np.where(self.mask.data.flatten() == 1, True, False)
    #     indices_flat = np.arange(0, np.count_nonzero(mask_arr), 1, dtype=int)

    #     idx_map = self.get_map(indices_flat, name="idx_flat")

    #     idx = idx_map.sel(lat=lat, lon=lon, method="nearest")
    #     if np.isnan(idx):
    #         gut.myprint(
    #             f"The lon {lon} lat {lat} index is not within mask, i.e. NaN!"
    #         )
    #         return idx
    #     else:
    #         return int(idx)
