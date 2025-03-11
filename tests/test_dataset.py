"""Create network of surface temperature data over land."""
import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from climnet.datasets.dataset import BaseDataset

PATH = os.path.dirname(os.path.abspath(__file__))


def test_dataset_load_fibonacci():
    """ """
    fname = PATH + f"/../input/Tair_test_data_grid_10_fibonacci.nc"
    dataset = BaseDataset(var_name='Tair', load_nc=fname)

    ref_ds = xr.open_dataset(
        PATH + f"/../input/Tair_test_data_grid_10_fibonacci.nc")

    # check consistency
    assert dataset.var_name == 'Tair'
    assert dataset.ds[dataset.var_name].shape == ref_ds['Tair'].shape
    assert dataset.grid_step == ref_ds.attrs['grid_step']
    assert dataset.grid_type == ref_ds.attrs['grid_type']
    assert dataset.lsm == bool(ref_ds.attrs['lsm'])
    assert dataset.ds.time.data[0] == ref_ds.time.data[0]
    assert dataset.ds.time.data[-1] == ref_ds.time.data[-1]
    return


def test_dataset_load_gaussian():
    return


def test_dataset_anomalies():
    return


def test_dataset_create_gaussian():
    return


def test_dataset_create_fibonacci():
    """
    """
    fname = PATH + f"/../input/Tair_test_data_grid_10.nc"
    dataset = BaseDataset(var_name="Tair", data_nc=fname, load_nc=None,
                          time_range=['2000-01-01', '2001-01-01'],
                          grid_step=10, grid_type='fibonacci')

    ref_ds = xr.open_dataset(
        PATH + f"/../input/Tair_test_data_grid_10_fibonacci.nc")

    # check consistency
    assert dataset.var_name == 'Tair'
    assert dataset.ds[dataset.var_name].shape == ref_ds['Tair'].shape
    assert dataset.grid_step == ref_ds.attrs['grid_step']
    assert dataset.grid_type == ref_ds.attrs['grid_type']
    assert dataset.lsm == bool(ref_ds.attrs['lsm'])
    assert dataset.ds.time.data[0] == ref_ds.time.data[0]
    assert dataset.ds.time.data[-1] == ref_ds.time.data[-1]

    return


if __name__ == "__main__":
    test_dataset_anomalies()
    test_dataset_create_fibonacci()
    test_dataset_create_gaussian()
    test_dataset_load_fibonacci()
    test_dataset_load_gaussian()
