"""Surface model (DSM) -> terrain model (DTM): remove trees, bushes and buildings.

Needed for ArcticDEM, Copernicus GLO-30, drone photogrammetry DSMs and phone scans,
which all measure the top of whatever stands on the ground.

Method: progressive morphological filter (Zhang et al. 2003, raster version): open the
surface with growing windows; a cell is 'object' if it sticks up above the opened surface
by more than a height threshold that grows with window size (so gentle banks and mounds
of 0.3-1 m survive, while trees and houses are removed). Object cells are then filled
from surrounding ground.

Limits: under closed forest canopy there is no ground in a DSM at all - the result is an
approximation and the report says so. Works well on fields, meadows, sparse forest.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .terrain import fill_nodata


def dsm_to_dtm(z: np.ndarray, cell_m: float, max_object_m: float = 20.0,
               slope: float = 0.15, dh0: float = 0.4, dh_max: float = 3.0) -> tuple[np.ndarray, dict]:
    z = fill_nodata(z.astype("float64"))
    ground = np.ones(z.shape, bool)
    surface = z.copy()
    w = 1
    windows = []
    while w * cell_m <= max_object_m:
        windows.append(w)
        w = w * 2 + 1
    prev_w = 0
    for w in windows:
        opened = ndi.grey_opening(surface, size=(w, w))
        dh = min(dh_max, dh0 + slope * (w - prev_w) * cell_m) if prev_w else dh0
        ground &= (surface - opened) <= dh
        surface = opened
        prev_w = w
    obj = ~ground
    # fill object cells from the surrounding ground: normalized convolution (a weighted
    # average of nearby ground), coarse-to-fine, so the patches are smooth, not terraced
    known = (~obj).astype("float64")
    zk = np.where(obj, 0.0, z)
    out = z.copy()
    todo = obj.copy()
    for sigma_m in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        s = max(0.7, sigma_m / cell_m)
        num, den = ndi.gaussian_filter(zk, s), ndi.gaussian_filter(known, s)
        ok = todo & (den > 0.05)
        out[ok] = num[ok] / den[ok]
        todo &= ~ok
        if not todo.any():
            break
    if todo.any():
        out[todo] = fill_nodata(np.where(todo, np.nan, out))[todo]
    info = {"object_share": round(float(obj.mean()), 3),
            "max_object_size_m": max_object_m,
            "method": "progressive morphological filter"}
    if info["object_share"] > 0.5:
        info["warning"] = ("More than half of the area was trees/buildings: the ground under "
                           "them is interpolated, small features there are not reliable.")
    return out.astype("float32"), info
