"""Terrain derivatives used in archaeological lidar work. All pure numpy/scipy.

Every function takes elevations in meters and the pixel size in meters.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve


def fill_nodata(z: np.ndarray) -> np.ndarray:
    """Replace NaN with the nearest valid value so filters don't explode."""
    mask = np.isnan(z)
    if not mask.any():
        return z
    if mask.all():
        raise ValueError("DEM window contains no data.")
    idx = ndi.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def slope_aspect(z: np.ndarray, csx: float, csy: float) -> tuple[np.ndarray, np.ndarray]:
    """Horn (1981) method. Slope in degrees; aspect in degrees clockwise from north
    (the direction the slope faces)."""
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * csx)   # east positive
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * csy)   # south positive
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360  # downslope direction
    return slope.astype("float32"), aspect.astype("float32")


def hillshade(z, csx, csy, azimuth=315.0, altitude=35.0, exaggeration=2.0):
    zp = np.pad(z * exaggeration, 1, mode="edge")
    dzdx = (zp[1:-1, 2:] - zp[1:-1, :-2]) / (2 * csx)
    dzdy = (zp[:-2, 1:-1] - zp[2:, 1:-1]) / (2 * csy)  # north positive
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    az = math.radians(90 - azimuth)
    alt = math.radians(altitude)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1).astype("float32")


def multi_hillshade(z, csx, csy):
    """Average of 16 light directions: no feature hides because it is parallel to the light."""
    return np.mean([hillshade(z, csx, csy, az) for az in range(0, 360, 360 // 16)], axis=0)


def local_relief(z: np.ndarray, cell_m: float, window_m: float = 35.0) -> np.ndarray:
    """Local Relief Model: elevation minus a local quadratic trend surface
    (2-D Savitzky-Golay fit in a window_m x window_m square).

    A quadratic fit follows hills, valleys and slope curvature, so what is left is
    micro-relief smaller than the window: negative = pits, ditches, sunken roads;
    positive = banks, walls, mounds."""
    n = max(7, int(round(window_m / cell_m)) | 1)
    h = n // 2
    yy, xx = np.mgrid[-h:h + 1, -h:h + 1].astype(float)
    a = np.stack([np.ones_like(xx), xx, yy, xx * xx, xx * yy, yy * yy], -1).reshape(-1, 6)
    kernel = np.linalg.pinv(a)[0].reshape(n, n)
    zp = np.pad(z.astype("float64"), h, mode="reflect")
    trend = fftconvolve(zp, kernel[::-1, ::-1], mode="valid")
    return (z - trend).astype("float32")


def tpi(z: np.ndarray, cell_m: float, radius_m: float) -> np.ndarray:
    """Topographic Position Index: + on ridges/elevations, - in hollows/valleys."""
    r = max(1, int(round(radius_m / cell_m)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    ring = ((xx ** 2 + yy ** 2) <= r * r).astype(float)
    ring[r, r] = 0
    zp = np.pad(z.astype("float64"), r, mode="reflect")
    mean = fftconvolve(zp, ring / ring.sum(), mode="valid")
    return (z - mean).astype("float32")


def sky_view_factor(z: np.ndarray, cell_m: float, radius_m: float = 10.0, directions: int = 16):
    """Share of the sky visible from each cell (Zakšek et al. 2011). 1 = open plain,
    lower = inside pits and ditches. Good for spotting shallow depressions."""
    n = max(2, int(round(radius_m / cell_m)))
    zp = np.pad(z, n, mode="edge")
    h, w = z.shape
    svf = np.zeros_like(z, dtype="float64")
    for k in range(directions):
        ang = 2 * math.pi * k / directions
        best = np.full(z.shape, -np.inf)
        for s in range(1, n + 1):
            dr, dc = -math.cos(ang) * s, math.sin(ang) * s
            ri, ci = int(round(dr)), int(round(dc))
            dist = math.hypot(ri, ci) * cell_m
            if dist == 0:
                continue
            shifted = zp[n + ri:n + ri + h, n + ci:n + ci + w]
            best = np.maximum(best, (shifted - z) / dist)
        svf += 1 - np.sin(np.arctan(np.maximum(best, 0)))
    return (svf / directions).astype("float32")


ASPECT_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def landform(tpi_small: float, tpi_large: float, slope_deg: float) -> str:
    """Simple Weiss (2001)-style classification of the spot."""
    if tpi_large > 1.0:
        return "ridge / hilltop" if slope_deg < 6 else "upper slope"
    if tpi_large < -1.0:
        return "valley bottom" if slope_deg < 3 else "lower slope"
    if slope_deg < 3:
        return "flat (terrace / plain)"
    return "mid slope"


def spot_metrics(z, slope, aspect, tpi_s, tpi_l, rc, cell_m, radius_m=15.0) -> dict:
    """Metrics at the user's chosen spot, averaged over a small disk to resist noise."""
    r0, c0 = rc
    rr = max(1, int(round(radius_m / cell_m)))
    yy, xx = np.ogrid[:z.shape[0], :z.shape[1]]
    disk = (yy - r0) ** 2 + (xx - c0) ** 2 <= rr * rr
    asp = aspect[disk]
    mean_aspect = (math.degrees(math.atan2(np.sin(np.radians(asp)).mean(),
                                           np.cos(np.radians(asp)).mean())) + 360) % 360
    s = float(np.median(slope[disk]))
    ts, tl = float(tpi_s[r0, c0]), float(tpi_l[r0, c0])
    return {
        "elevation_m": round(float(z[r0, c0]), 2),
        "slope_deg": round(s, 1),
        "aspect_deg": round(mean_aspect, 0),
        "aspect": ASPECT_NAMES[int(((mean_aspect + 22.5) % 360) // 45)],
        "tpi_50m": round(ts, 2),
        "tpi_250m": round(tl, 2),
        "landform": landform(ts, tl, s),
        "height_above_window_min_m": round(float(z[r0, c0] - np.nanmin(z)), 1),
        "relief_in_window_m": round(float(np.nanmax(z) - np.nanmin(z)), 1),
    }
