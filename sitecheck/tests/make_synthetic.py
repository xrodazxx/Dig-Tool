"""Synthetic 1 m DEM with known 'archaeology', written as a GeoTIFF (EPSG:4326).

Ground truth (meters from the upper-left corner, x right / y down):
  - river valley along the bottom (y ~ 440..500)
  - terrace with a south-facing slope down to the river
  - 3 house pits: (150,200) d=6 depth .7 | (175,215) d=5 depth .5 | (320,180) d=4 depth .4
  - rectangular foundation outline 14x9 m, wall bank 0.35 m high, centre (240,250), rotated 20 deg
  - straight sunken road 220 m long, 0.5 m deep, from (40,120) to (260,90)
  - curved natural gully on the slope (should be flagged as curved)
  - lidar-like noise 3 cm + vegetation bumps
"""
import math
import sys

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import ImageFileDirectory_v2

N = 500
LAT0, LON0 = 60.50, 24.90          # upper-left corner (Finland, why not)
DY = 1 / 110_574
DX = 1 / (111_320 * math.cos(math.radians(LAT0 - N * DY / 2)))


def build(seed=1):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:N, 0:N].astype(float)
    z = 120 + 0.004 * x                                   # gentle regional tilt
    z -= np.clip((y - 300) / 140, 0, 1) ** 1.5 * 12        # slope down to river
    z -= 2.5 * np.exp(-((y - 480) / 18) ** 2)              # river channel
    z += 1.5 * np.sin(x / 90) * np.exp(-((y - 150) / 200) ** 2)  # broad undulation

    for cx, cy, d, depth in [(150, 200, 6, .7), (175, 215, 5, .5), (320, 180, 4, .4)]:
        r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (d / 2) ** 2
        z -= depth * np.clip(1 - r2, 0, 1) ** 0.7
        z += 0.12 * np.exp(-((np.sqrt(r2) - 1.25) / 0.3) ** 2)   # spoil ring

    # rectangular foundation (wall outline)
    a = math.radians(20)
    u = (x - 240) * math.cos(a) + (y - 250) * math.sin(a)
    v = -(x - 240) * math.sin(a) + (y - 250) * math.cos(a)
    outer = (np.abs(u) <= 7.5) & (np.abs(v) <= 5)
    inner = (np.abs(u) <= 6) & (np.abs(v) <= 3.5)
    from scipy.ndimage import gaussian_filter
    z += gaussian_filter((outer & ~inner).astype(float), 0.7) * 0.35

    # straight sunken road
    p0, p1 = np.array([40, 120.]), np.array([260, 90.])
    t = np.clip(((x - p0[0]) * (p1 - p0)[0] + (y - p0[1]) * (p1 - p0)[1]) / ((p1 - p0) ** 2).sum(), 0, 1)
    dist = np.hypot(x - (p0[0] + t * (p1 - p0)[0]), y - (p0[1] + t * (p1 - p0)[1]))
    z -= 0.5 * np.clip(1 - (dist / 2.5) ** 2, 0, 1)

    # curved natural gully on the slope
    gx = 400 + 25 * np.sin((y - 300) / 30)
    z -= 0.6 * np.exp(-((x - gx) / 2.5) ** 2) * ((y > 300) & (y < 470))

    z += gaussian_filter(rng.normal(0, 0.15, z.shape), 2)   # vegetation / micro-noise
    z += rng.normal(0, 0.03, z.shape)                        # sensor noise
    return z.astype("float32")


def save_geotiff(z, path):
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (DX, DY, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, LON0, LAT0, 0.0)
    # GeoKeyDirectory: version 1.1.0, 3 keys: model=geographic, raster=pixel-is-area, EPSG 4326
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    ifd.tagtype[34735] = 3   # SHORT
    Image.fromarray(z, mode="F").save(path, tiffinfo=ifd)


def spot_lonlat(px_x, px_y):
    return LON0 + (px_x + .5) * DX, LAT0 - (px_y + .5) * DY


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_dem.tif"
    save_geotiff(build(), out)
    lon, lat = spot_lonlat(200, 220)
    print(f"wrote {out}; suggested spot: --point {lat:.6f},{lon:.6f}")
