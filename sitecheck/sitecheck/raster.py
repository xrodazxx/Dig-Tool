"""Loading a DEM (GeoTIFF) and cutting the analysis window around the chosen spot.

Uses rasterio + pyproj when installed (recommended: handles any CRS and
compression). Falls back to a minimal Pillow-based GeoTIFF reader for simple
files in geographic coordinates (EPSG:4326), so the core works with almost
no dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import proj

try:  # optional, recommended
    import rasterio
    from rasterio.windows import from_bounds
except ImportError:  # pragma: no cover
    rasterio = None

try:
    from pyproj import Transformer
except ImportError:  # pragma: no cover
    Transformer = None


@dataclass
class Dem:
    """Elevation grid plus everything needed to turn pixels into meters and coordinates."""
    z: np.ndarray            # float32 elevations, NaN = no data
    x0: float                # x of the upper-left corner of the upper-left pixel (CRS units)
    y0: float                # y of that corner
    dx: float                # pixel width (CRS units)
    dy: float                # pixel height (CRS units, positive)
    epsg: int | None         # CRS code, 4326 = geographic degrees
    source: str = "local file"
    meta: dict = field(default_factory=dict)

    @property
    def geographic(self) -> bool:
        return self.epsg in (4326, 4258, 4269, 4019)

    def cell_size_m(self) -> tuple[float, float]:
        """Pixel size in meters (x, y). Degrees are converted at the grid's center latitude."""
        if not self.geographic:
            return abs(self.dx), abs(self.dy)
        lat = self.y0 - self.dy * self.z.shape[0] / 2
        return (abs(self.dx) * 111_320 * math.cos(math.radians(lat)),
                abs(self.dy) * 110_574)

    def pixel_to_crs(self, row: float, col: float) -> tuple[float, float]:
        return self.x0 + (col + 0.5) * self.dx, self.y0 - (row + 0.5) * self.dy

    def crs_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return (self.y0 - y) / self.dy - 0.5, (x - self.x0) / self.dx - 0.5

    # --- WGS84 <-> grid ------------------------------------------------------
    def _transformers(self):
        """(forward, inverse) callables lon,lat <-> x,y; work on scalars and arrays."""
        if self.geographic:
            ident = lambda a, b: (a, b)
            return ident, ident
        if Transformer is not None and self.epsg:
            fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
            inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)
            return fwd.transform, inv.transform
        if proj.supported(self.epsg):
            return (lambda lon, lat: proj.forward(self.epsg, lon, lat),
                    lambda x, y: proj.inverse(self.epsg, x, y))
        if rasterio is not None and self.epsg:   # rasterio ships its own PROJ
            from rasterio.warp import transform as _tr

            def _mk(src, dst):
                def f(a, b):
                    xs, ys = _tr(f"EPSG:{src}", f"EPSG:{dst}", np.atleast_1d(a).ravel(), np.atleast_1d(b).ravel())
                    xs, ys = np.asarray(xs), np.asarray(ys)
                    return (float(xs[0]), float(ys[0])) if np.ndim(a) == 0 else (xs, ys)
                return f
            return _mk(4326, self.epsg), _mk(self.epsg, 4326)
        raise RuntimeError(
            f"DEM is in EPSG:{self.epsg}. Install pyproj (pip install pyproj) "
            "to convert lat/lon to this projection.")

    def lonlat_to_pixel(self, lon, lat):
        fwd, _ = self._transformers()
        return self.crs_to_pixel(*fwd(lon, lat))

    def pixel_to_lonlat(self, row, col):
        _, inv = self._transformers()
        return inv(*self.pixel_to_crs(row, col))


# --------------------------------------------------------------------------- loading

def load_dem(path: str) -> Dem:
    if rasterio is not None:
        return _load_rasterio(path)
    return _load_pillow(path)


def _load_rasterio(path: str) -> Dem:
    with rasterio.open(path) as ds:
        z = ds.read(1, masked=True).astype("float32").filled(np.nan)
        t = ds.transform
        if t.b != 0 or t.d != 0:
            raise ValueError("Rotated rasters are not supported; reproject the file first.")
        epsg = ds.crs.to_epsg() if ds.crs else None
        return Dem(z, t.c, t.f, t.a, -t.e, epsg, source=path,
                   meta={"driver": ds.driver, "crs_wkt": ds.crs.to_wkt() if ds.crs else None})


def _load_pillow(path: str) -> Dem:
    """Minimal GeoTIFF reader: single band, north-up, tiepoint + pixel scale tags."""
    from PIL import Image
    im = Image.open(path)
    z = np.array(im, dtype="float32")
    tags = im.tag_v2
    scale = tags.get(33550)
    tie = tags.get(33922)
    if not scale or not tie:
        raise ValueError(f"{path}: no georeferencing tags. Install rasterio for full GeoTIFF support.")
    keys = tags.get(34735) or ()
    epsg = None
    for i in range(4, len(keys), 4):  # GeoKeyDirectory entries: id, location, count, value
        if keys[i] in (3072, 2048) and keys[i + 1] == 0:
            epsg = int(keys[i + 3])
            if keys[i] == 3072:
                break
    nodata = tags.get(42113)
    if nodata is not None:
        try:
            z[z == float(str(nodata).strip("\x00 "))] = np.nan
        except ValueError:
            pass
    i, j, _, x, y, _ = tie[:6]
    sx, sy = scale[0], scale[1]
    return Dem(z, x - i * sx, y + j * sy, sx, sy, epsg, source=path, meta={"reader": "pillow"})


# --------------------------------------------------------------------------- window

def crop_around(dem: Dem, lon: float, lat: float, radius_m: float) -> tuple[Dem, tuple[int, int]]:
    """Cut a square window of +/- radius_m around the spot. Returns the window and the
    spot's (row, col) inside it."""
    r, c = dem.lonlat_to_pixel(lon, lat)
    csx, csy = dem.cell_size_m()
    hr, hc = int(math.ceil(radius_m / csy)), int(math.ceil(radius_m / csx))
    r0, r1 = max(0, int(r) - hr), min(dem.z.shape[0], int(r) + hr + 1)
    c0, c1 = max(0, int(c) - hc), min(dem.z.shape[1], int(c) + hc + 1)
    if r0 >= r1 or c0 >= c1 or not (0 <= r < dem.z.shape[0] and 0 <= c < dem.z.shape[1]):
        raise ValueError("The chosen point is outside the DEM.")
    sub = Dem(dem.z[r0:r1, c0:c1].copy(),
              dem.x0 + c0 * dem.dx, dem.y0 - r0 * dem.dy, dem.dx, dem.dy,
              dem.epsg, dem.source, dict(dem.meta))
    return sub, (int(round(r)) - r0, int(round(c)) - c0)


def save_geotiff(z: np.ndarray, x0: float, y0: float, dx: float, dy: float, path: str,
                 epsg: int = 4326):
    """Write a float32 north-up GeoTIFF. (x0, y0) = upper-left corner, dx/dy = pixel size
    in CRS units, epsg = CRS code. NaN becomes -9999 nodata."""
    data = np.where(np.isnan(z), -9999.0, z).astype("float32")
    if rasterio is not None:
        from rasterio.transform import from_origin
        with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1,
                           dtype="float32", crs=f"EPSG:{epsg}", nodata=-9999.0,
                           transform=from_origin(x0, y0, dx, dy), compress="deflate") as ds:
            ds.write(data, 1)
        return
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (float(dx), float(dy), 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, float(x0), float(y0), 0.0)
    geographic = epsg in (4326, 4258, 4269)
    # GeoKeyDirectory: model type (2=geographic, 1=projected), raster=pixel-is-area, CRS key
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 2 if geographic else 1, 1025, 0, 1, 1,
                  2048 if geographic else 3072, 0, 1, int(epsg))
    ifd.tagtype[34735] = 3
    ifd[42113] = "-9999"
    Image.fromarray(data, mode="F").save(path, tiffinfo=ifd)


def save_geotiff_wgs84(z, lon0, lat0, dx, dy, path):
    save_geotiff(z, lon0, lat0, dx, dy, path, 4326)


def mosaic(dems: list[Dem]) -> Dem:
    """Join tiles on the same grid (same CRS and pixel size), e.g. NLS or ArcticDEM tiles."""
    if len(dems) == 1:
        return dems[0]
    d0 = dems[0]
    for d in dems[1:]:
        if d.epsg != d0.epsg or abs(d.dx - d0.dx) > 1e-9 * max(1, abs(d0.dx)):
            raise ValueError("tiles have different CRS or resolution")
    x0 = min(d.x0 for d in dems)
    y0 = max(d.y0 for d in dems)
    x1 = max(d.x0 + d.dx * d.z.shape[1] for d in dems)
    y1 = min(d.y0 - d.dy * d.z.shape[0] for d in dems)
    w = int(round((x1 - x0) / d0.dx))
    h = int(round((y0 - y1) / d0.dy))
    z = np.full((h, w), np.nan, dtype="float32")
    for d in dems:
        r = int(round((y0 - d.y0) / d0.dy))
        c = int(round((d.x0 - x0) / d0.dx))
        blk = z[r:r + d.z.shape[0], c:c + d.z.shape[1]]
        src = d.z[:blk.shape[0], :blk.shape[1]]
        blk[np.isnan(blk)] = src[np.isnan(blk)]
    return Dem(z, x0, y0, d0.dx, d0.dy, d0.epsg, source="mosaic of %d tiles" % len(dems))


def interpolation_check(z: np.ndarray) -> dict:
    """Detect 'fake' resolution: a coarse DEM resampled (bilinear) to a fine grid looks
    detailed but is piecewise linear, so most second differences along rows/cols are ~0.
    Real lidar is never that smooth. Returns the share of 'too linear' pixels and the
    estimated true pixel size factor."""
    z = z.astype("float64")
    ok = ~np.isnan(z)
    d2r = np.abs(z[:, 2:] - 2 * z[:, 1:-1] + z[:, :-2])[ok[:, 2:] & ok[:, :-2] & ok[:, 1:-1]]
    d2c = np.abs(z[2:] - 2 * z[1:-1] + z[:-2])[ok[2:] & ok[:-2] & ok[1:-1]]
    if d2r.size < 100:
        return {"linear_share": None, "factor": None}
    tol = max(2e-4, 4e-7 * float(np.nanmax(np.abs(z))))  # float32 rounding at this height
    share = float(min((d2r < tol).mean(), (d2c < tol).mean()))
    factor = 1 / (1 - share) if share < 0.99 else 100.0
    return {"linear_share": round(share, 3), "factor": round(factor, 1)}
