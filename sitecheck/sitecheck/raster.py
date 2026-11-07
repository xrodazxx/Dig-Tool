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
        if self.geographic:
            return None, None
        if Transformer is None:
            raise RuntimeError(
                f"DEM is in EPSG:{self.epsg}. Install pyproj (pip install pyproj) "
                "to convert lat/lon to this projection.")
        fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        return fwd, inv

    def lonlat_to_pixel(self, lon: float, lat: float) -> tuple[float, float]:
        fwd, _ = self._transformers()
        x, y = (lon, lat) if fwd is None else fwd.transform(lon, lat)
        return self.crs_to_pixel(x, y)

    def pixel_to_lonlat(self, row: float, col: float) -> tuple[float, float]:
        x, y = self.pixel_to_crs(row, col)
        _, inv = self._transformers()
        return (x, y) if inv is None else inv.transform(x, y)


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


def save_geotiff_wgs84(z: np.ndarray, lon0: float, lat0: float, dx: float, dy: float, path: str):
    """Write a float32 GeoTIFF in EPSG:4326 (upper-left corner lon0/lat0, pixel size in degrees)."""
    if rasterio is not None:
        from rasterio.transform import from_origin
        with rasterio.open(path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1,
                           dtype="float32", crs="EPSG:4326", nodata=-9999.0,
                           transform=from_origin(lon0, lat0, dx, dy), compress="deflate") as ds:
            ds.write(np.where(np.isnan(z), -9999.0, z).astype("float32"), 1)
        return
    from PIL import Image
    from PIL.TiffImagePlugin import ImageFileDirectory_v2
    ifd = ImageFileDirectory_v2()
    ifd[33550] = (dx, dy, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, lon0, lat0, 0.0)
    ifd[34735] = (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326)
    ifd.tagtype[34735] = 3
    ifd[42113] = "-9999"
    Image.fromarray(np.where(np.isnan(z), -9999.0, z).astype("float32"), mode="F").save(path, tiffinfo=ifd)
