"""Historical map (or any georeferenced image) under the candidates.

Old maps are often the strongest evidence: Russian PGM/Mende/RKKA sheets, Finnish
Senate maps, OS 6-inch maps, USGS historical topos. Georeference a scan once in QGIS
(free: Raster > Georeferencer) and save it as GeoTIFF; then pass --overlay map.tif.

Works with any CRS the DEM code understands (4326, built-in TM/UTM, anything with pyproj).
"""
from __future__ import annotations

import numpy as np

from .raster import Dem


def load_rgb(path: str) -> tuple[np.ndarray, Dem]:
    """Returns an (h, w, 3) uint8 image and a Dem-like georeference holder."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            bands = ds.read()[:3] if ds.count >= 3 else np.repeat(ds.read(1)[None], 3, axis=0)
            img = np.moveaxis(bands, 0, -1)
            t = ds.transform
            ref = Dem(np.zeros((ds.height, ds.width), "float32"), t.c, t.f, t.a, -t.e,
                      ds.crs.to_epsg() if ds.crs else None, source=path)
    except ImportError:
        from PIL import Image
        im = Image.open(path)
        img = np.array(im.convert("RGB"))
        ref = _georef_only(path, img.shape[:2])
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, [1, 99])
        img = np.clip((img - lo) / max(hi - lo, 1e-9) * 255, 0, 255).astype(np.uint8)
    return img, ref


def _georef_only(path, shape):
    from PIL import Image
    tags = Image.open(path).tag_v2
    scale, tie = tags.get(33550), tags.get(33922)
    if not scale or not tie:
        raise ValueError(f"{path}: not a GeoTIFF (no georeference). Save it from QGIS Georeferencer as GeoTIFF.")
    keys = tags.get(34735) or ()
    epsg = None
    for i in range(4, len(keys), 4):
        if keys[i] in (3072, 2048) and keys[i + 1] == 0:
            epsg = int(keys[i + 3])
    i, j, _, x, y, _ = tie[:6]
    return Dem(np.zeros(shape, "float32"), x - i * scale[0], y + j * scale[1], scale[0], scale[1], epsg, source=path)


def sample_onto(window: Dem, img: np.ndarray, ref: Dem) -> np.ndarray:
    """Resample the image onto the DEM window grid (nearest neighbour). Pixels outside
    the map become transparent (alpha 0)."""
    h, w = window.z.shape
    rows, cols = np.mgrid[0:h, 0:w].astype("float64")
    lon, lat = window.pixel_to_lonlat(rows.ravel(), cols.ravel())
    r, c = ref.lonlat_to_pixel(np.asarray(lon), np.asarray(lat))
    r, c = np.round(r).astype(int), np.round(c).astype(int)
    inside = (r >= 0) & (c >= 0) & (r < img.shape[0]) & (c < img.shape[1])
    out = np.zeros((h * w, 4), np.uint8)
    out[inside, :3] = img[r[inside], c[inside]]
    out[inside, 3] = 255
    return out.reshape(h, w, 4)
