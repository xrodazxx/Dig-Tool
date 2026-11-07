"""Point cloud -> DEM grid, for people without national lidar (e.g. Russia):

  - drone photogrammetry (OpenDroneMap / WebODM, free and offline: export the point cloud
    as LAS/LAZ, or use its dtm.tif directly with --dem)
  - drone lidar, phone lidar scans (iPhone/iPad Pro apps export LAS, PLY or XYZ)
  - any XYZ / CSV text file

Formats: .xyz/.txt/.csv (x y z, any separator), .ply (ASCII or binary little-endian),
.las/.laz (needs `pip install laspy` / `laspy[lazrs]`).

Coordinates:
  - projected (UTM etc.): pass --epsg (e.g. 32637 for UTM 37N)
  - local scanner coordinates in meters (phone apps): pass --anchor LAT,LON; the scan's
    origin (or its centre with --center) is placed there, x = east, y = north.
"""
from __future__ import annotations

import math
import re
import struct

import numpy as np

from .ground import dsm_to_dtm
from .raster import save_geotiff
from .terrain import fill_nodata


def read_points(path: str) -> tuple[np.ndarray, dict]:
    low = path.lower()
    if low.endswith((".las", ".laz")):
        try:
            import laspy
        except ImportError:
            raise RuntimeError("LAS/LAZ needs laspy: pip install 'laspy[lazrs]'")
        las = laspy.read(path)
        pts = np.column_stack([las.x, las.y, las.z]).astype("float64")
        info = {"format": "las", "points": len(pts)}
        try:
            cls = np.asarray(las.classification)
            if (cls == 2).sum() > 0.05 * len(cls):   # already ground-classified
                info["ground_classified"] = True
                info["ground_points"] = int((cls == 2).sum())
                pts = pts[cls == 2]
        except Exception:  # noqa: BLE001 - classification is optional
            pass
        try:
            crs = las.header.parse_crs()
            if crs is not None and crs.to_epsg():
                info["epsg"] = crs.to_epsg()
        except Exception:  # noqa: BLE001
            pass
        return pts, info
    if low.endswith(".ply"):
        return _read_ply(path), {"format": "ply"}
    data = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)
            if len(nums) >= 3:
                data.append([float(v) for v in nums[:3]])
    if not data:
        raise ValueError(f"{path}: no x y z rows found")
    return np.array(data), {"format": "xyz text"}


def _read_ply(path):
    with open(path, "rb") as fh:
        header, props, n, fmt = [], [], 0, "ascii"
        in_vertex = False
        while True:
            line = fh.readline().decode("ascii", "ignore").strip()
            header.append(line)
            if line.startswith("format"):
                fmt = line.split()[1]
            elif line.startswith("element"):
                in_vertex = line.split()[1] == "vertex"
                if in_vertex:
                    n = int(line.split()[2])
            elif line.startswith("property") and in_vertex:
                props.append(line.split()[1:])
            elif line == "end_header":
                break
        names = [p[-1] for p in props]
        ix, iy, iz = names.index("x"), names.index("y"), names.index("z")
        if fmt == "ascii":
            rows = [fh.readline().split() for _ in range(n)]
            arr = np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows])
            return arr
        if fmt != "binary_little_endian":
            raise ValueError("PLY: only ascii and binary_little_endian are supported")
        codes = {"float": "f", "float32": "f", "double": "d", "float64": "d", "uchar": "B", "uint8": "B",
                 "char": "b", "int8": "b", "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                 "int": "i", "int32": "i", "uint": "I", "uint32": "I"}
        if any(p[0] == "list" for p in props):
            raise ValueError("PLY: list properties in vertex element are not supported")
        dtype = np.dtype([(p[-1], "<" + codes[p[0]]) for p in props])
        rec = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype, count=n)
        return np.column_stack([rec["x"], rec["y"], rec["z"]]).astype("float64")


def grid(points: np.ndarray, cell: float) -> tuple[np.ndarray, float, float]:
    """Lowest point per cell (best guess of ground under grass/bushes). Returns z, x0, y0
    (upper-left corner)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x0, y1 = x.min(), y.max()
    cols = ((x - x0) / cell).astype(int)
    rows = ((y1 - y) / cell).astype(int)
    h, w = rows.max() + 1, cols.max() + 1
    if h * w > 25_000_000:
        raise ValueError(f"grid would be {w}x{h}; use a larger --cell")
    out = np.full(h * w, np.inf)
    np.minimum.at(out, rows * w + cols, z)
    out[np.isinf(out)] = np.nan
    return out.reshape(h, w), x0, y1


def points_to_dem(path, out_path, cell=0.5, epsg=None, anchor=None, center=False,
                  ground_filter=True, max_object_m=10.0, log=print) -> dict:
    pts, info = read_points(path)
    if len(pts) < 100:
        raise ValueError("too few points")
    epsg = epsg or info.get("epsg")
    if anchor is None and epsg is None:
        raise ValueError("unknown coordinates: pass --epsg (projected) or --anchor LAT,LON (local scan)")
    if anchor is not None:
        if center:
            pts[:, :2] -= pts[:, :2].mean(axis=0)
    log(f"{len(pts):,} points, extent {np.ptp(pts[:, 0]):.1f} x {np.ptp(pts[:, 1]):.1f} m")
    z, x0, y0 = grid(pts, cell)
    holes = float(np.isnan(z).mean())
    z = fill_nodata(z)
    gf = None
    if ground_filter and not info.get("ground_classified"):
        z, gf = dsm_to_dtm(z, cell, max_object_m=max_object_m)
    if anchor is not None:
        lat, lon = anchor
        mlat, mlon = 110_574.0, 111_320.0 * math.cos(math.radians(lat))
        save_geotiff(z, lon + x0 / mlon, lat + y0 / mlat, cell / mlon, cell / mlat, out_path, 4326)
    else:
        save_geotiff(z, x0, y0, cell, cell, out_path, int(epsg))
    return {"file": out_path, "label": f"point cloud {info['format']} gridded at {cell} m",
            "cell_m": cell, "empty_cells_filled": round(holes, 3), "ground_filter": gf, **info}
