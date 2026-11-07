"""Built-in Transverse Mercator (Krüger series, mm accuracy) so the common national
grids work even without pyproj:

  EPSG:3067  ETRS-TM35FIN (Finland, NLS data)
  EPSG:326zz / 327zz  WGS84 UTM north / south (drone data, ArcticDEM-free areas)
  EPSG:258zz ETRS89 UTM (Europe)
  EPSG:3006  SWEREF99 TM (Sweden)

ETRS89 and WGS84 differ by well under a metre in these regions, which is irrelevant for
locating a spot. British National Grid (27700) needs a datum shift: use pyproj for it.
"""
from __future__ import annotations

import math

import numpy as np

A_WGS84, F_WGS84 = 6378137.0, 1 / 298.257223563
A_GRS80, F_GRS80 = 6378137.0, 1 / 298.257222101


def tm_params(epsg: int):
    """(a, f, lon0_deg, k0, false_easting, false_northing) or None."""
    if epsg == 3067:
        return A_GRS80, F_GRS80, 27.0, 0.9996, 500000.0, 0.0
    if epsg == 3006:
        return A_GRS80, F_GRS80, 15.0, 0.9996, 500000.0, 0.0
    if 32601 <= epsg <= 32660:
        return A_WGS84, F_WGS84, -183 + 6 * (epsg - 32600), 0.9996, 500000.0, 0.0
    if 32701 <= epsg <= 32760:
        return A_WGS84, F_WGS84, -183 + 6 * (epsg - 32700), 0.9996, 500000.0, 10000000.0
    if 25828 <= epsg <= 25838:
        return A_GRS80, F_GRS80, -183 + 6 * (epsg - 25800), 0.9996, 500000.0, 0.0
    return None


def supported(epsg) -> bool:
    return epsg is not None and tm_params(int(epsg)) is not None


def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _series(f):
    n = f / (2 - f)
    A = (1 / (1 + n)) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = [n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16 + 41 * n ** 4 / 180,
             13 * n ** 2 / 48 - 3 * n ** 3 / 5 + 557 * n ** 4 / 1440,
             61 * n ** 3 / 240 - 103 * n ** 4 / 140,
             49561 * n ** 4 / 161280]
    beta = [n / 2 - 2 * n ** 2 / 3 + 37 * n ** 3 / 96 - n ** 4 / 360,
            n ** 2 / 48 + n ** 3 / 15 - 437 * n ** 4 / 1440,
            17 * n ** 3 / 480 - 37 * n ** 4 / 840,
            4397 * n ** 4 / 161280]
    delta = [2 * n - 2 * n ** 2 / 3 - 2 * n ** 3 + 116 * n ** 4 / 45,
             7 * n ** 2 / 3 - 8 * n ** 3 / 5 - 227 * n ** 4 / 45,
             56 * n ** 3 / 15 - 136 * n ** 4 / 35,
             4279 * n ** 4 / 630]
    return n, A, alpha, beta, delta


def forward(epsg, lon, lat):
    """lon/lat degrees (scalars or arrays) -> easting, northing."""
    a, f, lon0, k0, fe, fn = tm_params(int(epsg))
    n, A, alpha, _, _ = _series(f)
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    phi, lam = np.radians(lat), np.radians(lon - lon0)
    e = math.sqrt(f * (2 - f))
    t = np.sinh(np.arctanh(np.sin(phi)) - e * np.arctanh(e * np.sin(phi)))
    xi_p = np.arctan2(t, np.cos(lam))
    eta_p = np.arctanh(np.sin(lam) / np.sqrt(1 + t * t))
    xi, eta = xi_p.copy(), eta_p.copy()
    for j, al in enumerate(alpha, 1):
        xi = xi + al * np.sin(2 * j * xi_p) * np.cosh(2 * j * eta_p)
        eta = eta + al * np.cos(2 * j * xi_p) * np.sinh(2 * j * eta_p)
    E = fe + k0 * a * A * eta
    N = fn + k0 * a * A * xi
    return (float(E), float(N)) if E.ndim == 0 else (E, N)


def inverse(epsg, x, y):
    """easting, northing -> lon, lat degrees."""
    a, f, lon0, k0, fe, fn = tm_params(int(epsg))
    n, A, _, beta, delta = _series(f)
    x, y = np.asarray(x, float), np.asarray(y, float)
    xi = (y - fn) / (k0 * a * A)
    eta = (x - fe) / (k0 * a * A)
    xi_p, eta_p = xi.copy(), eta.copy()
    for j, be in enumerate(beta, 1):
        xi_p = xi_p - be * np.sin(2 * j * xi) * np.cosh(2 * j * eta)
        eta_p = eta_p - be * np.cos(2 * j * xi) * np.sinh(2 * j * eta)
    chi = np.arcsin(np.sin(xi_p) / np.cosh(eta_p))
    phi = chi.copy()
    for j, de in enumerate(delta, 1):
        phi = phi + de * np.sin(2 * j * chi)
    lam = np.arctan2(np.sinh(eta_p), np.cos(xi_p))
    lon, lat = lon0 + np.degrees(lam), np.degrees(phi)
    return (float(lon), float(lat)) if lon.ndim == 0 else (lon, lat)
