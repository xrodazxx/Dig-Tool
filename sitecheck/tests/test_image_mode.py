"""Screenshot mode against ground truth: the synthetic terrain is rendered as a fake
satellite photo (sun-lit, grass colour, patchy albedo, JPEG), then analysed.

    python tests/test_image_mode.py [runs]

The learned model is switched off here: it was trained on real screenshots and (rightly, on real
data) distrusts shading-only hollows, which is all this artificial photo has. This test checks the
hand-made detectors; real-data accuracy comes from `sitecheck evaluate <marked folder>`.
"""
import math
import os
import sys
import tempfile
import warnings

import cv2
import numpy as np

os.environ["SITECHECK_NO_MODEL"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import make_synthetic as syn  # noqa: E402
from sitecheck.image_mode import run_images  # noqa: E402
from sitecheck.terrain import hillshade  # noqa: E402

warnings.filterwarnings("ignore")


def fake_photo(seed, sun_az=165, sun_el=28):
    rng = np.random.default_rng(seed)
    z = syn.build(seed).astype("float64")
    hs = hillshade(z, 1.0, 1.0, azimuth=sun_az, altitude=sun_el, exaggeration=1.0)
    from scipy.ndimage import gaussian_filter
    albedo = 1 + gaussian_filter(rng.normal(0, 0.25, z.shape), 6) + rng.normal(0, 0.06, z.shape)
    base = np.array([0.28, 0.45, 0.22])                      # meadow green
    img = np.clip(hs[..., None] * albedo[..., None] * base * 1.9, 0, 1)
    up = cv2.resize((img * 255).astype(np.uint8), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)  # 0.333 m/px
    path = os.path.join(tempfile.gettempdir(), f"fake_photo_{seed}.jpg")
    cv2.imwrite(path, cv2.cvtColor(up, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return path


def main(runs=10):
    pits_truth = [(150, 200), (175, 215), (320, 180)]
    tot_p, tot_c, tot_false = 0, 0, 0
    for seed in range(1, runs + 1):
        path = fake_photo(seed)
        out = os.path.join(tempfile.gettempdir(), f"img_test_{seed}")
        rep = run_images([path], out, lat=60.5, lon=24.9, mpp=1 / 3, spot_px=(600, 660))
        F = rep["features"]
        hit = lambda x, y: any(math.hypot(f["x_m"] - (x - 200), f["y_m"] - (220 - y)) < 4 for f in F)
        p = sum(hit(x, y) for x, y in pits_truth)
        near_truth = lambda f: any(math.hypot(f["x_m"] - (x - 200), f["y_m"] - (220 - y)) < 8
                                   for x, y in pits_truth + [(240, 250)]) or \
            abs((f["y_m"] - (220 - 120)) + (f["x_m"] + 160) * 30 / 220) < 6   # along the road
        false = sum(1 for f in F if not near_truth(f))
        sun = rep["data"]["per_image"][0]["sun_azimuth_used"]
        print(f"run {seed:2d}: pits {p}/3, other candidates {false:2d}, sun estimated {sun} (true 165)")
        tot_p += p
        tot_c += 1
        tot_false += false
    print(f"\n{runs} runs: pits found {tot_p}/{3 * runs}, false candidates {tot_false / runs:.1f} per image")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
