"""Writes report.json, prompt.md and the pictures the user (and an external AI) looks at."""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"pit": "#e8452c", "rectangle": "#1f6fff", "linear": "#f2b300"}


def write_outputs(report, layers, rc, cell, features, out_dir):
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "prompt.md"), "w", encoding="utf-8") as fh:
        fh.write(build_prompt(report))
    np.savez_compressed(os.path.join(out_dir, "layers.npz"),
                        **{k: v for k, v in layers.items() if v is not None},
                        spot_rc=np.array(rc), cell=np.array(cell))
    with open(os.path.join(out_dir, "features_full.json"), "w", encoding="utf-8") as fh:
        json.dump(features, fh, indent=1)   # includes pixel geometry used for drawing
    render(layers, rc, cell, features, out_dir)


def _extent(shape, rc, cell):
    h, w = shape
    return [(-0.5 - rc[1]) * cell, (w - 0.5 - rc[1]) * cell,
            (rc[0] - h + 0.5) * cell, (rc[0] + 0.5) * cell]


def _robust(a, p=2):
    lo, hi = np.nanpercentile(a, [p, 100 - p])
    return lo, hi


def _draw_features(ax, features, rc, cell, labels=True):
    to_xy = lambda r, c: ((c - rc[1]) * cell, (rc[0] - r) * cell)
    for f in features:
        if f.get("user_status") == "rejected":
            continue
        col = COLORS[f["type"]]
        style = "-" if f.get("user_status") == "confirmed" else "--"
        if f["type"] == "pit":
            x, y = to_xy(f["row"], f["col"])
            ax.add_patch(plt.Circle((x, y), max(f["diameter_m"], 4) / 2 + 2, fill=False, ec=col,
                                    lw=1.4, ls=style))
        elif f["type"] == "rectangle" and "corners_px" in f:
            pts = [to_xy(r, c) for c, r in f["corners_px"]]
            ax.add_patch(plt.Polygon(pts, fill=False, ec=col, lw=1.4, ls=style))
            x, y = to_xy(f["row"], f["col"])
        else:
            (r1, c1), (r2, c2) = f["ends_px"]
            (x1, y1), (x2, y2) = to_xy(r1, c1), to_xy(r2, c2)
            ax.plot([x1, x2], [y1, y2], color=col, lw=1.2, ls=style, alpha=.9)
            x, y = to_xy(f["row"], f["col"])
        if labels:
            ax.annotate(str(f["id"]), (x, y), xytext=(6, 6), textcoords="offset points", color="white",
                        fontsize=8, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=.85))


def render(layers, rc, cell, features, out_dir):
    ext = _extent(layers["z"].shape, rc, cell)

    # 1) main picture: local relief over hillshade, with numbered candidates
    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    from scipy.ndimage import gaussian_filter
    lrm_s = gaussian_filter(layers["lrm"], max(0.7, 0.7 / cell))
    lim = max(4 * float(np.median(np.abs(lrm_s))) * 1.48, 0.08)
    ax.imshow(lrm_s, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=ext)
    ax.imshow(layers["hillshade"], cmap="gray", extent=ext, vmin=0, vmax=1, alpha=.45)
    _draw_features(ax, features, rc, cell)
    ax.plot(0, 0, marker="+", ms=22, mew=2.5, color="black")
    ax.plot(0, 0, marker="+", ms=18, mew=1.2, color="white")
    ax.set_title("Local relief (red = raised, blue = sunken) over hillshade\n"
                 "numbered = candidates, dashed = unreviewed, + = your spot")
    ax.set_xlabel("meters east of the spot")
    ax.set_ylabel("meters north of the spot")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "candidates.png"))
    plt.close(fig)

    if layers.get("overlay") is not None:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
        ax.imshow(layers["hillshade"], cmap="gray", extent=ext, vmin=0, vmax=1)
        ax.imshow(layers["overlay"], extent=ext, alpha=.85)
        _draw_features(ax, features, rc, cell)
        ax.plot(0, 0, marker="+", ms=22, mew=2.5, color="black")
        ax.plot(0, 0, marker="+", ms=18, mew=1.2, color="white")
        ax.set_title("Historical map with candidates (+ = your spot)")
        ax.set_xlabel("meters east of the spot")
        ax.set_ylabel("meters north of the spot")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "overlay.png"))
        plt.close(fig)

    # 2) four standard visualisations for the eye and for a multimodal AI
    panels = [("Multi-direction hillshade (contrast-stretched)", layers["hillshade"], "gray",
               _robust(layers["hillshade"], 0.5)),
              ("Local relief model, m", layers["lrm"], "RdBu_r", (-lim, lim)),
              ("Slope, degrees", layers["slope"], "magma_r", _robust(layers["slope"])),
              ("Sky-view factor (dark = enclosed)" if layers["svf"] is not None else "Elevation, m",
               layers["svf"] if layers["svf"] is not None else layers["z"],
               "gray" if layers["svf"] is not None else "terrain",
               _robust(layers["svf"] if layers["svf"] is not None else layers["z"]))]
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=100)
    for ax, (title, arr, cmap, (lo, hi)) in zip(axs.flat, panels):
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi, extent=ext)
        ax.plot(0, 0, marker="+", ms=14, mew=2, color="cyan")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.02)
    fig.suptitle("Terrain visualisations around the spot (axes in meters)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "terrain_panels.png"))
    plt.close(fig)


def build_prompt(report) -> str:
    sc = report["score"]
    lines = [
        "# Site assessment request",
        "",
        "I am checking whether a specific spot was once a settlement, farmstead, work site or other "
        "place of human activity. Below is a JSON report produced by an offline tool from a terrain "
        "model (DEM / lidar) and my own field knowledge. Attached images:",
        "- `candidates.png` - local relief model over hillshade; numbered shapes are automatic "
        "candidates (dashed = I have not checked them yet), `+` is the spot;",
        "- `terrain_panels.png` - hillshade, local relief, slope, sky-view factor.",
    ]
    if report["data"].get("overlay"):
        lines.append(f"- `overlay.png` - the historical map `{report['data']['overlay']}` with the same candidates.")
    if report["data"].get("user_image"):
        lines.append(f"- `{report['data']['user_image']}` - my own image of the area.")
    lines += [
        "",
        "Please:",
        "1. Interpret each numbered feature: what it most likely is (house pit, cellar, foundation, "
        "road, ditch, field boundary, charcoal pit, crater, natural feature...) and why.",
        "2. Say whether together they suggest a settlement, a single farmstead, a seasonal or work "
        "site, or nothing - and roughly which period, using my finds and the landscape.",
        "3. If a settlement is likely, estimate its size (households / inhabitants) and explain the basis.",
        "4. List what would most change your conclusion (what I should check on the ground or on old maps).",
        "5. Be explicit about uncertainty. Treat `confidence` as shape quality, not as proof, "
        f"and the score ({sc['points']}/100, {sc['band']}) as a crude heuristic.",
        "",
        "Field notes: values with source `user` come from me; `computed` values were calculated "
        "from the DEM; `x_m`/`y_m` are meters east/north of the spot.",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=1,
                   default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        "```",
        "",
    ]
    return "\n".join(lines)
