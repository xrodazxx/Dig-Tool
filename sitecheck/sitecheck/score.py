"""Transparent evidence score: a sum of points, every point explained.

It answers "how many independent signs of past human presence are there?",
not "what is the probability". Missing data gives 0 points, never a penalty
(except when the user states a negative, e.g. an old map shows nothing)."""
from __future__ import annotations

import json
import math
from importlib import resources


def load_weights(path: str | None = None) -> dict:
    if path:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(resources.files("sitecheck").joinpath("weights.json").read_text("utf-8"))


def _band_points(value, table):
    for lim, pts in table:
        if value < lim:
            return pts
    return 0


def _distance_factor(f, w):
    """Features right at the spot matter most; far ones only add context."""
    d = f.get("distance_from_spot_m", 0)
    for lim, k in w.get("feature_distance_factor", [[1e12, 1.0]]):
        if d < lim:
            return k
    return 0.0


def compute(spot: dict, water: dict, features: list[dict], ctx: dict, w: dict) -> dict:
    items = []

    def add(factor, points, why, source):
        if points:
            items.append({"factor": factor, "points": round(points, 1), "why": why, "source": source})

    # --- landscape ---------------------------------------------------------------
    d = water.get("distance_m")
    if d is not None:
        add("water_distance", _band_points(d, w["water_distance_m"]),
            f"water ({water.get('kind', '?')}) at {d:.0f} m", water.get("source", "?"))
    if spot.get("landform") == "valley bottom":
        add("flood_risk", w["valley_bottom_flood_risk"], "spot is on a valley bottom", "computed:DEM")
    if spot.get("slope_deg") is not None:
        add("slope", _band_points(spot["slope_deg"], w["slope_deg"]),
            f"slope {spot['slope_deg']} deg", "computed:DEM")
        if spot["slope_deg"] >= 2 and spot.get("aspect") in w["sunny_aspects"]:
            add("aspect", w["sunny_aspect"], f"faces {spot['aspect']}", "computed:DEM")
    add("landform", w["good_landform"].get(spot.get("landform"), 0), spot.get("landform"), "computed:DEM")

    # --- micro-relief features -----------------------------------------------------
    mult = w["feature_multiplier"]

    def feat_points(kind, cfg, extra=lambda f: True):
        pts, used = 0.0, []
        for f in features:
            if f["type"] != kind or f.get("inside_feature") or not extra(f):
                continue
            m = mult.get(f.get("user_status", "unreviewed"), 0) * _distance_factor(f, w)
            if f.get("n_images", 1) > 1 and len(f.get("seen_in_images", [1, 2])) == 1 \
                    and f.get("user_status") != "confirmed":
                m *= 0.5          # screenshots: seen in only one of several pictures
            # several pictures: the list is ranked by how many pictures show a spot; for points the
            # earlier rule stays (strongest picture + 0.2 for every further picture)
            strength = f["confidence"]
            if "confidence_best_picture" in f:
                strength = max(strength, f["confidence_best_picture"] + 0.2 * (len(f.get("seen_in_images", [1])) - 1))
            if m and (strength >= cfg["min_confidence"] or f.get("user_status") == "confirmed"):
                pts += cfg["each"] * m
                used.append(f["id"])
        return min(round(pts, 1), cfg["max"]), used

    p, ids = feat_points("pit", w["pit"])
    add("pits", p, f"pits #{', #'.join(map(str, ids))}", "computed:DEM + user review")
    p, ids = feat_points("rectangle", w["rectangle"])
    add("rectangles", p, f"rectangular features #{', #'.join(map(str, ids))}", "computed:DEM + user review")
    cfg = w["linear_straight"]
    p, ids = feat_points("linear", cfg, lambda f: f.get("straightness", 0) >= cfg["min_straightness"])
    add("straight_lines", p, f"straight linear features #{', #'.join(map(str, ids))}", "computed:DEM + user review")

    cb = w["pit_cluster_bonus"]
    pits = [f for f in features if f["type"] == "pit" and f.get("user_status") != "rejected"
            and "x_m" in f]
    for a in pits:
        near = [b for b in pits if math.hypot(a["x_m"] - b["x_m"], a["y_m"] - b["y_m"]) <= cb["within_m"]]
        if len(near) >= cb["min_pits"]:
            add("pit_cluster", cb["points"], f"{len(near)} pits within {cb['within_m']} m (a row of houses?)",
                "computed:DEM")
            break

    # --- screenshot marks: fields and woodland -------------------------------------------------
    live = [f for f in features if f.get("user_status") != "rejected"]
    soil = [f for f in live if f.get("kind") in ("soil_dark", "soil_light", "crop_green", "crop_parched")]
    if soil:
        # Mottled soil gives colour patches anywhere (tested on two marked "nothing ever stood here"
        # fields: 12/12 points). A patch that comes back in several years is worth more.
        n_img = max([f.get("n_images", 1) for f in soil] + [1])
        lasting = [f for f in soil if len(f.get("seen_in_images", [1])) >= 2] if n_img > 1 else []
        clustered = any(f.get("cluster_size", 0) >= 4 for f in (lasting or soil))
        if lasting:
            pts = min(10, 2.5 * len(lasting)) * (1.0 if clustered else 0.6)
            why = f"{len(lasting)} soil/crop marks seen in 2+ pictures" + (" in a cluster" if clustered else "")
        else:
            pts = min(4, 1.0 * len(soil))
            why = f"{len(soil)} soil/crop marks in one picture only (patchy soil does this too)"
        add("soil_crop_marks", pts, why, "computed:image")
    grid = [f for f in live if f.get("kind") == "tree_row" and f.get("parallel_rows", 0) >= 3]
    if grid:
        add("tree_grid", 6, f"{grid[0]['parallel_rows']} parallel tree rows (planted grid)", "computed:image")
    woods = [f for f in live if f.get("kind") in ("forest_clearing", "canopy_anomaly")]
    if woods:
        add("woodland_anomalies", min(6, 2 * len(woods)), f"{len(woods)} clearings / unusual tree patches",
            "computed:image")

    rws = [f for f in live if f.get("type") == "row" and f.get("chance", 1) <= 0.20]
    if rws:
        best = min(rws, key=lambda f: f["chance"])
        pts = 15 if best["chance"] <= 0.03 else 10 if best["chance"] < 0.10 else 3
        if best.get("along_cultivation"):
            pts /= 2
        # a row in ONE picture: on 23 marked hard empty fields (patchy, streaky soil) 4 such rows
        # passed the chance test, so one picture alone is worth at most 6 points
        one_pic = len(best.get("seen_in_images", [1, 2])) < 2
        if one_pic:
            pts = min(pts, 6)
        if any(f.get("parallel_rows", 0) >= 2 for f in rws):
            pts += 5
        add("regular_rows", pts, f"row of {best['n_patches']} patches every {best['spacing_m']:.0f} m "
            f"(random chance {best['chance'] * 100:.0f}%" + (", one picture only)" if one_pic else ")"),
            "computed:image")

    # --- what the user knows -----------------------------------------------------------
    f = ctx["finds"]
    if f["any_finds_nearby"] == "yes":
        add("finds", w["finds_yes"], "finds reported nearby", "user")
        diag = set(f["categories"]) & set(w["finds_diagnostic_categories"])
        if diag:
            add("finds_diagnostic", w["finds_diagnostic"], "settlement-type finds: " + ", ".join(sorted(diag)), "user")
        if f.get("distance_from_spot_m") is not None and f["distance_from_spot_m"] <= w["finds_close_m"]:
            add("finds_close", w["finds_close"], f"finds within {f['distance_from_spot_m']:.0f} m", "user")
    if ctx["legends"]["any_within_30km"] == "yes":
        add("legend", w["legend_yes"], "local legend within 30 km", "user")
    hm = ctx["historical_maps"]["structure_shown"]
    if hm == "yes":
        add("historical_map", w["historical_map_yes"], "old map shows a structure here", "user")
    elif hm == "no":
        add("historical_map", w["historical_map_no"], "old map shows nothing here", "user")
    if ctx["water"]["other_settlements_on_same_water"] == "yes":
        add("same_water", w["same_water_settlements"], "other villages on the same water", "user")

    total = max(0, min(100, sum(i["points"] for i in items)))
    band = next(lbl for lim, lbl in w["bands"] if total < lim)
    unknown = [k for k, v in (("finds", f["any_finds_nearby"]), ("historical_map", hm),
                              ("water", "known" if d is not None else "unknown")) if v == "unknown"]
    return {
        "points": round(total, 1),
        "band": band,
        "kind": "heuristic evidence score (0-100), NOT a probability",
        "weights_version": w["version"],
        "contributions": sorted(items, key=lambda i: -abs(i["points"])),
        "unknown_inputs": unknown,
    }
