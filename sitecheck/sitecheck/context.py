"""The questionnaire: what only the user knows. Saved as context.json and reused."""
from __future__ import annotations

import json

FIND_CATEGORIES = ["coins", "pottery_sherds", "arrowheads", "tools", "nails", "implements",
                   "horseshoes", "buttons_buckles", "brick_tile", "slag", "glass", "clay_pipes",
                   "bone", "jewellery", "other"]

PERIODS = ["prehistoric", "iron_age", "roman", "early_medieval",
           "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17", "c18", "c19", "c20"]

LAND_USE = ["forest", "ploughed_field", "pasture_meadow", "abandoned_field", "wetland",
            "built_up", "unknown"]

TEMPLATE = {
    "geography": {"country": "", "region": "", "nearest_village": ""},
    "water": {
        "known_water_nearby": "unknown",          # yes / no / unknown
        "water_kind": "",                         # river, stream, lake, spring, swamp
        "user_distance_m": None,                  # only if you know better than the map
        "other_settlements_on_same_water": "unknown",
    },
    "finds": {
        "any_finds_nearby": "unknown",            # yes / no / unknown
        "distance_from_spot_m": None,
        "periods": [],                            # from PERIODS
        "categories": [],                         # from FIND_CATEGORIES
        "description": "",
    },
    "legends": {"any_within_30km": "unknown", "text": ""},
    "historical_maps": {
        "structure_shown": "unknown",             # yes / no / unknown
        "map_name_year": "",
        "what_is_shown": "",                      # e.g. "farmstead 'Ivanovka' on 1860s map"
    },
    "land_use": "unknown",                        # from LAND_USE
    "notes": "",
}


def load(path: str | None) -> dict:
    ctx = json.loads(json.dumps(TEMPLATE))
    if path:
        with open(path, encoding="utf-8") as fh:
            _merge(ctx, json.load(fh))
    return ctx


def _merge(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v


# --------------------------------------------------------------------------- interactive

def _ask(q, default=""):
    a = input(f"{q}" + (f" [{default}]" if default else "") + ": ").strip()
    return a or default


def _yn(q):
    a = _ask(q + " (y/n/?)", "?").lower()
    return {"y": "yes", "yes": "yes", "n": "no", "no": "no"}.get(a, "unknown")


def _choose_many(q, options):
    print(q)
    for i, o in enumerate(options, 1):
        print(f"  {i:2d}. {o}")
    raw = _ask("numbers separated by commas (empty = none)")
    out = []
    for part in raw.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(options):
            out.append(options[int(part) - 1])
    return out


def interactive() -> dict:
    c = json.loads(json.dumps(TEMPLATE))
    print("Answer what you know. Enter = skip / unknown.\n")
    g = c["geography"]
    g["country"] = _ask("Country")
    g["region"] = _ask("Region")
    g["nearest_village"] = _ask("Nearest large village")

    w = c["water"]
    w["known_water_nearby"] = _yn("Do you know of a river/stream/lake/spring nearby?")
    if w["known_water_nearby"] == "yes":
        w["water_kind"] = _ask("What kind (river, stream, lake, spring, swamp)")
        d = _ask("Distance in meters, if known")
        w["user_distance_m"] = float(d) if d.replace(".", "", 1).isdigit() else None
        w["other_settlements_on_same_water"] = _yn("Other villages on the same water?")

    f = c["finds"]
    f["any_finds_nearby"] = _yn("Any finds near the spot?")
    if f["any_finds_nearby"] == "yes":
        d = _ask("Distance from the spot in meters")
        f["distance_from_spot_m"] = float(d) if d.replace(".", "", 1).isdigit() else None
        f["periods"] = _choose_many("Periods / centuries of the finds:", PERIODS)
        f["categories"] = _choose_many("Categories:", FIND_CATEGORIES)
        f["description"] = _ask("Short description")

    l = c["legends"]
    l["any_within_30km"] = _yn("Local legends about a settlement/treasure within 30 km?")
    if l["any_within_30km"] == "yes":
        l["text"] = _ask("Retell briefly")

    h = c["historical_maps"]
    h["structure_shown"] = _yn("Does an old map show a building/village at the spot?")
    if h["structure_shown"] != "unknown":
        h["map_name_year"] = _ask("Which map / year")
        h["what_is_shown"] = _ask("What is shown")

    lu = _choose_many("Land use now (pick one):", LAND_USE)
    c["land_use"] = lu[0] if lu else "unknown"
    c["notes"] = _ask("Anything else")
    return c
