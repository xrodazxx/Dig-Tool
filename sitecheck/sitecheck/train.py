"""Train the learned spot detector on marked examples (sitecheck train FOLDER).

Folder layout = the labelled pack: <category>/<area>/ with a clean picture (clear/original/plane),
a copy with red marks (mark/mix/quad in the name), other seasons, and optionally a .txt with
coordinates. Every area is first tested with a model trained WITHOUT it (honest numbers), then
the final model is trained on everything and saved as JSON (numpy-only at run time).
Needs scikit-learn for training only.
"""
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from . import evaluate as EV, groundtruth as G, learned as LM

PARAMS = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)


def collect(roots, skip_categories=()):
    """Candidate spots + features + labels for every picture of every marked area.
    Places marked as empty (pictures straight in a category folder) give only negatives."""
    roots = [roots] if isinstance(roots, str) else list(roots)
    rows = []
    for root in roots:
        tag = os.path.basename(os.path.normpath(root)) if len(roots) > 1 else ""
        for e in G.discover(root):
            if e["category"] in skip_categories:
                continue
            ref, gt, seasons = G.prepare(e)
            key = (tag + "/" if tag else "") + f"{e['category']}_{e['area']}"
            for s in seasons:
                if not s["ok"]:
                    continue
                rgb = s["rgb"]
                valid = rgb.max(-1) > 0.02
                cands, X = LM.candidates_and_features(rgb, valid, lat=e["lat"])
                if not cands:
                    continue
                XA = LM.augment(cands, X)
                scored, found = EV.match([{**c, "confidence": 0} for c in cands], gt, rgb.shape[:2],
                                         keep_ignored=True)
                keep = np.array([x[2] is not None for x in scored])      # drop spots on roads / in yards
                # spots on things the user marked as NOT a site (bushes, animals) weigh 3x in training
                la = EV._in_mask_rc(gt.get("lookalikes") or [], rgb.shape[:2]) if gt.get("lookalikes") else None
                look = np.array([bool(la is not None and la[int(min(c["row"], rgb.shape[0] - 1)),
                                                           int(min(c["col"], rgb.shape[1] - 1))])
                                 for c in cands])
                rows.append({"key": key, "pic": s["name"], "cands": [c for c, k in zip(cands, keep) if k],
                             "XA": XA[keep], "y": np.array([x[1] for x in scored])[keep],
                             "ids": [x[2] for x in scored if x[2] is not None],
                             "w": np.where(look[keep] & ~np.array([x[1] for x in scored])[keep], 3.0, 1.0),
                             "n_obj": len(gt["objects"]), "negative": bool(e.get("negative"))})
                print(f"  {key} {s['name']}: {int(keep.sum())} spots, {int(rows[-1]['y'].sum())} on marked objects"
                      + (" (empty place)" if e.get("negative") else ""), flush=True)
    return rows


def export(m):
    trees = []
    for preds in m._predictors:
        nodes = preds[0].nodes
        trees.append({"f": nodes["feature_idx"].astype(int).tolist(), "t": nodes["num_threshold"].astype(float).tolist(),
                      "l": nodes["left"].astype(int).tolist(), "r": nodes["right"].astype(int).tolist(),
                      "leaf": nodes["is_leaf"].astype(int).tolist(), "v": nodes["value"].astype(float).tolist()})
    return {"type": "gbt", "augment": True, "baseline": float(np.ravel(m._baseline_prediction)[0]), "trees": trees}


def main(roots, out=None, cv_dir=None):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
    except ImportError:
        raise SystemExit("training needs scikit-learn:  pip install scikit-learn   (using the model does not)")
    rows = collect(roots)
    if not rows:
        raise SystemExit(f"no marked areas found in {roots}")
    keys = sorted({r["key"] for r in rows if not r["negative"]})     # empty places: training only
    if len(keys) < 3:
        raise SystemExit(f"only {len(keys)} marked area(s): mark at least 3 (better 10+) so the model can be "
                         "tested on areas it has not seen")
    res = []
    for k in keys:                                   # honest test: never trained on the tested area
        tr = [r for r in rows if r["key"] != k]
        m = HistGradientBoostingClassifier(**PARAMS).fit(np.vstack([r["XA"] for r in tr]),
                                                         np.concatenate([r["y"] for r in tr]),
                                                         sample_weight=np.concatenate([r["w"] for r in tr]))
        if cv_dir:                                   # keep held-out models for end-to-end tests
            os.makedirs(cv_dir, exist_ok=True)
            with open(os.path.join(cv_dir, k.replace("/", "__") + ".json"), "w") as fh:
                json.dump(export(m), fh)
        for r in rows:
            if r["key"] != k:
                continue
            p = m.predict_proba(r["XA"])[:, 1]
            scored = [(float(a), bool(b), c) for a, b, c in zip(p, r["y"], r["ids"])]
            ap = EV.average_precision(scored, r["n_obj"])
            top = sorted(scored, key=lambda x: -x[0])[:30]
            rec = len({i for s in top for i in s[2] if i != -1}) / max(1, r["n_obj"])
            res.append((k, r["pic"], ap, rec, np.mean([s[1] for s in top])))
            print(f"  held out {k:20s} {r['pic']:18s} AP {ap:.2f}  found in top-30 {rec:.2f}  precision top-30 {res[-1][4]:.2f}")
    a = np.array([x[2:] for x in res])
    print(f"\nArea-by-area validation: AP {a[:, 0].mean():.2f}, found in top-30 {a[:, 1].mean():.2f}, "
          f"precision top-30 {a[:, 2].mean():.2f}")
    # with several packs: the harder test - train on the other packs only (new person, new imagery)
    packs = sorted({r["key"].split("/")[0] for r in rows}) if all("/" in r["key"] for r in rows) else []
    by_pack = {}
    if len(packs) > 1:
        li = LM.feature_names().index("log_resp")
        for pk in packs:
            tr = [r for r in rows if not r["key"].startswith(pk + "/")]
            m = HistGradientBoostingClassifier(**PARAMS).fit(np.vstack([r["XA"] for r in tr]),
                                                             np.concatenate([r["y"] for r in tr]),
                                                             sample_weight=np.concatenate([r["w"] for r in tr]))
            te = [r for r in rows if r["key"].startswith(pk + "/") and not r["negative"]]
            ap_m = [EV.average_precision([(float(p_), bool(b), c) for p_, b, c in
                                          zip(m.predict_proba(r["XA"])[:, 1], r["y"], r["ids"])], r["n_obj"]) for r in te]
            ap_s = [EV.average_precision([(float(p_), bool(b), c) for p_, b, c in
                                          zip(r["XA"][:, li], r["y"], r["ids"])], r["n_obj"]) for r in te]
            by_pack[pk] = {"AP_model": round(float(np.mean(ap_m)), 3), "AP_spot_strength": round(float(np.mean(ap_s)), 3),
                           "pictures": len(te)}
            print(f"trained without pack {pk}: AP {by_pack[pk]['AP_model']:.2f} on it "
                  f"(ranking by spot strength alone: {by_pack[pk]['AP_spot_strength']:.2f})")
    m = HistGradientBoostingClassifier(**PARAMS).fit(np.vstack([r["XA"] for r in rows]),
                                                     np.concatenate([r["y"] for r in rows]),
                                                     sample_weight=np.concatenate([r["w"] for r in rows]))
    model = export(m)
    model.update({"trained_on": keys, "pictures": len(rows), "features": LM.feature_names(),
                  "context": LM.CONTEXT_KEYS,
                  "validation": {"AP": round(float(a[:, 0].mean()), 3), "found_top30": round(float(a[:, 1].mean()), 3),
                                 "precision_top30": round(float(a[:, 2].mean()), 3),
                                 "other_pack": by_pack or None}})
    # self-check: our numpy predictor must reproduce scikit-learn exactly
    r0 = rows[0]
    from sitecheck.learned import _tree_raw
    raw = np.full(len(r0["XA"]), model["baseline"]) + sum(_tree_raw(r0["XA"], t) for t in model["trees"])
    ours = 1 / (1 + np.exp(-raw))
    sk = m.predict_proba(r0["XA"])[:, 1]
    print(f"numpy predictor vs scikit-learn: max difference {np.abs(ours - sk).max():.2e}")
    out = out or LM.MODEL_PATH
    with open(out, "w") as fh:
        json.dump(model, fh)
    print(f"saved {out} ({os.path.getsize(out) // 1024} KB)")

    return model
