"""Use a visible sparse map region to choose a distant point in embedding space.

Set SCREENSHOT_SELECTION after inspecting analysis/map-before.png. The point in the
embedding space is the unit vector minimizing a smooth maximum cosine to every
real video vector, constrained to stay close to the selected rim blend.
"""
import json
import pickle
from collections import Counter

import numpy as np
import torch
from scipy.spatial import Delaunay

from common import DATA, RAW, ROOT, place

# Replace with the center of a visually identified hole in analysis/map-before.png.
SCREENSHOT_SELECTION = (520, 355)  # visible gap between Jade/Green and Liquid/Finance groups
SCREENSHOT_SIZE = (1280, 720)
MAX_INGREDIENTS = 8
STEPS = 500
BETA = 40.0
STAY = 0.86


def circumcircle(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        return None, np.inf
    aa, bb, cc = a @ a, b @ b, c @ c
    o = np.array([
        (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d,
        (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d,
    ])
    return o, float(np.linalg.norm(a - o))


def screenshot_to_map(px, py, xy):
    W, H = SCREENSHOT_SIZE
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    avail_w, avail_h = W - 60, H - 180
    k = min(avail_w / (x1 - x0), avail_h / (y1 - y0))
    bx = W / 2 - k * (x0 + x1) / 2
    by = 130 + avail_h / 2 + k * (y0 + y1) / 2
    return np.array([(px - bx) / k, (by - py) / k])


def choose_hole(xy, X, labels, selected):
    tri = Delaunay(xy)
    candidates = []
    for simplex in tri.simplices:
        center, radius = circumcircle(*xy[simplex])
        if center is None or tri.find_simplex(center) < 0:
            continue
        if np.linalg.norm(center - selected) > radius:
            continue
        angles = np.sort(np.arctan2(xy[simplex, 1] - center[1], xy[simplex, 0] - center[0]))
        max_gap = np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max()
        if max_gap > np.deg2rad(200):
            continue
        d = np.linalg.norm(xy - center, axis=1)
        order = list(map(int, simplex)) + [int(i) for i in np.argsort(d) if d[i] <= radius * 1.45 and i not in simplex]
        per, ingredients = Counter(), []
        for i in order:
            if per[int(labels[i])] < 2:
                ingredients.append(i)
                per[int(labels[i])] += 1
            if len(ingredients) == MAX_INGREDIENTS:
                break
        if len(ingredients) >= 3:
            coherence = float((X[ingredients] @ X[ingredients].T).sum() - len(ingredients)) / (len(ingredients) * (len(ingredients) - 1))
            candidates.append((radius ** 1.4 * max(coherence - 0.35, 0.02), center, radius, ingredients, coherence))
    if not candidates:
        raise SystemExit("No enclosed Delaunay hole covers that screenshot coordinate; choose a more central visible gap.")
    return max(candidates, key=lambda item: item[0])


def push_away(X, anchor):
    Xt = torch.as_tensor(X, dtype=torch.float32)
    a = torch.as_tensor(anchor, dtype=torch.float32)
    v = a.clone().requires_grad_(True)
    opt = torch.optim.Adam([v], lr=0.01)
    for _ in range(STEPS):
        u = v / v.norm()
        crowd = torch.logsumexp(BETA * (Xt @ u), 0) / BETA
        drift = torch.relu(STAY - u @ a)
        loss = crowd + 2000 * drift ** 2
        opt.zero_grad(); loss.backward(); opt.step()
    return (v / v.norm()).detach().numpy().astype(np.float32)


def main():
    if SCREENSHOT_SELECTION is None:
        raise SystemExit("Set SCREENSHOT_SELECTION to the center of the chosen empty region in analysis/map-before.png")
    videos = json.loads((DATA / "videos.json").read_text())
    X = np.load(DATA / "embeddings.npy")
    xy = np.load(RAW / "map_xy.npy")
    labels = np.load(RAW / "labels.npy")
    names = {c["id"]: c["name"] for c in json.loads((ROOT / "docs" / "data.json").read_text())["clusters"]}
    selected = screenshot_to_map(*SCREENSHOT_SELECTION, xy)
    score, center, radius, ingredients, coherence = choose_hole(xy, X, labels, selected)
    blend = X[ingredients].mean(0); blend /= np.linalg.norm(blend)
    target = push_away(X, blend)
    proj = json.loads((RAW / "projection.json").read_text())
    reducer = pickle.load(open(RAW / "umap.pkl", "rb"))
    landing = place(reducer, target[None], proj)[0]
    sims = X @ target
    order = np.argsort(-sims)
    before = float((X @ blend).max()); after = float(sims.max())
    selected_name = " / ".join(sorted({names[int(labels[i])] for i in ingredients}))
    out = {
        "selected_at_screenshot_px": list(SCREENSHOT_SELECTION), "selected_map_xy": selected.tolist(),
        "center": center.tolist(), "radius": radius, "score": score, "coherence": coherence,
        "topics": selected_name, "ingredients": [int(i) for i in ingredients],
        "target_map_xy": landing.tolist(), "max_cosine_blend_before": before,
        "max_cosine_target_after": after, "target_cosine_anchor": float(target @ blend),
        "nearest": [{"i": int(i), "cosine": float(sims[i]), "title": videos[i]["title"], "description": videos[i].get("description", ""), "tags": videos[i].get("tags", [])} for i in order[:12]],
        "method": f"Delaunay empty circle selected from screenshot at pixel {SCREENSHOT_SELECTION}; rim embeddings were averaged to anchor the concept. Adam gradient descent ({STEPS} steps) minimized logsumexp(40 × cosine) over all {len(videos)} normalized {X.shape[1]}-dimensional video vectors, constrained to cosine ≥ {STAY:.2f} from the rim blend.",
    }
    (DATA / "gap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    np.save(DATA / "gap_target.npy", target)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
