"""Turn the five browser annotations into optimized semantic gap targets."""
import json
import pickle

import numpy as np

from common import DATA, RAW, ROOT, place
from find_gap import choose_hole, push_away

# The attached screenshot is displayed at 1.5x; browser comment rectangles use
# CSS pixels. These centers come directly from the five browser comment bounds.
SCREENSHOT_SIZE = (562, 840)
SCREENSHOT_SELECTIONS = [
    (288.34765625, 514.998046875),
    (491.1640625, 467.51171875),
    (205.53515625, 303.65234375),
    (200.5, 446.546875),
    (278.43359375, 360.35546875),
]


def screenshot_to_map(px, py, xy):
    width, height = SCREENSHOT_SIZE
    mobile = width < 760
    pad_top, pad_bottom = (110, 140) if mobile else (130, 50)
    avail_w, avail_h = width - 60, height - pad_top - pad_bottom
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    scale = min(avail_w / (x1 - x0), avail_h / (y1 - y0))
    base_x = width / 2 - scale * (x0 + x1) / 2
    base_y = pad_top + avail_h / 2 + scale * (y0 + y1) / 2
    return np.array([(px - base_x) / scale, (base_y - py) / scale])


def main():
    videos = json.loads((DATA / "videos.json").read_text())
    site = json.loads((ROOT / "docs" / "data.json").read_text())
    X = np.load(DATA / "embeddings.npy")
    xy = np.load(RAW / "map_xy.npy")
    labels = np.load(RAW / "labels.npy")
    cluster_names = {c["id"]: c["name"] for c in site["clusters"]}
    reducer = pickle.load(open(RAW / "umap.pkl", "rb"))
    projection = json.loads((RAW / "projection.json").read_text())

    gaps, targets = [], []
    for number, pixel in enumerate(SCREENSHOT_SELECTIONS, start=1):
        selected = screenshot_to_map(*pixel, xy)
        score, center, radius, ingredients, coherence = choose_hole(xy, X, labels, selected)
        anchor = X[ingredients].mean(0)
        anchor /= np.linalg.norm(anchor)
        target = push_away(X, anchor)
        sims = X @ target
        order = np.argsort(-sims)
        landing = place(reducer, target[None], projection)[0]
        topics = sorted({cluster_names[int(labels[i])] for i in ingredients})
        gaps.append({
            "id": f"comment-{number}",
            "comment_number": number,
            "selected_at_screenshot_px": list(pixel),
            "selected_map_xy": selected.tolist(),
            "center": center.tolist(),
            "radius": radius,
            "score": score,
            "coherence": coherence,
            "topics": topics,
            "ingredients": [int(i) for i in ingredients],
            "target_map_xy": landing.tolist(),
            "max_cosine_blend_before": float((X @ anchor).max()),
            "max_cosine_target_after": float(sims.max()),
            "target_cosine_anchor": float(target @ anchor),
            "nearest": [
                {"i": int(i), "cosine": float(sims[i]), "title": videos[i]["title"]}
                for i in order[:12]
            ],
            "method": (
                f"Delaunay empty circle selected from browser comment {number} "
                f"at CSS pixel {pixel}; rim embeddings were averaged to anchor the target. "
                f"Adam gradient descent ({500} steps) minimized logsumexp({40} × cosine) "
                f"over all {len(videos)} normalized {X.shape[1]}-dimensional video vectors, "
                f"constrained to cosine ≥ {0.86:.2f} from the rim blend."
            ),
        })
        targets.append(target)

    (DATA / "comment_gaps.json").write_text(json.dumps(gaps, ensure_ascii=False, indent=2))
    np.save(DATA / "comment_gap_targets.npy", np.stack(targets))
    for gap in gaps:
        print(json.dumps({k: gap[k] for k in (
            "id", "selected_map_xy", "center", "radius", "target_map_xy",
            "topics", "max_cosine_target_after"
        )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
