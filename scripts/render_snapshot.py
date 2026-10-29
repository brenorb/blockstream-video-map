"""Render a shareable 1280x720 overview of the browser map layout for inspection."""
import json
import math
import colorsys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "docs" / "data.json").read_text())


def main(path="analysis/map-before.png", include_ideas=True):
    posts, clusters = DATA["posts"], DATA["clusters"]
    ghosts = DATA.get("ghosts", []) if include_ideas else []
    hue = {}
    for c in clusters:
        ang = math.atan2(c["y"], c["x"])
        h = ((ang / (2 * math.pi)) * 360 + 360 + 200) % 360 / 360
        hue[c["id"]] = colorsys.hls_to_rgb(h, .62, .85)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#05070c")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="#05070c")
    ax.set_xlim(-1.32, 1.32); ax.set_ylim(-1.05, 1.05); ax.axis("off")
    rng = np.random.default_rng(7)
    ax.scatter(rng.uniform(-1.3, 1.3, 220), rng.uniform(-1.0, 1.0, 220), s=2, c="#9aaac3", alpha=.22, linewidths=0)
    for p in posts:
        color = hue[p["cluster"]]
        ax.scatter(p["x"], p["y"], s=130, color=color, alpha=.10, linewidths=0)
        ax.scatter(p["x"], p["y"], s=27, color=color, alpha=.55, linewidths=0)
        ax.scatter(p["x"], p["y"], s=4, color="#ffffff", alpha=.85, linewidths=0)
    for c in clusters:
        ax.text(c["x"], c["y"], c["name"].upper(), color=hue[c["id"]], fontsize=7,
                ha="center", va="center", family="DejaVu Sans", weight="bold",
                bbox={"facecolor":"#05070c", "alpha":.55, "edgecolor":"none", "pad":2})
    for gi, g in enumerate(ghosts):
        ax.scatter(g["x"], g["y"], marker="*", s=330, color="#ffb347", edgecolor="#fff4dc", linewidth=1.2, zorder=8)
        if g.get("annotation_number") is not None:
            ax.text(g["x"], g["y"], str(g["annotation_number"]), color="#3b250b", fontsize=8,
                    weight="bold", ha="center", va="center", zorder=9)
    ax.text(.02, .965, "01   ━━━   MAP OF MEANING", transform=ax.transAxes, color="#e8edf5", fontsize=9, weight="bold", va="top")
    ax.text(.02, .925, "BLOCKSTREAM VIDEOS", transform=ax.transAxes, color="#f3f6fb", fontsize=25, weight="bold", va="top")
    st = DATA["stats"]
    ax.text(.02, .865, f"{st['n']} videos, {st['years'][0]}–{st['years'][1]}, placed by title, description and metadata.",
            transform=ax.transAxes, color="#9aa5b6", fontsize=9, va="top")
    ax.text(.02, .025, f"{DATA['model']}  ·  UMAP  ·  descriptions and metadata only", transform=ax.transAxes,
            color="#717c90", fontsize=7, va="bottom")
    out = ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="#05070c")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
