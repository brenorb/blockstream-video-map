"""Embed each hypothetical concept and compare it with its selected gap target."""
import json
import numpy as np

from common import DATA, embed


def text(v):
    return "\n".join(x for x in [
        v["title"], v.get("description", ""),
        "Categories: " + ", ".join(v.get("categories", [])),
        "Topics: " + ", ".join(v.get("tags", [])),
    ] if x)


def main():
    ideas = json.loads((DATA / "hypothetical_video.json").read_text())
    real = json.loads((DATA / "videos.json").read_text())
    X = np.load(DATA / "embeddings.npy")
    G = embed([text(v) for v in ideas])
    np.save(DATA / "ghost_embeddings.npy", G)
    video_stats = json.loads((DATA.parent / "docs" / "data.json").read_text())["stats"]
    old_gap = json.loads((DATA / "gap.json").read_text())
    comment_gaps = json.loads((DATA / "comment_gaps.json").read_text())
    comment_targets = np.load(DATA / "comment_gap_targets.npy")
    gap_by_id = {gap["id"]: (gap, comment_targets[i]) for i, gap in enumerate(comment_gaps)}
    targets, reports = [], []
    for i, idea in enumerate(ideas):
        if idea["gap_id"] == "original":
            gap, target = old_gap, np.load(DATA / "gap_target.npy")
        else:
            gap, target = gap_by_id[idea["gap_id"]]
        targets.append(target)
        video_sims = X @ G[i]
        nearest = int(video_sims.argmax())
        idea.update({
            "target_cosine": round(float(G[i] @ target), 3),
            "twin": round(float(video_sims[nearest]), 3),
            "median_real_nearest_similarity": round(float(video_stats["median_twin"]), 3),
            "nearest_real_title": real[nearest]["title"],
            "nearest_real_id": nearest,
            "max_cosine_target": round(float(gap["max_cosine_target_after"]), 3),
            "target_map_xy": gap["target_map_xy"],
            "gap_center": gap["center"],
            "gap_radius": gap["radius"],
            "ingredients": gap["ingredients"],
            "ingredient_sims": {
                str(j): round(float(X[j] @ G[i]), 3) for j in gap["ingredients"]
            },
            "method": gap["method"],
            "hypothetical": True,
            "place_at_target": True,
        })
        reports.append({
            "gap_id": idea["gap_id"], "title": idea["title"],
            "target_cosine": idea["target_cosine"], "twin": idea["twin"],
            "nearest_real_title": idea["nearest_real_title"],
            "target_xy": gap["target_map_xy"],
        })
    (DATA / "ghosts.json").write_text(json.dumps(ideas, ensure_ascii=False, indent=2))
    np.save(DATA / "ghost_targets.npy", np.stack(targets))
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
