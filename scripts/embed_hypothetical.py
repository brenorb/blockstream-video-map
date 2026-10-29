"""Embed the labeled hypothetical concept and report its alignment and distance."""
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
    gap = json.loads((DATA / "gap.json").read_text())
    target = np.load(DATA / "gap_target.npy")
    sims = X @ G[0]
    nearest = int(sims.argmax())
    video_stats = json.loads((DATA.parent / "docs" / "data.json").read_text())["stats"]
    idea = ideas[0]
    idea.update({
        "title": idea["title"], "description": idea["description"],
        "tags": idea["tags"], "region": idea["region"],
        "target_cosine": round(float(G[0] @ target), 3),
        "twin": round(float(sims[nearest]), 3),
        "median_real_nearest_similarity": round(float(video_stats["median_twin"]), 3),
        "nearest_real_title": real[nearest]["title"],
        "nearest_real_id": nearest,
        "max_cosine_target": round(float(gap["max_cosine_target_after"]), 3),
        "hypothetical": True,
    })
    (DATA / "ghosts.json").write_text(json.dumps(ideas, ensure_ascii=False, indent=2))
    print(json.dumps({k: idea[k] for k in ["title", "target_cosine", "twin", "median_real_nearest_similarity", "nearest_real_title", "max_cosine_target"]}, indent=2))


if __name__ == "__main__":
    main()
