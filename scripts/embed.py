"""Embed public video descriptions and metadata with Qwen3-Embedding."""
import json

import numpy as np

from common import DATA, embed


def text(v):
    # No audio, transcript, thumbnail, or frame content is used.
    return "\n".join(x for x in [
        v["title"], v.get("description", ""),
        "Categories: " + ", ".join(v.get("categories", [])),
        "Topics: " + ", ".join(v.get("tags", [])),
    ] if x)


def main():
    videos = json.loads((DATA / "videos.json").read_text())
    X = embed([text(v) for v in videos])
    np.save(DATA / "embeddings.npy", X)
    print("embeddings", X.shape)


if __name__ == "__main__":
    main()
