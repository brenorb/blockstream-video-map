"""Cluster and project embeddings, then export the static-site data."""
import json
import pickle
from collections import Counter
from datetime import date

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from common import DATA, MODEL_NAME, RAW, SITE, place

N_CLUSTERS = 12
K_NEIGHBORS = 8


def terms_for(videos, labels):
    docs = [" ".join(videos[i]["title"] + " " + videos[i].get("description", "") for i in np.where(labels == c)[0]) for c in range(N_CLUSTERS)]
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.75, min_df=1, token_pattern=r"[A-Za-z][A-Za-z0-9\-]+")
    M = tf.fit_transform(docs).toarray()
    vocab = np.array(tf.get_feature_names_out())
    return [list(vocab[np.argsort(-M[c])[:6]]) for c in range(N_CLUSTERS)]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    SITE.mkdir(parents=True, exist_ok=True)
    videos = json.loads((DATA / "videos.json").read_text())
    X = np.load(DATA / "embeddings.npy")
    n = len(videos)
    if n != len(X):
        raise ValueError(f"video/embedding count mismatch: {n} vs {len(X)}")
    S = X @ X.T
    np.fill_diagonal(S, -1)
    nn_idx = np.argsort(-S, axis=1)[:, :K_NEIGHBORS]
    nn_sim = S[np.arange(n), nn_idx[:, 0]]
    closer = np.array([(nn_sim > s).mean() for s in nn_sim])
    labels = KMeans(N_CLUSTERS, n_init=20, random_state=0).fit_predict(X)
    names_path = DATA / "cluster_names.json"
    terms = terms_for(videos, labels)
    names = json.loads(names_path.read_text()) if names_path.exists() else {}

    import umap
    reducer = umap.UMAP(n_neighbors=min(20, max(5, n - 1)), min_dist=0.10, metric="cosine", random_state=42)
    Y = reducer.fit_transform(X)
    with open(RAW / "umap.pkl", "wb") as f:
        pickle.dump(reducer, f)
    center = Y.mean(0)
    scale = float(np.abs(Y - center).max())
    Yn = (Y - center) / scale
    (RAW / "projection.json").write_text(json.dumps({"center": center.tolist(), "scale": scale}))
    np.save(RAW / "map_xy.npy", Yn)

    clusters = []
    for c in range(N_CLUSTERS):
        idx = np.where(labels == c)[0]
        tags = Counter(t for i in idx for t in videos[i].get("tags", [])).most_common(4)
        clusters.append({
            "id": c, "name": names.get(str(c), {}).get("name") or ", ".join(terms[c][:3]),
            "blurb": names.get(str(c), {}).get("blurb", ", ".join(terms[c][:6])),
            "size": int(len(idx)), "terms": terms[c], "top_tags": [t for t, _ in tags],
            "x": float(Yn[idx, 0].mean()), "y": float(Yn[idx, 1].mean()),
            "typical_twin": float(np.median(nn_sim[idx])),
            "examples": [videos[i]["title"] for i in idx[np.argsort(-nn_sim[idx])][:4]],
        })

    out = []
    for i, v in enumerate(videos):
        out.append({
            "i": i, "id": v.get("id"), "title": v["title"], "description": v.get("description", ""),
            "url": v["url"], "date": v.get("date", ""), "channel_id": v.get("channel_id"), "tags": v.get("tags", []),
            "categories": v.get("categories", []), "duration": v.get("duration"),
            "views": v.get("view_count"), "likes": v.get("like_count"),
            "channel": v.get("channel"), "live_status": v.get("live_status"),
            "availability": v.get("availability"), "cluster": int(labels[i]),
            "x": round(float(Yn[i, 0]), 5), "y": round(float(Yn[i, 1]), 5),
            "nn": [[int(j), round(float(S[i, j]), 3)] for j in nn_idx[i]],
            "twin": round(float(nn_sim[i]), 3), "closer": round(float(closer[i]), 4),
        })

    ghosts = []
    gpath, gvpath = DATA / "ghosts.json", DATA / "ghost_embeddings.npy"
    if gpath.exists() and gvpath.exists():
        G = np.load(gvpath)
        text_xy = place(reducer, G, {"center": center, "scale": scale})
        for gi, g in enumerate(json.loads(gpath.read_text())):
            sims = X @ G[gi]
            order = np.argsort(-sims)[:K_NEIGHBORS]
            intended = np.asarray(g.get("target_map_xy", text_xy[gi]), dtype=float)
            marker_xy = intended if g.get("place_at_target") else text_xy[gi]
            ghosts.append({
                **g, "x": round(float(marker_xy[0]), 5), "y": round(float(marker_xy[1]), 5),
                "text_x": round(float(text_xy[gi, 0]), 5), "text_y": round(float(text_xy[gi, 1]), 5),
                "nn": [[int(j), round(float(sims[j]), 3)] for j in order],
                "twin": round(float(sims[order[0]]), 3),
                "closer": round(float((nn_sim > sims[order[0]]).mean()), 4),
            })

    (SITE / "data.json").write_text(json.dumps({
        "generated": date.today().isoformat(), "model": MODEL_NAME,
        "source": "https://www.youtube.com/@Blockstream/videos",
        "stats": {"n": n, "median_twin": float(np.median(nn_sim)), "p5_twin": float(np.percentile(nn_sim, 5)),
                  "p95_twin": float(np.percentile(nn_sim, 95)),
                  "years": [min(v["date"][:4] for v in videos if v.get("date")), max(v["date"][:4] for v in videos if v.get("date"))]},
        "clusters": clusters, "posts": out, "ghosts": ghosts,
    }, ensure_ascii=False, separators=(",", ":")))
    np.save(RAW / "labels.npy", labels)
    print(f"wrote {SITE/'data.json'} videos={n} clusters={N_CLUSTERS}")
    for c in clusters:
        print(f"[{c['id']:2}] n={c['size']:3} {c['top_tags']} | {c['name']}")


if __name__ == "__main__":
    main()
