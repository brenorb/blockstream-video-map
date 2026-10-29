# Blockstream Video Map

An interactive semantic map of the public videos on [Blockstream’s YouTube channel](https://www.youtube.com/@Blockstream/videos). Real videos are colored by topic. A larger gold star marks one explicitly hypothetical concept generated for a sparse region of the map.

This is an independent project and is not affiliated with Blockstream.

## Source and coverage

The source is the official `@Blockstream` videos playlist (`UCZNt3fZazX9cwWcC9vjDJ4Q`). The scraper records titles, descriptions, upload dates, duration, view and like counts when available, channel, tags, categories, live status, availability, and YouTube URL. It does not request video, audio, subtitles, thumbnails, or frames. The exact number of entries returned and any errors are saved in `data/scrape_report.json`.

YouTube can change public counts and availability. The site represents the playlist as returned at the capture date in the report.

## Method

1. `scripts/scrape.py` reads metadata and descriptions from the public channel with `yt-dlp`.
2. `scripts/embed.py` embeds each title, description, category, and tag with [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), a 22.7M-parameter Apache-2.0 sentence embedding model. It does not use transcripts or visual media.
3. `scripts/build_map.py` groups embeddings into topics and projects them to two dimensions with cosine UMAP. Similarity and nearest-video links use cosine similarity in the original embedding space.
4. `docs/` is a static interactive site. Search, zoom, topic coloring, selection cards, descriptions, metadata, and nearest neighbors all run in the browser.
5. A screenshot of the rendered map is saved in `analysis/`. Its visible sparse regions guide the candidate search. The chosen location is then optimized mathematically to lower its maximum cosine similarity to all real video embeddings while remaining anchored to the selected region. A hypothetical video concept is written from the neighboring videos’ descriptions and metadata, embedded with the same model, and shown as a larger gold star. It is clearly marked as generated and is not represented as an actual Blockstream video.

UMAP axes have no intrinsic labels: proximity is an approximate 2D view of semantic similarity, while the exact neighbor scores are computed in the original high-dimensional embedding space.

## Rebuild

Requires Python 3.12+, `uv`, and `yt-dlp`.

```sh
uv sync
uv run python scripts/scrape.py
uv run python scripts/embed.py
uv run python scripts/build_map.py
# Review analysis/map-before.png and select a visible sparse region.
uv run python scripts/find_gap.py
uv run python scripts/embed_hypothetical.py
uv run python scripts/build_map.py
uv run python scripts/render_snapshot.py
python3 -m http.server --directory docs 8765
```

GitHub Actions deploys the static `docs/` directory to GitHub Pages on pushes to `main`.
