"""Scrape public metadata and descriptions from the official Blockstream YouTube channel.

No video, audio, subtitles, thumbnails, or frames are requested. Requires yt-dlp.
"""
import json
import argparse
import shutil
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHANNEL = "https://www.youtube.com/@Blockstream/videos"
FIELDS = ["id", "title", "description", "upload_date", "duration", "view_count", "like_count", "channel", "channel_id", "webpage_url", "categories", "tags", "live_status", "availability"]
FORMAT = chr(31).join(f"%({field})j" for field in FIELDS)


def main():
    if not shutil.which("yt-dlp"):
        raise SystemExit("yt-dlp is required (brew install yt-dlp or pip install yt-dlp)")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-tsv", help="parse a reduced metadata capture produced by yt-dlp --print")
    parser.add_argument("--manifest", help="channel manifest JSON for --from-tsv mode")
    opts = parser.parse_args()
    if opts.from_tsv:
        stdout = Path(opts.from_tsv).read_text()
        manifest_data = json.loads(Path(opts.manifest).read_text()) if opts.manifest else {}
        stderr, returncode = "", 0
    else:
        manifest = subprocess.run(["yt-dlp", "--flat-playlist", "--dump-single-json", "--no-warnings", CHANNEL], text=True, capture_output=True)
        manifest_data = json.loads(manifest.stdout)
        stdout = None
    expected_ids = {e["id"] for e in manifest_data.get("entries", []) if e.get("id")}
    args = ["yt-dlp", "--skip-download", "--ignore-errors", "--no-warnings", "--no-check-formats", "--print", FORMAT, CHANNEL]
    proc = subprocess.run(args, text=True, capture_output=True) if stdout is None else None
    if proc:
        stdout, stderr, returncode = proc.stdout, proc.stderr, proc.returncode
    videos, errors = [], []
    for line in stdout.splitlines():
        try:
            sep = chr(31) if chr(31) in line else r"\t"
            cells = line.split(sep)
            # Older yt-dlp capture used literal \\t separators. Rejoin any such
            # escaped tabs inside the JSON-encoded description field.
            if sep == r"\t" and len(cells) > len(FIELDS):
                cells = cells[:2] + [sep.join(cells[2:-11])] + cells[-11:]
            if len(cells) != len(FIELDS):
                raise ValueError(f"expected {len(FIELDS)} fields; got {len(cells)}")
            row = [None if cell in ("", "NA") else json.loads(cell) for cell in cells]
            p = dict(zip(FIELDS, row))
            if p["id"]:
                p["url"] = p.pop("webpage_url") or f"https://www.youtube.com/watch?v={p['id']}"
                raw_date = (p.pop("upload_date") or "")[:8]
                p["date"] = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:8]}" if len(raw_date) == 8 else ""
                p["description"] = p["description"] or ""
                p["tags"] = p["tags"] or []
                p["categories"] = p["categories"] or []
                videos.append(p)
        except Exception as exc:
            errors.append({"line": line[:240], "error": str(exc)})
    dedup = {v["id"]: v for v in videos}
    videos = sorted(dedup.values(), key=lambda v: (v["date"], v["id"]))
    DATA.mkdir(exist_ok=True)
    (DATA / "videos.json").write_text(json.dumps(videos, ensure_ascii=False, indent=1))
    scraped_ids = {v["id"] for v in videos}
    report = {"source": CHANNEL, "channel": manifest_data.get("channel") or "Blockstream", "channel_id": manifest_data.get("channel_id") or "UCZNt3fZazX9cwWcC9vjDJ4Q", "scraped_at": date.today().isoformat(), "playlist_entries": len(expected_ids) if expected_ids else manifest_data.get("playlist_count"), "metadata_records": len(videos), "videos_with_description": sum(bool(v["description"]) for v in videos), "missing_metadata_ids": sorted(expected_ids - scraped_ids), "metadata_parse_errors": errors, "yt_dlp_returncode": returncode, "yt_dlp_stderr_tail": stderr[-3000:]}
    (DATA / "scrape_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not videos:
        raise SystemExit("No videos were scraped")

if __name__ == "__main__":
    main()
