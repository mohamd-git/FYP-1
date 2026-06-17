"""
fetch_rail_track.py — point the demo corridor at a REAL railway line.
=====================================================================
Pulls actual rail geometry near Kuala Lumpur from OpenStreetMap (Overpass API),
picks the longest contiguous line, walks ~TARGET_LEN metres along it (densified
to ~5 m spacing), and writes it to data/track.csv (lat,lng,chainage_m). The map's
OSM tiles draw that rail, so the AGV + defect pins now sit on a visible track.

Run:  python fetch_rail_track.py
"""
from __future__ import annotations

import csv
import json
import math
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AGV = Path(__file__).resolve().parent
CENTER = (3.1390, 101.6869)     # central Kuala Lumpur
RADIUS = 3000                   # metres to search
TARGET_LEN = 200.0              # metres of corridor to lay down
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def hav(a, b):
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, [a[0], a[1], b[0], b[1]])
    dla, dlo = la2 - la1, lo2 - lo1
    h = math.sin(dla / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlo / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def fetch():
    q = ('[out:json][timeout:60];'
         '(way["railway"="rail"](around:%d,%f,%f);'
         'way["railway"="light_rail"](around:%d,%f,%f););'
         'out geom;') % (RADIUS, CENTER[0], CENTER[1], RADIUS, CENTER[0], CENTER[1])
    body = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for url in ENDPOINTS:
        try:
            req = urllib.request.Request(url, data=body, headers={
                "User-Agent": "agv-poc/1.0",
                "Content-Type": "application/x-www-form-urlencoded",
            })
            return json.loads(urllib.request.urlopen(req, timeout=80).read())
        except urllib.error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode(errors="replace")
            except Exception:
                pass
            k = raw.find("Error")
            print(f"  (HTTP {e.code} {url}: {(raw[k:k+220] if k >= 0 else raw[:220]).strip()})")
            last = e
        except Exception as e:
            print(f"  (endpoint failed: {url} -> {e})")
            last = e
    raise SystemExit(f"all Overpass endpoints failed: {last}")


def main():
    print("Querying OpenStreetMap for real rail near Kuala Lumpur ...")
    data = fetch()
    ways = [e for e in data.get("elements", []) if e.get("type") == "way" and e.get("geometry")]
    if not ways:
        raise SystemExit("no rail lines returned for that area")

    def way_len(w):
        g = w["geometry"]
        return sum(hav((g[i]["lat"], g[i]["lon"]), (g[i + 1]["lat"], g[i + 1]["lon"])) for i in range(len(g) - 1))

    best = max(ways, key=way_len)
    g = [(p["lat"], p["lon"]) for p in best["geometry"]]
    print(f"  picked rail way id={best.get('id')} ({len(g)} nodes, ~{way_len(best):.0f} m available)")

    out = [(g[0][0], g[0][1], 0.0)]
    chain = 0.0
    for i in range(len(g) - 1):
        a, b = g[i], g[i + 1]
        seg = hav(a, b)
        if seg <= 0:
            continue
        steps = max(1, int(seg // 5))          # ~5 m spacing
        for s in range(1, steps + 1):
            f = s / steps
            lat, lng = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
            chain += hav((out[-1][0], out[-1][1]), (lat, lng))
            out.append((lat, lng, chain))
        if chain >= TARGET_LEN:
            break

    csv_path = AGV / "data" / "track.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lat", "lng", "chainage_m"])
        for lat, lng, ch in out:
            w.writerow([f"{lat:.7f}", f"{lng:.7f}", f"{ch:.1f}"])

    mid = out[len(out) // 2]
    print(f"\nwrote {len(out)} points -> {csv_path}  (length {out[-1][2]:.1f} m)")
    print(f"START   {out[0][0]:.6f}, {out[0][1]:.6f}")
    print(f"MAP_MID {mid[0]:.6f}, {mid[1]:.6f}    <-- use for config map_start")


if __name__ == "__main__":
    main()
