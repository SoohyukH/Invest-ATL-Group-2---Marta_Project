"""
Complete MARTA + Event Venue + Local Business Analysis Pipeline
Author: You

This script performs the full spatial analysis workflow used in the DataLab project:

1. Load MARTA Stops (shapefile)
2. Load event venues (CSV). If the file does not exist, create a template automatically.
3. Load local business locations (CSV). If the file does not exist, create a template.
4. Compute the nearest MARTA stop to each event venue, with distances in meters and miles.
5. Build accessibility buffers (0.5-mile and 1-mile) around event venues based on nearby MARTA stops.
6. Count how many businesses fall inside each buffer (grouped by category).
7. Export results to CSV + GeoJSON + interactive HTML map (Folium).

This script produces 100% ready-to-use deliverables for analysis, mapping, and reporting.
"""

import os
import math
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point



# ======================================================
# CONFIGURATION (edit only if needed)
# ======================================================

MILES_TO_METERS = 1609.34
BUFFER_RADII_METERS = [0.5 * MILES_TO_METERS, 1.0 * MILES_TO_METERS]  # 0.5 mile, 1 mile

ROOT = os.path.dirname(os.path.abspath(__file__))

MARTA_SHP = os.path.join(ROOT, "data", "MARTA_Stops.shp")

BUSINESS_CSV = os.path.join(ROOT, "businesses.csv")

# Output directory
OUT_DIR = os.path.join(ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Paths for all output files
OUT_STOPS_GEOJSON = os.path.join(OUT_DIR, "MARTA_Stops.geojson")
OUT_STOPS_CSV    = os.path.join(OUT_DIR, "MARTA_Stops.csv")
OUT_NEAREST      = os.path.join(OUT_DIR, "proximity_summary.csv")
OUT_BIZ_COUNTS   = os.path.join(OUT_DIR, "business_counts_by_buffer.csv")
OUT_MAP          = os.path.join(OUT_DIR, "event_marta_access_map.html")



# ======================================================
# 1. Load Data
# ======================================================

def load_marta_stops(shp_path):
    """Load MARTA shapefile and return both EPSG:4326 and EPSG:3857 versions."""
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    gdf = gpd.read_file(shp_path)

    # Assign CRS if missing
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)

    gdf_4326 = gdf.to_crs(epsg=4326)
    gdf_3857 = gdf_4326.to_crs(epsg=3857)

    return gdf_4326, gdf_3857


def load_or_create_businesses(csv_path):
    """Load business dataset or create a template CSV (name, lat, lon, category)."""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame([
            {"name": "Sample Coffee", "lat": 33.7570, "lon": -84.3970, "category": "Cafe"},
            {"name": "Sample Retail", "lat": 33.7560, "lon": -84.3990, "category": "Retail"},
        ])
        df.to_csv(csv_path, index=False)
        print("[INFO] Created template businesses.csv")

    gdf_4326 = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    return gdf_4326


def load_event_venues():
    """
    Load event venues directly from a hard-coded list.
    No CSV file needed.

    You can edit the list below to match the venues you care about.
    """
    data = [
        {
            "venue": "Mercedes-Benz Stadium",
            "lat": 33.7554,
            "lon": -84.4009,
            "event_type": "FIFA / Super Bowl"
        },
        {
            "venue": "State Farm Arena",
            "lat": 33.7573,
            "lon": -84.3963,
            "event_type": "NBA / Events"
        },
        {
            "venue": "Bobby Dodd Stadium",
            "lat": 33.7725,
            "lon": -84.3929,
            "event_type": "NCAA"
        },
    ]

    df = pd.DataFrame(data)

    gdf_4326 = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    gdf_3857 = gdf_4326.to_crs(epsg=3857)

    return gdf_4326, gdf_3857



# ======================================================
# 2. Compute nearest MARTA stop for each venue
# ======================================================

def guess_name_column(gdf):
    """Best guess for the stop-name field."""
    candidates = ["STOP_NAME", "Stop_Name", "NAME", "STATION", "STOPNAME"]
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def compute_nearest_stations(venues_3857, stops_3857):
    """Compute nearest MARTA stop for each event venue."""
    name_col = guess_name_column(stops_3857)

    nearest_ids, nearest_names, dist_m = [], [], []

    for _, v in venues_3857.iterrows():
        d = stops_3857.geometry.distance(v.geometry)
        idx_min = d.idxmin()
        nearest_ids.append(idx_min)
        dist_m.append(float(d[idx_min]))

        if name_col:
            nearest_names.append(stops_3857.loc[idx_min, name_col])
        else:
            nearest_names.append(str(idx_min))

    result = venues_3857.copy()
    result["nearest_stop_id"] = nearest_ids
    result["nearest_stop_name"] = nearest_names
    result["nearest_stop_dist_m"] = dist_m
    result["nearest_stop_dist_miles"] = result["nearest_stop_dist_m"] / MILES_TO_METERS

    return result.to_crs(epsg=4326).drop(columns=["geometry"])



# ======================================================
# 3. Build buffers (0.5 mile, 1 mile)
# ======================================================

def build_accessibility_buffers(stops_3857, venues_3857, radii_list):
    """Create accessibility polygons around event venues using nearby MARTA stops."""
    buffer_records = []

    for _, v in venues_3857.iterrows():
        # All stops within max radius
        d = stops_3857.geometry.distance(v.geometry)
        close_stops = stops_3857[d <= max(radii_list)]

        if close_stops.empty:
            continue

        for r in radii_list:
            within_r = close_stops[close_stops.geometry.distance(v.geometry) <= r]
            if within_r.empty:
                continue

            union_poly = unary_union(within_r.geometry.buffer(r).tolist())

            buffer_records.append({
                "venue": v["venue"],
                "event_type": v.get("event_type", ""),
                "radius_m": r,
                "geometry": union_poly
            })

    return gpd.GeoDataFrame(buffer_records, crs=stops_3857.crs).to_crs(epsg=4326)



# ======================================================
# 4. Count businesses inside each buffer
# ======================================================

def count_businesses(buffers_4326, biz_4326):
    rows = []

    for _, b in buffers_4326.iterrows():
        mask = biz_4326.geometry.within(b.geometry)
        subset = biz_4326[mask]

        if subset.empty:
            rows.append({
                "venue": b["venue"],
                "radius_m": b["radius_m"],
                "radius_miles": b["radius_m"] / MILES_TO_METERS,
                "category": "__ALL__",
                "count": 0
            })
        else:
            grp = subset.groupby("category").size().reset_index(name="count")
            for _, r in grp.iterrows():
                rows.append({
                    "venue": b["venue"],
                    "radius_m": b["radius_m"],
                    "radius_miles": b["radius_m"] / MILES_TO_METERS,
                    "category": r["category"],
                    "count": int(r["count"])
                })

    return pd.DataFrame(rows)



# ======================================================
# 5. Folium map (optional)
# ======================================================

def generate_map(venues_4326, stops_4326, buffers_4326, out_file):
    try:
        import folium
    except:
        print("[WARNING] Folium not installed → skipping HTML map.")
        return

    m = folium.Map(
        location=[venues_4326.geometry.y.mean(), venues_4326.geometry.x.mean()],
        zoom_start=13
    )

    # Venues
    for _, v in venues_4326.iterrows():
        folium.Marker(
            [v.geometry.y, v.geometry.x],
            popup=v["venue"],
            tooltip=v["venue"],
            icon=folium.Icon(icon="flag")
        ).add_to(m)

    # MARTA stops
    for _, s in stops_4326.iterrows():
        folium.CircleMarker(
            [s.geometry.y, s.geometry.x], radius=2
        ).add_to(m)

    # Buffers
    for _, b in buffers_4326.iterrows():
        gj = gpd.GeoSeries([b.geometry], crs=buffers_4326.crs).to_json()
        miles = round(b["radius_m"] / MILES_TO_METERS, 2)

        folium.GeoJson(
            data=gj,
            name=f"{b['venue']} - {miles} mile buffer"
        ).add_to(m)

    m.save(out_file)
    print(f"[INFO] Saved map → {out_file}")



# ======================================================
# 6. MAIN
# ======================================================

def main():
    # Load data
    stops_4326, stops_3857 = load_marta_stops(MARTA_SHP)
    venues_4326, venues_3857 = load_event_venues()
    biz_4326 = load_or_create_businesses(BUSINESS_CSV)

    # Nearest MARTA stop
    nearest_df = compute_nearest_stations(venues_3857, stops_3857)
    nearest_df.to_csv(OUT_NEAREST, index=False)

    # Buffers
    buffers_4326 = build_accessibility_buffers(stops_3857, venues_3857, BUFFER_RADII_METERS)

    # Business counts
    biz_counts = count_businesses(buffers_4326, biz_4326)
    biz_counts.to_csv(OUT_BIZ_COUNTS, index=False)

    # Export MARTA stops for external tools
    stops_4326.to_file(OUT_STOPS_GEOJSON, driver="GeoJSON")
    stops_4326.drop(columns=["geometry"]).to_csv(OUT_STOPS_CSV, index=False)

    # Map
    generate_map(venues_4326, stops_4326, buffers_4326, OUT_MAP)

    print("\n[✓] Analysis complete.")
    print("[✓] CSV and map files generated successfully.")


if __name__ == "__main__":
    main()