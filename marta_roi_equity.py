import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- PATHS ----------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

M_PER_MILE = 1609.34  # meters in one mile


# ---------- HELPERS ----------
def need(path, msg):
    """Check that a file exists; if not, print a message and exit."""
    if not os.path.exists(path):
        print(f"[MISSING] {path}\n→ {msg}")
        sys.exit(1)


def minmax_norm(s):
    """Normalize a pandas Series to the range [0,1]."""
    s = s.astype(float)
    if len(s) == 0:
        return s
    smin, smax = s.min(), s.max()
    if smax == smin:
        return np.zeros(len(s))
    return (s - smin) / (smax - smin)


def nearest_stops_for_all(biz_df, stops_df, batch_size=400):
    """
    For each business, find the nearest MARTA stop using haversine distance.
    Uses only numpy (no spatial index).
    """
    # convert stop coordinates to radians
    stop_lats = np.radians(stops_df["stop_lat"].to_numpy())
    stop_lons = np.radians(stops_df["stop_lon"].to_numpy())

    N = len(biz_df)
    all_idx = np.empty(N, dtype=int)
    all_dist = np.empty(N, dtype=float)

    R = 6371000.0  # Earth radius in meters

    for start_i in range(0, N, batch_size):
        end_i = min(start_i + batch_size, N)

        lat_deg = biz_df["latitude"].values[start_i:end_i]
        lon_deg = biz_df["longitude"].values[start_i:end_i]

        lat1 = np.radians(lat_deg)[:, None]  # shape (B,1)
        lon1 = np.radians(lon_deg)[:, None]  # shape (B,1)

        lat2 = stop_lats[None, :]            # shape (1,M)
        lon2 = stop_lons[None, :]            # shape (1,M)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        d = R * c  # distances in meters, shape (B, M)

        idx = d.argmin(axis=1)                     # nearest stop index
        dist = d[np.arange(d.shape[0]), idx]       # nearest distance

        all_idx[start_i:end_i] = idx
        all_dist[start_i:end_i] = dist

    print(f"[INFO] Computed nearest stops for {N} businesses.")
    return all_idx, all_dist


# ---------- MAIN ----------
def main():
    # 1) required files
    biz_path   = os.path.join(DATA, "Atlanta_Business_License_Records_2025(Sheet1).csv")
    stops_path = os.path.join(DATA, "MARTA_Stops.shp")

    need(biz_path,   "Put Atlanta_Business_License_Records_2025(Sheet1).csv in data/")
    need(stops_path, "Put MARTA_Stops.* (5 files) in data/")

    # 2) load data (latin1 인코딩은 에러 안 남)
    print("[INFO] Loading business license data...")
    biz = pd.read_csv(biz_path, encoding="latin1", low_memory=False)

    print("[INFO] Loading MARTA stops shapefile...")
    import geopandas as gpd
    stops = gpd.read_file(stops_path)

    # 3) clean businesses: pick name/category + coords
    # name: prefer company_dba, else company_name
    if "company_dba" in biz.columns:
        biz["name"] = biz["company_dba"].fillna(biz["company_name"])
    else:
        biz["name"] = biz["company_name"]

    if "naics_name" in biz.columns:
        biz["category"] = biz["naics_name"]
    else:
        biz["category"] = biz.get("license_classification", "")

    # latitude / longitude columns already exist
    biz["latitude"]  = pd.to_numeric(biz["latitude"],  errors="coerce")
    biz["longitude"] = pd.to_numeric(biz["longitude"], errors="coerce")
    biz_clean = biz.dropna(subset=["latitude", "longitude"]).copy()
    print(f"[INFO] Businesses with coordinates: {len(biz_clean)}")

    # 4) clean stops: keep coords + name
    stops["stop_lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops_clean = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
    stops_clean["stop_name"] = stops_clean["stop_name"].fillna("Unknown stop")
    print(f"[INFO] MARTA stops with coordinates: {len(stops_clean)}")

    # 5) nearest stop for each business
    idx, dist_m = nearest_stops_for_all(biz_clean, stops_clean, batch_size=400)
    nearest_names = stops_clean.iloc[idx]["stop_name"].values

    # 6) build final DataFrame
    df = biz_clean[["name", "category", "latitude", "longitude"]].copy()
    df["nearest_stop_name"] = nearest_names
    df["dist_to_stop_m"] = dist_m
    df["dist_to_stop_miles"] = df["dist_to_stop_m"] / M_PER_MILE

    # accessibility: closer = higher
    df["stop_prox_norm"] = 1 - minmax_norm(df["dist_to_stop_m"])
    df["access_score"] = df["stop_prox_norm"]

    # 7) save outputs
    out_csv = os.path.join(OUT, "businesses_with_access_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SAVED] {out_csv}")

    # scatter plot
    plt.figure(figsize=(6, 4))
    plt.scatter(df["dist_to_stop_miles"], df["access_score"], s=3, alpha=0.4)
    plt.xlabel("Miles to nearest MARTA stop")
    plt.ylabel("Accessibility score (0–1)")
    plt.title("Transit distance vs Accessibility")
    plt.tight_layout()
    plot_path = os.path.join(OUT, "scatter_distance_vs_score.png")
    plt.savefig(plot_path, dpi=160)
    print(f"[SAVED] {plot_path}")


if __name__ == "__main__":
    main()