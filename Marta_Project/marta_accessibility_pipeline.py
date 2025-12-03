"""
MARTA Accessibility & Economic Development Analysis Pipeline
------------------------------------------------------------

This script:

1. Loads data:
   - MARTA stops (shapefile)
   - Event venues (CSV)
   - Atlanta business license records (CSV)
   - MARTA bus & train ridership (CSV)

2. Geocodes businesses (if lat/lon are missing) using Nominatim via geopy.

3. Converts everything into GeoDataFrames (EPSG:4326, EPSG:3857).

4. Computes:
   - Nearest MARTA stop per business
   - Business counts within 0.5 and 1 mile buffer per venue
   - Business density per venue
   - Category breakdowns by venue
   - Station-level business density & correlations with ridership

5. Generates visualizations:
   - Bar chart: businesses within 0.5 vs 1 mile per venue
   - Heatmap-style density comparison
   - Pie charts for category distribution per venue
   - Ridership trend line
   - Scatter plots: ridership vs business density

6. Builds an interactive folium map:
   - MARTA stops
   - Event venues
   - Business density buffers

7. Exports:
   - CSV summaries
   - PNG charts
   - HTML map

How to run locally (from project root):

    python marta_accessibility_pipeline.py

Make sure you have the required packages installed:

    pip install pandas geopandas shapely pyproj geopy matplotlib seaborn folium
"""

import os
import time
from typing import Optional, Tuple, Dict

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

import matplotlib.pyplot as plt
import seaborn as sns
import folium


# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# Input file paths
MARTA_STOPS_SHP = os.path.join(DATA_DIR, "MARTA_Stops.shp")
BUS_RIDERSHIP_CSV = os.path.join(
    DATA_DIR, "MARTA_Bus_Ridership_2023_20250912(Route ridership).csv"
)
TRAIN_RIDERSHIP_CSV = os.path.join(
    DATA_DIR, "MARTA_Train_Ridership_2023_20250917(Total trips by date, station).csv"
)
BUSINESS_CSV = os.path.join(
    DATA_DIR, "Atlanta_Business_License_Records_2025(Sheet1).csv"
)

# Output paths
GEOCODE_CACHE_CSV = os.path.join(OUTPUT_DIR, "geocode_cache.csv")
BUSINESSES_WITH_COORDS_CSV = os.path.join(
    OUTPUT_DIR, "businesses_with_coordinates.csv"
)
BUSINESSES_WITH_MARTA_CSV = os.path.join(
    OUTPUT_DIR, "businesses_with_nearest_marta.csv"
)
VENUE_BUSINESS_SUMMARY_CSV = os.path.join(
    OUTPUT_DIR, "venue_business_summary.csv"
)
VENUE_CATEGORY_BREAKDOWN_CSV = os.path.join(
    OUTPUT_DIR, "venue_category_breakdown.csv"
)
STATION_DENSITY_CSV = os.path.join(
    OUTPUT_DIR, "station_business_density_and_ridership.csv"
)

FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")

FOLIUM_MAP_HTML = os.path.join(MAPS_DIR, "marta_accessibility_map.html")

# Distance in meters (approx)
METER_PER_MILE = 1609.34
BUFFER_DISTANCES_MILES = [0.5, 1.0]
BUFFER_DISTANCES_METERS = [d * METER_PER_MILE for d in BUFFER_DISTANCES_MILES]

# Column names assumed in business license CSV
BUSINESS_LAT_COL = "latitude"
BUSINESS_LON_COL = "longitude"

# Category column for breakdown (adjust if you prefer naics_name)
BUSINESS_CATEGORY_COL = "license_classification"


# -----------------------------
# Utility functions
# -----------------------------

def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MAPS_DIR, exist_ok=True)


def load_marta_stops(shp_path: str) -> gpd.GeoDataFrame:
    """Load MARTA stops shapefile as GeoDataFrame (EPSG:4326 assumed)."""
    gdf = gpd.read_file(shp_path)
    # If CRS is missing, assume WGS84
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    return gdf


def load_venues(_: str = None) -> gpd.GeoDataFrame:
    """
    Load event venues.

    네가 말한 3개 경기장(머세데스 벤츠, 스테이트팜, 바비 도드)을
    코드 안에서 바로 정의해서 GeoDataFrame으로 만든다.

    나중에 venue 추가/수정하고 싶으면 아래 data 리스트만 고치면 됨.
    """

    data = [
        {
            "venue": "Mercedes-Benz Stadium",
            "latitude": 33.755403,
            "longitude": -84.400992,
            "city": "Atlanta",
            "state": "GA",
        },
        {
            "venue": "State Farm Arena",
            "latitude": 33.757347,
            "longitude": -84.396284,
            "city": "Atlanta",
            "state": "GA",
        },
        {
            "venue": "Bobby Dodd Stadium",
            "latitude": 33.772553,
            "longitude": -84.392709,
            "city": "Atlanta",
            "state": "GA",
        },
    ]

    df = pd.DataFrame(data)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326",
    )
    return gdf

def load_businesses(csv_path: str) -> pd.DataFrame:
    """Load Atlanta business license records."""
    df = pd.read_csv(csv_path)
    return df


def init_geocoder(user_agent: str = "marta_accessibility_geocoder") -> Tuple[Nominatim, RateLimiter]:
    """Initialize Nominatim geocoder with a rate limiter."""
    geolocator = Nominatim(user_agent=user_agent)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    return geolocator, geocode


def load_geocode_cache(cache_path: str) -> Dict[str, Tuple[float, float]]:
    """Load geocode cache mapping address -> (lat, lon)."""
    if not os.path.exists(cache_path):
        return {}
    cache_df = pd.read_csv(cache_path)
    cache_df = cache_df.dropna(subset=["latitude", "longitude"])
    return {
        row["address"]: (row["latitude"], row["longitude"])
        for _, row in cache_df.iterrows()
    }


def save_geocode_cache(cache: Dict[str, Tuple[float, float]], cache_path: str):
    """Save geocode cache to CSV."""
    rows = [
        {"address": addr, "latitude": lat, "longitude": lon}
        for addr, (lat, lon) in cache.items()
    ]
    cache_df = pd.DataFrame(rows)
    cache_df.to_csv(cache_path, index=False)


def build_address(row: pd.Series) -> Optional[str]:
    """
    Build a full address string for geocoding from a row in the business dataset.

    Adjust logic depending on which columns are most reliable in your CSV.
    """

    # If there's already a clean full address column, use it first
    for col in ["address_api", "full_address"]:
        if col in row and isinstance(row[col], str) and row[col].strip():
            return row[col]

    # Fallback: build from components
    components = []

    for col in ["address_line1", "address_line2"]:
        if col in row and isinstance(row[col], str):
            components.append(row[col].strip())

    # City, state, zip might be in separate columns or embedded
    city = row.get("city", "Atlanta")
    state = row.get("state", "GA")
    zipcode = row.get("zip_code", row.get("postal_code", ""))

    address_parts = components + [str(city), str(state)]
    if isinstance(zipcode, (str, int, float)) and str(zipcode).strip():
        address_parts.append(str(zipcode))

    # Filter out empty pieces
    address_parts = [p for p in address_parts if p and p != "nan"]

    if not address_parts:
        return None

    return ", ".join(address_parts)


def geocode_businesses_if_needed(
    df: pd.DataFrame,
    cache_path: str = GEOCODE_CACHE_CSV,
    max_new_geocodes: Optional[int] = None,
) -> pd.DataFrame:
    """
    Geocode businesses that are missing lat/lon.

    - If latitude/longitude columns exist & mostly filled, we reuse them.
    - If they don't exist or many missing, we geocode using Nominatim.

    Parameters
    ----------
    df : pd.DataFrame
        Business dataset.
    cache_path : str
        Path to CSV cache of previously geocoded addresses.
    max_new_geocodes : int or None
        To avoid hammering Nominatim while testing, you can set a limit
        on how many *new* addresses to geocode.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with latitude & longitude columns filled where possible.
    """

    # If lat/lon already exist and are largely filled, we may just return
    lat_exists = BUSINESS_LAT_COL in df.columns
    lon_exists = BUSINESS_LON_COL in df.columns

    if lat_exists and lon_exists:
        missing = df[BUSINESS_LAT_COL].isna() | df[BUSINESS_LON_COL].isna()
        missing_count = missing.sum()
        print(f"[INFO] Found existing latitude/longitude. Missing rows: {missing_count}")
    else:
        missing = pd.Series([True] * len(df))
        missing_count = len(df)
        df[BUSINESS_LAT_COL] = None
        df[BUSINESS_LON_COL] = None
        print("[INFO] No latitude/longitude columns found. Need to geocode all rows.")

    if missing_count == 0:
        print("[INFO] No geocoding needed.")
        return df

    # Load cache
    cache = load_geocode_cache(cache_path)
    print(f"[INFO] Loaded geocode cache with {len(cache)} entries.")

    # Initialize geocoder
    geolocator, geocode = init_geocoder()

    new_geocoded = 0

    for idx, row in df[missing].iterrows():
        address = build_address(row)
        if not address:
            continue

        # Try cache first
        if address in cache:
            lat, lon = cache[address]
            df.at[idx, BUSINESS_LAT_COL] = lat
            df.at[idx, BUSINESS_LON_COL] = lon
            continue

        # Stop if we hit max_new_geocodes (for testing)
        if max_new_geocodes is not None and new_geocoded >= max_new_geocodes:
            break

        try:
            location = geocode(address)
            if location:
                lat, lon = location.latitude, location.longitude
                df.at[idx, BUSINESS_LAT_COL] = lat
                df.at[idx, BUSINESS_LON_COL] = lon
                cache[address] = (lat, lon)
                new_geocoded += 1
                print(f"[GEOCODED] {address} -> ({lat:.5f}, {lon:.5f})")
            else:
                print(f"[GEOCODE FAIL] {address}")
        except Exception as e:
            print(f"[ERROR] Geocoding failed for '{address}': {e}")
            time.sleep(2)

    # Save updated cache
    save_geocode_cache(cache, cache_path)
    print(f"[INFO] Saved geocode cache with {len(cache)} entries.")

    return df


def df_to_geodf(df: pd.DataFrame, lat_col: str, lon_col: str) -> gpd.GeoDataFrame:
    """Convert a DataFrame with lat/lon into a GeoDataFrame (EPSG:4326)."""
    valid = df.dropna(subset=[lat_col, lon_col]).copy()

    valid["geometry"] = [
        Point(xy) for xy in zip(valid[lon_col].astype(float), valid[lat_col].astype(float))
    ]
    gdf = gpd.GeoDataFrame(valid, geometry="geometry", crs="EPSG:4326")
    return gdf


def to_3857(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Project GeoDataFrame to EPSG:3857."""
    return gdf.to_crs(epsg=3857)


# -----------------------------
# Spatial analysis functions
# -----------------------------

def compute_nearest_marta_for_businesses(
    businesses_gdf_3857: gpd.GeoDataFrame,
    stops_gdf_3857: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    For each business, find the nearest MARTA stop and distance in meters.
    """
    nearest = gpd.sjoin_nearest(
        businesses_gdf_3857,
        stops_gdf_3857[["stop_id", "stop_name", "geometry"]],
        how="left",
        distance_col="dist_to_stop_m",
    )
    return nearest


def summarize_venues_business_buffers(
    venues_gdf_3857: gpd.GeoDataFrame,
    businesses_gdf_3857: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    For each venue and each buffer (0.5, 1 mile), compute:
    - number of businesses
    - business density (count / area)
    """

    records = []

    for _, venue in venues_gdf_3857.iterrows():
        venue_name = venue["venue"]

        for radius_m, radius_miles in zip(BUFFER_DISTANCES_METERS, BUFFER_DISTANCES_MILES):
            buffer_geom = venue.geometry.buffer(radius_m)

            in_buffer = businesses_gdf_3857[businesses_gdf_3857.geometry.within(buffer_geom)]

            count = len(in_buffer)
            area_sq_m = buffer_geom.area
            density_per_sq_km = count / (area_sq_m / 1_000_000) if area_sq_m > 0 else None

            records.append(
                {
                    "venue": venue_name,
                    "buffer_miles": radius_miles,
                    "buffer_meters": radius_m,
                    "business_count": count,
                    "business_density_per_sq_km": density_per_sq_km,
                }
            )

    summary_df = pd.DataFrame(records)
    return summary_df


def summarize_venue_category_breakdown(
    venues_gdf_3857: gpd.GeoDataFrame,
    businesses_gdf_3857: gpd.GeoDataFrame,
    category_col: str = BUSINESS_CATEGORY_COL,
    buffer_meters: float = BUFFER_DISTANCES_METERS[1],  # default 1 mile
) -> pd.DataFrame:
    """
    For each venue, compute category counts (%) within a given buffer.
    Returns a long-format DataFrame: [venue, category, count, share].
    """

    records = []

    for _, venue in venues_gdf_3857.iterrows():
        venue_name = venue["venue"]
        buffer_geom = venue.geometry.buffer(buffer_meters)

        in_buffer = businesses_gdf_3857[businesses_gdf_3857.geometry.within(buffer_geom)]
        total = len(in_buffer)

        if total == 0:
            continue

        # ✅ 카테고리별 count를 안전하게 계산
        # value_counts 결과를 category/count 두 컬럼으로 확실히 만든다
        cat_counts = (
            in_buffer[category_col]
            .fillna("Unknown")
            .value_counts()                      # Series: index=category, values=count
            .rename_axis("category")
            .reset_index(name="count")           # DataFrame: [category, count]
        )

        # venue 이름 붙이기
        cat_counts["venue"] = venue_name

        # count를 숫자로 강제 변환 (혹시라도 타입 꼬인 것 방지)
        cat_counts["count"] = pd.to_numeric(cat_counts["count"], errors="coerce").fillna(0).astype(int)

        # 비율 계산
        cat_counts["share"] = cat_counts["count"] / float(total)

        records.append(cat_counts)

    if not records:
        return pd.DataFrame(columns=["venue", "category", "count", "share"])

    result = pd.concat(records, ignore_index=True)
    result = result[["venue", "category", "count", "share"]]
    return result


def compute_station_business_density(
    stops_gdf_4326: gpd.GeoDataFrame,
    businesses_gdf_3857: gpd.GeoDataFrame,
    train_ridership_csv: str,
) -> pd.DataFrame:
    """
    Compute business density around train stations and join with ridership.

    Assumes:
    - Train ridership CSV has columns: 'date', 'station', 'trips' (adjust as needed).
    - Train stations in the stops shapefile contain 'STATION' in stop_name.

    Returns a DataFrame with:
    - station_name
    - businesses_within_0_5mi
    - businesses_within_1mi
    - avg_daily_ridership
    - correlations can be computed from this table.
    """

    # 1. Filter stops to station-like names (heuristic)
    station_gdf = stops_gdf_4326[
        stops_gdf_4326["stop_name"].str.contains("STATION", case=False, na=False)
    ].copy()

    if station_gdf.empty:
        print("[WARN] No station-like stops found. Check stop_name pattern.")
        return pd.DataFrame()

    station_gdf_3857 = to_3857(station_gdf)

    # 2. Business counts within 0.5 and 1 mile for each station
    records = []
    for _, row in station_gdf_3857.iterrows():
        station_name = row["stop_name"]
        station_id = row.get("stop_id", None)

        row_rec = {"station_name": station_name, "stop_id": station_id}

        for radius_m, radius_miles in zip(BUFFER_DISTANCES_METERS, BUFFER_DISTANCES_MILES):
            buffer_geom = row.geometry.buffer(radius_m)
            count = businesses_gdf_3857.geometry.within(buffer_geom).sum()
            key = f"businesses_within_{radius_miles}mi"
            row_rec[key] = int(count)

        records.append(row_rec)

    density_df = pd.DataFrame(records)

    # 3. Load train ridership and compute avg daily ridership per station
    ridership = pd.read_csv(train_ridership_csv)
    # Try to guess column names; adjust if needed
    # Example expected columns: 'Date', 'Station', 'Total_Trips'
    ridership_cols = {c.lower(): c for c in ridership.columns}

    # Flexible mapping
    date_col = ridership_cols.get("date", ridership.columns[0])
    station_col = ridership_cols.get("station", ridership.columns[1])
    trips_col_candidates = [
        k for k in ridership_cols.keys()
        if "trip" in k or "ridership" in k or "total" in k
    ]
    if trips_col_candidates:
        trips_col = ridership_cols[trips_col_candidates[0]]
    else:
        trips_col = ridership.columns[-1]

    ridership[date_col] = pd.to_datetime(ridership[date_col])
    ridership["station_clean"] = ridership[station_col].str.upper().str.strip()

    ridership_summary = (
        ridership.groupby("station_clean")[trips_col]
        .mean()
        .reset_index()
        .rename(columns={trips_col: "avg_daily_ridership"})
    )

    # 4. Join density_df with ridership_summary on cleaned station_name
    density_df["station_clean"] = density_df["station_name"].str.upper().str.strip()
    merged = density_df.merge(
        ridership_summary, on="station_clean", how="left"
    )

    return merged


# -----------------------------
# Plotting functions
# -----------------------------

def plot_venue_business_bar(summary_df: pd.DataFrame, output_path: str):
    """Bar chart: businesses within 0.5 vs 1 mile for each venue."""
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary_df,
        x="venue",
        y="business_count",
        hue="buffer_miles",
    )
    plt.title("Businesses within 0.5 vs 1 mile of each venue")
    plt.ylabel("Number of businesses")
    plt.xlabel("Venue")
    plt.legend(title="Buffer (miles)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_venue_density_heatmap(summary_df: pd.DataFrame, output_path: str):
    """
    Heatmap-style density comparison:
    Rows = venues, Columns = buffer size, Values = business density.
    """
    pivot = summary_df.pivot_table(
        index="venue",
        columns="buffer_miles",
        values="business_density_per_sq_km",
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f")
    plt.title("Business Density (per sq km) by Venue and Buffer")
    plt.ylabel("Venue")
    plt.xlabel("Buffer (miles)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_venue_category_pies(
    category_df: pd.DataFrame,
    output_dir: str,
    min_share: float = 0.03,  # 이 비율보다 작은 카테고리는 "Other"로 묶기 (3%)
):
    """
    Pie chart for category distribution per venue.

    - 너무 많은 카테고리가 겹쳐 보이는 걸 막기 위해,
      min_share 이하 카테고리는 Other로 묶는다.
    - 라벨은 파이 안에 쓰지 않고, 오른쪽 legend로만 표시한다.
    """

    venues = category_df["venue"].unique()

    for venue in venues:
        subset = category_df[category_df["venue"] == venue].copy()

        if subset.empty:
            continue

        # 작은 카테고리들은 Other로 묶기
        big = subset[subset["share"] >= min_share].copy()
        small = subset[subset["share"] < min_share].copy()

        if not small.empty:
            other_row = pd.DataFrame(
                {
                    "venue": [venue],
                    "category": ["Other"],
                    "count": [small["count"].sum()],
                    "share": [small["share"].sum()],
                }
            )
            big = pd.concat([big, other_row], ignore_index=True)

        big = big.sort_values("share", ascending=False)

        # 파이 차트 그리기 (라벨은 비율만)
        plt.figure(figsize=(8, 6))
        wedges, texts, autotexts = plt.pie(
            big["share"],
            labels=None,              # 라벨은 legend로
            autopct="%1.1f%%",
            startangle=140,
        )

        plt.title(f"Business Category Distribution within 1 mile of {venue}")

        # 오른쪽 legend에 카테고리 이름 표시
        plt.legend(
            wedges,
            big["category"],
            title="Category",
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
        )

        plt.tight_layout()

        out_path = os.path.join(
            output_dir, f"category_pie_{venue.replace(' ', '_')}.png"
        )
        plt.savefig(out_path, dpi=300)
        plt.close()


def plot_ridership_trend(train_ridership_csv: str, output_path: str):
    """Ridership trend line graph (sum over all stations by date)."""
    df = pd.read_csv(train_ridership_csv)
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", df.columns[0])

    trips_col_candidates = [
        k for k in cols.keys()
        if "trip" in k or "ridership" in k or "total" in k
    ]
    if trips_col_candidates:
        trips_col = cols[trips_col_candidates[0]]
    else:
        trips_col = df.columns[-1]

    df[date_col] = pd.to_datetime(df[date_col])
    daily = df.groupby(date_col)[trips_col].sum().reset_index()

    plt.figure(figsize=(12, 6))
    plt.plot(daily[date_col], daily[trips_col])
    plt.title("MARTA Train Ridership Over Time (Total Trips per Day)")
    plt.xlabel("Date")
    plt.ylabel("Total Trips")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_ridership_vs_density(
    station_df: pd.DataFrame,
    output_path_prefix: str,
):
    """Scatter plots of ridership vs business density (0.5, 1 mile)."""
    for radius in BUFFER_DISTANCES_MILES:
        col = f"businesses_within_{radius}mi"
        if col not in station_df.columns:
            continue

        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=station_df,
            x=col,
            y="avg_daily_ridership",
            ci=None,
        )
        plt.title(f"Avg Daily Ridership vs Businesses within {radius} mile")
        plt.xlabel(f"Businesses within {radius} mile")
        plt.ylabel("Average Daily Ridership")
        plt.tight_layout()
        out_path = f"{output_path_prefix}_ridership_vs_density_{radius}mi.png"
        plt.savefig(out_path, dpi=300)
        plt.close()


# -----------------------------
# Folium map
# -----------------------------

def build_folium_map(
    venues_gdf_4326: gpd.GeoDataFrame,
    stops_gdf_4326: gpd.GeoDataFrame,
    businesses_gdf_4326: gpd.GeoDataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
):
    """
    Build an interactive folium map showing:
    - MARTA stops
    - Event venues
    - Business buffers around venues (1 mile)
    """
    # Center on average venue location
    center_lat = venues_gdf_4326.geometry.y.mean()
    center_lon = venues_gdf_4326.geometry.x.mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add MARTA stops (small blue markers)
    for _, row in stops_gdf_4326.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        name = row.get("stop_name", "MARTA Stop")
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            popup=name,
            color="blue",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    # Add venues (larger red markers)
    for _, row in venues_gdf_4326.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        venue_name = row["venue"]
        folium.Marker(
            location=[lat, lon],
            popup=venue_name,
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    # Add 1-mile buffers around venues with different colors
    venues_3857 = to_3857(venues_gdf_4326)

    for _, row in venues_3857.iterrows():
        venue_name = row["venue"]
        buffer_geom = row.geometry.buffer(BUFFER_DISTANCES_METERS[1])  # 1 mile
        buffer_4326 = gpd.GeoSeries([buffer_geom], crs=venues_3857.crs).to_crs(epsg=4326)

        folium.GeoJson(
            buffer_4326.__geo_interface__,
            name=f"{venue_name} - 1 mile buffer",
            style_function=lambda x: {
                "fillColor": "#ff7800",
                "color": "#ff7800",
                "weight": 1,
                "fillOpacity": 0.1,
            },
        ).add_to(m)

    # Optional: add business points (may be heavy if many)
    # Here we add only those within 1 mile of any venue for clarity
    venues_buffer_union_3857 = venues_3857.geometry.buffer(BUFFER_DISTANCES_METERS[1]).unary_union
    businesses_3857 = to_3857(businesses_gdf_4326)
    in_any_buffer = businesses_3857[businesses_3857.geometry.within(venues_buffer_union_3857)]
    in_any_buffer_4326 = in_any_buffer.to_crs(epsg=4326)

    for _, row in in_any_buffer_4326.iterrows():
        lat = row.geometry.y
        lon = row.geometry.x
        name = row.get("company_name", "Business")
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            popup=name,
            color="green",
            fill=True,
            fill_opacity=0.6,
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ensure_directories()

    # 1. Load core datasets
    print("[STEP 1] Loading data...")
    marta_stops = load_marta_stops(MARTA_STOPS_SHP)
    venues = load_venues()
    businesses_df = load_businesses(BUSINESS_CSV)

    print(f"[INFO] Loaded {len(marta_stops)} MARTA stops.")
    print(f"[INFO] Loaded {len(venues)} venues.")
    print(f"[INFO] Loaded {len(businesses_df)} businesses (raw).")

    # 2. Geocode businesses (if needed)
    print("[STEP 2] Geocoding businesses (if needed)...")
    # For first run, you may want max_new_geocodes=100 to test quickly.
    businesses_df = geocode_businesses_if_needed(
        businesses_df,
        cache_path=GEOCODE_CACHE_CSV,
        max_new_geocodes=None,  # set to small int for testing
    )
    businesses_df.to_csv(BUSINESSES_WITH_COORDS_CSV, index=False)
    print(f"[INFO] Saved businesses_with_coordinates to {BUSINESSES_WITH_COORDS_CSV}")

    # 3. Convert to GeoDataFrames (EPSG:4326 -> 3857)
    print("[STEP 3] Converting to GeoDataFrames...")
    businesses_gdf_4326 = df_to_geodf(businesses_df, BUSINESS_LAT_COL, BUSINESS_LON_COL)
    venues_gdf_4326 = venues  # already in 4326
    marta_stops_4326 = marta_stops.to_crs(epsg=4326)

    businesses_gdf_3857 = to_3857(businesses_gdf_4326)
    venues_gdf_3857 = to_3857(venues_gdf_4326)
    marta_stops_3857 = to_3857(marta_stops_4326)

    # 4. Nearest MARTA stop per business
    print("[STEP 4] Computing nearest MARTA stop for each business...")
    businesses_with_marta = compute_nearest_marta_for_businesses(
        businesses_gdf_3857, marta_stops_3857
    )
    businesses_with_marta.to_csv(BUSINESSES_WITH_MARTA_CSV, index=False)
    print(f"[INFO] Saved businesses_with_nearest_marta to {BUSINESSES_WITH_MARTA_CSV}")

    # 5. Business counts & density per venue
    print("[STEP 5] Summarizing business counts & density per venue...")
    venue_summary = summarize_venues_business_buffers(
        venues_gdf_3857, businesses_gdf_3857
    )
    venue_summary.to_csv(VENUE_BUSINESS_SUMMARY_CSV, index=False)
    print(f"[INFO] Saved venue_business_summary to {VENUE_BUSINESS_SUMMARY_CSV}")

    # 6. Category breakdown (within 1 mile)
    print("[STEP 6] Computing category breakdown per venue (1 mile)...")
    category_breakdown = summarize_venue_category_breakdown(
        venues_gdf_3857, businesses_gdf_3857, category_col=BUSINESS_CATEGORY_COL
    )
    category_breakdown.to_csv(VENUE_CATEGORY_BREAKDOWN_CSV, index=False)
    print(f"[INFO] Saved venue_category_breakdown to {VENUE_CATEGORY_BREAKDOWN_CSV}")

    # 7. Station-level density & ridership correlation
    print("[STEP 7] Computing station-level business density & ridership...")
    station_density = compute_station_business_density(
        marta_stops_4326, businesses_gdf_3857, TRAIN_RIDERSHIP_CSV
    )
    if not station_density.empty:
        station_density.to_csv(STATION_DENSITY_CSV, index=False)
        print(f"[INFO] Saved station_business_density_and_ridership to {STATION_DENSITY_CSV}")

        # Optional: print simple correlations
        for radius in BUFFER_DISTANCES_MILES:
            col = f"businesses_within_{radius}mi"
            if col in station_density.columns:
                corr = station_density[[col, "avg_daily_ridership"]].corr().iloc[0, 1]
                print(f"[CORR] radius={radius}mi, Pearson corr={corr:.3f}")

    # 8. Plots
    print("[STEP 8] Generating charts...")

    # Bar chart: businesses within 0.5 vs 1 mile per venue
    bar_chart_path = os.path.join(FIGURES_DIR, "venue_business_counts_bar.png")
    plot_venue_business_bar(venue_summary, bar_chart_path)

    # Heatmap: density comparison
    heatmap_path = os.path.join(FIGURES_DIR, "venue_business_density_heatmap.png")
    plot_venue_density_heatmap(venue_summary, heatmap_path)

    # Pie charts: category distribution per venue
    plot_venue_category_pies(
        category_breakdown,
        FIGURES_DIR,
    )

    # Ridership trend line
    ridership_trend_path = os.path.join(FIGURES_DIR, "train_ridership_trend.png")
    plot_ridership_trend(TRAIN_RIDERSHIP_CSV, ridership_trend_path)

    # Ridership vs density scatter plots (if we have station_density)
    if not station_density.empty:
        ridership_vs_density_prefix = os.path.join(FIGURES_DIR, "station")
        plot_ridership_vs_density(station_density, ridership_vs_density_prefix)

    # 9. Folium map
    print("[STEP 9] Building folium map...")
    build_folium_map(
        venues_gdf_4326=venues_gdf_4326,
        stops_gdf_4326=marta_stops_4326,
        businesses_gdf_4326=businesses_gdf_4326,
        summary_df=venue_summary,
        output_path=FOLIUM_MAP_HTML,
    )

    print("[DONE] Pipeline finished successfully.")


if __name__ == "__main__":
    main()