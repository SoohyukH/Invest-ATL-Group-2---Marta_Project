import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, "outputs")

df = pd.read_csv(os.path.join(OUT, "businesses_with_access_metrics.csv"))

# 1) Summary
print("=== Overall ===")
print("Total businesses:", len(df))
print("Avg distance to nearest stop (miles):", df["dist_to_stop_miles"].mean())

# Ration by distance range
bins = [0, 0.05, 0.1, 0.25, 0.5, 999]
labels = ["0–0.05", "0.05–0.1", "0.1–0.25", "0.25–0.5", "0.5+"]
df["dist_bucket"] = pd.cut(df["dist_to_stop_miles"], bins=bins, labels=labels)

print("\n=== Distance buckets (% of businesses) ===")
print((df["dist_bucket"].value_counts(normalize=True) * 100).round(1))

# 2) ratio by category
def cat_stats(keyword):
    sub = df[df["category"].str.contains(keyword, na=False)]
    if len(sub) == 0:
        print(f"\n[{keyword}] not found")
        return
    print(f"\n=== {keyword} ===")
    print("Count:", len(sub))
    print("Avg distance (miles):", sub["dist_to_stop_miles"].mean())
    print("Avg access score:", sub["access_score"].mean())

cat_stats("Full-Service Restaurants")
cat_stats("Beauty Salons")
cat_stats("Child Day Care Services")

# 3) TOP 20_near_stops
restaurants = df[df["category"].str.contains("Full-Service Restaurants", na=False)]
top20 = restaurants.sort_values("dist_to_stop_miles").head(20)
out_path = os.path.join(OUT, "restaurants_top20_near_stop.csv")
top20.to_csv(out_path, index=False)
print(f"\nSaved example table: {out_path}")