import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# RAW MAHARASHTRA DATASET (43 URBAN CENTERS)
raw_data = {
    "city": [
        "Mumbai", "Pune", "Nagpur", "Thane", "Pimpri-Chinchwad", "Nashik",
        "Kalyan-Dombivli", "Vasai-Virar", "Chhatrapati Sambhajinagar", "Navi Mumbai",
        "Solapur", "Mira-Bhayandar", "Bhiwandi-Nizampur", "Amravati", "Nanded-Waghala",
        "Kolhapur", "Ulhasnagar", "Sangli-Miraj-Kupwad", "Malegaon", "Jalgaon",
        "Akola", "Latur", "Dhule", "Ahilyanagar", "Chandrapur",
        "Parbhani", "Ichalkaranji", "Jalna", "Ambernath", "Bhusawal",
        "Panvel", "Badlapur", "Beed", "Gondia", "Satara",
        "Barshi", "Yavatmal", "Achalpur", "Dharashiva", "Nandurbar",
        "Wardha", "Udgir", "Hinganghat"
    ],
    "monthly_salary_inr": [
        85000, 78000, 48000, 72000, 75000, 45000, 52000, 48000, 42000, 78000,
        35000, 55000, 38000, 34000, 32000, 38000, 42000, 35000, 28000, 31000,
        32000, 33000, 30000, 36000, 34000, 29000, 33000, 31000, 44000, 29000,
        62000, 46000, 28000, 30000, 37000, 28000, 29000, 27000, 29000, 28000,
        31000, 28000, 29000
    ],
    "cost_of_living_index": [
        88.5, 65.2, 44.0, 70.4, 62.0, 42.1, 55.0, 50.0, 39.5, 72.1,
        32.1, 58.0, 41.0, 31.0, 30.0, 36.2, 45.0, 33.0, 26.0, 29.0,
        30.0, 31.0, 28.0, 34.0, 32.0, 27.0, 30.0, 29.0, 46.0, 27.0,
        60.0, 48.0, 26.0, 28.0, 35.0, 25.0, 27.0, 24.0, 26.0, 25.0,
        29.0, 25.0, 26.0
    ],
    "rent_index": [
        92.0, 58.0, 33.0, 65.0, 54.0, 31.0, 42.0, 38.0, 28.0, 68.0,
        18.0, 46.0, 29.0, 17.0, 16.0, 22.0, 32.0, 19.0, 12.0, 15.0,
        16.0, 17.0, 14.0, 20.0, 18.0, 13.0, 17.0, 15.0, 33.0, 13.0,
        50.0, 35.0, 12.0, 14.0, 21.0, 11.0, 13.0, 10.0, 12.0, 11.0,
        15.0, 11.0, 12.0
    ],
    "avg_aqi": [
        168, 98, 78, 125, 102, 62, 138, 130, 85, 112,
        58, 142, 155, 52, 65, 54, 148, 56, 72, 70,
        64, 61, 68, 75, 92, 63, 60, 88, 140, 74,
        118, 132, 59, 50, 48, 52, 55, 49, 53, 58,
        62, 54, 82
    ],
    "internet_speed_mbps": [
        110, 115, 80, 98, 108, 75, 85, 82, 68, 105,
        55, 88, 65, 50, 48, 60, 72, 52, 40, 45,
        48, 50, 42, 58, 52, 42, 46, 45, 70, 42,
        95, 78, 40, 44, 58, 38, 42, 36, 40, 38,
        46, 38, 40
    ],
    "healthcare_score": [
        82.0, 85.0, 72.0, 75.0, 80.0, 65.0, 68.0, 62.0, 60.0, 78.0,
        52.0, 64.0, 50.0, 50.0, 46.0, 58.0, 55.0, 54.0, 38.0, 44.0,
        48.0, 46.0, 42.0, 52.0, 48.0, 40.0, 45.0, 42.0, 56.0, 40.0,
        70.0, 60.0, 38.0, 42.0, 56.0, 36.0, 40.0, 34.0, 38.0, 35.0,
        45.0, 36.0, 38.0
    ],
    "safety_score": [
        78.0, 72.0, 70.0, 80.0, 74.0, 84.0, 75.0, 72.0, 76.0, 81.0,
        85.0, 76.0, 62.0, 86.0, 80.0, 88.0, 68.0, 84.0, 70.0, 78.0,
        80.0, 82.0, 76.0, 80.0, 72.0, 78.0, 82.0, 75.0, 72.0, 76.0,
        79.0, 76.0, 84.0, 85.0, 89.0, 86.0, 83.0, 87.0, 85.0, 82.0,
        84.0, 83.0, 80.0
    ],
    "traffic_commute_index": [
        92.0, 85.0, 42.0, 74.0, 78.0, 38.0, 70.0, 65.0, 35.0, 60.0,
        25.0, 68.0, 58.0, 22.0, 24.0, 28.0, 62.0, 26.0, 32.0, 28.0,
        25.0, 24.0, 22.0, 30.0, 28.0, 20.0, 24.0, 26.0, 55.0, 22.0,
        52.0, 48.0, 18.0, 18.0, 22.0, 16.0, 18.0, 15.0, 16.0, 18.0,
        20.0, 16.0, 22.0
    ],
    "green_cover_index": [
        25.0, 48.0, 75.0, 52.0, 45.0, 68.0, 38.0, 42.0, 42.0, 55.0,
        38.0, 35.0, 28.0, 62.0, 50.0, 70.0, 30.0, 64.0, 32.0, 45.0,
        52.0, 48.0, 40.0, 55.0, 68.0, 42.0, 50.0, 40.0, 36.0, 42.0,
        58.0, 44.0, 45.0, 65.0, 72.0, 42.0, 58.0, 60.0, 46.0, 50.0,
        62.0, 44.0, 55.0
    ]
}

df = pd.DataFrame(raw_data)

# Standardization & Cleaning
df["city_clean"] = df["city"].astype(str).str.strip().str.lower().str.title()
df["city_clean"] = df["city_clean"].replace({
    "Aurangabad": "Chhatrapati Sambhajinagar",
    "Ahmednagar": "Ahilyanagar",
    "Osmanabad": "Dharashiva"
})

# Feature Scaling
df["log_salary"] = np.log1p(df["monthly_salary_inr"])
df["log_cost"] = np.log1p(df["cost_of_living_index"])
df["log_rent"] = np.log1p(df["rent_index"])

feature_cols = [
    "log_salary", "log_cost", "log_rent", "avg_aqi",
    "internet_speed_mbps", "healthcare_score", "safety_score",
    "traffic_commute_index", "green_cover_index"
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

# K-Means Machine Learning
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster_id"] = kmeans.fit_predict(X_scaled)

archetype_map = {
    0: "Tier-1 Metro Core (High Income / High Stress)",
    1: "Emerging Industrial & Satellite Hub",
    2: "Balanced Tech & Growth Hub",
    3: "Low-Cost / High-Eco Livable City"
}
df["cluster_archetype"] = df["cluster_id"].map(archetype_map)

# PCA Dimensionality Reduction
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)
df["pca_x"] = pca_coords[:, 0]
df["pca_y"] = pca_coords[:, 1]

# Save output CSV file
df.to_csv("maharashtra_all_cities_ml.csv", index=False)
print("SUCCESS: Processed dataset saved as 'maharashtra_all_cities_ml.csv'.")