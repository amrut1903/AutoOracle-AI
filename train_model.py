import pickle, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── 1. Load ───────────────────────────────────────────────
DATA_PATH = "dataset/car_dekho.csv"
print("🚗 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   Rows before cleaning : {len(df)}")

# ── 2. Feature Engineering ────────────────────────────────
df["brand"]        = df["name"].apply(lambda x: x.split()[0])
current_year       = pd.Timestamp.now().year
df["car_age"]      = current_year - df["year"]

# engineered features
df["km_per_year"]  = df["km_driven"] / (df["car_age"] + 1)
df["log_km"]       = np.log1p(df["km_driven"])
df["age_sq"]       = df["car_age"] ** 2
df["km_age_ratio"] = df["km_driven"] * df["car_age"]

# ── 3. Clean ──────────────────────────────────────────────
df = df[df["selling_price"] > 10000]
df = df[df["km_driven"]     < 300000]
df = df[df["car_age"]       < 25]
df = df[df["selling_price"] < df["selling_price"].quantile(0.99)]
print(f"   Rows after cleaning  : {len(df)}")

# ── 4. Target (log-transform) ─────────────────────────────
df["price_log"] = np.log1p(df["selling_price"])

# ── 5. Features ───────────────────────────────────────────
features = [
    "brand", "fuel", "seller_type", "transmission", "owner",  # categorical
    "car_age", "km_driven", "km_per_year", "log_km",           # numeric
    "age_sq", "km_age_ratio"
]
categorical = ["brand", "fuel", "seller_type", "transmission", "owner"]
numeric     = ["car_age", "km_driven", "km_per_year", "log_km", "age_sq", "km_age_ratio"]

X = df[features]
y = df["price_log"]

# ── 6. Dropdown values for UI ─────────────────────────────
cat_unique = {
    "Brand":        sorted(df["brand"].unique().tolist()),
    "Fuel Type":    sorted(df["fuel"].unique().tolist()),
    "Seller Type":  sorted(df["seller_type"].unique().tolist()),
    "Transmission": sorted(df["transmission"].unique().tolist()),
    "Owner":        sorted(df["owner"].unique().tolist()),
}

# ── 7. Preprocessor ───────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ("num", "passthrough", numeric)
])

# ── 8. Model (best config from tuning) ───────────────────
model = GradientBoostingRegressor(
    n_estimators   = 1500,
    learning_rate  = 0.015,
    max_depth      = 7,
    subsample      = 0.85,
    max_features   = 0.8,
    min_samples_split = 4,
    min_samples_leaf  = 3,
    random_state   = 42
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model",        model)
])

# ── 9. Train ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\n🚀 Training model (this takes ~60 seconds)...")
pipeline.fit(X_train, y_train)

# ── 10. Evaluate ──────────────────────────────────────────
y_pred_log  = pipeline.predict(X_test)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

r2   = r2_score(y_test_real, y_pred_real)
mae  = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print("\n📊 Results:")
print(f"   R² Score : {r2:.4f}")
print(f"   MAE      : ₹{mae:,.0f}")
print(f"   RMSE     : ₹{rmse:,.0f}")

# ── 11. Save ──────────────────────────────────────────────
with open("model_artifacts.pkl", "wb") as f:
    pickle.dump({
        "pipeline":   pipeline,
        "r2":         round(float(r2),   4),
        "mae":        round(float(mae),  2),
        "rmse":       round(float(rmse), 2),
        "cat_unique": cat_unique,
    }, f)

print("\n✅ Saved → model_artifacts.pkl")
print(f"   Best R² achieved: {r2:.4f}")