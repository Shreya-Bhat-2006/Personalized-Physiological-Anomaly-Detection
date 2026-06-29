import pandas as pd
import numpy as np
import time
import os
import joblib
import gc
import json
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# -------------------------
# SET RANDOM SEED FOR REPRODUCIBILITY
# -------------------------
np.random.seed(42)

# -------------------------
# FUNCTION: MEASURE MODEL SIZE (consistent with saving)
# -------------------------
def measure_model_size(model, temp_file):
    """Measure serialized model size in KB using same method as saving"""
    joblib.dump(model, temp_file)
    size_bytes = os.path.getsize(temp_file)
    os.remove(temp_file)
    return size_bytes / 1024  # KB

# -------------------------
# FUNCTION: MEASURE INFERENCE TIME (1000 runs for stability)
# -------------------------
def measure_inference_time(model, X_data, n_runs=1000):
    """
    Measure average inference time in ms.
    Predicts ALL data each run for stable measurement.
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        if hasattr(model, 'predict'):
            model.predict(X_data)
        elif hasattr(model, 'fit_predict'):
            model.fit_predict(X_data)
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)

# -------------------------
# FUNCTION: SAVE MODEL
# -------------------------
def save_model(model, filepath):
    """Save model using joblib"""
    joblib.dump(model, filepath)
    print(f"  ✓ Saved: {filepath}")

# -------------------------
# LOAD DATASET
# -------------------------
print("="*70)
print("PHASE 1: IMPLEMENTATION CHARACTERISTICS")
print("="*70)
print()

df = pd.read_csv("feature_engineered_dataset.csv")

print(f"Dataset shape: {df.shape}")
print()

# -------------------------
# FEATURES (13 features)
# -------------------------
features = [
    "HR_Zscore",
    "Sleep_Zscore",
    "Steps_Zscore",
    "Deep_Sleep",
    "Calories",
    "Moderate_Activity",
    "Vigorous_Activity",
    "Fluid_Intake",
    "Stress",
    "Fatigue",
    "Mood",
    "Readiness",
    "Restlessness"
]

print(f"Number of features: {len(features)}")
print()

# -------------------------
# PREPARE DATA (Use all 132 samples - NO SPLIT)
# -------------------------
print("Preparing data...")

# Handle missing values
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(df[features])

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

print(f"Training samples: {len(X_scaled)}")
print()

# -------------------------
# SAVE PREPROCESSING OBJECTS
# -------------------------
print("Saving preprocessing objects...")
save_model(imputer, "imputer.pkl")
save_model(scaler, "scaler.pkl")
save_model(features, "feature_list.pkl")
print()

# -------------------------
# TRAIN AND MEASURE ALL MODELS
# -------------------------

# Store results
results = []

# -------------------------
# 1. ISOLATION FOREST
# -------------------------
print("="*70)
print("1. Isolation Forest")
print("="*70)

# Model parameters
IF_ESTIMATORS = 100
IF_CONTAMINATION = 0.05
IF_RANDOM_STATE = 42

print(f"Parameters:")
print(f"  n_estimators: {IF_ESTIMATORS}")
print(f"  contamination: {IF_CONTAMINATION}")
print(f"  random_state: {IF_RANDOM_STATE}")
print()

print("Training...")
gc.collect()

# Create and train model (ONCE)
iso_model = IsolationForest(
    n_estimators=IF_ESTIMATORS,
    contamination=IF_CONTAMINATION,
    random_state=IF_RANDOM_STATE,
    max_samples='auto'
)

# Measure training time
start = time.perf_counter()
iso_model.fit(X_scaled)
train_time = (time.perf_counter() - start) * 1000

# Measure model size (using same method as saving)
model_size = measure_model_size(iso_model, "temp_isolation_forest.pkl")

# Measure inference time (1000 runs for stability)
inference_time = measure_inference_time(iso_model, X_scaled, n_runs=1000)

print(f"  Training Time: {train_time:.2f} ms")
print(f"  Model Size: {model_size:.2f} KB")
print(f"  Inference Time: {inference_time:.4f} ms")

# Save model
save_model(iso_model, "isolation_forest.pkl")

results.append({
    "Model": "Isolation Forest",
    "Training Time (ms)": train_time,
    "Inference Time (ms)": inference_time,
    "Model Size (KB)": model_size,
    "Training Samples": len(X_scaled),
    "Number of Features": len(features)
})

print()

# -------------------------
# 2. DBSCAN
# -------------------------
print("="*70)
print("2. DBSCAN")
print("="*70)

# Model parameters
DBSCAN_EPS = 3.5
DBSCAN_MIN_SAMPLES = 5

print(f"Parameters:")
print(f"  eps: {DBSCAN_EPS}")
print(f"  min_samples: {DBSCAN_MIN_SAMPLES}")
print(f"  metric: euclidean")
print()

print("Training...")
gc.collect()

# Create and train model (ONCE)
dbscan_model = DBSCAN(
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
    metric='euclidean'
)

# Measure training time
start = time.perf_counter()
dbscan_model.fit(X_scaled)
train_time = (time.perf_counter() - start) * 1000

# Measure model size (using same method as saving)
model_size = measure_model_size(dbscan_model, "temp_dbscan.pkl")

# Define prediction function for DBSCAN (Custom Heuristic)
def predict_dbscan(X_data, X_new, eps, min_samples):
    """Nearest-cluster assignment heuristic for DBSCAN"""
    cluster_labels = dbscan_model.labels_
    non_noise_mask = cluster_labels != -1
    X_clusters = X_data[non_noise_mask]
    
    if len(X_clusters) == 0:
        return -1
    
    unique_clusters = np.unique(cluster_labels[non_noise_mask])
    
    for cluster_id in unique_clusters:
        cluster_points = X_clusters[cluster_labels[non_noise_mask] == cluster_id]
        distances = np.linalg.norm(cluster_points - X_new, axis=1)
        if np.min(distances) <= eps:
            return 1
    
    distances_to_all = np.linalg.norm(X_data - X_new, axis=1)
    if np.sum(distances_to_all <= eps) >= min_samples:
        return 1
    
    return -1

# Measure inference time for DBSCAN (Custom Heuristic)
# Note: DBSCAN has no native predict() - this measures the custom heuristic
times = []
for _ in range(1000):
    start = time.perf_counter()
    for x in X_scaled:
        predict_dbscan(X_scaled, x, DBSCAN_EPS, DBSCAN_MIN_SAMPLES)
    times.append((time.perf_counter() - start) * 1000)
inference_time = np.mean(times)

print(f"  Training Time: {train_time:.2f} ms")
print(f"  Model Size: {model_size:.2f} KB")
print(f"  Inference Time (Custom Heuristic): {inference_time:.4f} ms")

# Save model
save_model(dbscan_model, "dbscan.pkl")

results.append({
    "Model": "DBSCAN (Custom Heuristic)",
    "Training Time (ms)": train_time,
    "Inference Time (ms)": inference_time,
    "Model Size (KB)": model_size,
    "Training Samples": len(X_scaled),
    "Number of Features": len(features)
})

print()

# -------------------------
# 3. LOCAL OUTLIER FACTOR
# -------------------------
print("="*70)
print("3. Local Outlier Factor (LOF)")
print("="*70)

# Model parameters
LOF_NEIGHBORS = 20
LOF_CONTAMINATION = 0.05
LOF_NOVELTY = True

print(f"Parameters:")
print(f"  n_neighbors: {LOF_NEIGHBORS}")
print(f"  contamination: {LOF_CONTAMINATION}")
print(f"  novelty: {LOF_NOVELTY}")
print()

print("Training...")
gc.collect()

# Create and train model (ONCE)
lof_model = LocalOutlierFactor(
    n_neighbors=LOF_NEIGHBORS,
    contamination=LOF_CONTAMINATION,
    novelty=LOF_NOVELTY
)

# Measure training time
start = time.perf_counter()
lof_model.fit(X_scaled)
train_time = (time.perf_counter() - start) * 1000

# Measure model size (using same method as saving)
model_size = measure_model_size(lof_model, "temp_lof.pkl")

# Measure inference time (1000 runs for stability)
inference_time = measure_inference_time(lof_model, X_scaled, n_runs=1000)

print(f"  Training Time: {train_time:.2f} ms")
print(f"  Model Size: {model_size:.2f} KB")
print(f"  Inference Time: {inference_time:.4f} ms")

# Save model
save_model(lof_model, "lof.pkl")

results.append({
    "Model": "LOF",
    "Training Time (ms)": train_time,
    "Inference Time (ms)": inference_time,
    "Model Size (KB)": model_size,
    "Training Samples": len(X_scaled),
    "Number of Features": len(features)
})

print()

# -------------------------
# CREATE METADATA
# -------------------------
metadata = {
    "dataset_rows": len(df),
    "features": len(features),
    "feature_names": features,
    "dbscan_eps": DBSCAN_EPS,
    "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
    "lof_neighbors": LOF_NEIGHBORS,
    "lof_contamination": LOF_CONTAMINATION,
    "lof_novelty": LOF_NOVELTY,
    "if_estimators": IF_ESTIMATORS,
    "if_contamination": IF_CONTAMINATION,
    "if_random_state": IF_RANDOM_STATE
}

with open("phase1_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata saved: phase1_metadata.json")
print()

# -------------------------
# COMPARISON TABLE
# -------------------------
print("="*70)
print("PHASE 1 RESULTS: COMPARISON TABLE")
print("="*70)

results_df = pd.DataFrame(results)

# Sort by Model name for cleaner display
results_df = results_df.sort_values("Model").reset_index(drop=True)

# Format for display
display_df = results_df.copy()
display_df["Training Time (ms)"] = display_df["Training Time (ms)"].apply(lambda x: f"{x:.2f}")
display_df["Inference Time (ms)"] = display_df["Inference Time (ms)"].apply(lambda x: f"{x:.4f}")
display_df["Model Size (KB)"] = display_df["Model Size (KB)"].apply(lambda x: f"{x:.2f}")

print(display_df.to_string(index=False))

# -------------------------
# SAVE RESULTS
# -------------------------
results_df.to_csv("phase1_results.csv", index=False)
print(f"\n✓ Results saved to: phase1_results.csv")

# Clean up temp files
for f in os.listdir('.'):
    if f.startswith('temp_') and f.endswith('.pkl'):
        try:
            os.remove(f)
        except:
            pass

print()
print("="*70)
print("PHASE 1 COMPLETE")
print("="*70)
print()
print("Files saved:")
print("  - imputer.pkl")
print("  - scaler.pkl")
print("  - feature_list.pkl")
print("  - isolation_forest.pkl")
print("  - dbscan.pkl")
print("  - lof.pkl")
print("  - phase1_results.csv")
print("  - phase1_metadata.json")
print()
print("Measurements:")
print(f"  - Training Time: Measured for each model")
print(f"  - Inference Time: Measured for each model (1000 runs)")
print(f"    Note: DBSCAN uses a Custom Heuristic (no native predict)")
print(f"  - Model Size: Measured for each model")
print(f"  - Training Samples: {len(X_scaled)}")
print(f"  - Number of Features: {len(features)}")
print()
print("Ready for Phase 2: Synthetic Dataset Evaluation")
print("="*70)