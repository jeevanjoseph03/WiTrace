import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==========================================================
# LOAD CSI FILE
# ==========================================================
def load_csi(filename):
    data = []
    with open(filename, "r") as f:
        for line in f:
            if "CSI_DATA:" in line:
                parts = line.replace("CSI_DATA:", "").strip().split()
                try:
                    values = [int(x) for x in parts]
                    data.append(values)
                except:
                    pass

    min_len = min(len(row) for row in data)
    data = [row[:min_len] for row in data]
    return np.array(data)


# ==========================================================
# PREPROCESSING
# ==========================================================
def preprocess_csi(data):
    # Remove static component per subcarrier
    data = data - np.mean(data, axis=0)

    # Convert to magnitude
    data = np.abs(data)

    # Smooth noise across time
    data = gaussian_filter1d(data, sigma=2, axis=0)

    return data


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(data):
    processed = preprocess_csi(data)

    # Mean energy over time
    mean_energy = np.mean(np.abs(data))

    # Temporal variance (activity intensity)
    temporal_variance = np.var(processed)

    # Motion centroid path
    centroid_path = []
    for t in range(processed.shape[0]):
        frame = processed[t]
        total = np.sum(frame)

        if total == 0:
            centroid = 0
        else:
            centroid = np.sum(np.arange(len(frame)) * frame) / total

        centroid_path.append(centroid)

    centroid_path = np.array(centroid_path)
    centroid_path = gaussian_filter1d(centroid_path, sigma=2)

    motion_variance = np.var(centroid_path)

    return mean_energy, temporal_variance, motion_variance


# ==========================================================
# STATISTICAL PRESENCE CLASSIFIER
# ==========================================================
def classify_scenario(features, empty_features):

    mean_energy, temp_var, motion_var = features
    empty_mean, empty_temp_var, empty_motion_var = empty_features

    # Z-score normalization vs empty baseline
    z_energy = (mean_energy - empty_mean) / (empty_mean + 1e-6)
    z_motion = (motion_var - empty_motion_var) / (empty_motion_var + 1e-6)

    # Decision logic (statistically separated thresholds)
    if z_motion < 0.5:
        return "NO PERSON DETECTED", "High"

    elif 0.5 <= z_motion < 3:
        return "PERSON PRESENT (STILL)", "Medium-High"

    elif 3 <= z_motion < 8:
        return "PERSON WALKING", "High"

    else:
        return "MULTIPLE PEOPLE / HIGH ACTIVITY", "Very High"


# ==========================================================
# LOAD DATASETS
# ==========================================================
print("Loading CSI datasets...")

empty = load_csi("../data/empty.txt")
occupied = load_csi("../data/occupied.txt")
walking = load_csi("../data/walking.txt")
multi_occ = load_csi("../data/multi_occ.txt")

datasets = [
    ("Empty Room", empty),
    ("Occupied (Still)", occupied),
    ("Walking", walking),
    ("Multiple People", multi_occ)
]

# ==========================================================
# EXTRACT FEATURES
# ==========================================================
feature_results = {}

for name, data in datasets:
    feature_results[name] = extract_features(data)

empty_features = feature_results["Empty Room"]

# ==========================================================
# PRINT CARD FORMAT RESULTS
# ==========================================================
print("\n" + "="*60)
print("        CSI PRESENCE DETECTION RESULTS")
print("="*60 + "\n")

for name in feature_results:

    mean_energy, temp_var, motion_var = feature_results[name]

    detection, confidence = classify_scenario(
        feature_results[name],
        empty_features
    )

    print("-"*50)
    print(f" SCENARIO: {name}")
    print("-"*50)
    print(f" Mean CSI Energy      : {mean_energy:.2f}")
    print(f" Temporal Variance    : {temp_var:.2f}")
    print(f" Motion Variance      : {motion_var:.2f}")
    print(f" Person Detection     : {detection}")
    print(f" Confidence Level     : {confidence}")
    print("-"*50 + "\n")

print("Detection Complete.")

