import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============================================
# LOAD CSI FILE
# ============================================
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

    if len(data) == 0:
        raise ValueError(f"No CSI data found in {filename}")

    # Make all rows equal length (subcarrier consistency)
    min_len = min(len(row) for row in data)
    data = [row[:min_len] for row in data]

    return np.array(data)


# ============================================
# NORMALIZE FOR DISPLAY (if needed later)
# ============================================
def normalize_for_display(data):
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        std = 1
    return (data - mean) / std


# ============================================
# PREPROCESS FOR MOTION TRACKING
# ============================================
def preprocess_csi(data):
    # Remove static component (per subcarrier)
    data = data - np.mean(data, axis=0)

    # Convert to magnitude
    data = np.abs(data)

    # Smooth noise over time
    data = gaussian_filter1d(data, sigma=2, axis=0)

    return data


# ============================================
# COMPUTE MOTION PATH (CENTROID METHOD)
# ============================================
def compute_motion_path(data):
    processed = preprocess_csi(data)

    path = []

    for t in range(processed.shape[0]):
        signal = processed[t]
        total_energy = np.sum(signal)

        if total_energy == 0:
            centroid = 0
        else:
            indices = np.arange(len(signal))
            centroid = np.sum(indices * signal) / total_energy

        path.append(centroid)

    path = np.array(path)

    # Smooth final motion curve
    path = gaussian_filter1d(path, sigma=2)

    return path


# ============================================
# LOAD DATASETS
# ============================================
print("Loading CSI datasets...")

empty = load_csi("../data/empty.txt")
occupied = load_csi("../data/occupied.txt")
walking = load_csi("../data/walking.txt")
multi_occ = load_csi("../data/multi_occ.txt")

# ============================================
# FORCE ALL DATASETS TO SAME TIME LENGTH
# ============================================
min_time = min(
    empty.shape[0],
    occupied.shape[0],
    walking.shape[0],
    multi_occ.shape[0]
)

empty = empty[:min_time]
occupied = occupied[:min_time]
walking = walking[:min_time]
multi_occ = multi_occ[:min_time]

print("All datasets trimmed to:", min_time, "time frames")

datasets = [
    ("Empty Room", empty),
    ("Occupied (Still)", occupied),
    ("Walking", walking),
    ("Multiple People", multi_occ)
]


# ============================================
# ENERGY COMPARISON (COMBINED GRAPH)
# ============================================
empty_energy = np.mean(np.abs(empty), axis=1)
occupied_energy = np.mean(np.abs(occupied), axis=1)
walking_energy = np.mean(np.abs(walking), axis=1)
multi_occ_energy = np.mean(np.abs(multi_occ), axis=1)

plt.figure(figsize=(12,6))
plt.plot(empty_energy, label="Empty Room")
plt.plot(occupied_energy, label="Occupied (Still)")
plt.plot(walking_energy, label="Walking")
plt.plot(multi_occ_energy, label="Multiple People")
plt.title("CSI Energy Comparison")
plt.xlabel("Time Frame")
plt.ylabel("CSI Energy")
plt.legend()
plt.grid()
plt.show()


# ============================================
# ENERGY GRAPHS — SEPARATE
# ============================================
energy_sets = [
    ("Empty Room", empty_energy),
    ("Occupied (Still)", occupied_energy),
    ("Walking", walking_energy),
    ("Multiple People", multi_occ_energy)
]

fig, axes = plt.subplots(2, 2, figsize=(14,10))

for ax, (name, energy) in zip(axes.flat, energy_sets):
    ax.plot(energy)
    ax.set_title(f"{name} — CSI Energy")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("CSI Energy")
    ax.grid(True)

plt.suptitle("CSI Energy — Individual Dataset Analysis", fontsize=14)
plt.tight_layout()
plt.show()


# ============================================
# MOTION TRACKING
# ============================================
motion_paths = []

for name, data in datasets:
    path = compute_motion_path(data)
    motion_paths.append((name, path))



# ============================================
# MOTION PATH COMPARISON GRAPH
# ============================================
plt.figure(figsize=(12,6))

for name, path in motion_paths:
    plt.plot(path, label=name)

plt.title("Motion Path Comparison Across Datasets")
plt.xlabel("Time Frame")
plt.ylabel("Detected Motion Position (Subcarrier Index)")
plt.legend()
plt.grid()
plt.show()

# ============================================
# SCATTER PLOTS — SEPARATE GRAPHS
# ============================================

scatter_sets = [
    ("Empty Room", empty_energy, "blue"),
    ("Occupied (Still)", occupied_energy, "green"),
    ("Walking", walking_energy, "orange"),
    ("Multiple People", multi_occ_energy, "red")
]

fig, axes = plt.subplots(2, 2, figsize=(14,10))

for ax, (name, energy, color) in zip(axes.flat, scatter_sets):
    ax.scatter(
        np.arange(len(energy)),
        energy,
        s=12,
        alpha=0.6,
        color=color
    )
    ax.set_title(f"{name} — CSI Energy Scatter")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("CSI Energy")
    ax.grid(True)

plt.suptitle("CSI Energy Scatter Comparison (All Scenarios)", fontsize=14)
plt.tight_layout()
plt.show()

