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

    min_len = min(len(row) for row in data)

    data = [row[:min_len] for row in data]

    return np.array(data)


# ============================================
# NORMALIZE FOR DISPLAY
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

    data = data - np.mean(data, axis=0)

    data = np.abs(data)

    data = gaussian_filter1d(data, sigma=2, axis=0)

    return data


# ============================================
# COMPUTE MOTION PATH
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

    path = gaussian_filter1d(path, sigma=2)

    return path


# ============================================
# LOAD DATASETS
# ============================================

print("Loading CSI datasets...")

empty = load_csi("../data/empty.txt")
occupied = load_csi("../data/occupied.txt")
walking = load_csi("../data/walking.txt")
wall = load_csi("../data/wall.txt")

datasets = [
    ("Empty Room", empty),
    ("Occupied (Still)", occupied),
    ("Walking", walking),
    ("Behind Wall", wall)
]

# ============================================
# ENERGY COMPARISON GRAPH
# ============================================
empty_energy = np.mean(np.abs(empty), axis=1)
occupied_energy = np.mean(np.abs(occupied), axis=1)
walking_energy = np.mean(np.abs(walking), axis=1)
wall_energy = np.mean(np.abs(wall), axis=1)

plt.figure(figsize=(12,6))
plt.plot(empty_energy, label="Empty Room")
plt.plot(occupied_energy, label="Occupied (Still)")
plt.plot(walking_energy, label="Walking")
plt.plot(wall_energy, label="Behind Wall")
plt.title("CSI Energy Comparison")
plt.xlabel("Time Frame")
plt.ylabel("CSI Energy")
plt.legend()
plt.grid()
plt.show()


# ============================================
# ENERGY GRAPHS — SEPARATE PLOTS
# ============================================

energy_sets = [
    ("Empty Room", np.mean(np.abs(empty), axis=1)),
    ("Occupied (Still)", np.mean(np.abs(occupied), axis=1)),
    ("Walking", np.mean(np.abs(walking), axis=1)),
    ("Behind Wall", np.mean(np.abs(wall), axis=1))
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (name, energy) in zip(axes.flat, energy_sets):
    ax.plot(energy)
    ax.set_title(f"{name} — CSI Energy")
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("CSI Energy")
    ax.grid(True)

plt.suptitle("CSI Energy — Individual Dataset Analysis", fontsize=15)
plt.tight_layout()
plt.show()




# ============================================
# HEATMAPS WITH PROPER ALIGNMENT
# ============================================

empty_n = normalize_for_display(empty)
occupied_n = normalize_for_display(occupied)
walking_n = normalize_for_display(walking)
wall_n = normalize_for_display(wall)

global_min = min(empty_n.min(), occupied_n.min(), walking_n.min(), wall_n.min())
global_max = max(empty_n.max(), occupied_n.max(), walking_n.max(), wall_n.max())

fig = plt.figure(figsize=(14,10))

gs = fig.add_gridspec(2, 3, width_ratios=[1,1,0.05])

axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[0,1]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[1,1])
]

cbar_ax = fig.add_subplot(gs[:,2])

normalized_sets = [
    ("Empty Room", empty_n),
    ("Occupied (Still)", occupied_n),
    ("Walking", walking_n),
    ("Behind Wall", wall_n)
]

for ax, (name, data) in zip(axes, normalized_sets):

    im = ax.imshow(
        data.T,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r',
        vmin=global_min,
        vmax=global_max
    )

    ax.set_title(name)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Subcarrier Index")

fig.colorbar(im, cax=cbar_ax)
cbar_ax.set_ylabel("Normalized CSI Amplitude")

plt.suptitle("CSI Heatmaps")

plt.show()


# ============================================
# MOTION TRACKING FOR ALL DATASETS
# ============================================

motion_paths = []

for name, data in datasets:

    path = compute_motion_path(data)

    motion_paths.append((name, path))


# ============================================
# HEATMAP + MOTION PATH OVERLAY
# ============================================

fig, axes = plt.subplots(2,2, figsize=(14,10))

for ax, (name, data), (_, path) in zip(axes.flat, datasets, motion_paths):

    display = normalize_for_display(data)

    im = ax.imshow(
        display.T,
        aspect='auto',
        origin='lower',
        cmap='RdBu_r'
    )

    ax.plot(path, color='lime', linewidth=2)

    ax.set_title(name)
    ax.set_xlabel("Time Frame")
    ax.set_ylabel("Subcarrier Index")

fig.colorbar(im, ax=axes)
plt.suptitle("Motion Tracking Comparison")

plt.show()


# ============================================
# MOTION PATH COMPARISON GRAPH
# ============================================

plt.figure(figsize=(12,6))

for name, path in motion_paths:

    plt.plot(path, label=name)

plt.title("Motion Path Comparison Across All Datasets")

plt.xlabel("Time Frame")
plt.ylabel("Detected Motion Position")

plt.legend()
plt.grid()

plt.show()
