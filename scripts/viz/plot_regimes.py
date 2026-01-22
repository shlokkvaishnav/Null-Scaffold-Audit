"""Plot regime discoveries: regimes, confidence, dynamics, and stability."""

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:
    print("Install cartopy: pip install cartopy")
    sys.exit(1)

from climate_discovery.config import CHECKPOINT_DIR, FIGURE_DIR, FUSED_NC, N_REGIMES
from climate_discovery.data.datasets import ClimateSpatialDataset
from climate_discovery.models.gating import GatingNetwork

# ======================================================
# Gating features
# ======================================================
FEATURES = [
    "lat_norm",
    "sin_lon",
    "cos_lon",
    "sst",
    "sss",
    "log_chl",
    "season_strength",
]

CHECKPOINT_PATH = CHECKPOINT_DIR / "gating_warmstart.pth"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    model = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return model


def get_probs(model, sample):
    img = sample["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        B, C, H, W = img.shape
        _, probs = model(img.permute(0, 2, 3, 1).reshape(-1, C))
    return probs.reshape(H, W, N_REGIMES).cpu().numpy()


# ======================================================
# FIGURE 1: Regimes + Confidence (Seasonal Snapshots)
# ======================================================
def plot_regimes_and_confidence(dataset, model):
    indices = [0, 6, 11]
    fig = plt.figure(figsize=(20, 12))

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        mask = sample["mask"].numpy()
        probs = get_probs(model, sample)

        regimes = np.argmax(probs, axis=2)
        confidence = np.max(probs, axis=2)

        ax = fig.add_subplot(2, 3, i + 1, projection=ccrs.Robinson())
        ax.set_title(f"Regimes (Step {idx})")
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="gray")
        ax.pcolormesh(
            dataset.ds.lon,
            dataset.ds.lat,
            np.ma.masked_where(~mask, regimes),
            transform=ccrs.PlateCarree(),
            cmap="tab10",
        )

        ax2 = fig.add_subplot(2, 3, i + 4, projection=ccrs.Robinson())
        ax2.set_title(f"Confidence (Step {idx})")
        ax2.coastlines()
        ax2.add_feature(cfeature.LAND, facecolor="gray")
        ax2.pcolormesh(
            dataset.ds.lon,
            dataset.ds.lat,
            np.ma.masked_where(~mask, confidence),
            transform=ccrs.PlateCarree(),
            cmap="plasma",
            vmin=0.4,
            vmax=1.0,
        )

    plt.tight_layout()
    path = FIGURE_DIR / "figure1_regimes_confidence.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)


# ======================================================
# FIGURE 2: Regime Transition Probability (Dynamic Fronts)
# ======================================================
def plot_transition_probability(dataset, model):
    T = len(dataset)
    sample0 = dataset[0]
    H, W = sample0["mask"].shape

    transitions = np.zeros((H, W))
    counts = np.zeros((H, W))

    prev = None
    for t in range(T):
        sample = dataset[t]
        mask = sample["mask"].numpy()
        probs = get_probs(model, sample)
        curr = np.argmax(probs, axis=2)

        if prev is not None:
            transitions += (curr != prev) * mask
            counts += mask

        prev = curr

    prob_change = transitions / (counts + 1e-6)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_title("Regime Transition Probability")
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="gray")
    m = ax.pcolormesh(
        dataset.ds.lon,
        dataset.ds.lat,
        prob_change,
        transform=ccrs.PlateCarree(),
        cmap="inferno",
        vmin=0,
        vmax=0.5,
    )
    plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.05)
    path = FIGURE_DIR / "figure2_transition_probability.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)


# ======================================================
# FIGURE 3: Latitudinal Regime Persistence
# ======================================================
def plot_latitudinal_persistence(dataset, model):
    T = len(dataset)
    lat = dataset.ds.lat.values

    persistence = np.zeros(len(lat))
    counts = np.zeros(len(lat))

    prev = None
    for t in range(T):
        sample = dataset[t]
        mask = sample["mask"].numpy()
        probs = get_probs(model, sample)
        curr = np.argmax(probs, axis=2)

        if prev is not None:
            same = (curr == prev) * mask
            persistence += same.sum(axis=1)
            counts += mask.sum(axis=1)

        prev = curr

    persistence /= (counts + 1e-6)

    plt.figure(figsize=(6, 5))
    plt.plot(lat, persistence)
    plt.xlabel("Latitude")
    plt.ylabel("Regime Persistence")
    plt.title("Latitudinal Regime Stability")
    plt.grid(True)

    path = FIGURE_DIR / "figure3_latitudinal_persistence.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)


# ======================================================
# FIGURE 4: Global Regime Usage (Entropy Insight)
# ======================================================
def plot_regime_usage(dataset, model):
    usage = np.zeros(N_REGIMES)

    for t in range(len(dataset)):
        sample = dataset[t]
        mask = sample["mask"].numpy()
        probs = get_probs(model, sample)
        for k in range(N_REGIMES):
            usage[k] += probs[..., k][mask].mean()

    usage /= usage.sum()

    plt.figure(figsize=(6, 4))
    plt.bar(range(N_REGIMES), usage)
    plt.xlabel("Regime ID")
    plt.ylabel("Mean Probability")
    plt.title("Global Regime Usage")

    path = FIGURE_DIR / "figure4_regime_usage.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)

# ======================================================
# ENSEMBLE UTILITIES
# ======================================================
ENSEMBLE_DIR = CHECKPOINT_DIR / "ensemble"

def load_ensemble_models():
    models = []
    for ckpt in sorted(ENSEMBLE_DIR.glob("seed_*/gating.pth")):
        m = GatingNetwork(input_dim=len(FEATURES), num_regimes=N_REGIMES).to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} ensemble members.")
    return models


def get_ensemble_probs(models, sample):
    img = sample["image"].unsqueeze(0).to(DEVICE)
    B, C, H, W = img.shape
    img_flat = img.permute(0, 2, 3, 1).reshape(-1, C)

    probs_all = []
    with torch.no_grad():
        for m in models:
            _, p = m(img_flat)
            probs_all.append(p.cpu().numpy())

    probs_mean = np.mean(probs_all, axis=0)
    return probs_mean.reshape(H, W, N_REGIMES), probs_all

# ======================================================
# FIGURE 5: Seasonal Mean Regimes (DJF vs JJA)
# ======================================================
def plot_seasonal_regimes(dataset, ensemble_models):
    seasons = {
        "DJF": [11, 0, 1],
        "JJA": [5, 6, 7],
    }

    fig = plt.figure(figsize=(14, 6))

    for i, (name, months) in enumerate(seasons.items()):
        probs_acc = []

        for t in months:
            probs_mean, _ = get_ensemble_probs(ensemble_models, dataset[t])
            probs_acc.append(probs_mean)

        mean_probs = np.mean(probs_acc, axis=0)
        regimes = np.argmax(mean_probs, axis=2)

        ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.Robinson())
        ax.set_title(f"{name} Mean Regimes")
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor="gray")
        ax.pcolormesh(
            dataset.ds.lon,
            dataset.ds.lat,
            regimes,
            transform=ccrs.PlateCarree(),
            cmap="tab10",
        )

    plt.tight_layout()
    path = FIGURE_DIR / "figure5_seasonal_regimes.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)

# ======================================================
# FIGURE 6: Ensemble Agreement Map
# ======================================================
def plot_ensemble_agreement(dataset, ensemble_models):
    sample = dataset[6]  # representative month
    probs_mean, probs_all = get_ensemble_probs(ensemble_models, sample)

    regime_maps = [
        np.argmax(p.reshape(*probs_mean.shape), axis=2)
        for p in probs_all
    ]

    regime_maps = np.stack(regime_maps, axis=0)
    agreement = np.mean(
        regime_maps == regime_maps[0:1], axis=0
    )

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_title("Ensemble Regime Agreement")
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="gray")
    m = ax.pcolormesh(
        dataset.ds.lon,
        dataset.ds.lat,
        agreement,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.05)

    path = FIGURE_DIR / "figure6_ensemble_agreement.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)


# ======================================================
# FIGURE 7: Front Displacement Magnitude
# ======================================================
def plot_front_displacement(dataset, ensemble_models):
    djf = [11, 0, 1]
    jja = [5, 6, 7]

    def seasonal_regime(months):
        acc = []
        for t in months:
            probs, _ = get_ensemble_probs(ensemble_models, dataset[t])
            acc.append(np.argmax(probs, axis=2))
        return np.mean(acc, axis=0)

    reg_djf = seasonal_regime(djf)
    reg_jja = seasonal_regime(jja)

    displacement = (reg_djf != reg_jja).astype(float)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_title("Seasonal Front Displacement (DJF → JJA)")
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="gray")
    m = ax.pcolormesh(
        dataset.ds.lon,
        dataset.ds.lat,
        displacement,
        transform=ccrs.PlateCarree(),
        cmap="magma",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.05)

    path = FIGURE_DIR / "figure7_front_displacement.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)

# ======================================================
# FIGURE 8: Seasonal Change in Regime Entropy (Dynamic Fronts)
# ======================================================
def plot_entropy_shift(dataset, ensemble_models):
    def entropy(p):
        return -np.sum(p * np.log(p + 1e-8), axis=2)

    seasons = {
        "DJF": [11, 0, 1],
        "JJA": [5, 6, 7],
    }

    entropy_maps = {}

    for name, months in seasons.items():
        ent_acc = []
        for t in months:
            probs_mean, _ = get_ensemble_probs(ensemble_models, dataset[t])
            ent_acc.append(entropy(probs_mean))
        entropy_maps[name] = np.mean(ent_acc, axis=0)

    delta_entropy = entropy_maps["JJA"] - entropy_maps["DJF"]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_title("Seasonal Change in Regime Entropy (JJA − DJF)")
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor="gray")

    m = ax.pcolormesh(
        dataset.ds.lon,
        dataset.ds.lat,
        delta_entropy,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=-0.3,
        vmax=0.3,
    )

    plt.colorbar(m, ax=ax, orientation="horizontal", pad=0.05, label="Δ Entropy")
    path = FIGURE_DIR / "figure8_entropy_shift.png"
    plt.savefig(path, dpi=300)
    print("Saved", path)


# ======================================================
# MAIN
# ======================================================
def main():
    model = load_model()
    dataset = ClimateSpatialDataset(str(FUSED_NC), FEATURES, mode="train")
    ensemble_models = load_ensemble_models()

    plot_regimes_and_confidence(dataset, model)
    plot_transition_probability(dataset, model)
    plot_latitudinal_persistence(dataset, model)
    plot_regime_usage(dataset, model)

    plot_seasonal_regimes(dataset, ensemble_models)
    plot_ensemble_agreement(dataset, ensemble_models)
    plot_front_displacement(dataset, ensemble_models)
    plot_entropy_shift(dataset, ensemble_models)


    print("\nALL DISCOVERY FIGURES GENERATED SUCCESSFULLY.\n")

if __name__ == "__main__":
    main()