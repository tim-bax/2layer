#!/usr/bin/env python3
"""
Scatter plot of a single random NMNIST sample.

Shows two views of the same sample:
  1. Spatial scatter (x, y) — colored by spike time, polarity by marker.
  2. Raster (time, pixel index) — colored by polarity.

NMNIST data path resolution order:
  $NMNIST_DATA_PATH > /share/neurocomputation/Tim/NMNIST_data
  > ~/Documents/NMNIST_data
"""
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import NMNISTDataLoader  # noqa: E402


def resolve_data_path() -> str:
    candidates = [
        os.environ.get("NMNIST_DATA_PATH"),
        "/share/neurocomputation/Tim/NMNIST_data",
        os.path.expanduser("~/Documents/NMNIST_data"),
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        "NMNIST data not found. Set NMNIST_DATA_PATH or place data at "
        "~/Documents/NMNIST_data"
    )


def pick_random_sample(data_path: str, split: str = "Train"):
    """Pick one random (label, file) pair from the dataset on disk."""
    split_dir = os.path.join(data_path, split)
    classes = sorted(d for d in os.listdir(split_dir)
                     if os.path.isdir(os.path.join(split_dir, d)))
    label = random.choice(classes)
    class_dir = os.path.join(split_dir, label)
    files = sorted(os.listdir(class_dir))
    fname = random.choice(files)
    return int(label), os.path.join(class_dir, fname)


def decode_pixel(pixel_idx: int, n_x: int = 34, n_y: int = 34):
    """Inverse of polarity*n_x*n_y + y*n_x + x ⇒ (polarity, y, x)."""
    polarity = pixel_idx // (n_x * n_y)
    rem = pixel_idx % (n_x * n_y)
    y = rem // n_x
    x = rem % n_x
    return polarity, y, x


def sample_to_events(image_pixels):
    """Convert list-of-arrays (per-pixel spike-times) to flat event arrays."""
    xs, ys, ts, ps = [], [], [], []
    for pixel_idx, spike_times in enumerate(image_pixels):
        if len(spike_times) == 0:
            continue
        pol, y, x = decode_pixel(pixel_idx)
        xs.extend([x] * len(spike_times))
        ys.extend([y] * len(spike_times))
        ts.extend(spike_times.tolist())
        ps.extend([pol] * len(spike_times))
    return (
        np.asarray(xs, dtype=np.int32),
        np.asarray(ys, dtype=np.int32),
        np.asarray(ts, dtype=np.float64),
        np.asarray(ps, dtype=np.int32),
    )


def main():
    random.seed()  # truly random each run
    data_path = resolve_data_path()
    print(f"NMNIST data: {data_path}")

    label, file_path = pick_random_sample(data_path, split="Train")
    print(f"Picked label {label}: {os.path.basename(file_path)}")

    loader = NMNISTDataLoader(data_path, duration_ms=300)
    image_pixels, _ = loader.load_image(file_path)
    xs, ys, ts, ps = sample_to_events(image_pixels)
    print(f"Total events: {len(ts)}")

    fig, (ax_sp, ax_ra) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1. Spatial scatter: (x, y) colored by time, polarity by marker
    on_mask = ps == 0
    off_mask = ps == 1
    sc_on = ax_sp.scatter(
        xs[on_mask], ys[on_mask], c=ts[on_mask], s=12, marker="o",
        cmap="viridis", alpha=0.8, label=f"ON ({on_mask.sum()})",
    )
    ax_sp.scatter(
        xs[off_mask], ys[off_mask], c=ts[off_mask], s=12, marker="x",
        cmap="viridis", alpha=0.8, label=f"OFF ({off_mask.sum()})",
    )
    ax_sp.set_xlim(-1, 34)
    ax_sp.set_ylim(34, -1)  # flip y so origin is top-left like an image
    ax_sp.set_aspect("equal")
    ax_sp.set_xlabel("x (pixel)")
    ax_sp.set_ylabel("y (pixel)")
    ax_sp.set_title(f"NMNIST sample — label {label} (spatial)")
    ax_sp.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(sc_on, ax=ax_sp, fraction=0.046, pad=0.04)
    cb.set_label("spike time (ms)")
    ax_sp.grid(True, alpha=0.2)

    # 2. Raster: time vs pixel index, colored by polarity
    pixel_idx_per_event = ps * (34 * 34) + ys * 34 + xs
    ax_ra.scatter(
        ts[on_mask], pixel_idx_per_event[on_mask], s=2,
        color="C0", alpha=0.6, label="ON",
    )
    ax_ra.scatter(
        ts[off_mask], pixel_idx_per_event[off_mask], s=2,
        color="C3", alpha=0.6, label="OFF",
    )
    ax_ra.axhline(34 * 34, color="k", linewidth=0.5, alpha=0.5)
    ax_ra.set_xlabel("time (ms)")
    ax_ra.set_ylabel("pixel index (ON: 0–1155, OFF: 1156–2311)")
    ax_ra.set_title(f"NMNIST sample — label {label} (raster)")
    ax_ra.set_xlim(0, 300)
    ax_ra.set_ylim(0, 34 * 34 * 2)
    ax_ra.legend(loc="upper right", fontsize=9, markerscale=3)
    ax_ra.grid(True, alpha=0.2)

    fig.tight_layout()
    out_path = os.path.join(_SCRIPT_DIR, "nmnist_sample.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
